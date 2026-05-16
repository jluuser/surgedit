#!/usr/bin/env python3
"""Build Stage-1 segment deletion pretraining data from clean FASTA sequences.

The generated task is: delete inserted residues from a biologically inspired
corrupted sequence to recover the original UniRef sequence.
"""

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict


DEFAULT_CONFIG = {
    "budgets": [0.1, 0.2, 0.3],
    "min_insert_segment_len": 1,
    "max_insert_segment_len": 40,
    "corruption_types": [
        "random_segment_insertion",
        "terminal_tail_insertion",
        "linker_like_insertion",
        "low_complexity_insertion",
        "local_duplication_insertion",
    ],
    "amino_acid_alphabet": "ACDEFGHIKLMNPQRSTVWY",
    "linker_like_tokens": ["G", "S", "P", "N", "Q"],
    "terminal_tail_bias": 0.2,
    "linker_like_bias": 0.25,
    "low_complexity_bias": 0.2,
    "local_duplication_bias": 0.25,
    "random_segment_bias": 0.1,
    "max_repeat_fraction": 0.6,
    "seed": 42,
}


TYPE_TO_BIAS = {
    "random_segment_insertion": "random_segment_bias",
    "terminal_tail_insertion": "terminal_tail_bias",
    "linker_like_insertion": "linker_like_bias",
    "low_complexity_insertion": "low_complexity_bias",
    "local_duplication_insertion": "local_duplication_bias",
}


def parse_scalar(value):
    value = value.strip()
    if not value:
        return ""
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def load_config(path):
    try:
        import yaml  # type: ignore

        with open(path) as handle:
            loaded = yaml.safe_load(handle) or {}
        config = dict(DEFAULT_CONFIG)
        config.update(loaded)
        return config
    except ImportError:
        config = dict(DEFAULT_CONFIG)
        current_key = None
        with open(path) as handle:
            for raw_line in handle:
                line = raw_line.rstrip()
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("- ") and current_key:
                    config.setdefault(current_key, []).append(parse_scalar(stripped[2:]))
                    continue
                if ":" not in stripped:
                    continue
                key, value = stripped.split(":", 1)
                key = key.strip()
                value = value.strip()
                current_key = key
                if value:
                    config[key] = parse_scalar(value)
                else:
                    config[key] = []
        return config


def read_fasta(path):
    records = []
    header = None
    chunks = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            records.append((header, "".join(chunks)))
    return records


def seq_id_from_header(header):
    return header.split(None, 1)[0]


def weighted_choice(corruption_types, config, rng):
    weights = [float(config.get(TYPE_TO_BIAS.get(name, ""), 0.0)) for name in corruption_types]
    if sum(weights) <= 0:
        weights = [1.0 for _ in corruption_types]
    return rng.choices(corruption_types, weights=weights, k=1)[0]


def random_segment(length, alphabet, rng):
    return "".join(rng.choice(alphabet) for _ in range(length))


def terminal_tail_segment(length, alphabet, rng):
    tail_tokens = list("GPSTNQDEKRS")
    segment = []
    for _ in range(length):
        if rng.random() < 0.65:
            segment.append(rng.choice(tail_tokens))
        else:
            segment.append(rng.choice(alphabet))
    return "".join(segment)


def linker_like_segment(length, linker_tokens, alphabet, rng):
    segment = []
    for _ in range(length):
        if rng.random() < 0.85:
            segment.append(rng.choice(linker_tokens))
        else:
            segment.append(rng.choice(alphabet))
    return "".join(segment)


def low_complexity_segment(length, alphabet, max_repeat_fraction, rng):
    if length <= 1:
        return rng.choice(alphabet)
    max_repeat = max(1, int(math.ceil(length * max_repeat_fraction)))
    for _ in range(100):
        token_count = rng.randint(2, min(4, len(alphabet)))
        tokens = rng.sample(alphabet, token_count)
        weights = [rng.random() + 0.2 for _ in tokens]
        segment = "".join(rng.choices(tokens, weights=weights, k=length))
        if max(Counter(segment).values()) <= max_repeat:
            return segment
    # Deterministic fallback that respects the repeat cap better than a homopolymer.
    tokens = rng.sample(alphabet, min(3, len(alphabet)))
    return "".join(tokens[i % len(tokens)] for i in range(length))


def local_duplication_segment(sequence, length, rng):
    if length <= len(sequence):
        source_start = rng.randint(0, len(sequence) - length)
        return sequence[source_start : source_start + length], source_start
    repeats = []
    while sum(len(part) for part in repeats) < length:
        source_len = min(len(sequence), length - sum(len(part) for part in repeats))
        source_start = rng.randint(0, len(sequence) - source_len)
        repeats.append(sequence[source_start : source_start + source_len])
    return "".join(repeats)[:length], None


def choose_insert_position(corruption_type, sequence, segment_len, source_start, rng):
    original_len = len(sequence)
    if corruption_type == "terminal_tail_insertion":
        return 0 if rng.random() < 0.5 else original_len
    if corruption_type == "local_duplication_insertion" and source_start is not None:
        center = source_start + segment_len
        low = max(0, center - 20)
        high = min(original_len, center + 20)
        return rng.randint(low, high)
    return rng.randint(0, original_len)


def build_inserted_segment(corruption_type, sequence, segment_len, config, rng):
    alphabet = list(str(config["amino_acid_alphabet"]))
    linker_tokens = list(config["linker_like_tokens"])
    source_start = None
    if corruption_type == "random_segment_insertion":
        inserted = random_segment(segment_len, alphabet, rng)
    elif corruption_type == "terminal_tail_insertion":
        inserted = terminal_tail_segment(segment_len, alphabet, rng)
    elif corruption_type == "linker_like_insertion":
        inserted = linker_like_segment(segment_len, linker_tokens, alphabet, rng)
    elif corruption_type == "low_complexity_insertion":
        inserted = low_complexity_segment(segment_len, alphabet, float(config["max_repeat_fraction"]), rng)
    elif corruption_type == "local_duplication_insertion":
        inserted, source_start = local_duplication_segment(sequence, segment_len, rng)
    else:
        raise ValueError("Unknown corruption type: {}".format(corruption_type))
    return inserted, source_start


def segment_lengths_for_budget(target_len, config, rng):
    min_len = int(config["min_insert_segment_len"])
    max_len = int(config["max_insert_segment_len"])
    remaining = target_len
    lengths = []
    while remaining > 0:
        if remaining <= max_len:
            seg_len = remaining
        else:
            seg_len = rng.randint(min_len, min(max_len, remaining))
        lengths.append(seg_len)
        remaining -= seg_len
    return lengths


def corrupt_sequence(seq_id, sequence, budget_ratio, config, rng):
    target_insert_len = int(math.floor(len(sequence) * budget_ratio))
    target_insert_len = max(1, target_insert_len)
    corruption_types = list(config["corruption_types"])
    original_pieces = [{"kind": "original", "text": aa, "orig_index": idx} for idx, aa in enumerate(sequence)]
    insertion_segments = []
    type_counts = Counter()

    for segment_idx, seg_len in enumerate(segment_lengths_for_budget(target_insert_len, config, rng)):
        corruption_type = weighted_choice(corruption_types, config, rng)
        inserted, source_start = build_inserted_segment(corruption_type, sequence, seg_len, config, rng)
        insert_pos_original = choose_insert_position(corruption_type, sequence, seg_len, source_start, rng)

        # Map original-coordinate insert position to current corrupted-coordinate slot.
        if insert_pos_original == len(sequence):
            current_insert_pos = len(original_pieces)
        else:
            current_insert_pos = next(
                i for i, piece in enumerate(original_pieces)
                if piece.get("kind") == "original" and piece.get("orig_index") == insert_pos_original
            )
        inserted_pieces = [
            {
                "kind": "inserted",
                "text": aa,
                "segment_idx": segment_idx,
                "corruption_type": corruption_type,
            }
            for aa in inserted
        ]
        original_pieces[current_insert_pos:current_insert_pos] = inserted_pieces
        corrupted_start = current_insert_pos
        corrupted_end_exclusive = current_insert_pos + len(inserted)
        insertion_segments.append(
            {
                "segment_idx": segment_idx,
                "corruption_type": corruption_type,
                "inserted_sequence": inserted,
                "inserted_length": len(inserted),
                "original_insert_pos": insert_pos_original,
                "corrupted_start": corrupted_start,
                "corrupted_end_exclusive": corrupted_end_exclusive,
                "source_start": source_start,
                "source_end_exclusive": None if source_start is None else source_start + len(inserted),
            }
        )
        type_counts[corruption_type] += 1

    corrupted_sequence = "".join(piece["text"] for piece in original_pieces)
    delete_labels = [1 if piece["kind"] == "inserted" else 0 for piece in original_pieces]
    keep_labels = [1 - value for value in delete_labels]
    recovered = "".join(aa for aa, label in zip(corrupted_sequence, delete_labels) if label == 0)
    inserted_total_length = sum(delete_labels)
    recovery_ok = (
        recovered == sequence
        and len(delete_labels) == len(corrupted_sequence)
        and len(corrupted_sequence) == len(sequence) + inserted_total_length
        and inserted_total_length == target_insert_len
    )
    return {
        "seq_id": seq_id,
        "original_sequence": sequence,
        "corrupted_sequence": corrupted_sequence,
        "budget_ratio": budget_ratio,
        "original_length": len(sequence),
        "corrupted_length": len(corrupted_sequence),
        "inserted_total_length": inserted_total_length,
        "insertion_segments": insertion_segments,
        "delete_labels": delete_labels,
        "keep_labels": keep_labels,
        "corruption_type_summary": dict(type_counts),
        "metadata": {
            "target_insert_len": target_insert_len,
            "recovery_ok": recovery_ok,
            "generator": "stage1_sequence_prior_segment_corruption",
        },
    }, recovery_ok


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def build_dataset(args):
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    rng = random.Random(int(config["seed"]))
    records = read_fasta(args.input_fasta)
    budgets = [float(value) for value in config["budgets"]]

    ensure_parent(args.out_jsonl)
    ensure_parent(args.summary_txt)
    debug_path = args.out_jsonl + ".debug_failed.jsonl"
    stats = Counter()
    budget_counts = Counter()
    type_counts = Counter()
    segment_length_counts = Counter()
    original_lengths = []
    corrupted_lengths = []
    inserted_fractions = []
    examples_by_type = {}
    failed_samples = []

    with open(args.out_jsonl, "w") as out_handle:
        for header, sequence in records:
            seq_id = seq_id_from_header(header)
            for budget in budgets:
                sample, recovery_ok = corrupt_sequence(seq_id, sequence, budget, config, rng)
                out_handle.write(json.dumps(sample, sort_keys=True) + "\n")
                stats["total_samples"] += 1
                if recovery_ok:
                    stats["recovery_pass"] += 1
                else:
                    stats["recovery_fail"] += 1
                    failed_samples.append(sample)
                budget_counts[str(budget)] += 1
                original_lengths.append(sample["original_length"])
                corrupted_lengths.append(sample["corrupted_length"])
                inserted_fractions.append(sample["inserted_total_length"] / float(sample["original_length"]))
                for segment in sample["insertion_segments"]:
                    corruption_type = segment["corruption_type"]
                    type_counts[corruption_type] += 1
                    segment_length_counts[segment["inserted_length"]] += 1
                    examples_by_type.setdefault(
                        corruption_type,
                        {
                            "seq_id": seq_id,
                            "budget_ratio": budget,
                            "inserted_length": segment["inserted_length"],
                            "inserted_sequence": segment["inserted_sequence"],
                            "original_insert_pos": segment["original_insert_pos"],
                        },
                    )

    if failed_samples:
        with open(debug_path, "w") as handle:
            for sample in failed_samples:
                handle.write(json.dumps(sample, sort_keys=True) + "\n")
    elif os.path.exists(debug_path):
        os.remove(debug_path)

    write_summary(
        args.summary_txt,
        args,
        config,
        records,
        stats,
        budget_counts,
        type_counts,
        segment_length_counts,
        original_lengths,
        corrupted_lengths,
        inserted_fractions,
        examples_by_type,
        debug_path if failed_samples else None,
    )
    return stats


def mean(values):
    return sum(values) / float(len(values)) if values else 0.0


def write_summary(
    path,
    args,
    config,
    records,
    stats,
    budget_counts,
    type_counts,
    segment_length_counts,
    original_lengths,
    corrupted_lengths,
    inserted_fractions,
    examples_by_type,
    debug_path,
):
    expected_types = list(config["corruption_types"])
    recovery_rate = stats["recovery_pass"] / float(stats["total_samples"]) if stats["total_samples"] else 0.0
    all_budgets_covered = all(budget_counts[str(float(value))] > 0 for value in config["budgets"])
    all_types_covered = all(type_counts[name] > 0 for name in expected_types)
    all_pass = (
        stats["total_samples"] == len(records) * len(config["budgets"])
        and recovery_rate == 1.0
        and all_budgets_covered
        and all_types_covered
        and stats["recovery_fail"] == 0
    )
    with open(path, "w") as handle:
        handle.write("Stage-1 deletion pretraining corruption summary\n\n")
        handle.write("input_fasta: {}\n".format(args.input_fasta))
        handle.write("config: {}\n".format(args.config))
        handle.write("out_jsonl: {}\n".format(args.out_jsonl))
        handle.write("seed: {}\n".format(config["seed"]))
        handle.write("num_input_sequences: {}\n".format(len(records)))
        handle.write("total_samples: {}\n".format(stats["total_samples"]))
        handle.write("recovery_check_pass_rate: {:.6f}\n".format(recovery_rate))
        handle.write("recovery_failures: {}\n".format(stats["recovery_fail"]))
        if debug_path:
            handle.write("debug_failed_jsonl: {}\n".format(debug_path))
        handle.write("\nSamples per budget:\n")
        for budget in config["budgets"]:
            handle.write("- {}: {}\n".format(float(budget), budget_counts[str(float(budget))]))
        handle.write("\nCorruption type counts:\n")
        for corruption_type in expected_types:
            handle.write("- {}: {}\n".format(corruption_type, type_counts[corruption_type]))
        handle.write("\nInserted segment length distribution:\n")
        for length, count in sorted(segment_length_counts.items()):
            handle.write("- {}: {}\n".format(length, count))
        handle.write("\nAverages:\n")
        handle.write("average_original_length: {:.3f}\n".format(mean(original_lengths)))
        handle.write("average_corrupted_length: {:.3f}\n".format(mean(corrupted_lengths)))
        handle.write("average_inserted_fraction: {:.6f}\n".format(mean(inserted_fractions)))
        handle.write("\nExamples of each corruption type:\n")
        for corruption_type in expected_types:
            handle.write("- {}: {}\n".format(corruption_type, json.dumps(examples_by_type.get(corruption_type, {}), sort_keys=True)))
        handle.write("\nQuality checks:\n")
        handle.write("delete_labels_match_corrupted_sequence: enforced_per_sample\n")
        handle.write("corrupted_length_matches_original_plus_inserted: enforced_per_sample\n")
        handle.write("all_budgets_covered: {}\n".format(all_budgets_covered))
        handle.write("all_corruption_types_covered: {}\n".format(all_types_covered))
        handle.write("\n{}\n".format("ALL_PASS" if all_pass else "WARN_CHECK_FAILED"))


def parse_args():
    parser = argparse.ArgumentParser(description="Build Stage-1 segment corruption JSONL.")
    parser.add_argument("--input_fasta", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--summary_txt", required=True)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    stats = build_dataset(args)
    print("Total samples: {}".format(stats["total_samples"]))
    print("Recovery pass: {}".format(stats["recovery_pass"]))
    print("Recovery fail: {}".format(stats["recovery_fail"]))
    print("Wrote {}".format(args.out_jsonl))
    print("Wrote {}".format(args.summary_txt))


if __name__ == "__main__":
    main()
