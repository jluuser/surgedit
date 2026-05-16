#!/usr/bin/env python3
"""Score candidate deletion segments with a trained Stage-1 deletion prior.

The Stage-1 model outputs token-level deletion probabilities conditioned on a
budget. This script aggregates token probabilities/logits into segment-level
utility features U_stage1(S) for downstream BioDel-Planner experiments.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from types import SimpleNamespace

import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_stage1_deletion_prior import build_model, encode_sequence  # noqa: E402


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def parse_budgets(text):
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def budget_tag(value):
    return "b{:02d}".format(int(round(float(value) * 100)))


def read_core_sequences(path):
    sequences = {}
    metadata = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get("accession") or row.get("protein_id")
            if not accession:
                continue
            sequences[accession] = row["sequence"]
            metadata[accession] = row
    return sequences, metadata


def read_segments(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    return rows, fields


def load_checkpoint(checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def load_model(checkpoint_path, device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    train_args = checkpoint["args"]
    model_args = SimpleNamespace(**train_args)
    vocab = checkpoint["vocab"]
    model = build_model(model_args, vocab)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, vocab, train_args, checkpoint


def score_sequence(model, vocab, sequence, budgets, device):
    input_ids = torch.tensor([encode_sequence(sequence, vocab)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.float32, device=device)
    scores = {}
    with torch.no_grad():
        for budget in budgets:
            budget_tensor = torch.tensor([[budget]], dtype=torch.float32, device=device)
            logits = model(input_ids, budget_tensor, attention_mask).squeeze(0).detach().cpu()
            probs = torch.sigmoid(logits)
            scores[budget] = {
                "logits": logits.tolist(),
                "probs": probs.tolist(),
            }
    return scores


def score_sequences_batched(model, vocab, accession_sequences, budgets, device, batch_size):
    """Score many sequences with padded mini-batches.

    The output matches score_sequence(): token-level logits/probs are trimmed
    back to the original sequence length for each accession.
    """
    accessions = list(accession_sequences)
    scores = {accession: {} for accession in accessions}
    pad_id = vocab.get("<pad>", 0)

    with torch.no_grad():
        for batch_start in range(0, len(accessions), batch_size):
            batch_accessions = accessions[batch_start : batch_start + batch_size]
            encoded = [encode_sequence(accession_sequences[acc], vocab) for acc in batch_accessions]
            lengths = [len(ids) for ids in encoded]
            max_len = max(lengths)
            input_ids = torch.full(
                (len(batch_accessions), max_len),
                pad_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros(
                (len(batch_accessions), max_len),
                dtype=torch.float32,
                device=device,
            )
            for row_idx, ids in enumerate(encoded):
                seq_tensor = torch.tensor(ids, dtype=torch.long, device=device)
                input_ids[row_idx, : len(ids)] = seq_tensor
                attention_mask[row_idx, : len(ids)] = 1.0

            for budget in budgets:
                budget_tensor = torch.full(
                    (len(batch_accessions), 1),
                    float(budget),
                    dtype=torch.float32,
                    device=device,
                )
                logits = model(input_ids, budget_tensor, attention_mask).detach().cpu()
                probs = torch.sigmoid(logits)
                for row_idx, accession in enumerate(batch_accessions):
                    length = lengths[row_idx]
                    scores[accession][budget] = {
                        "logits": logits[row_idx, :length].tolist(),
                        "probs": probs[row_idx, :length].tolist(),
                    }
    return scores


def mean(values):
    return sum(values) / float(len(values)) if values else 0.0


def aggregate_segment(values, start, end):
    segment_values = values[start : end + 1]
    if not segment_values:
        return {"mean": "", "sum": "", "max": ""}
    return {
        "mean": mean(segment_values),
        "sum": sum(segment_values),
        "max": max(segment_values),
    }


def format_float(value):
    if value == "":
        return ""
    return "{:.6f}".format(float(value))


def write_summary(path, args, summary):
    ensure_parent(path)
    with open(path, "w") as handle:
        handle.write("Stage-1 segment utility scoring summary\n\n")
        handle.write("checkpoint: {}\n".format(args.checkpoint))
        handle.write("segments_csv: {}\n".format(args.segments_csv))
        handle.write("core_csv: {}\n".format(args.core_csv))
        handle.write("out_csv: {}\n".format(args.out_csv))
        handle.write("budgets: {}\n".format(",".join(str(x) for x in summary["budgets"])))
        handle.write("default_budget: {}\n".format(args.default_budget))
        handle.write("device: {}\n\n".format(args.device))
        for key in [
            "input_segments",
            "output_segments",
            "unique_accessions_in_segments",
            "scored_accessions",
            "missing_sequence_segments",
            "length_mismatch_segments",
            "oob_segments",
        ]:
            handle.write("{}: {}\n".format(key, summary[key]))
        handle.write("\nStage-1 checkpoint metrics:\n")
        for key, value in summary["checkpoint_metrics"].items():
            handle.write("- {}: {}\n".format(key, value))
        handle.write("\nSegment utility distribution by default budget:\n")
        handle.write("mean: {:.6f}\n".format(summary["default_mean_score"]))
        handle.write("min: {:.6f}\n".format(summary["default_min_score"]))
        handle.write("max: {:.6f}\n".format(summary["default_max_score"]))
        handle.write("\n{}\n".format("STAGE1_SEGMENT_UTILITY_PASS" if summary["pass"] else "STAGE1_SEGMENT_UTILITY_WARN"))


def score_segments(args):
    device = torch.device(args.device)
    budgets = parse_budgets(args.budgets)
    if args.default_budget not in budgets:
        budgets.append(args.default_budget)
    budgets = sorted(set(budgets))

    model, vocab, train_args, checkpoint = load_model(args.checkpoint, device)
    sequences, _ = read_core_sequences(args.core_csv)
    rows, fields = read_segments(args.segments_csv)
    if args.limit_segments is not None:
        rows = rows[: args.limit_segments]

    accessions = sorted({row["accession"] for row in rows if row.get("accession")})
    if args.limit_proteins is not None:
        allowed = set(accessions[: args.limit_proteins])
        rows = [row for row in rows if row.get("accession") in allowed]
        accessions = sorted(allowed)

    accession_sequences = {}
    missing_accessions = []
    for accession in accessions:
        sequence = sequences.get(accession)
        if sequence is None:
            missing_accessions.append(accession)
            continue
        accession_sequences[accession] = sequence
    token_scores = score_sequences_batched(
        model,
        vocab,
        accession_sequences,
        budgets,
        device,
        max(1, int(args.batch_size)),
    )

    added_fields = []
    for budget in budgets:
        tag = budget_tag(budget)
        added_fields.extend(
            [
                "stage1_mean_prob_{}".format(tag),
                "stage1_sum_prob_{}".format(tag),
                "stage1_max_prob_{}".format(tag),
                "stage1_mean_logit_{}".format(tag),
            ]
        )
    default_tag = budget_tag(args.default_budget)
    added_fields.extend(["stage1_utility_score", "stage1_utility_budget", "stage1_scoring_status"])
    out_fields = fields + [field for field in added_fields if field not in fields]

    stats = Counter()
    default_scores = []
    ensure_parent(args.out_csv)
    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            stats["input_segments"] += 1
            accession = row.get("accession")
            sequence = sequences.get(accession)
            if accession not in token_scores:
                row["stage1_scoring_status"] = "missing_sequence"
                stats["missing_sequence_segments"] += 1
                writer.writerow(row)
                continue
            start = int(row["seg_start"])
            end = int(row["seg_end"])
            protein_length = int(row["protein_length"])
            if len(sequence) != protein_length:
                row["stage1_scoring_status"] = "length_mismatch"
                stats["length_mismatch_segments"] += 1
                writer.writerow(row)
                continue
            if start < 0 or end >= len(sequence) or start > end:
                row["stage1_scoring_status"] = "segment_oob"
                stats["oob_segments"] += 1
                writer.writerow(row)
                continue
            for budget in budgets:
                tag = budget_tag(budget)
                probs = token_scores[accession][budget]["probs"]
                logits = token_scores[accession][budget]["logits"]
                prob_stats = aggregate_segment(probs, start, end)
                logit_stats = aggregate_segment(logits, start, end)
                row["stage1_mean_prob_{}".format(tag)] = format_float(prob_stats["mean"])
                row["stage1_sum_prob_{}".format(tag)] = format_float(prob_stats["sum"])
                row["stage1_max_prob_{}".format(tag)] = format_float(prob_stats["max"])
                row["stage1_mean_logit_{}".format(tag)] = format_float(logit_stats["mean"])
            default_value = row.get("stage1_mean_prob_{}".format(default_tag), "")
            row["stage1_utility_score"] = default_value
            row["stage1_utility_budget"] = args.default_budget
            row["stage1_scoring_status"] = "success"
            if default_value != "":
                default_scores.append(float(default_value))
            stats["output_segments"] += 1
            writer.writerow(row)

    checkpoint_metrics = checkpoint.get("val_metrics", {})
    summary = {
        "budgets": budgets,
        "input_segments": stats["input_segments"],
        "output_segments": stats["output_segments"],
        "unique_accessions_in_segments": len(accessions),
        "scored_accessions": len(token_scores),
        "missing_sequence_segments": stats["missing_sequence_segments"],
        "length_mismatch_segments": stats["length_mismatch_segments"],
        "oob_segments": stats["oob_segments"],
        "checkpoint_metrics": {
            "best_epoch": checkpoint.get("best_epoch"),
            "val_token_ap": checkpoint_metrics.get("token_ap"),
            "val_token_auc": checkpoint_metrics.get("token_auc"),
            "val_token_f1": checkpoint_metrics.get("token_f1"),
            "val_loss": checkpoint_metrics.get("loss"),
        },
        "default_mean_score": mean(default_scores),
        "default_min_score": min(default_scores) if default_scores else 0.0,
        "default_max_score": max(default_scores) if default_scores else 0.0,
        "pass": stats["output_segments"] > 0 and stats["missing_sequence_segments"] == 0 and stats["oob_segments"] == 0,
    }
    write_summary(args.summary_txt, args, summary)
    print("Scored accessions: {}".format(summary["scored_accessions"]))
    print("Output segments: {}".format(summary["output_segments"]))
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.summary_txt))


def parse_args():
    parser = argparse.ArgumentParser(description="Score BioPrior segments with a trained Stage-1 deletion prior.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--segments_csv", required=True)
    parser.add_argument("--core_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--summary_txt", required=True)
    parser.add_argument("--budgets", default="0.1,0.2,0.3")
    parser.add_argument("--default_budget", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit_proteins", type=int, default=None)
    parser.add_argument("--limit_segments", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    score_segments(parse_args())
