#!/usr/bin/env python3
"""Score ProteinGym deletion segments with the Stage-1 deletion prior."""

import argparse
import csv
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


def read_csv(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def parse_budgets(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def budget_tag(value):
    return "b{:02d}".format(int(round(float(value) * 100)))


def load_checkpoint(checkpoint_path, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_model(checkpoint_path, device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    model_args = SimpleNamespace(**checkpoint["args"])
    vocab = checkpoint["vocab"]
    model = build_model(model_args, vocab)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, vocab, checkpoint


def score_sequences(model, vocab, sequences, budgets, device, batch_size):
    pad_id = vocab.get("<pad>", 0)
    keys = list(sequences)
    out = {key: {} for key in keys}
    with torch.no_grad():
        for start in range(0, len(keys), batch_size):
            batch_keys = keys[start : start + batch_size]
            encoded = [encode_sequence(sequences[key], vocab) for key in batch_keys]
            lengths = [len(x) for x in encoded]
            max_len = max(lengths)
            input_ids = torch.full((len(batch_keys), max_len), pad_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros((len(batch_keys), max_len), dtype=torch.float32, device=device)
            for i, ids in enumerate(encoded):
                tensor = torch.tensor(ids, dtype=torch.long, device=device)
                input_ids[i, : len(ids)] = tensor
                attention_mask[i, : len(ids)] = 1.0
            for budget in budgets:
                budget_tensor = torch.full((len(batch_keys), 1), float(budget), dtype=torch.float32, device=device)
                logits = model(input_ids, budget_tensor, attention_mask).detach().cpu()
                probs = torch.sigmoid(logits)
                for i, key in enumerate(batch_keys):
                    length = lengths[i]
                    out[key][budget] = {
                        "logits": logits[i, :length].tolist(),
                        "probs": probs[i, :length].tolist(),
                    }
    return out


def mean(values):
    return sum(values) / len(values) if values else 0.0


def segment_stats(values, start, end):
    vals = values[start : end + 1]
    if not vals:
        return "", "", ""
    return mean(vals), sum(vals), max(vals)


def fmt(value):
    if value == "":
        return ""
    return "{:.6f}".format(float(value))


def write_summary(path, args, rows, status_counts):
    ensure_parent(path)
    source_counts = Counter(row["source"] for row in rows)
    success = sum(1 for row in rows if row.get("stage1_scoring_status") == "success")
    values = [float(row["stage1_utility_score"]) for row in rows if row.get("stage1_scoring_status") == "success" and row.get("stage1_utility_score")]
    with open(path, "w") as handle:
        handle.write("ProteinGym deletion Stage-1 scoring summary\n\n")
        handle.write("input_csv: {}\n".format(args.input_csv))
        handle.write("checkpoint: {}\n".format(args.checkpoint))
        handle.write("out_csv: {}\n".format(args.out_csv))
        handle.write("budgets: {}\n".format(args.budgets))
        handle.write("default_budget: {}\n".format(args.default_budget))
        handle.write("device: {}\n\n".format(args.device))
        handle.write("input_rows: {}\n".format(len(rows)))
        handle.write("source_counts: {}\n".format(dict(source_counts)))
        handle.write("status_counts: {}\n".format(dict(status_counts)))
        handle.write("scored_rows: {}\n".format(success))
        handle.write("mean_stage1_utility: {:.6f}\n".format(mean(values)))
        handle.write("min_stage1_utility: {:.6f}\n".format(min(values) if values else 0.0))
        handle.write("max_stage1_utility: {:.6f}\n".format(max(values) if values else 0.0))
        handle.write("\nPROTEINGYM_STAGE1_SCORING_PASS\n")


def main():
    parser = argparse.ArgumentParser(description="Score ProteinGym deletion benchmark with Stage-1 model.")
    parser.add_argument("--input_csv", default="results/proteingym_deletion_benchmark/proteingym_single_segment_deletions.csv")
    parser.add_argument("--checkpoint", default="checkpoints/stage1_deletion_prior_100k/best_model.pt")
    parser.add_argument("--out_csv", default="results/proteingym_deletion_benchmark/proteingym_single_segment_deletions_stage1_scored.csv")
    parser.add_argument("--summary_txt", default="results/proteingym_deletion_benchmark/proteingym_single_segment_deletions_stage1_scored_summary.txt")
    parser.add_argument("--budgets", default="0.1,0.2,0.3")
    parser.add_argument("--default_budget", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=1200)
    args = parser.parse_args()

    rows, fields = read_csv(args.input_csv)
    budgets = sorted(set(parse_budgets(args.budgets) + [args.default_budget]))
    device = torch.device(args.device)
    model, vocab, _ = load_model(args.checkpoint, device)

    sequences = {}
    for row in rows:
        seq = row["target_sequence"]
        if len(seq) <= args.max_len:
            sequences[seq] = seq
    token_scores = score_sequences(model, vocab, sequences, budgets, device, max(1, args.batch_size))

    added = []
    for budget in budgets:
        tag = budget_tag(budget)
        added.extend([
            "stage1_mean_prob_{}".format(tag),
            "stage1_sum_prob_{}".format(tag),
            "stage1_max_prob_{}".format(tag),
            "stage1_mean_logit_{}".format(tag),
        ])
    added.extend(["stage1_utility_score", "stage1_utility_budget", "stage1_scoring_status"])

    status_counts = Counter()
    for row in rows:
        seq = row["target_sequence"]
        start = int(row["deletion_start"])
        end = int(row["deletion_end"])
        if len(seq) > args.max_len:
            row["stage1_scoring_status"] = "target_too_long"
            status_counts[row["stage1_scoring_status"]] += 1
            for field in added:
                row.setdefault(field, "")
            continue
        if seq not in token_scores:
            row["stage1_scoring_status"] = "missing_sequence_score"
            status_counts[row["stage1_scoring_status"]] += 1
            for field in added:
                row.setdefault(field, "")
            continue
        if start < 0 or end >= len(seq) or start > end:
            row["stage1_scoring_status"] = "segment_oob"
            status_counts[row["stage1_scoring_status"]] += 1
            for field in added:
                row.setdefault(field, "")
            continue
        for budget in budgets:
            tag = budget_tag(budget)
            scores = token_scores[seq][budget]
            pmean, psum, pmax = segment_stats(scores["probs"], start, end)
            lmean, _, _ = segment_stats(scores["logits"], start, end)
            row["stage1_mean_prob_{}".format(tag)] = fmt(pmean)
            row["stage1_sum_prob_{}".format(tag)] = fmt(psum)
            row["stage1_max_prob_{}".format(tag)] = fmt(pmax)
            row["stage1_mean_logit_{}".format(tag)] = fmt(lmean)
        default_tag = budget_tag(args.default_budget)
        row["stage1_utility_score"] = row["stage1_mean_prob_{}".format(default_tag)]
        row["stage1_utility_budget"] = args.default_budget
        row["stage1_scoring_status"] = "success"
        status_counts["success"] += 1

    ensure_parent(args.out_csv)
    out_fields = fields + [field for field in added if field not in fields]
    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)
    write_summary(args.summary_txt, args, rows, status_counts)
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.summary_txt))


if __name__ == "__main__":
    main()
