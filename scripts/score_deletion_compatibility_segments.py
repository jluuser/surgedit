#!/usr/bin/env python3
"""Score deletion-candidate CSV rows with a trained compatibility model."""

import argparse
import csv
import json
import math
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from bioprior.deletion_compatibility import (  # noqa: E402
    FeatureSpec,
    TabularDeletionCompatibilityModel,
    featurize_row,
    safe_float,
)


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def read_rows(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path, rows, fields):
    ensure_parent(path)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    spec = FeatureSpec.from_dict(checkpoint["feature_spec"])
    config = dict(checkpoint["model_config"])
    model = TabularDeletionCompatibilityModel(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.10)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, spec, checkpoint


def score_rows(model, spec, rows, device, batch_size):
    logits = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            x = torch.tensor([featurize_row(row, spec) for row in batch_rows], dtype=torch.float32, device=device)
            logits.extend(model(x).detach().cpu().tolist())
    for row, logit in zip(rows, logits):
        prob = sigmoid(float(logit))
        prior_risk = safe_float(row.get("risk_upper"), safe_float(row.get("risk_point"), 0.0)) or 0.0
        learned_risk = max(0.0, min(1.0, 1.0 - prob))
        blended_risk = max(0.0, min(1.0, 0.50 * learned_risk + 0.50 * prior_risk))
        row["compatibility_logit"] = "{:.6f}".format(float(logit))
        row["compatibility_score"] = "{:.6f}".format(prob)
        row["compatibility_risk"] = "{:.6f}".format(learned_risk)
        row["compatibility_blended_risk"] = "{:.6f}".format(blended_risk)
        row["compatibility_biodel_score"] = "{:.6f}".format(prob - blended_risk)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Score deletion candidates with a trained compatibility model.")
    parser.add_argument("--model", default="results/deletion_compatibility_model/best_model.pt")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_summary", default=None)
    args = parser.parse_args()

    rows, fields = read_rows(args.input_csv)
    if not rows:
        raise RuntimeError("No rows found in {}".format(args.input_csv))
    device = torch.device(args.device)
    model, spec, checkpoint = load_model(args.model, device)
    rows = score_rows(model, spec, rows, device, args.batch_size)
    new_fields = list(fields)
    for field in [
        "compatibility_logit",
        "compatibility_score",
        "compatibility_risk",
        "compatibility_blended_risk",
        "compatibility_biodel_score",
    ]:
        if field not in new_fields:
            new_fields.append(field)
    write_csv(args.out_csv, rows, new_fields)
    summary_path = args.out_summary or os.path.splitext(args.out_csv)[0] + "_summary.txt"
    with open(summary_path, "w") as handle:
        values = [float(row["compatibility_score"]) for row in rows]
        risks = [float(row["compatibility_blended_risk"]) for row in rows]
        handle.write("Deletion compatibility scoring summary\n\n")
        handle.write("model: {}\n".format(args.model))
        handle.write("input_csv: {}\n".format(args.input_csv))
        handle.write("rows: {}\n".format(len(rows)))
        handle.write("feature_dim: {}\n".format(checkpoint["model_config"]["input_dim"]))
        handle.write("mean_compatibility_score: {:.6f}\n".format(sum(values) / len(values)))
        handle.write("mean_compatibility_blended_risk: {:.6f}\n".format(sum(risks) / len(risks)))
        handle.write("\nBIODEL_DELETION_COMPATIBILITY_SCORING_PASS\n")
    print("Wrote {}".format(args.out_csv))


if __name__ == "__main__":
    main()
