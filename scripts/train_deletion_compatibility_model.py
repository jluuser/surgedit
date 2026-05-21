#!/usr/bin/env python3
"""Train a lightweight deletion-compatibility model from ProteinGym indels.

The model is the first trainable BioDel-Cert core.  It learns a favorable
deletion score from public DMS deletion fitness and existing BioPrior features.
It is deliberately tabular so it can be trained quickly before a larger
PLM-embedding model is introduced.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from bioprior.deletion_compatibility import (  # noqa: E402
    TabularDeletionCompatibilityModel,
    build_feature_spec,
    featurize_row,
    model_config,
    safe_float,
)


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)


def write_json(path, data):
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def stable_hash(text):
    value = 2166136261
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def split_name(row, train_frac, val_frac, seed):
    key = row.get("DMS_id") or row.get("UniProt_ID") or row.get("target_sha1") or row.get("example_id") or ""
    value = stable_hash("{}:{}".format(seed, key)) / float(0xFFFFFFFF)
    if value < train_frac:
        return "train"
    if value < train_frac + val_frac:
        return "val"
    return "test"


def filtered_rows(rows, args):
    out = []
    for row in rows:
        y = safe_float(row.get(args.target_column))
        if y is None:
            continue
        if args.require_structure and row.get("structure_feature_status") != "success":
            continue
        if args.require_swissprot and "success" not in row.get("swissprot_feature_status", ""):
            continue
        if safe_float(row.get("deletion_len"), 0.0) is None:
            continue
        out.append(row)
    if args.max_rows and len(out) > args.max_rows:
        rng = random.Random(args.seed)
        rng.shuffle(out)
        out = out[: args.max_rows]
    return out


def add_targets(rows, target_column):
    by_assay = defaultdict(list)
    for row in rows:
        by_assay[row.get("DMS_id", "")].append(row)
    for _, group in by_assay.items():
        values = [safe_float(row.get(target_column)) for row in group]
        values = [value for value in values if value is not None]
        if not values:
            continue
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / float(max(1, len(values) - 1))
        std = math.sqrt(var) if var > 1e-12 else 1.0
        sorted_vals = sorted(values)
        cutoff = sorted_vals[max(0, int(math.floor(0.50 * (len(sorted_vals) - 1))))]
        for row in group:
            value = safe_float(row.get(target_column))
            if value is None:
                row["_target_regression"] = ""
                row["_target_binary"] = ""
                continue
            row["_target_regression"] = "{:.8f}".format((value - mean) / std)
            row["_target_binary"] = "1" if value >= cutoff else "0"


class CompatibilityDataset(Dataset):
    def __init__(self, rows, feature_spec):
        self.rows = rows
        self.feature_spec = feature_spec

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            "x": torch.tensor(featurize_row(row, self.feature_spec), dtype=torch.float32),
            "y_reg": torch.tensor(float(row["_target_regression"]), dtype=torch.float32),
            "y_bin": torch.tensor(float(row["_target_binary"]), dtype=torch.float32),
            "DMS_id": row.get("DMS_id", ""),
            "example_id": row.get("example_id", ""),
        }


def collate(batch):
    return {
        "x": torch.stack([item["x"] for item in batch], dim=0),
        "y_reg": torch.stack([item["y_reg"] for item in batch], dim=0),
        "y_bin": torch.stack([item["y_bin"] for item in batch], dim=0),
    }


def rankdata(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    dy = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    xvals, yvals = zip(*pairs)
    return pearson(rankdata(list(xvals)), rankdata(list(yvals)))


def binary_metrics(labels, scores):
    pairs = [(int(y), float(s)) for y, s in zip(labels, scores) if y is not None and s is not None]
    if not pairs:
        return None, None
    positives = sum(y for y, _ in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None, None
    sorted_pairs = sorted(pairs, key=lambda p: p[1])
    ranks = rankdata([score for _, score in sorted_pairs])
    pos_rank_sum = sum(rank for rank, (label, _) in zip(ranks, sorted_pairs) if label == 1)
    auroc = (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    desc = sorted(pairs, key=lambda p: p[1], reverse=True)
    tp = 0
    precision_sum = 0.0
    for idx, (label, _) in enumerate(desc, start=1):
        if label == 1:
            tp += 1
            precision_sum += tp / idx
    return auroc, precision_sum / positives


def fmt(value):
    if value is None:
        return ""
    return "{:.6f}".format(float(value))


def evaluate_model(model, rows, feature_spec, device, batch_size):
    model.eval()
    scores = []
    loader = DataLoader(CompatibilityDataset(rows, feature_spec), batch_size=batch_size, shuffle=False, collate_fn=collate)
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            scores.extend(model(x).detach().cpu().tolist())
    targets = [safe_float(row.get("_target_regression")) for row in rows]
    labels = [safe_float(row.get("_target_binary")) for row in rows]
    auroc, auprc = binary_metrics(labels, scores)
    return {
        "n": len(rows),
        "pearson": pearson(scores, targets),
        "spearman": spearman(scores, targets),
        "auroc": auroc,
        "auprc": auprc,
    }


def train_epoch(model, loader, optimizer, device, regression_weight):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        y_reg = batch["y_reg"].to(device)
        y_bin = batch["y_bin"].to(device)
        pred = model(x)
        loss = regression_weight * F.mse_loss(pred, y_reg) + F.binary_cross_entropy_with_logits(pred, y_bin)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach().cpu()) * x.shape[0]
        n += x.shape[0]
    return total / float(max(1, n))


def main():
    parser = argparse.ArgumentParser(description="Train deletion compatibility model from ProteinGym DMS indels.")
    parser.add_argument("--input_csv", default="results/proteingym_v13_indel/proteingym_v13_single_segment_deletions_biodel_features_stage1_finetuned_certified.csv")
    parser.add_argument("--out_dir", default="results/deletion_compatibility_model")
    parser.add_argument("--target_column", default="DMS_score")
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--require_structure", action="store_true")
    parser.add_argument("--require_swissprot", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--regression_weight", type=float, default=0.35)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    ensure_dir(args.out_dir)

    rows = filtered_rows(read_rows(args.input_csv), args)
    if not rows:
        raise RuntimeError("No usable rows found in {}".format(args.input_csv))
    add_targets(rows, args.target_column)
    splits = {"train": [], "val": [], "test": []}
    for row in rows:
        splits[split_name(row, args.train_frac, args.val_frac, args.seed)].append(row)
    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise RuntimeError("Empty split after grouped split: {}".format({k: len(v) for k, v in splits.items()}))

    feature_spec = build_feature_spec(splits["train"])
    device = torch.device(args.device)
    model = TabularDeletionCompatibilityModel(
        input_dim=feature_spec.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        CompatibilityDataset(splits["train"], feature_spec),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    log_rows = []
    best_val = -999.0
    best_path = os.path.join(args.out_dir, "best_model.pt")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, args.regression_weight)
        val_metrics = evaluate_model(model, splits["val"], feature_spec, device, args.batch_size)
        test_metrics = evaluate_model(model, splits["test"], feature_spec, device, args.batch_size)
        val_score = val_metrics["spearman"] if val_metrics["spearman"] is not None else -999.0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{"val_" + key: value for key, value in val_metrics.items()},
            **{"test_" + key: value for key, value in test_metrics.items()},
        }
        log_rows.append(row)
        print(
            "epoch {epoch}: train_loss={loss:.5f} val_spearman={val} val_auprc={auprc}".format(
                epoch=epoch,
                loss=train_loss,
                val=fmt(val_metrics["spearman"]),
                auprc=fmt(val_metrics["auprc"]),
            )
        )
        if val_score > best_val:
            best_val = val_score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_spec": feature_spec.to_dict(),
                    "model_config": model_config(feature_spec.input_dim, args.hidden_dim, args.num_layers, args.dropout),
                    "args": vars(args),
                    "split_counts": {key: len(value) for key, value in splits.items()},
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    final_metrics = {key: evaluate_model(model, split_rows, feature_spec, device, args.batch_size) for key, split_rows in splits.items()}
    write_json(
        os.path.join(args.out_dir, "metrics.json"),
        {
            "split_counts": {key: len(value) for key, value in splits.items()},
            "feature_columns": feature_spec.numeric_columns,
            "feature_dim": feature_spec.input_dim,
            "metrics": final_metrics,
        },
    )
    with open(os.path.join(args.out_dir, "train_log.csv"), "w", newline="") as handle:
        fields = sorted({key for row in log_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in log_rows:
            writer.writerow({key: fmt(value) if isinstance(value, float) else value for key, value in row.items()})
    with open(os.path.join(args.out_dir, "report.txt"), "w") as handle:
        handle.write("Deletion compatibility training report\n\n")
        handle.write("input_csv: {}\n".format(args.input_csv))
        handle.write("split_counts: {}\n".format({key: len(value) for key, value in splits.items()}))
        handle.write("feature_dim: {}\n".format(feature_spec.input_dim))
        for split, metrics in final_metrics.items():
            handle.write(
                "{}: spearman={} auroc={} auprc={} n={}\n".format(
                    split,
                    fmt(metrics["spearman"]),
                    fmt(metrics["auroc"]),
                    fmt(metrics["auprc"]),
                    metrics["n"],
                )
            )
        handle.write("\nBIODEL_DELETION_COMPATIBILITY_TRAINING_PASS\n")
    print("Wrote {}".format(best_path))


if __name__ == "__main__":
    main()
