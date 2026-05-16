#!/usr/bin/env python3
"""Train a lightweight budget-conditioned deletion utility prior.

This Stage-1 model learns to identify inserted redundant residues in corrupted
UniRef sequences. It is a general deletion utility prior, not the final safe
deletion policy.
"""

import argparse
import csv
import json
import math
import os
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)


def build_vocab():
    tokens = [PAD_TOKEN, UNK_TOKEN] + list(AMINO_ACIDS)
    stoi = {token: idx for idx, token in enumerate(tokens)}
    return {"tokens": tokens, "stoi": stoi, "pad_id": stoi[PAD_TOKEN], "unk_id": stoi[UNK_TOKEN]}


def encode_sequence(sequence, vocab):
    stoi = vocab["stoi"]
    unk_id = vocab["unk_id"]
    return [stoi.get(aa, unk_id) for aa in sequence]


def load_jsonl(path, max_len, max_samples=None):
    samples = []
    skipped = Counter()
    with open(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            if max_samples is not None and len(samples) >= max_samples:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped["json_decode_error"] += 1
                continue
            required = [
                "corrupted_sequence",
                "budget_ratio",
                "delete_labels",
                "inserted_total_length",
                "seq_id",
                "original_length",
                "corrupted_length",
            ]
            if any(key not in row for key in required):
                skipped["missing_required_field"] += 1
                continue
            sequence = row["corrupted_sequence"]
            labels = row["delete_labels"]
            if len(sequence) != len(labels):
                skipped["label_length_mismatch"] += 1
                continue
            if len(sequence) != int(row["corrupted_length"]):
                skipped["corrupted_length_mismatch"] += 1
                continue
            if len(sequence) > max_len:
                skipped["too_long"] += 1
                continue
            if not sequence:
                skipped["empty_sequence"] += 1
                continue
            samples.append(
                {
                    "seq_id": str(row["seq_id"]),
                    "sequence": sequence,
                    "budget_ratio": float(row["budget_ratio"]),
                    "labels": [float(x) for x in labels],
                    "inserted_total_length": float(row["inserted_total_length"]),
                    "original_length": int(row["original_length"]),
                    "corrupted_length": int(row["corrupted_length"]),
                    "line_no": line_no,
                }
            )
    return samples, skipped


def split_by_seq_id(samples, train_frac, val_frac, seed):
    seq_ids = sorted({sample["seq_id"] for sample in samples})
    rng = random.Random(seed)
    rng.shuffle(seq_ids)
    n = len(seq_ids)
    n_train = max(1, int(n * train_frac)) if n else 0
    n_val = max(1, int(n * val_frac)) if n - n_train > 1 else max(0, n - n_train)
    if n_train + n_val >= n and n > 2:
        n_val = 1
        n_train = n - 2
    train_ids = set(seq_ids[:n_train])
    val_ids = set(seq_ids[n_train : n_train + n_val])
    test_ids = set(seq_ids[n_train + n_val :])
    if not test_ids and val_ids and len(seq_ids) > 2:
        moved = sorted(val_ids)[-1]
        val_ids.remove(moved)
        test_ids.add(moved)
    splits = {"train": [], "val": [], "test": []}
    for sample in samples:
        if sample["seq_id"] in train_ids:
            splits["train"].append(sample)
        elif sample["seq_id"] in val_ids:
            splits["val"].append(sample)
        elif sample["seq_id"] in test_ids:
            splits["test"].append(sample)
    return splits, {"train": train_ids, "val": val_ids, "test": test_ids}


class DeletionDataset(Dataset):
    def __init__(self, samples, vocab):
        self.samples = samples
        self.vocab = vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "input_ids": encode_sequence(sample["sequence"], self.vocab),
            "labels": sample["labels"],
            "budget_ratio": sample["budget_ratio"],
            "inserted_total_length": sample["inserted_total_length"],
            "seq_id": sample["seq_id"],
            "length": len(sample["sequence"]),
        }


def collate_batch(batch, pad_id):
    max_len = max(item["length"] for item in batch)
    input_ids = []
    labels = []
    mask = []
    budgets = []
    target_counts = []
    lengths = []
    seq_ids = []
    for item in batch:
        length = item["length"]
        pad = max_len - length
        input_ids.append(item["input_ids"] + [pad_id] * pad)
        labels.append(item["labels"] + [0.0] * pad)
        mask.append([1.0] * length + [0.0] * pad)
        budgets.append([float(item["budget_ratio"])])
        target_counts.append(float(item["inserted_total_length"]))
        lengths.append(float(length))
        seq_ids.append(item["seq_id"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "mask": torch.tensor(mask, dtype=torch.float32),
        "budget_ratio": torch.tensor(budgets, dtype=torch.float32),
        "target_counts": torch.tensor(target_counts, dtype=torch.float32),
        "lengths": torch.tensor(lengths, dtype=torch.float32),
        "seq_ids": seq_ids,
    }


class TransformerDeletionPrior(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, hidden_dim, num_layers, num_heads, dropout, pad_id):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.budget_embedding = nn.Linear(1, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, 1)
        self.max_len = max_len

    def forward(self, input_ids, budget_ratio, attention_mask):
        batch, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)
        x = x + self.budget_embedding(budget_ratio).unsqueeze(1)
        x = self.dropout(x)
        key_padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.head(x).squeeze(-1)


class BiLSTMDeletionPrior(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_id):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.budget_embedding = nn.Linear(1, embed_dim)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, budget_ratio, attention_mask):
        x = self.token_embedding(input_ids) + self.budget_embedding(budget_ratio).unsqueeze(1)
        x = self.dropout(x)
        x, _ = self.encoder(x)
        return self.head(x).squeeze(-1)


class CNNDeletionPrior(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_id):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.budget_embedding = nn.Linear(1, embed_dim)
        layers = []
        in_channels = embed_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, budget_ratio, attention_mask):
        x = self.token_embedding(input_ids) + self.budget_embedding(budget_ratio).unsqueeze(1)
        x = x.transpose(1, 2)
        x = self.encoder(x).transpose(1, 2)
        return self.head(x).squeeze(-1)


def build_model(args, vocab):
    vocab_size = len(vocab["tokens"])
    pad_id = vocab["pad_id"]
    if args.model_type == "transformer":
        return TransformerDeletionPrior(
            vocab_size,
            args.max_len,
            args.embed_dim,
            args.hidden_dim,
            args.num_layers,
            args.num_heads,
            args.dropout,
            pad_id,
        )
    if args.model_type == "bilstm":
        return BiLSTMDeletionPrior(vocab_size, args.embed_dim, args.hidden_dim, args.num_layers, args.dropout, pad_id)
    if args.model_type == "cnn":
        return CNNDeletionPrior(vocab_size, args.embed_dim, args.hidden_dim, args.num_layers, args.dropout, pad_id)
    raise ValueError("Unsupported model_type: {}".format(args.model_type))


def compute_loss(logits, batch, lambda_budget):
    labels = batch["labels"]
    mask = batch["mask"]
    token_loss = F.binary_cross_entropy_with_logits(logits[mask > 0], labels[mask > 0])
    probs = torch.sigmoid(logits) * mask
    pred_counts = probs.sum(dim=1)
    budget_loss = torch.abs(pred_counts - batch["target_counts"]).div(batch["lengths"].clamp_min(1.0)).mean()
    return token_loss + lambda_budget * budget_loss, token_loss.detach(), budget_loss.detach()


def binary_metrics(labels, probs):
    labels = list(labels)
    probs = list(probs)
    preds = [1 if p >= 0.5 else 0 for p in probs]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / float(precision + recall) if precision + recall else 0.0
    auc = None
    ap = None
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if len(set(labels)) > 1:
            auc = float(roc_auc_score(labels, probs))
            ap = float(average_precision_score(labels, probs))
    except Exception:
        auc = None
        ap = None
    return {"token_auc": auc, "token_ap": ap, "token_f1": f1, "precision": precision, "recall": recall}


def run_epoch(model, loader, device, optimizer, lambda_budget, train):
    model.train(train)
    total_loss = 0.0
    total_batches = 0
    all_labels = []
    all_probs = []
    count_abs_errors = []
    num_tokens = 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            logits = model(batch["input_ids"], batch["budget_ratio"], batch["mask"])
            loss, _, _ = compute_loss(logits, batch, lambda_budget)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_batches += 1
            probs = torch.sigmoid(logits).detach()
            mask = batch["mask"] > 0
            all_labels.extend(batch["labels"][mask].detach().cpu().tolist())
            all_probs.extend(probs[mask].detach().cpu().tolist())
            pred_counts = (probs * batch["mask"]).sum(dim=1)
            count_abs_errors.extend(torch.abs(pred_counts - batch["target_counts"]).detach().cpu().tolist())
            num_tokens += int(mask.sum().detach().cpu())
    metrics = binary_metrics(all_labels, all_probs) if all_labels else {
        "token_auc": None,
        "token_ap": None,
        "token_f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    metrics["loss"] = total_loss / float(total_batches) if total_batches else 0.0
    metrics["deletion_count_mae"] = sum(count_abs_errors) / float(len(count_abs_errors)) if count_abs_errors else 0.0
    metrics["num_samples"] = len(loader.dataset)
    metrics["num_tokens"] = num_tokens
    return metrics


def dataset_stats(samples):
    lengths = [len(sample["sequence"]) for sample in samples]
    positives = sum(sum(sample["labels"]) for sample in samples)
    tokens = sum(lengths)
    return {
        "total_samples": len(samples),
        "max_length": max(lengths) if lengths else 0,
        "mean_length": sum(lengths) / float(len(lengths)) if lengths else 0.0,
        "positive_delete_label_ratio": positives / float(tokens) if tokens else 0.0,
        "total_tokens": tokens,
    }


def write_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


def train(args):
    set_seed(args.seed)
    ensure_dir(args.out_dir)
    ensure_dir(args.ckpt_dir)
    vocab = build_vocab()
    samples, skipped = load_jsonl(args.train_jsonl, args.max_len, args.max_samples)
    if not samples:
        raise RuntimeError("No usable samples loaded from {}".format(args.train_jsonl))
    splits, split_ids = split_by_seq_id(samples, args.train_frac, args.val_frac, args.seed)
    stats = dataset_stats(samples)
    print("total samples: {}".format(len(samples)))
    print("skipped samples: {}".format(dict(skipped)))
    print("train/val/test seq_id counts: {}/{}/{}".format(len(split_ids["train"]), len(split_ids["val"]), len(split_ids["test"])))
    print("train/val/test sample counts: {}/{}/{}".format(len(splits["train"]), len(splits["val"]), len(splits["test"])))
    print("max length / mean length: {} / {:.3f}".format(stats["max_length"], stats["mean_length"]))
    print("positive delete label ratio: {:.6f}".format(stats["positive_delete_label_ratio"]))

    loaders = {}
    for split_name in ["train", "val", "test"]:
        dataset = DeletionDataset(splits[split_name], vocab)
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split_name == "train"),
            num_workers=args.num_workers,
            collate_fn=lambda batch, pad_id=vocab["pad_id"]: collate_batch(batch, pad_id),
        )

    device = torch.device(args.device)
    model = build_model(args, vocab).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_log_path = os.path.join(args.out_dir, "train_log.csv")
    best_score = -float("inf")
    best_epoch = None
    best_path = os.path.join(args.ckpt_dir, "best_model.pt")

    with open(train_log_path, "w", newline="") as handle:
        fields = [
            "epoch",
            "train_loss",
            "val_loss",
            "val_token_auc",
            "val_token_ap",
            "val_token_f1",
            "val_precision",
            "val_recall",
            "val_deletion_count_mae",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, loaders["train"], device, optimizer, args.lambda_budget, train=True)
            val_metrics = run_epoch(model, loaders["val"], device, optimizer, args.lambda_budget, train=False)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_token_auc": val_metrics["token_auc"],
                "val_token_ap": val_metrics["token_ap"],
                "val_token_f1": val_metrics["token_f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_deletion_count_mae": val_metrics["deletion_count_mae"],
            }
            writer.writerow(row)
            handle.flush()
            print(
                "epoch {epoch}: train_loss={train_loss:.5f} val_loss={val_loss:.5f} val_ap={val_token_ap} val_f1={val_token_f1:.5f}".format(
                    **row
                )
            )
            score = val_metrics["token_ap"] if val_metrics["token_ap"] is not None else -val_metrics["loss"]
            if score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "vocab": vocab,
                        "best_epoch": best_epoch,
                        "val_metrics": val_metrics,
                    },
                    best_path,
                )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, loaders["test"], device, optimizer=None, lambda_budget=args.lambda_budget, train=False)
    output_test_metrics = {
        "token_auc": test_metrics["token_auc"],
        "token_ap": test_metrics["token_ap"],
        "token_f1": test_metrics["token_f1"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "deletion_count_mae": test_metrics["deletion_count_mae"],
        "num_test_samples": test_metrics["num_samples"],
        "num_test_tokens": test_metrics["num_tokens"],
        "best_epoch": best_epoch,
    }
    write_json(os.path.join(args.out_dir, "test_metrics.json"), output_test_metrics)
    config = vars(args).copy()
    config.update(
        {
            "skipped_samples": dict(skipped),
            "dataset_stats": stats,
            "split_seq_id_counts": {key: len(value) for key, value in split_ids.items()},
            "split_sample_counts": {key: len(value) for key, value in splits.items()},
        }
    )
    write_json(os.path.join(args.out_dir, "config.json"), config)
    write_json(os.path.join(args.out_dir, "feature_or_vocab.json"), vocab)
    print("best_model: {}".format(best_path))
    print("test_metrics: {}".format(os.path.join(args.out_dir, "test_metrics.json")))


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage-1 deletion utility prior.")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda_budget", type=float, default=0.1)
    parser.add_argument("--model_type", choices=["transformer", "bilstm", "cnn"], default="transformer")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=1200)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional CPU/debug cap on loaded samples.")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
