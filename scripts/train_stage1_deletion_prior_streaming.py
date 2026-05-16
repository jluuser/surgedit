#!/usr/bin/env python3
"""Train Stage-1 deletion prior directly from full UniRef50 with streaming corruption.

This avoids materializing a full corrupted JSONL file. The dataset is streamed
from FASTA, filtered, split deterministically by sequence hash, and corrupted
on the fly for each configured budget.
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
import random
import time
from collections import Counter
from datetime import timedelta
from math import ceil

try:
    from contextlib import nullcontext
except ImportError:
    class nullcontext(object):
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DistributedDataParallel
from torch.utils.data import DataLoader, IterableDataset

try:
    from torch.distributed.algorithms.join import Join
except ImportError:
    Join = None

from build_stage1_deletion_pretrain_dataset import corrupt_sequence, load_config
from train_stage1_deletion_prior import (
    AMINO_ACIDS,
    binary_metrics,
    build_model,
    build_vocab,
    collate_batch,
    compute_loss,
    encode_sequence,
    set_seed,
)


STANDARD_AA = set(AMINO_ACIDS)


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)


def write_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


def setup_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if not enabled:
        return {
            "enabled": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "device": torch.device(args.device),
        }
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build")
    if torch.cuda.is_available() and (args.device.startswith("cuda") or args.device == "auto"):
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = args.ddp_backend or "nccl"
    else:
        device = torch.device("cpu" if args.device == "auto" else args.device)
        backend = args.ddp_backend or "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, timeout=timedelta(hours=args.ddp_timeout_hours))
    return {
        "enabled": True,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
    }


def cleanup_distributed(state):
    if state.get("enabled") and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(state):
    return int(state.get("rank", 0)) == 0


def rank_print(state, message):
    if is_main_process(state):
        print(message)


def barrier(state):
    if state.get("enabled") and dist.is_initialized():
        dist.barrier()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_initial_checkpoint(model, path, device, strict=True, expected_vocab=None):
    checkpoint = torch_load(path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError("checkpoint does not contain model_state_dict: {}".format(path))
    if expected_vocab is not None and checkpoint.get("vocab") is not None and checkpoint["vocab"] != expected_vocab:
        raise ValueError("checkpoint vocab does not match current vocab: {}".format(path))
    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"], strict=strict)
    return checkpoint


def stream_fasta(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        header = None
        chunks = []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)


def seq_id_from_header(header):
    return header.split(None, 1)[0]


def stable_int(text):
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16)


def split_for_sequence(seq_hash, train_frac, val_frac):
    value = int(seq_hash[:12], 16) / float(16 ** 12)
    if value < train_frac:
        return "train"
    if value < train_frac + val_frac:
        return "val"
    return "test"


def encode_sample(row, vocab):
    return {
        "seq_id": str(row["seq_id"]),
        "sequence": row["corrupted_sequence"],
        "input_ids": encode_sequence(row["corrupted_sequence"], vocab),
        "length": len(row["corrupted_sequence"]),
        "budget_ratio": float(row["budget_ratio"]),
        "labels": [float(x) for x in row["delete_labels"]],
        "inserted_total_length": float(row["inserted_total_length"]),
        "original_length": int(row["original_length"]),
        "corrupted_length": int(row["corrupted_length"]),
    }


class StreamingUniRefCorruptionDataset(IterableDataset):
    def __init__(
        self,
        input_fasta,
        config,
        split,
        train_frac,
        val_frac,
        min_len,
        max_original_len,
        max_corrupted_len,
        seed,
        max_records=None,
        max_samples=None,
        exclude_nonstandard_aa=True,
        shard_index=0,
        shard_count=1,
        vocab=None,
    ):
        super().__init__()
        self.input_fasta = input_fasta
        self.config = dict(config)
        self.split = split
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.min_len = min_len
        self.max_original_len = max_original_len
        self.max_corrupted_len = max_corrupted_len
        self.seed = seed
        self.max_records = max_records
        self.max_samples = max_samples
        self.exclude_nonstandard_aa = exclude_nonstandard_aa
        self.shard_index = int(shard_index)
        self.shard_count = max(1, int(shard_count))
        self.vocab = vocab

    def __iter__(self):
        if self.vocab is None:
            raise RuntimeError("StreamingUniRefCorruptionDataset requires vocab for encoded samples")
        emitted = 0
        scanned = 0
        budgets = [float(value) for value in self.config["budgets"]]
        for header, raw_sequence in stream_fasta(self.input_fasta):
            scanned += 1
            if self.max_records is not None and scanned > self.max_records:
                break
            sequence = "".join(raw_sequence.split()).upper()
            if len(sequence) < self.min_len or len(sequence) > self.max_original_len:
                continue
            if self.exclude_nonstandard_aa and not set(sequence).issubset(STANDARD_AA):
                continue
            seq_hash = hashlib.sha1(sequence.encode("ascii")).hexdigest()
            if split_for_sequence(seq_hash, self.train_frac, self.val_frac) != self.split:
                continue
            if self.shard_count > 1 and stable_int(seq_hash) % self.shard_count != self.shard_index:
                continue
            seq_id = seq_id_from_header(header)
            base_seed = self.seed + stable_int(seq_hash)
            for budget_idx, budget in enumerate(budgets):
                sample_seed = base_seed + int(round(budget * 1000)) + budget_idx * 1000003
                rng = random.Random(sample_seed)
                sample, recovery_ok = corrupt_sequence(seq_id, sequence, budget, self.config, rng)
                if not recovery_ok:
                    continue
                if sample["corrupted_length"] > self.max_corrupted_len:
                    continue
                yield encode_sample(sample, self.vocab)
                emitted += 1
                if self.max_samples is not None and emitted >= self.max_samples:
                    return


def make_loader(args, config, vocab, split, max_samples=None, shard_index=0, shard_count=1):
    dataset = StreamingUniRefCorruptionDataset(
        input_fasta=args.input_fasta,
        config=config,
        split=split,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        min_len=args.min_len,
        max_original_len=args.max_original_len,
        max_corrupted_len=args.max_len,
        seed=args.seed,
        max_records=args.max_records,
        max_samples=max_samples,
        exclude_nonstandard_aa=not args.allow_nonstandard_aa,
        shard_index=shard_index,
        shard_count=shard_count,
        vocab=vocab,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=lambda batch, pad_id=vocab["pad_id"]: collate_batch(batch, pad_id),
    )


def estimate_steps_for_split(args, config, split, shard_index, shard_count):
    """Pre-scan FASTA to estimate DDP streaming steps for one epoch.

    This uses the same sequence-level filters and split/shard assignment as the
    training dataset. It intentionally does not run corruption; rare failed
    corruptions or max-corrupted-length skips can make the real step count a
    little lower.
    """

    budgets = [float(value) for value in config["budgets"]]
    shard_count = max(1, int(shard_count))
    shard_index = int(shard_index)
    count = 0
    scanned = 0
    kept_sequences = 0
    start_time = time.time()
    for header, raw_sequence in stream_fasta(args.input_fasta):
        scanned += 1
        if args.max_records is not None and scanned > args.max_records:
            break
        sequence = "".join(raw_sequence.split()).upper()
        if len(sequence) < args.min_len or len(sequence) > args.max_original_len:
            continue
        if not args.allow_nonstandard_aa and not set(sequence).issubset(STANDARD_AA):
            continue
        seq_hash = hashlib.sha1(sequence.encode("ascii")).hexdigest()
        if split_for_sequence(seq_hash, args.train_frac, args.val_frac) != split:
            continue
        shard = stable_int(seq_hash) % shard_count
        if shard != shard_index:
            continue
        count += len(budgets)
        kept_sequences += 1

    if split == "train" and args.max_train_samples_per_epoch is not None:
        count = min(count, args.max_train_samples_per_epoch)
    steps = int(ceil(count / float(max(1, args.batch_size))))
    return {
        "split": split,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "scanned_records": scanned,
        "kept_sequences": kept_sequences,
        "sample_count": count,
        "steps": steps,
        "elapsed_sec": time.time() - start_time,
    }


def train_one_epoch(
    model,
    loader,
    device,
    optimizer,
    lambda_budget,
    max_steps=None,
    log_every=100,
    rank=0,
    use_ddp_join=False,
    total_steps_hint=None,
    global_step_start=0,
    eval_every_steps=None,
    periodic_eval_fn=None,
):
    model.train(True)
    total_loss = 0.0
    steps = 0
    samples = 0
    tokens = 0
    start_time = time.time()
    if use_ddp_join and Join is None:
        raise RuntimeError("DDP full-stream training needs torch.distributed.algorithms.join.Join")
    context = Join([model]) if use_ddp_join else nullcontext()
    with context:
        for batch in loader:
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            logits = model(batch["input_ids"], batch["budget_ratio"], batch["mask"])
            loss, _, _ = compute_loss(logits, batch, lambda_budget)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            steps += 1
            samples += int(batch["input_ids"].shape[0])
            tokens += int((batch["mask"] > 0).sum().detach().cpu())
            total_loss += float(loss.detach().cpu())
            global_step = int(global_step_start) + steps
            if rank == 0 and log_every and steps % log_every == 0:
                elapsed = max(1e-6, time.time() - start_time)
                steps_per_sec = steps / elapsed
                samples_per_sec = samples / elapsed
                tokens_per_sec = tokens / elapsed
                if total_steps_hint:
                    remaining = max(0, int(total_steps_hint) - steps)
                    eta_sec = remaining / max(steps_per_sec, 1e-9)
                    progress = "{:.2f}%".format(100.0 * steps / float(max(1, int(total_steps_hint))))
                    eta = format_duration(eta_sec)
                else:
                    progress = "unknown"
                    eta = "unknown"
                print(
                    "train step {step} loss={loss:.5f} samples={samples} tokens={tokens} "
                    "elapsed={elapsed} progress={progress} eta={eta} "
                    "speed={steps_per_sec:.3f} steps/s {samples_per_sec:.1f} samples/s {tokens_per_sec:.1f} tokens/s".format(
                        step=steps,
                        loss=total_loss / steps,
                        samples=samples,
                        tokens=tokens,
                        elapsed=format_duration(elapsed),
                        progress=progress,
                        eta=eta,
                        steps_per_sec=steps_per_sec,
                        samples_per_sec=samples_per_sec,
                        tokens_per_sec=tokens_per_sec,
                    )
                )
            if periodic_eval_fn is not None and eval_every_steps and global_step % int(eval_every_steps) == 0:
                periodic_eval_fn(global_step, {
                    "loss": total_loss / float(steps) if steps else 0.0,
                    "steps": steps,
                    "samples": samples,
                    "tokens": tokens,
                })
                model.train(True)
            if max_steps is not None and steps >= max_steps:
                break
    return {
        "loss": total_loss / float(steps) if steps else 0.0,
        "steps": steps,
        "samples": samples,
        "tokens": tokens,
    }


def format_duration(seconds):
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return "{}h{}m{}s".format(hours, minutes, secs)
    if minutes:
        return "{}m{}s".format(minutes, secs)
    return "{}s".format(secs)


def evaluate(model, loader, device, lambda_budget):
    model.train(False)
    total_loss = 0.0
    steps = 0
    samples = 0
    tokens = 0
    all_labels = []
    all_probs = []
    count_abs_errors = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            logits = model(batch["input_ids"], batch["budget_ratio"], batch["mask"])
            loss, _, _ = compute_loss(logits, batch, lambda_budget)
            probs = torch.sigmoid(logits)
            mask = batch["mask"] > 0
            all_labels.extend(batch["labels"][mask].detach().cpu().tolist())
            all_probs.extend(probs[mask].detach().cpu().tolist())
            pred_counts = (probs * batch["mask"]).sum(dim=1)
            count_abs_errors.extend(torch.abs(pred_counts - batch["target_counts"]).detach().cpu().tolist())
            total_loss += float(loss.detach().cpu())
            steps += 1
            samples += int(batch["input_ids"].shape[0])
            tokens += int(mask.sum().detach().cpu())
    metrics = binary_metrics(all_labels, all_probs) if all_labels else {
        "token_auc": None,
        "token_ap": None,
        "token_f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    metrics["loss"] = total_loss / float(steps) if steps else 0.0
    metrics["deletion_count_mae"] = sum(count_abs_errors) / float(len(count_abs_errors)) if count_abs_errors else 0.0
    metrics["num_samples"] = samples
    metrics["num_tokens"] = tokens
    metrics["steps"] = steps
    return metrics


def save_checkpoint(path, model, args, vocab, epoch, val_metrics, global_step=None):
    ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "model_state_dict": unwrap_model(model).state_dict(),
            "args": vars(args),
            "vocab": vocab,
            "best_epoch": epoch,
            "global_step": global_step,
            "val_metrics": val_metrics,
            "streaming_full_uniref50": True,
        },
        path,
    )


def write_log_header(path):
    ensure_dir(os.path.dirname(path))
    fields = [
        "event",
        "epoch",
        "global_step",
        "train_loss",
        "train_steps",
        "train_samples",
        "train_tokens",
        "val_loss",
        "val_token_auc",
        "val_token_ap",
        "val_token_f1",
        "val_precision",
        "val_recall",
        "val_deletion_count_mae",
        "val_samples",
        "val_tokens",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
    return fields


def append_log(path, fields, row):
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writerow(row)


def train(args):
    set_seed(args.seed)
    distributed = setup_distributed(args)
    if distributed["enabled"] and args.eval_every_steps and args.steps_per_epoch is None:
        raise RuntimeError("DDP periodic validation requires --steps_per_epoch so all ranks enter validation together")
    if is_main_process(distributed):
        ensure_dir(args.out_dir)
        ensure_dir(args.ckpt_dir)
    config = load_config(args.config)
    config["seed"] = args.seed
    vocab = build_vocab()
    device = distributed["device"]
    model = build_model(args, vocab).to(device)
    if args.init_checkpoint:
        checkpoint = load_initial_checkpoint(
            model,
            args.init_checkpoint,
            device,
            strict=not args.init_non_strict,
            expected_vocab=vocab,
        )
        rank_print(
            distributed,
            "initialized model from {} (best_epoch={}, global_step={})".format(
                args.init_checkpoint,
                checkpoint.get("best_epoch"),
                checkpoint.get("global_step"),
            ),
        )
    if distributed["enabled"]:
        ddp_kwargs = {}
        if device.type == "cuda":
            ddp_kwargs.update({"device_ids": [distributed["local_rank"]], "output_device": distributed["local_rank"]})
        model = DistributedDataParallel(model, **ddp_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score = -float("inf")
    best_epoch = None
    best_step = None
    global_step = 0
    best_path = os.path.join(args.ckpt_dir, "best_model.pt")
    last_path = os.path.join(args.ckpt_dir, "last_model.pt")
    log_path = os.path.join(args.out_dir, "train_log.csv")
    fields = write_log_header(log_path) if is_main_process(distributed) else None
    barrier(distributed)
    estimated_train_steps = None
    if args.estimate_steps_before_train:
        rank_print(distributed, "estimating train epoch steps from {}".format(args.input_fasta))
        estimate = estimate_steps_for_split(
            args,
            config,
            "train",
            distributed["rank"],
            distributed["world_size"],
        )
        estimated_train_steps = estimate["steps"]
        if is_main_process(distributed):
            print(
                "rank0 estimated train samples: {}; steps: {}; scan_elapsed={}".format(
                    estimate["sample_count"],
                    estimate["steps"],
                    format_duration(estimate["elapsed_sec"]),
                )
            )
            write_json(os.path.join(args.out_dir, "train_step_estimate.json"), estimate)
        barrier(distributed)

    def run_validation(event, epoch, current_global_step, train_metrics, max_val_samples):
        nonlocal best_score, best_epoch, best_step
        barrier(distributed)
        if is_main_process(distributed):
            print("{}: evaluating val split at global_step {}".format(event, current_global_step))
            val_loader = make_loader(args, config, vocab, "val", max_samples=max_val_samples)
            val_metrics = evaluate(unwrap_model(model), val_loader, device, args.lambda_budget)
            row = {
                "event": event,
                "epoch": epoch,
                "global_step": current_global_step,
                "train_loss": train_metrics["loss"],
                "train_steps": train_metrics["steps"],
                "train_samples": train_metrics["samples"],
                "train_tokens": train_metrics["tokens"],
                "val_loss": val_metrics["loss"],
                "val_token_auc": val_metrics["token_auc"],
                "val_token_ap": val_metrics["token_ap"],
                "val_token_f1": val_metrics["token_f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_deletion_count_mae": val_metrics["deletion_count_mae"],
                "val_samples": val_metrics["num_samples"],
                "val_tokens": val_metrics["num_tokens"],
            }
            append_log(log_path, fields, row)
            print(
                "{event} epoch={epoch} global_step={global_step}: train_loss={train_loss:.5f} "
                "val_loss={val_loss:.5f} val_ap={val_token_ap} val_auc={val_token_auc} "
                "val_f1={val_token_f1:.5f} val_recall={val_recall:.5f}".format(**row)
            )
            score = val_metrics["token_ap"] if val_metrics["token_ap"] is not None else -val_metrics["loss"]
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_step = current_global_step
                save_checkpoint(best_path, model, args, vocab, epoch, val_metrics, global_step=current_global_step)
                print("saved best checkpoint: {}".format(best_path))
            if args.save_last_checkpoint:
                save_checkpoint(last_path, model, args, vocab, epoch, val_metrics, global_step=current_global_step)
        barrier(distributed)

    for epoch in range(1, args.epochs + 1):
        rank_print(distributed, "epoch {}: streaming train split from {}".format(epoch, args.input_fasta))
        train_loader = make_loader(
            args,
            config,
            vocab,
            "train",
            max_samples=args.max_train_samples_per_epoch,
            shard_index=distributed["rank"] if distributed["enabled"] else 0,
            shard_count=distributed["world_size"] if distributed["enabled"] else 1,
        )
        train_metrics = train_one_epoch(
            model,
            train_loader,
            device,
            optimizer,
            args.lambda_budget,
            max_steps=args.steps_per_epoch,
            log_every=args.log_every,
            rank=distributed["rank"],
            use_ddp_join=distributed["enabled"] and not args.disable_ddp_join and not args.eval_every_steps,
            total_steps_hint=args.steps_per_epoch or args.estimated_steps_per_epoch or estimated_train_steps,
            global_step_start=global_step,
            eval_every_steps=args.eval_every_steps,
            periodic_eval_fn=(
                lambda step, metrics, current_epoch=epoch: run_validation(
                    "step_eval",
                    current_epoch,
                    step,
                    metrics,
                    args.periodic_max_val_samples or args.max_val_samples,
                )
            ) if args.eval_every_steps else None,
        )
        global_step += train_metrics["steps"]
        run_validation("epoch_end", epoch, global_step, train_metrics, args.max_val_samples)

    if is_main_process(distributed):
        print("loading best checkpoint for test evaluation")
        checkpoint = torch_load(best_path, map_location=device)
        unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
        test_loader = make_loader(args, config, vocab, "test", max_samples=args.max_test_samples)
        test_metrics = evaluate(unwrap_model(model), test_loader, device, args.lambda_budget)
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
            "best_step": best_step,
            "streaming_full_uniref50": True,
            "distributed_world_size": distributed["world_size"],
        }
        write_json(os.path.join(args.out_dir, "test_metrics.json"), output_test_metrics)
        write_json(os.path.join(args.out_dir, "config.json"), vars(args))
        write_json(os.path.join(args.out_dir, "feature_or_vocab.json"), vocab)
        print("best_model: {}".format(best_path))
        print("test_metrics: {}".format(os.path.join(args.out_dir, "test_metrics.json")))
    barrier(distributed)
    cleanup_distributed(distributed)


def parse_args():
    parser = argparse.ArgumentParser(description="Stream-train Stage-1 deletion utility prior from full UniRef50.")
    parser.add_argument("--input_fasta", default="/public/home/zhangyangroup/chengshiz/keyuan.zhou/data/raw/uniref50/uniref50.fasta.gz")
    parser.add_argument("--config", default="configs/stage1_corruption.yaml")
    parser.add_argument("--out_dir", default="results/stage1_deletion_prior_full_uniref50_stream")
    parser.add_argument("--ckpt_dir", default="checkpoints/stage1_deletion_prior_full_uniref50_stream")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--init_checkpoint", default=None, help="Optional checkpoint to initialize/fine-tune from.")
    parser.add_argument("--init_non_strict", action="store_true", help="Allow missing/unexpected keys when loading --init_checkpoint.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ddp_backend", default=None, help="Distributed backend. Defaults to nccl on CUDA and gloo on CPU.")
    parser.add_argument("--ddp_timeout_hours", type=float, default=6.0)
    parser.add_argument("--disable_ddp_join", action="store_true", help="Disable DDP Join context for uneven streaming shards.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda_budget", type=float, default=0.1)
    parser.add_argument("--model_type", choices=["transformer", "bilstm", "cnn"], default="transformer")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=1200)
    parser.add_argument("--min_len", type=int, default=80)
    parser.add_argument("--max_original_len", type=int, default=800)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--max_records", type=int, default=None, help="Debug cap on raw FASTA records scanned.")
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="Debug/time cap. Omit for a full streaming pass.")
    parser.add_argument("--max_train_samples_per_epoch", type=int, default=None, help="Optional train sample cap; omit for full split.")
    parser.add_argument("--max_val_samples", type=int, default=30000)
    parser.add_argument("--max_test_samples", type=int, default=30000)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--eval_every_steps", type=int, default=None, help="Run validation and checkpointing every N optimizer steps.")
    parser.add_argument("--periodic_max_val_samples", type=int, default=None, help="Validation sample cap for --eval_every_steps; defaults to --max_val_samples.")
    parser.add_argument("--save_last_checkpoint", action="store_true", help="Also write last_model.pt after each validation.")
    parser.add_argument("--estimated_steps_per_epoch", type=int, default=None, help="Optional progress/ETA hint for full streaming runs.")
    parser.add_argument("--estimate_steps_before_train", action="store_true", help="Pre-scan FASTA to estimate total train steps and enable ETA.")
    parser.add_argument("--allow_nonstandard_aa", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
