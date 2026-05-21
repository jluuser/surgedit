#!/usr/bin/env python3
"""Train a SCISOR-style Stage-1 deletion proposer from streaming corrupted proteins.

This keeps the SCISOR diffusion objective and reverse-deletion posterior, but
uses the Biodel corruption stream as the training source and the learned model
as a high-recall deletion proposal prior for downstream certified planning.
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
import random
import time
from collections import Counter, OrderedDict
from datetime import timedelta
from math import ceil

try:
    from contextlib import nullcontext
except ImportError:  # pragma: no cover
    class nullcontext(object):
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

try:
    from torch.distributed.algorithms.join import Join
except ImportError:  # pragma: no cover
    Join = None

from SCISOR.alignments import get_deletion_log_alignments
from SCISOR.shortening_scud import ShorteningSCUD

from build_stage1_deletion_pretrain_dataset import corrupt_sequence, load_config
from train_stage1_deletion_prior import binary_metrics


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_SCISOR_CKPT = "/public/home/zhangyangroup/chengshiz/.cache/shortening_scud/SCISOR_U90_S.ckpt"
DEFAULT_P0 = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit/p0.pt"
DEFAULT_HF_CACHE = "/public/home/zhangyangroup/chengshiz/.cache/huggingface/hub"


def resolve_local_hf_snapshot(model_name, cache_dir=DEFAULT_HF_CACHE):
    """Return a local HF snapshot path when available, else the original model id."""
    cache_dir = os.environ.get("BIODEL_HF_CACHE", cache_dir)
    if os.path.isdir(model_name):
        return model_name
    if "/" not in model_name:
        return model_name
    namespace, repo = model_name.split("/", 1)
    repo_dir = os.path.join(cache_dir, "models--{}--{}".format(namespace, repo))
    snapshots_dir = os.path.join(repo_dir, "snapshots")
    refs_main = os.path.join(repo_dir, "refs", "main")
    if not os.path.isdir(snapshots_dir):
        return model_name
    snapshot = None
    if os.path.exists(refs_main):
        with open(refs_main) as handle:
            ref = handle.read().strip()
        candidate = os.path.join(snapshots_dir, ref)
        if os.path.isdir(candidate):
            snapshot = candidate
    if snapshot is None:
        candidates = [
            os.path.join(snapshots_dir, name)
            for name in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, name))
        ]
        if candidates:
            snapshot = sorted(candidates)[-1]
    if snapshot and os.path.exists(os.path.join(snapshot, "config.json")):
        return snapshot
    return model_name


class FAESMBaseNoFA(nn.Module):
    """FAESM wrapper that forces PyTorch SDPA instead of flash-attention."""

    def __init__(self, hf_model_name="esm2_t6_8M_UR50D", **kwargs):
        super().__init__()
        from SCISOR.esm import FAEsmForMaskedLM

        print("Using FAESM model {} with use_fa=False".format(hf_model_name))
        conditioning_dim = kwargs.get("d_embedding", 128)
        pretrained = kwargs.get("pretrained", True)
        model_name_or_path = resolve_local_hf_snapshot("facebook/{}".format(hf_model_name))
        print("Loading FAESM backbone from {}".format(model_name_or_path))
        self.faesm = FAEsmForMaskedLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            use_fa=False,
            conditioning_dim=conditioning_dim,
            load_pretrained_weights=pretrained,
        )
        self.embed_dim = self.faesm.esm.embeddings.word_embeddings.embedding_dim
        self.proj = nn.Linear(self.embed_dim, 1)

    def forward(self, x, t, input_mask=None, S=None):
        cond = t if S is None else S
        embeddings = self.faesm(
            input_ids=x,
            attention_mask=input_mask,
            conditioning=cond,
        )["last_hidden_state"]
        return self.proj(embeddings).squeeze()


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)


def write_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


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


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_sequence(sequence, tokenizer):
    return tokenizer(sequence).input_ids


class LRUSequenceEncoder:
    def __init__(self, tokenizer, cache_size=200000, use_fast=True):
        self.tokenizer = tokenizer
        self.cache_size = int(cache_size or 0)
        self.use_fast = bool(use_fast)
        self.cache = OrderedDict()
        self.cls_id = getattr(tokenizer, "cls_token_id", None)
        self.eos_id = getattr(tokenizer, "eos_token_id", None)
        self.char_to_id = {}
        if self.use_fast and self.cls_id is not None and self.eos_id is not None:
            for aa in STANDARD_AA:
                token_id = tokenizer.convert_tokens_to_ids(aa)
                if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
                    self.use_fast = False
                    break
                self.char_to_id[aa] = int(token_id)
        else:
            self.use_fast = False

    def _encode_uncached(self, sequence):
        if self.use_fast:
            try:
                return [self.cls_id] + [self.char_to_id[aa] for aa in sequence] + [self.eos_id]
            except KeyError:
                pass
        return encode_sequence(sequence, self.tokenizer)

    def __call__(self, sequence):
        if self.cache_size <= 0:
            return self._encode_uncached(sequence)
        cached = self.cache.get(sequence)
        if cached is not None:
            self.cache.move_to_end(sequence)
            return cached
        encoded = self._encode_uncached(sequence)
        self.cache[sequence] = encoded
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return encoded


class SCISORCollator:
    def __init__(self, pad_id, tokenizer, use_fast_tokenizer=True, tokenizer_cache_size=200000, batch_alignments=True):
        self.pad_id = pad_id
        self.encoder = LRUSequenceEncoder(
            tokenizer,
            cache_size=tokenizer_cache_size,
            use_fast=use_fast_tokenizer,
        )
        self.batch_alignments = bool(batch_alignments)

    def __call__(self, items):
        return pad_batch(items, self.pad_id, self.encoder, batch_alignments=self.batch_alignments)


def pad_batch(items, pad_id, encoder, batch_alignments=True):
    x0_ids = [encoder(item["original_sequence"]) for item in items]
    xt_ids = [encoder(item["corrupted_sequence"]) for item in items]
    labels = [[0.0] + [float(v) for v in item["delete_labels"]] + [0.0] for item in items]
    budgets = [[float(item["budget_ratio"])] for item in items]
    s_values = [float(item["inserted_total_length"]) for item in items]
    seq_ids = [item["seq_id"] for item in items]
    original_lengths = [int(item["original_length"]) for item in items]
    corrupted_lengths = [int(item["corrupted_length"]) for item in items]

    log_alignments = []
    max_x0 = max(len(ids) for ids in x0_ids)
    max_xt = max(len(ids) for ids in xt_ids)
    x0_padded = []
    xt_padded = []
    mask = []
    labels_padded = []
    for x0, xt, lab in zip(x0_ids, xt_ids, labels):
        x0_padded.append(x0 + [pad_id] * (max_x0 - len(x0)))
        xt_padded.append(xt + [pad_id] * (max_xt - len(xt)))
        mask.append([1.0] * len(xt) + [0.0] * (max_xt - len(xt)))
        labels_padded.append(lab + [0.0] * (max_xt - len(lab)))

    x0_tensor = torch.tensor(x0_padded, dtype=torch.long)
    xt_tensor = torch.tensor(xt_padded, dtype=torch.long)
    if batch_alignments:
        log_alignments = get_deletion_log_alignments(x0_tensor, xt_tensor, pad_id=pad_id)[0]
    else:
        log_alignments = []
        for x0, xt in zip(x0_ids, xt_ids):
            x0_single = torch.tensor([x0], dtype=torch.long)
            xt_single = torch.tensor([xt], dtype=torch.long)
            align = get_deletion_log_alignments(x0_single, xt_single, pad_id=pad_id)[0]
            log_alignments.append(align.squeeze(0).tolist())
        log_alignments = torch.tensor(
            [align + [float("-inf")] * (max_xt - len(align)) for align in log_alignments],
            dtype=torch.float32,
        )

    return {
        "x0": x0_tensor,
        "x_t": xt_tensor,
        "mask": torch.tensor(mask, dtype=torch.float32),
        "labels": torch.tensor(labels_padded, dtype=torch.float32),
        "budget_ratio": torch.tensor(budgets, dtype=torch.float32),
        "inserted_total_length": torch.tensor(s_values, dtype=torch.float32),
        "log_alignments": log_alignments.float(),
        "seq_ids": seq_ids,
        "original_lengths": torch.tensor(original_lengths, dtype=torch.long),
        "corrupted_lengths": torch.tensor(corrupted_lengths, dtype=torch.long),
    }


class StreamingSCISORCorruptionDataset(IterableDataset):
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
        length_bucket_size=0,
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
        self.length_bucket_size = int(length_bucket_size or 0)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        worker_count = int(worker.num_workers) if worker is not None else 1
        effective_shard_index = self.shard_index * worker_count + worker_id
        effective_shard_count = self.shard_count * worker_count
        max_samples = self.max_samples
        if max_samples is not None and worker_count > 1:
            max_samples = int(ceil(float(max_samples) / float(worker_count)))
        emitted = 0
        scanned = 0
        budgets = [float(value) for value in self.config["budgets"]]
        bucket = []

        def flush_bucket():
            nonlocal emitted, bucket
            if self.length_bucket_size > 0:
                bucket.sort(key=lambda item: item["corrupted_length"])
            for buffered in bucket:
                if max_samples is not None and emitted >= max_samples:
                    break
                emitted += 1
                yield buffered
            bucket = []

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
            if effective_shard_count > 1 and stable_int(seq_hash) % effective_shard_count != effective_shard_index:
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
                item = {
                    "seq_id": seq_id,
                    "original_sequence": sample["original_sequence"],
                    "corrupted_sequence": sample["corrupted_sequence"],
                    "delete_labels": sample["delete_labels"],
                    "budget_ratio": budget,
                    "inserted_total_length": sample["inserted_total_length"],
                    "original_length": sample["original_length"],
                    "corrupted_length": sample["corrupted_length"],
                }
                if self.length_bucket_size > 0:
                    bucket.append(item)
                    if len(bucket) >= self.length_bucket_size:
                        for buffered in flush_bucket():
                            yield buffered
                        if max_samples is not None and emitted >= max_samples:
                            return
                else:
                    yield item
                    emitted += 1
                    if max_samples is not None and emitted >= max_samples:
                        return
        if bucket:
            for buffered in flush_bucket():
                yield buffered


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


def barrier(state):
    if state.get("enabled") and dist.is_initialized():
        dist.barrier()


def is_main_process(state):
    return int(state.get("rank", 0)) == 0


def rank_print(state, message):
    if is_main_process(state):
        print(message)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_stage1_model(args, device):
    model = ShorteningSCUD.load_from_checkpoint(args.checkpoint, map_location=device)
    model.to(device)
    model.eval()
    for attr, value in {
        "_unk_token": "<unk>",
        "_cls_token": "<cls>",
        "_bos_token": "<cls>",
        "_eos_token": "<eos>",
        "_sep_token": "<eos>",
        "_pad_token": "<pad>",
        "_mask_token": "<mask>",
        "_additional_special_tokens": [],
    }.items():
        if not hasattr(model.tokenizer, attr):
            setattr(model.tokenizer, attr, value)
    model.p0 = torch.load(args.p0, map_location=device)
    rate = 1 / 1.1
    model.alpha = lambda t: (1 - t) ** rate
    model.beta = lambda t: rate / (1 - t)
    model.log_alpha = lambda t: -torch.log(model.alpha(t))
    return model


def use_preprocessed_forward(model):
    """Route DDP calls through the precomputed-alignment training objective."""

    model.forward = model.forward_preprocessed
    return model


def mean(values):
    return sum(values) / float(len(values)) if values else 0.0


def format_duration(seconds):
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return "{}h{}m{}s".format(hours, minutes, secs)
    if minutes:
        return "{}m{}s".format(minutes, secs)
    return "{}s".format(secs)


def write_log_header(path):
    ensure_dir(os.path.dirname(path))
    fields = [
        "event",
        "epoch",
        "global_step",
        "train_loss",
        "train_vb_loss",
        "train_samples",
        "train_tokens",
        "val_loss",
        "val_vb_loss",
        "val_token_ap",
        "val_token_auc",
        "val_token_f1",
        "val_precision",
        "val_recall",
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


def build_loaders(args, config, tokenizer, pad_id, split, max_samples=None, shard_index=0, shard_count=1):
    dataset = StreamingSCISORCorruptionDataset(
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
        length_bucket_size=args.length_bucket_size if split == "train" else args.eval_length_bucket_size,
    )
    collator = SCISORCollator(
        pad_id,
        tokenizer,
        use_fast_tokenizer=not args.disable_fast_tokenizer,
        tokenizer_cache_size=args.tokenizer_cache_size,
        batch_alignments=not args.disable_batch_alignments,
    )
    num_workers = int(args.num_workers if split == "train" else args.eval_num_workers)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": num_workers,
        "collate_fn": collator,
        "pin_memory": bool(args.pin_memory),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
    return DataLoader(
        dataset,
        **loader_kwargs,
    )


def estimate_steps_for_split(args, config, split, shard_index, shard_count):
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


def compute_metrics_from_batch(model, batch, device):
    x_t = batch["x_t"].to(device)
    t = batch["budget_ratio"].squeeze(-1).to(device).clamp(0.05, 0.95)
    S = batch["inserted_total_length"].to(device)
    labels = batch["labels"].to(device)
    mask = batch["mask"].to(device)
    logits = model.model_predict(x_t.long(), t.float(), None, S.float())
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    logits = logits.masked_fill(mask <= 0, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return labels, probs


def evaluate(model, loader, device, non_blocking=False):
    model.eval()
    total_loss = 0.0
    total_vb = 0.0
    steps = 0
    samples = 0
    tokens = 0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device, non_blocking=non_blocking) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            x0 = batch["x0"]
            x_t = batch["x_t"]
            t = batch["budget_ratio"].squeeze(-1).clamp(0.05, 0.95)
            S = batch["inserted_total_length"]
            log_alignments = batch["log_alignments"]
            loss, info = model.forward_preprocessed(x0.long(), t.float(), S.float(), x_t.long(), log_alignments.float())
            labels, probs = compute_metrics_from_batch(model, batch, device)
            mask = batch["mask"] > 0
            all_labels.extend(labels[mask].detach().cpu().tolist())
            all_probs.extend(probs[mask].detach().cpu().tolist())
            total_loss += float(loss.detach().cpu())
            total_vb += float(info["vb_loss"])
            steps += 1
            samples += int(x0.shape[0])
            tokens += int(mask.sum().detach().cpu())
    metrics = binary_metrics(all_labels, all_probs) if all_labels else {
        "token_auc": None,
        "token_ap": None,
        "token_f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    metrics["loss"] = total_loss / float(steps) if steps else 0.0
    metrics["vb_loss"] = total_vb / float(steps) if steps else 0.0
    metrics["num_samples"] = samples
    metrics["num_tokens"] = tokens
    metrics["steps"] = steps
    return metrics


def save_checkpoint(path, model, args, epoch, val_metrics, global_step=None):
    ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "state_dict": unwrap_model(model).state_dict(),
            "args": vars(args),
            "best_epoch": epoch,
            "global_step": global_step,
            "val_metrics": val_metrics,
            "stage1_mode": "scisor_style_deletion_proposer",
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train a SCISOR-style Stage-1 deletion proposer.")
    parser.add_argument("--input_fasta", default="/public/home/zhangyangroup/chengshiz/keyuan.zhou/data/raw/uniref50/uniref50.fasta.gz")
    parser.add_argument("--config", default="configs/stage1_corruption.yaml")
    parser.add_argument("--checkpoint", default=DEFAULT_SCISOR_CKPT)
    parser.add_argument("--p0", default=DEFAULT_P0)
    parser.add_argument("--out_dir", default="results/stage1_scisor_style_full_uniref50_stream")
    parser.add_argument("--ckpt_dir", default="checkpoints/stage1_scisor_style_full_uniref50_stream")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ddp_backend", default=None)
    parser.add_argument("--ddp_timeout_hours", type=float, default=6.0)
    parser.add_argument("--disable_ddp_join", action="store_true")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--min_len", type=int, default=80)
    parser.add_argument("--max_original_len", type=int, default=800)
    parser.add_argument("--max_len", type=int, default=1200)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--max_train_samples_per_epoch", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--length_bucket_size", type=int, default=0)
    parser.add_argument("--eval_length_bucket_size", type=int, default=0)
    parser.add_argument("--tokenizer_cache_size", type=int, default=200000)
    parser.add_argument("--disable_fast_tokenizer", action="store_true")
    parser.add_argument("--disable_batch_alignments", action="store_true")
    parser.add_argument("--max_val_samples", type=int, default=30000)
    parser.add_argument("--max_test_samples", type=int, default=30000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every_steps", type=int, default=None)
    parser.add_argument("--periodic_max_val_samples", type=int, default=None)
    parser.add_argument("--save_last_checkpoint", action="store_true")
    parser.add_argument("--estimated_steps_per_epoch", type=int, default=None)
    parser.add_argument("--estimate_steps_before_train", action="store_true")
    parser.add_argument("--allow_nonstandard_aa", action="store_true")
    parser.add_argument(
        "--disable-fa",
        action="store_true",
        help="Instantiate FAESM with PyTorch SDPA instead of flash-attention paths.",
    )
    args = parser.parse_args()

    if args.disable_fa:
        import SCISOR.continuous_time_diffusion as ctd

        ctd.FAESM_Base = FAESMBaseNoFA

    set_seed(args.seed)
    config = load_config(args.config)
    config["seed"] = args.seed
    distributed = setup_distributed(args)
    if distributed["enabled"] and args.eval_every_steps and args.steps_per_epoch is None:
        raise RuntimeError("DDP periodic validation requires --steps_per_epoch so all ranks enter validation together")
    if is_main_process(distributed):
        ensure_dir(args.out_dir)
        ensure_dir(args.ckpt_dir)

    device = distributed["device"]
    model = load_stage1_model(args, device)
    model = use_preprocessed_forward(model)
    if distributed["enabled"]:
        ddp_kwargs = {"find_unused_parameters": True}
        if device.type == "cuda":
            ddp_kwargs.update({"device_ids": [distributed["local_rank"]], "output_device": distributed["local_rank"]})
        model = DDP(model, **ddp_kwargs)
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
                    estimate["sample_count"], estimate["steps"], format_duration(estimate["elapsed_sec"])
                )
            )
            write_json(os.path.join(args.out_dir, "train_step_estimate.json"), estimate)
        barrier(distributed)

    def run_validation(event, epoch, current_global_step, train_metrics, max_val_samples):
        nonlocal best_score, best_epoch, best_step
        barrier(distributed)
        if is_main_process(distributed):
            print("{}: evaluating val split at global_step {}".format(event, current_global_step))
            val_loader = build_loaders(
                args,
                config,
                unwrap_model(model).tokenizer,
                unwrap_model(model).tokenizer.pad_token_id,
                "val",
                max_samples=max_val_samples,
            )
            val_metrics = evaluate(unwrap_model(model), val_loader, device, non_blocking=args.pin_memory)
            row = {
                "event": event,
                "epoch": epoch,
                "global_step": current_global_step,
                "train_loss": train_metrics["loss"],
                "train_vb_loss": train_metrics["vb_loss"],
                "train_samples": train_metrics["samples"],
                "train_tokens": train_metrics["tokens"],
                "val_loss": val_metrics["loss"],
                "val_vb_loss": val_metrics["vb_loss"],
                "val_token_ap": val_metrics["token_ap"],
                "val_token_auc": val_metrics["token_auc"],
                "val_token_f1": val_metrics["token_f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
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
                save_checkpoint(best_path, model, args, epoch, val_metrics, global_step=current_global_step)
                print("saved best checkpoint: {}".format(best_path))
            if args.save_last_checkpoint:
                save_checkpoint(last_path, model, args, epoch, val_metrics, global_step=current_global_step)
        barrier(distributed)

    for epoch in range(1, args.epochs + 1):
        rank_print(distributed, "epoch {}: streaming train split from {}".format(epoch, args.input_fasta))
        train_loader = build_loaders(
            args,
            config,
            unwrap_model(model).tokenizer,
            unwrap_model(model).tokenizer.pad_token_id,
            "train",
            max_samples=args.max_train_samples_per_epoch,
            shard_index=distributed["rank"] if distributed["enabled"] else 0,
            shard_count=distributed["world_size"] if distributed["enabled"] else 1,
        )
        total_loss = 0.0
        total_vb = 0.0
        steps = 0
        samples = 0
        tokens = 0
        start_time = time.time()
        total_steps_hint = args.steps_per_epoch or args.estimated_steps_per_epoch or estimated_train_steps
        if Join is None and distributed["enabled"] and not args.disable_ddp_join and not args.eval_every_steps:
            raise RuntimeError("DDP full-stream training needs torch.distributed.algorithms.join.Join")
        context = Join([model]) if (distributed["enabled"] and not args.disable_ddp_join and not args.eval_every_steps) else nullcontext()
        with context:
            for batch in train_loader:
                batch_tokens = int(batch["mask"].sum().item())
                batch = {
                    key: value.to(device, non_blocking=args.pin_memory) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                x0 = batch["x0"].long()
                x_t = batch["x_t"].long()
                t = batch["budget_ratio"].squeeze(-1).float().clamp(0.05, 0.95)
                S = batch["inserted_total_length"].float()
                log_alignments = batch["log_alignments"].float()
                optimizer.zero_grad(set_to_none=True)
                loss, info = model(x0, t, S, x_t, log_alignments)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                steps += 1
                samples += int(x0.shape[0])
                tokens += batch_tokens
                total_loss += float(loss.detach().cpu())
                total_vb += float(info["vb_loss"])
                global_step += 1
                if is_main_process(distributed) and args.log_every and steps % args.log_every == 0:
                    elapsed = max(1e-6, time.time() - start_time)
                    steps_per_sec = steps / elapsed
                    samples_per_sec = samples / elapsed
                    tokens_per_sec = tokens / elapsed
                    if total_steps_hint:
                        remaining = max(0, int(total_steps_hint) - steps)
                        eta = format_duration(remaining / max(steps_per_sec, 1e-9))
                        progress = "{:.2f}%".format(100.0 * steps / float(max(1, int(total_steps_hint))))
                    else:
                        eta = "unknown"
                        progress = "unknown"
                    print(
                        "train step {step} loss={loss:.5f} vb={vb:.5f} samples={samples} tokens={tokens} "
                        "elapsed={elapsed} progress={progress} eta={eta} "
                        "speed={steps_per_sec:.3f} steps/s {samples_per_sec:.1f} samples/s {tokens_per_sec:.1f} tokens/s".format(
                            step=steps,
                            loss=total_loss / float(steps),
                            vb=total_vb / float(steps),
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
                if args.eval_every_steps and global_step % int(args.eval_every_steps) == 0:
                    run_validation(
                        "step_eval",
                        epoch,
                        global_step,
                        {
                            "loss": total_loss / float(max(1, steps)),
                            "vb_loss": total_vb / float(max(1, steps)),
                            "samples": samples,
                            "tokens": tokens,
                        },
                        args.periodic_max_val_samples or args.max_val_samples,
                    )
                    model.train(True)
                if args.steps_per_epoch is not None and steps >= args.steps_per_epoch:
                    break
        train_metrics = {
            "loss": total_loss / float(max(1, steps)),
            "vb_loss": total_vb / float(max(1, steps)),
            "samples": samples,
            "tokens": tokens,
        }
        run_validation("epoch_end", epoch, global_step, train_metrics, args.max_val_samples)

    if is_main_process(distributed):
        print("loading best checkpoint for test evaluation")
        checkpoint = torch_load(best_path, map_location=device)
        unwrap_model(model).load_state_dict(checkpoint["state_dict"])
        test_loader = build_loaders(
            args,
            config,
            unwrap_model(model).tokenizer,
            unwrap_model(model).tokenizer.pad_token_id,
            "test",
            max_samples=args.max_test_samples,
        )
        test_metrics = evaluate(unwrap_model(model), test_loader, device, non_blocking=args.pin_memory)
        output_test_metrics = {
            "token_auc": test_metrics["token_auc"],
            "token_ap": test_metrics["token_ap"],
            "token_f1": test_metrics["token_f1"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "num_test_samples": test_metrics["num_samples"],
            "num_test_tokens": test_metrics["num_tokens"],
            "best_epoch": best_epoch,
            "best_step": best_step,
            "stage1_mode": "scisor_style_deletion_proposer",
        }
        write_json(os.path.join(args.out_dir, "test_metrics.json"), output_test_metrics)
        write_json(os.path.join(args.out_dir, "config.json"), vars(args))
        write_json(os.path.join(args.out_dir, "feature_or_vocab.json"), {
            "checkpoint": args.checkpoint,
            "p0": args.p0,
            "tokenizer": str(type(unwrap_model(model).tokenizer)),
        })
        print("best_model: {}".format(best_path))
        print("test_metrics: {}".format(os.path.join(args.out_dir, "test_metrics.json")))

    barrier(distributed)
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
