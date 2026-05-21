#!/usr/bin/env python3
"""Convert a SCISOR-style Stage-1 deletion posterior into interval proposals.

Stage 1 is trained with SCISOR's reverse-deletion posterior objective, so its
native output is a token-level deletion distribution.  The certified planner
expects contiguous deletion intervals.  This script bridges the two: score each
protein with the trained Stage-1 model, smooth token scores, extract high-score
runs, and write BioDel-compatible interval candidates.
"""

import argparse
import csv
import gzip
import json
import os
import sys
from collections import Counter

import torch
import torch.nn.functional as F


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_stage1_scisor_style_streaming import (  # noqa: E402
    DEFAULT_HF_CACHE,
    FAESMBaseNoFA,
    resolve_local_hf_snapshot,
    torch_load,
)
from SCISOR.shortening_scud import ShorteningSCUD  # noqa: E402


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


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


def accession_from_header(header):
    return header.split(None, 1)[0].split("|", 1)[0]


def set_tokenizer_defaults(tokenizer):
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
        if not hasattr(tokenizer, attr):
            setattr(tokenizer, attr, value)


def load_model(args, device):
    if args.disable_fa:
        import SCISOR.continuous_time_diffusion as ctd

        ctd.FAESM_Base = FAESMBaseNoFA
    checkpoint = torch_load(args.checkpoint, map_location=device)
    train_args = checkpoint.get("args", {})
    base_checkpoint = args.base_checkpoint or train_args.get("checkpoint")
    if not base_checkpoint:
        raise ValueError(
            "Could not infer the base SCISOR checkpoint. Pass --base_checkpoint."
        )
    model = ShorteningSCUD.load_from_checkpoint(base_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.to(device)
    model.eval()
    set_tokenizer_defaults(model.tokenizer)
    model.p0 = torch.load(args.p0, map_location=device)
    rate = 1 / 1.1
    model.alpha = lambda t: (1 - t) ** rate
    model.beta = lambda t: rate / (1 - t)
    model.log_alpha = lambda t: -torch.log(model.alpha(t))
    return model, checkpoint


def token_scores(model, sequences, budget, device):
    encoded = [model.tokenizer(sequence).input_ids for sequence in sequences]
    max_len = max(len(ids) for ids in encoded)
    x = torch.full(
        (len(encoded), max_len),
        model.tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    for row_idx, ids in enumerate(encoded):
        x[row_idx, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    valid_mask = (
        (x != model.tokenizer.pad_token_id)
        & (x != model.tokenizer.cls_token_id)
        & (x != model.tokenizer.eos_token_id)
    )
    s_values = torch.tensor(
        [max(1, int(round(len(sequence) * float(budget)))) for sequence in sequences],
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        if model.window_size is not None and x.shape[1] > model.window_size:
            probs = model.predict_with_windows(x, None, s_values)
        else:
            logits = model.model_predict(x, None, None, s_values)
            logits = logits.masked_fill(~valid_mask, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
    scores = []
    for row_idx, sequence in enumerate(sequences):
        row_scores = probs[row_idx][valid_mask[row_idx]].detach().cpu().tolist()
        scores.append(row_scores[: len(sequence)])
    return scores


def moving_average(values, window):
    if window <= 1 or len(values) <= 1:
        return list(values)
    tensor = torch.tensor(values, dtype=torch.float32).view(1, 1, -1)
    pad = window // 2
    padded = F.pad(tensor, (pad, pad), mode="replicate")
    kernel = torch.ones(1, 1, window, dtype=torch.float32) / float(window)
    return F.conv1d(padded, kernel).view(-1)[: len(values)].tolist()


def merge_close_intervals(intervals, max_gap):
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        prev = merged[-1]
        if start - prev[1] - 1 <= max_gap:
            prev[1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def propose_intervals_for_scores(scores, args):
    if not scores:
        return []
    smoothed = moving_average(scores, args.smooth_window)
    ranked = sorted(smoothed)
    quantile_index = int(round((len(ranked) - 1) * float(args.score_quantile)))
    threshold = ranked[max(0, min(len(ranked) - 1, quantile_index))]
    raw = []
    start = None
    for idx, score in enumerate(smoothed):
        if score >= threshold:
            if start is None:
                start = idx
        elif start is not None:
            raw.append((start, idx - 1))
            start = None
    if start is not None:
        raw.append((start, len(smoothed) - 1))
    intervals = merge_close_intervals(raw, args.merge_gap)
    filtered = []
    for start, end in intervals:
        length = end - start + 1
        if length < args.min_interval_len:
            continue
        if length > args.max_interval_len:
            chunk_start = start
            while chunk_start <= end:
                chunk_end = min(end, chunk_start + args.max_interval_len - 1)
                if chunk_end - chunk_start + 1 >= args.min_interval_len:
                    filtered.append((chunk_start, chunk_end))
                chunk_start = chunk_end + 1
            continue
        filtered.append((start, end))
    filtered.sort(
        key=lambda item: (
            sum(scores[item[0] : item[1] + 1]) / float(item[1] - item[0] + 1),
            item[1] - item[0] + 1,
        ),
        reverse=True,
    )
    return filtered[: args.max_intervals_per_protein]


def fmt(value):
    return "{:.6f}".format(float(value))


def build_row(accession, sequence, scores, start, end, rank, args):
    seg_scores = scores[start : end + 1]
    length = end - start + 1
    mean_score = sum(seg_scores) / float(max(1, length))
    sum_score = sum(seg_scores)
    max_score = max(seg_scores) if seg_scores else 0.0
    is_terminal = start == 0 or end == len(sequence) - 1
    missing_structure_penalty = 0.0 if is_terminal else args.sequence_only_structure_penalty
    missing_annotation_penalty = args.sequence_only_annotation_penalty
    low_plddt_penalty = 0.0
    uncertainty = missing_structure_penalty + missing_annotation_penalty + low_plddt_penalty
    evidence_confidence = max(0.0, min(1.0, 1.0 - uncertainty))
    return {
        "accession": accession,
        "protein_length": len(sequence),
        "seg_start": start,
        "seg_end": end,
        "seg_len": length,
        "proposal_source": "stage1_scisor_style_posterior",
        "biological_rationale": "learned SCISOR-style deletion posterior interval proposal",
        "stage1_interval_rank": rank,
        "stage1_posterior_mean": fmt(mean_score),
        "stage1_posterior_sum": fmt(sum_score),
        "stage1_posterior_max": fmt(max_score),
        "stage1_utility_score": fmt(mean_score),
        "stage1_utility_budget": fmt(args.budget),
        "stage1_scoring_status": "success",
        "n_protected_overlap": 0,
        "protected_overlap_fraction": fmt(0.0),
        "n_shadow_overlap": 0,
        "shadow_overlap_fraction": fmt(0.0),
        "mean_contact_density_8A": "",
        "max_contact_density_8A": "",
        "mean_motif_contact_count_8A": "",
        "max_motif_contact_count_8A": "",
        "mean_pLDDT": "",
        "min_pLDDT": "",
        "low_pLDDT_fraction": "",
        "high_pLDDT_fraction": "",
        "terminal_overlap_fraction": fmt(length / float(max(1, len(sequence))) if is_terminal else 0.0),
        "boundary_ca_distance": "",
        "closure_type": "terminal" if is_terminal else "sequence_only_unknown",
        "closure_friendly_8A": "True" if is_terminal else "False",
        "terminal_tail_score": fmt(1.0 if is_terminal else 0.0),
        "surface_flexible_loop_score": fmt(0.0),
        "disorder_like_score": fmt(0.0),
        "linker_compressibility_score": fmt(mean_score),
        "functional_core_risk_score": fmt(0.0),
        "structural_core_risk_score": fmt(0.0 if is_terminal else args.sequence_only_structural_risk),
        "motif_shadow_risk_score": fmt(0.0),
        "geometric_closure_score": fmt(1.0 if is_terminal else 0.0),
        "final_bioprior_score": fmt(mean_score),
        "hard_reject": "False",
        "reject_reason": "",
        "risk_point": fmt(0.0 if is_terminal else args.sequence_only_structural_risk),
        "risk_lower": fmt(0.0),
        "risk_upper": fmt(min(1.0, max(0.0, (0.0 if is_terminal else args.sequence_only_structural_risk) + uncertainty))),
        "risk_interval_width": fmt(uncertainty),
        "evidence_level": "sequence_only",
        "evidence_confidence": fmt(evidence_confidence),
        "has_annotation_evidence": "False",
        "has_structure_evidence": "False",
        "missing_annotation_penalty": fmt(missing_annotation_penalty),
        "missing_structure_penalty": fmt(missing_structure_penalty),
        "low_plddt_uncertainty_penalty": fmt(low_plddt_penalty),
        "risk_certificate_status": "sequence_only_stage1_proposal_uncalibrated",
        "risk_upper_favorable_score": fmt(1.0 - min(1.0, max(0.0, (0.0 if is_terminal else args.sequence_only_structural_risk) + uncertainty))),
        "risk_certified_biodel_score": fmt(mean_score * evidence_confidence),
        "risk_certificate_components": json.dumps(
            {
                "functional": 0.0,
                "motif_shadow": 0.0,
                "structural": 0.0 if is_terminal else args.sequence_only_structural_risk,
                "closure": 0.0 if is_terminal else 1.0,
                "missing_annotation_penalty": missing_annotation_penalty,
                "missing_structure_penalty": missing_structure_penalty,
            },
            sort_keys=True,
        ),
    }


def write_summary(path, args, stats, checkpoint):
    ensure_parent(path)
    val_metrics = checkpoint.get("val_metrics", {})
    with open(path, "w") as handle:
        handle.write("SCISOR-style Stage-1 interval proposal summary\n\n")
        handle.write("checkpoint: {}\n".format(args.checkpoint))
        handle.write("base_checkpoint: {}\n".format(args.base_checkpoint or checkpoint.get("args", {}).get("checkpoint", "")))
        handle.write("input_fasta: {}\n".format(args.input_fasta))
        handle.write("out_csv: {}\n".format(args.out_csv))
        handle.write("budget: {}\n".format(args.budget))
        handle.write("score_quantile: {}\n".format(args.score_quantile))
        handle.write("smooth_window: {}\n".format(args.smooth_window))
        handle.write("max_intervals_per_protein: {}\n".format(args.max_intervals_per_protein))
        handle.write("disable_fa: {}\n\n".format(args.disable_fa))
        for key in [
            "input_records",
            "kept_records",
            "skipped_nonstandard",
            "skipped_too_short",
            "skipped_too_long",
            "proteins_with_proposals",
            "proposal_rows",
        ]:
            handle.write("{}: {}\n".format(key, stats[key]))
        handle.write("\nCheckpoint validation metrics:\n")
        for key in sorted(val_metrics):
            handle.write("- {}: {}\n".format(key, val_metrics[key]))
        handle.write("\n{}\n".format("STAGE1_SCISOR_INTERVAL_PROPOSAL_PASS" if stats["proposal_rows"] > 0 else "STAGE1_SCISOR_INTERVAL_PROPOSAL_WARN"))


def propose(args):
    if args.disable_fa:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = torch.device(args.device)
    model, checkpoint = load_model(args, device)
    fieldnames = [
        "accession",
        "protein_length",
        "seg_start",
        "seg_end",
        "seg_len",
        "proposal_source",
        "biological_rationale",
        "stage1_interval_rank",
        "stage1_posterior_mean",
        "stage1_posterior_sum",
        "stage1_posterior_max",
        "stage1_utility_score",
        "stage1_utility_budget",
        "stage1_scoring_status",
        "n_protected_overlap",
        "protected_overlap_fraction",
        "n_shadow_overlap",
        "shadow_overlap_fraction",
        "mean_contact_density_8A",
        "max_contact_density_8A",
        "mean_motif_contact_count_8A",
        "max_motif_contact_count_8A",
        "mean_pLDDT",
        "min_pLDDT",
        "low_pLDDT_fraction",
        "high_pLDDT_fraction",
        "terminal_overlap_fraction",
        "boundary_ca_distance",
        "closure_type",
        "closure_friendly_8A",
        "terminal_tail_score",
        "surface_flexible_loop_score",
        "disorder_like_score",
        "linker_compressibility_score",
        "functional_core_risk_score",
        "structural_core_risk_score",
        "motif_shadow_risk_score",
        "geometric_closure_score",
        "final_bioprior_score",
        "hard_reject",
        "reject_reason",
        "risk_point",
        "risk_lower",
        "risk_upper",
        "risk_interval_width",
        "evidence_level",
        "evidence_confidence",
        "has_annotation_evidence",
        "has_structure_evidence",
        "missing_annotation_penalty",
        "missing_structure_penalty",
        "low_plddt_uncertainty_penalty",
        "risk_certificate_status",
        "risk_upper_favorable_score",
        "risk_certified_biodel_score",
        "risk_certificate_components",
    ]
    ensure_parent(args.out_csv)
    stats = Counter()
    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        batch = []
        for header, raw_sequence in stream_fasta(args.input_fasta):
            stats["input_records"] += 1
            if args.max_records is not None and stats["input_records"] > args.max_records:
                break
            sequence = "".join(raw_sequence.split()).upper()
            if len(sequence) < args.min_len:
                stats["skipped_too_short"] += 1
                continue
            if len(sequence) > args.max_len:
                stats["skipped_too_long"] += 1
                continue
            if args.exclude_nonstandard_aa and not set(sequence).issubset(STANDARD_AA):
                stats["skipped_nonstandard"] += 1
                continue
            batch.append((accession_from_header(header), sequence))
            if len(batch) >= args.batch_size:
                write_batch(writer, model, batch, args, device, stats)
                batch = []
        if batch:
            write_batch(writer, model, batch, args, device, stats)
    write_summary(args.summary_txt, args, stats, checkpoint)
    print("Kept proteins: {}".format(stats["kept_records"]))
    print("Proposal rows: {}".format(stats["proposal_rows"]))
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.summary_txt))


def write_batch(writer, model, batch, args, device, stats):
    accessions = [item[0] for item in batch]
    sequences = [item[1] for item in batch]
    all_scores = token_scores(model, sequences, args.budget, device)
    for accession, sequence, scores in zip(accessions, sequences, all_scores):
        stats["kept_records"] += 1
        intervals = propose_intervals_for_scores(scores, args)
        if intervals:
            stats["proteins_with_proposals"] += 1
        for rank, (start, end) in enumerate(intervals, start=1):
            writer.writerow(build_row(accession, sequence, scores, start, end, rank, args))
            stats["proposal_rows"] += 1


def parse_args():
    parser = argparse.ArgumentParser(description="Propose contiguous deletion intervals from a SCISOR-style Stage-1 model.")
    parser.add_argument("--checkpoint", required=True, help="Trained Stage-1 checkpoint saved by train_stage1_scisor_style_streaming.py.")
    parser.add_argument("--base_checkpoint", default=None, help="Original SCISOR .ckpt used to instantiate the architecture.")
    parser.add_argument("--p0", default="p0.pt")
    parser.add_argument("--input_fasta", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--summary_txt", required=True)
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument("--score_quantile", type=float, default=0.85)
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--merge_gap", type=int, default=2)
    parser.add_argument("--min_interval_len", type=int, default=3)
    parser.add_argument("--max_interval_len", type=int, default=40)
    parser.add_argument("--max_intervals_per_protein", type=int, default=20)
    parser.add_argument("--min_len", type=int, default=40)
    parser.add_argument("--max_len", type=int, default=800)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--exclude_nonstandard_aa", action="store_true")
    parser.add_argument("--disable-fa", action="store_true")
    parser.add_argument("--hf_cache", default=DEFAULT_HF_CACHE, help="Documented for reproducibility; local snapshot resolution uses the shared default cache.")
    parser.add_argument("--sequence_only_annotation_penalty", type=float, default=0.15)
    parser.add_argument("--sequence_only_structure_penalty", type=float, default=0.20)
    parser.add_argument("--sequence_only_structural_risk", type=float, default=0.10)
    return parser.parse_args()


if __name__ == "__main__":
    propose(parse_args())
