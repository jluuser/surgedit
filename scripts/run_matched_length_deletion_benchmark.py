#!/usr/bin/env python3
"""Run a matched-length deletion benchmark across BioDel-Cert, SCISOR, and Raygun.

Protocol:
1. Use the BioDel-Cert default profile to define a target deletion count for each
   protein on the certified subset.
2. Evaluate SCISOR and Raygun against the same per-protein target length.
3. Score only deletion-quality metrics that are comparable across methods:
   protected overlap, motif-shadow overlap, motif-contact overlap, closure
   unfriendly rate, and exact-length success.

The benchmark is intentionally selective: proteins for which BioDel-Cert abstains
are excluded from the matched-length comparison and reported separately as
coverage.
"""

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from statistics import mean, StatisticsError

import torch
from Bio import pairwise2 as pw


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAYGUN_ROOT = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/raygun"
if RAYGUN_ROOT not in sys.path:
    sys.path.insert(0, RAYGUN_ROOT)

from raygun.pll import get_PLL, penalizerepeats  # noqa: E402
from raygun.pretrained import raygun_2_2mil_800M, raygun_4_4mil_800M, raygun_8_8mil_800M  # noqa: E402


RAYGUN_CKPT_MAP = {
    "https://zenodo.org/records/15447158/files/model-may-16.ckpt?download=1": os.path.expanduser(
        "~/.cache/torch/hub/checkpoints/model-may-16.ckpt"
    ),
    "https://zenodo.org/records/15578824/files/may30-chkpoint-trained-on-4.4m.ckpt?download=1": os.path.expanduser(
        "~/.cache/torch/hub/checkpoints/may30-chkpoint-trained-on-4.4m.ckpt"
    ),
    "https://zenodo.org/records/17253788/files/species_function_aware_sep30_val_blosum_0.9856.ckpt?download=1": os.path.expanduser(
        "~/.cache/torch/hub/checkpoints/species_function_aware_sep30_val_blosum_0.9856.ckpt"
    ),
}
_TORCH_HUB_LOAD_STATE_DICT_FROM_URL = torch.hub.load_state_dict_from_url


def load_state_dict_from_url_local(url, *args, **kwargs):
    local_path = RAYGUN_CKPT_MAP.get(url)
    if local_path and os.path.exists(local_path):
        return torch.load(local_path, map_location=kwargs.get("map_location"))
    return _TORCH_HUB_LOAD_STATE_DICT_FROM_URL(url, *args, **kwargs)


torch.hub.load_state_dict_from_url = load_state_dict_from_url_local


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames=None):
    ensure_parent(path)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


TABLE_FIELDNAMES = [
    "method",
    "accession",
    "original_length",
    "target_delete_len",
    "target_delete_ratio",
    "target_new_len",
    "achieved_delete_len",
    "achieved_delete_ratio",
    "fill_ratio",
    "protected_overlap_residues",
    "protected_overlap_rate",
    "shadow_overlap_residues",
    "shadow_overlap_rate",
    "motif_contact_overlap_residues",
    "motif_contact_overlap_rate",
    "closure_unfriendly_len",
    "closure_unfriendly_rate",
    "length_success",
    "sequence_identity",
    "pll_score",
    "deleted_positions",
    "selection_status",
    "auto_profile",
    "source",
    "raygun_noise",
]


def append_csv_rows(path, rows, fieldnames):
    ensure_parent(path)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def read_existing_accession_methods(path):
    done = set()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return done
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            accession = row.get("accession")
            method = row.get("method")
            if accession and method:
                done.add((accession, method))
    return done


def safe_mean(values):
    values = list(values)
    if not values:
        return ""
    try:
        return mean(values)
    except StatisticsError:
        return ""


def parse_positions(text):
    if not text:
        return set()
    return {int(token) for token in text.split(";") if token.strip()}


def write_single_fasta(path, header, sequence):
    ensure_parent(path)
    with open(path, "w") as handle:
        handle.write(">{}\n{}\n".format(header, sequence))


def read_split_proteins(path):
    proteins = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get("accession") or row.get("protein_id")
            if not accession:
                continue
            seq = row.get("sequence", "")
            length = int(row.get("length") or len(seq))
            proteins[accession] = {
                "accession": accession,
                "sequence": seq,
                "length": length,
                "protected": parse_positions(row.get("protected_positions", "")),
                "row": row,
            }
    return proteins


def base_header(header):
    return (header or "").split("|", 1)[0]


def read_residue_priors(path):
    priors = defaultdict(lambda: {"shadow": set(), "motif_contact": set()})
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get("accession") or row.get("protein_id")
            if not accession:
                continue
            pos_value = row.get("residue_index_0based") or row.get("position")
            if pos_value is None or pos_value == "":
                continue
            pos = int(float(pos_value))
            if row.get("is_motif_shadow_8A") == "True":
                priors[accession]["shadow"].add(pos)
            if int(float(row.get("motif_contact_count_8A", 0) or 0)) > 0:
                priors[accession]["motif_contact"].add(pos)
    return dict(priors)


def read_selected_lengths(path, profile_name="default"):
    selected = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("auto_profile") != profile_name:
                continue
            accession = row.get("accession")
            if not accession:
                continue
            selected[accession] = {
                "selection_status": row.get("selection_status", ""),
                "selected_len": int(float(row.get("selected_len") or 0)),
                "actual_delete_ratio": float(row.get("actual_delete_ratio") or 0.0),
                "protein_length": int(float(row.get("protein_length") or 0)),
            }
    return selected


def read_selected_segments(path, profile_name="default"):
    grouped = defaultdict(list)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("auto_profile") != profile_name:
                continue
            if row.get("selection_status") != "certified":
                continue
            accession = row.get("accession")
            if not accession:
                continue
            grouped[accession].append(row)
    return dict(grouped)


def read_scisor_candidate_rows(path):
    grouped = defaultdict(list)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get("accession") or row.get("protein_id")
            if not accession:
                continue
            grouped[accession].append(
                {
                    "protein_id": accession,
                    "start": int(float(row.get("seg_start") or row.get("start") or 0)),
                    "end": int(float(row.get("seg_end") or row.get("end") or 0)),
                    "is_terminal_deletion": str(row.get("closure_type", "")).strip() == "terminal",
                    "closure_friendly": str(row.get("closure_friendly_8A", "")).strip() == "True",
                    "boundary_ca_distance": row.get("boundary_ca_distance", ""),
                }
            )
    return dict(grouped)


def write_scisor_candidate_csv(path, rows):
    ensure_parent(path)
    fieldnames = [
        "protein_id",
        "start",
        "end",
        "is_terminal_deletion",
        "closure_friendly",
        "boundary_ca_distance",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def subprocess_env(args):
    env = os.environ.copy()
    if args.scisor_offline_hf:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
    if args.clear_proxy:
        for key in [
            "HF_ENDPOINT",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
        ]:
            env.pop(key, None)
    return env


def annotate_segment(row):
    start = int(float(row.get("seg_start") or 0))
    end = int(float(row.get("seg_end") or 0))
    seg_len = int(float(row.get("seg_len") or (end - start + 1)))
    row = dict(row)
    row["_start"] = start
    row["_end"] = end
    row["_seg_len"] = seg_len
    return row


def deleted_positions_from_segments(rows):
    deleted = set()
    for row in rows:
        start = int(float(row.get("seg_start") or row.get("_start") or 0))
        end = int(float(row.get("seg_end") or row.get("_end") or 0))
        deleted.update(range(start, end + 1))
    return sorted(deleted)


def parse_pdb_ca_coords(path):
    coords = []
    seen = set()
    with open(path, "rb") as handle:
        for raw_line in handle:
            if not raw_line.startswith(b"ATOM"):
                continue
            line = raw_line.decode("ascii", "ignore")
            if line[12:16].strip() != "CA":
                continue
            altloc = line[16:17]
            if altloc not in (" ", "A"):
                continue
            key = (line[21:22], line[22:26].strip(), line[26:27])
            if key in seen:
                continue
            seen.add(key)
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return coords


def deleted_segments(positions):
    positions = sorted(set(positions))
    if not positions:
        return []
    segments = []
    start = prev = positions[0]
    for pos in positions[1:]:
        if pos == prev + 1:
            prev = pos
        else:
            segments.append((start, prev))
            start = prev = pos
    segments.append((start, prev))
    return segments


def dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def closure_unfriendly_len(accession, length, positions, structure_dir, cutoff=8.0):
    pdb_path = os.path.join(structure_dir, accession + ".pdb")
    if not os.path.exists(pdb_path):
        return 0
    coords = parse_pdb_ca_coords(pdb_path)
    if len(coords) != length:
        return 0
    unfriendly_len = 0
    for start, end in deleted_segments(positions):
        if start == 0 or end == length - 1:
            continue
        left = start - 1
        right = end + 1
        if left < 0 or right >= len(coords):
            continue
        if dist(coords[left], coords[right]) > cutoff:
            unfriendly_len += end - start + 1
    return unfriendly_len


def align_and_infer_deletions(original, generated):
    alignments = pw.align.globalms(original, generated, 2, -1, -10, -0.5, one_alignment_only=True)
    if not alignments:
        return []
    aligned = alignments[0]
    aligned_orig = aligned.seqA
    aligned_gen = aligned.seqB
    deleted = []
    orig_index = 0
    for a, b in zip(aligned_orig, aligned_gen):
        if a != "-":
            if b == "-":
                deleted.append(orig_index)
            orig_index += 1
    return deleted


def read_deletions_json(path):
    with open(path) as handle:
        payload = json.load(handle)
    by_header = {}
    for entry in payload:
        header = entry.get("header", "")
        positions = []
        for key in ("deletion_positions_zero_based", "deleted_positions", "deleted_positions_zero_based"):
            if key in entry and isinstance(entry[key], list):
                positions = [int(x) for x in entry[key]]
                break
        by_header[header] = positions
        by_header[base_header(header)] = positions
    return by_header


def method_metrics(
    method,
    accession,
    original,
    target_delete_len,
    deleted_positions,
    protected_positions,
    shadow_positions,
    motif_contact_positions,
    structure_dir,
    generated_sequence=None,
    pll_score=None,
):
    deleted_set = set(int(x) for x in deleted_positions)
    deleted_count = len(deleted_set)
    target_delete_len = int(target_delete_len)
    target_delete_len = max(1, target_delete_len)
    fill_ratio = deleted_count / float(target_delete_len)
    deleted_ratio = deleted_count / float(len(original) or 1)
    protected_overlap = len(deleted_set & protected_positions)
    shadow_overlap = len(deleted_set & shadow_positions)
    motif_contact_overlap = len(deleted_set & motif_contact_positions)
    closure_len = closure_unfriendly_len(accession, len(original), deleted_set, structure_dir)
    closure_rate = closure_len / float(deleted_count or 1)
    identity = None
    if generated_sequence is not None:
        alignments = pw.align.globalms(original, generated_sequence, 2, -1, -10, -0.5, one_alignment_only=True)
        if alignments:
            aligned = alignments[0]
            match = 0
            aligned_len = 0
            for a, b in zip(aligned.seqA, aligned.seqB):
                if a == "-" or b == "-":
                    continue
                aligned_len += 1
                if a == b:
                    match += 1
            identity = match / float(aligned_len or 1)
    return {
        "method": method,
        "accession": accession,
        "original_length": len(original),
        "target_delete_len": target_delete_len,
        "target_delete_ratio": target_delete_len / float(len(original) or 1),
        "achieved_delete_len": deleted_count,
        "achieved_delete_ratio": deleted_ratio,
        "fill_ratio": fill_ratio,
        "protected_overlap_residues": protected_overlap,
        "protected_overlap_rate": protected_overlap / float(deleted_count or 1),
        "shadow_overlap_residues": shadow_overlap,
        "shadow_overlap_rate": shadow_overlap / float(deleted_count or 1),
        "motif_contact_overlap_residues": motif_contact_overlap,
        "motif_contact_overlap_rate": motif_contact_overlap / float(deleted_count or 1),
        "closure_unfriendly_len": closure_len,
        "closure_unfriendly_rate": closure_rate,
        "length_success": int(deleted_count == target_delete_len),
        "sequence_identity": "" if identity is None else "{:.6f}".format(identity),
        "pll_score": "" if pll_score is None else "{:.6f}".format(pll_score),
        "deleted_positions": ";".join(str(pos) for pos in sorted(deleted_set)),
    }


def summarize_rows(rows):
    if not rows:
        return {}
    deleted_counts = [int(row["achieved_delete_len"]) for row in rows]
    return {
        "method": rows[0]["method"],
        "n_proteins": len(rows),
        "mean_original_length": mean(float(row["original_length"]) for row in rows),
        "mean_target_delete_len": mean(float(row["target_delete_len"]) for row in rows),
        "mean_target_delete_ratio": mean(float(row["target_delete_ratio"]) for row in rows),
        "mean_achieved_delete_len": mean(float(row["achieved_delete_len"]) for row in rows),
        "mean_achieved_delete_ratio": mean(float(row["achieved_delete_ratio"]) for row in rows),
        "mean_fill_ratio": mean(float(row["fill_ratio"]) for row in rows),
        "protected_overlap_residues": sum(int(row["protected_overlap_residues"]) for row in rows),
        "protected_overlap_rate": sum(int(row["protected_overlap_residues"]) for row in rows) / float(sum(deleted_counts) or 1),
        "shadow_overlap_residues": sum(int(row["shadow_overlap_residues"]) for row in rows),
        "shadow_overlap_rate": sum(int(row["shadow_overlap_residues"]) for row in rows) / float(sum(deleted_counts) or 1),
        "motif_contact_overlap_residues": sum(int(row["motif_contact_overlap_residues"]) for row in rows),
        "motif_contact_overlap_rate": sum(int(row["motif_contact_overlap_residues"]) for row in rows)
        / float(sum(deleted_counts) or 1),
        "closure_unfriendly_len": sum(int(row["closure_unfriendly_len"]) for row in rows),
        "closure_unfriendly_rate": sum(int(row["closure_unfriendly_len"]) for row in rows) / float(sum(deleted_counts) or 1),
        "length_success_rate": mean(float(row["length_success"]) for row in rows),
        "proteins_with_any_protected_violation": sum(1 for row in rows if int(row["protected_overlap_residues"]) > 0),
        "proteins_with_any_shadow_overlap": sum(1 for row in rows if int(row["shadow_overlap_residues"]) > 0),
        "proteins_with_any_motif_contact_overlap": sum(1 for row in rows if int(row["motif_contact_overlap_residues"]) > 0),
        "proteins_with_any_closure_unfriendly": sum(1 for row in rows if int(row["closure_unfriendly_len"]) > 0),
        "mean_sequence_identity": safe_mean(float(row["sequence_identity"]) for row in rows if row["sequence_identity"] != ""),
        "mean_pll_score": safe_mean(float(row["pll_score"]) for row in rows if row["pll_score"] != ""),
    }


class RaygunRunner:
    def __init__(self, model_name, device, samples, noise, numcycles, penalize_repeats, randomize_noise=False):
        self.device = torch.device(device)
        self.samples = int(samples)
        self.noise = float(noise)
        self.numcycles = int(numcycles)
        self.penalize_repeats = bool(penalize_repeats)
        self.randomize_noise = bool(randomize_noise)

        import esm

        self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm_model = self.esm_model.to(self.device)
        self.esm_model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()

        if model_name == "8.8M":
            self.raygun = raygun_8_8mil_800M().to(self.device)
        elif model_name == "4.4M":
            self.raygun = raygun_4_4mil_800M().to(self.device)
        elif model_name == "2.2M":
            self.raygun = raygun_2_2mil_800M().to(self.device)
        else:
            raise ValueError("Unknown Raygun model {}".format(model_name))
        self.raygun.eval()

    def embed(self, accession, sequence):
        data = [(accession, sequence)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        with torch.no_grad():
            emb = self.esm_model(tokens, repr_layers=[33], return_contacts=False)["representations"][33][:, 1:-1]
        return emb

    def score_sequence(self, sequence):
        pll = get_PLL(sequence, self.esm_model, self.alphabet, self.batch_converter, device=self.device)
        pll = pll / abs(-0.406 * len(sequence) + 1.363)
        if self.penalize_repeats:
            pll = pll * penalizerepeats(sequence)
        return float(pll)

    def generate(self, accession, sequence, target_len):
        emb = self.embed(accession, sequence)
        target_tensor = torch.tensor([int(target_len)], dtype=torch.long, device=self.device)
        candidates = []
        with torch.no_grad():
            for _ in range(self.samples):
                nratio = random.random() * self.noise if self.randomize_noise else self.noise
                result = self.raygun(
                    emb,
                    target_lengths=target_tensor,
                    noise=nratio,
                    return_logits_and_seqs=True,
                )
                generated = result["generated-sequences"][0]
                pll = self.score_sequence(generated)
                candidates.append(
                    {
                        "sequence": generated,
                        "pll_score": pll,
                        "noise": nratio,
                    }
                )
        return max(candidates, key=lambda item: item["pll_score"])


def run_scisor_single(
    python,
    accession,
    sequence,
    target_delete_len,
    input_fasta,
    out_dir,
    test_csv,
    residue_priors_csv,
    segment_candidates_rows,
    execute,
    env,
):
    ensure_dir(out_dir)
    candidate_csv = os.path.join(out_dir, "{}_segment_candidates.csv".format(accession))
    write_scisor_candidate_csv(candidate_csv, segment_candidates_rows)
    write_single_fasta(input_fasta, accession, sequence)
    # SCISOR computes the number of deletions as ceil(length * pct / 100), so we
    # nudge the percentage slightly below the exact ratio to hit the intended count.
    shrink_pct = 100.0 * max(0.0, float(target_delete_len) - 1e-3) / float(len(sequence) or 1)
    cmd = [
        python,
        "scripts/run_scisor_shrink.py",
        "--input",
        input_fasta,
        "--output-dir",
        out_dir,
        "--shrink-pct",
        "{:.6f}".format(shrink_pct),
        "--temperature",
        "0.0",
        "--max-sequences",
        "1",
        "--disable-fa",
        "--motif_csv",
        test_csv,
        "--protect_motif",
        "--structure_priors",
        residue_priors_csv,
        "--protect_shadow",
        "--shadow_penalty",
        "0.2",
        "--shadow_mode",
        "multiply",
        "--segment_candidates",
        candidate_csv,
        "--protect_closure",
        "--closure_penalty",
        "0.2",
        "--closure_mode",
        "multiply",
    ]
    if execute:
        subprocess.run(cmd, check=True, env=env)
    else:
        print(" ".join(cmd))
    return os.path.join(out_dir, "deletions.json"), os.path.join(out_dir, "shrunk_sequences.fasta")


def read_fasta(path):
    records = []
    header = None
    chunks = []
    with open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
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


def run(args):
    proteins = read_split_proteins(args.split_csv)
    priors = read_residue_priors(args.residue_priors_csv)
    target_lengths = read_selected_lengths(args.selected_lengths_csv, profile_name=args.profile)
    selected_segments = read_selected_segments(args.selected_segments_csv, profile_name=args.profile)
    candidate_segments = read_scisor_candidate_rows(args.segment_candidates_csv)

    total_proteins = len(proteins)
    certified = {acc: item for acc, item in target_lengths.items() if item.get("selection_status") == "certified" and int(item.get("selected_len", 0)) > 0}
    if args.limit is not None:
        certified = dict(list(sorted(certified.items()))[: int(args.limit)])
    accessions = sorted(certified)
    if not accessions:
        raise SystemExit("No certified proteins found for profile '{}'.".format(args.profile))

    incremental_path = os.path.join(args.out_dir, "matched_length_incremental_rows.csv")
    done = read_existing_accession_methods(incremental_path) if args.resume else set()

    selected_rows = []
    for accession in accessions:
        protein = proteins[accession]
        seg_rows = [annotate_segment(row) for row in selected_segments.get(accession, [])]
        deleted_positions = deleted_positions_from_segments(seg_rows)
        metrics = method_metrics(
            method="BioDel-Cert default",
            accession=accession,
            original=protein["sequence"],
            target_delete_len=certified[accession]["selected_len"],
            deleted_positions=deleted_positions,
            protected_positions=protein["protected"],
            shadow_positions=priors.get(accession, {}).get("shadow", set()),
            motif_contact_positions=priors.get(accession, {}).get("motif_contact", set()),
            structure_dir=args.structure_dir,
        )
        metrics["selection_status"] = certified[accession]["selection_status"]
        metrics["auto_profile"] = args.profile
        metrics["source"] = "BioDel-Cert selected segments"
        selected_rows.append(metrics)

    raygun_runner = None
    if args.execute:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available in the current environment.")
        device = "cuda:{}".format(args.raygun_device)
        raygun_runner = RaygunRunner(
            model_name=args.raygun_model,
            device=device,
            samples=args.raygun_samples,
            noise=args.raygun_noise,
            numcycles=args.raygun_numcycles,
            penalize_repeats=args.raygun_penalize_repeats,
            randomize_noise=args.raygun_randomize_noise,
        )

    outputs = []
    child_env = subprocess_env(args)
    scisor_methods = [item.strip() for item in args.scisor_modes.split(",") if item.strip()]
    scisor_method_labels = {
        "nomask": "SCISOR no-mask",
        "hardmask": "SCISOR hardmask",
        "hardmask_shadow02": "SCISOR hardmask+shadow02",
    }

    for accession in accessions:
        protein = proteins[accession]
        target_delete_len = certified[accession]["selected_len"]
        target_new_len = len(protein["sequence"]) - target_delete_len

        if args.execute:
            protein_root = os.path.join(args.out_dir, accession)
            ensure_dir(protein_root)
            accession_rows = []

            for mode in scisor_methods:
                method_label = scisor_method_labels.get(mode, mode)
                if args.resume and (accession, method_label) in done:
                    continue
                scisor_dir = os.path.join(protein_root, "scisor", mode)
                input_fasta = os.path.join(protein_root, "{}.fasta".format(accession))
                scisor_json = os.path.join(scisor_dir, "deletions.json")
                scisor_fasta = os.path.join(scisor_dir, "shrunk_sequences.fasta")
                if not (
                    args.resume
                    and os.path.exists(scisor_json)
                    and os.path.getsize(scisor_json) > 0
                    and os.path.exists(scisor_fasta)
                    and os.path.getsize(scisor_fasta) > 0
                ):
                    scisor_json, scisor_fasta = run_scisor_single(
                        python=args.python,
                        accession=accession,
                        sequence=protein["sequence"],
                        target_delete_len=target_delete_len,
                        input_fasta=input_fasta,
                        out_dir=scisor_dir,
                        test_csv=args.split_csv,
                        residue_priors_csv=args.residue_priors_csv,
                        segment_candidates_rows=candidate_segments.get(accession, []),
                        execute=True,
                        env=child_env,
                    )
                deletions_map = read_deletions_json(scisor_json)
                generated_records = read_fasta(scisor_fasta)
                if not generated_records:
                    raise RuntimeError("SCISOR produced no FASTA records for {}".format(accession))
                generated_seq = generated_records[0][1]
                deleted_positions = deletions_map.get(accession, [])
                metrics = method_metrics(
                    method=method_label,
                    accession=accession,
                    original=protein["sequence"],
                    target_delete_len=target_delete_len,
                    deleted_positions=deleted_positions,
                    protected_positions=protein["protected"],
                    shadow_positions=priors.get(accession, {}).get("shadow", set()),
                    motif_contact_positions=priors.get(accession, {}).get("motif_contact", set()),
                    structure_dir=args.structure_dir,
                    generated_sequence=generated_seq,
                )
                metrics["source"] = scisor_json
                metrics["target_new_len"] = target_new_len
                outputs.append(metrics)
                accession_rows.append(metrics)

            raygun_method = "Raygun {}".format(args.raygun_model)
            if args.resume and (accession, raygun_method) in done:
                if accession_rows:
                    append_csv_rows(incremental_path, accession_rows, TABLE_FIELDNAMES)
                    print("Wrote incremental rows for {}".format(accession), flush=True)
                continue
            raygun_dir = os.path.join(protein_root, "raygun")
            ensure_dir(raygun_dir)
            best = raygun_runner.generate(accession, protein["sequence"], target_new_len)
            generated_seq = best["sequence"]
            deleted_positions = align_and_infer_deletions(protein["sequence"], generated_seq)
            metrics = method_metrics(
                method="Raygun {}".format(args.raygun_model),
                accession=accession,
                original=protein["sequence"],
                target_delete_len=target_delete_len,
                deleted_positions=deleted_positions,
                protected_positions=protein["protected"],
                shadow_positions=priors.get(accession, {}).get("shadow", set()),
                motif_contact_positions=priors.get(accession, {}).get("motif_contact", set()),
                structure_dir=args.structure_dir,
                generated_sequence=generated_seq,
                pll_score=best["pll_score"],
            )
            metrics["source"] = raygun_dir
            metrics["target_new_len"] = target_new_len
            metrics["raygun_noise"] = best["noise"]
            outputs.append(metrics)
            accession_rows.append(metrics)
            append_csv_rows(incremental_path, accession_rows, TABLE_FIELDNAMES)
            print("Wrote incremental rows for {}".format(accession), flush=True)

    incremental_rows = []
    if os.path.exists(incremental_path) and os.path.getsize(incremental_path) > 0:
        incremental_rows = read_csv(incremental_path)
    all_rows = selected_rows + incremental_rows
    table_path = os.path.join(args.out_dir, "matched_length_comparison_table.csv")
    summary_path = os.path.join(args.out_dir, "matched_length_comparison_summary.csv")
    report_path = os.path.join(args.out_dir, "matched_length_comparison_report.md")
    write_csv(table_path, all_rows, fieldnames=TABLE_FIELDNAMES)

    summary_rows = []
    for method in sorted({row["method"] for row in all_rows}):
        method_rows = [row for row in all_rows if row["method"] == method]
        summary = summarize_rows(method_rows)
        summary_rows.append(summary)
    write_csv(summary_path, summary_rows)

    with open(report_path, "w") as handle:
        handle.write("# Matched-Length Deletion Benchmark\n\n")
        handle.write("This benchmark matches the per-protein deletion lengths selected by BioDel-Cert (default profile) and evaluates SCISOR and Raygun on the same targets.\n\n")
        handle.write("## Coverage\n\n")
        handle.write("- total_test_proteins: {}\n".format(total_proteins))
        handle.write("- certified_subset: {}\n".format(len(accessions)))
        handle.write("- coverage: {:.6f}\n\n".format(len(accessions) / float(total_proteins or 1)))
        handle.write("## Method Summary\n\n")
        handle.write("| Method | Proteins | Mean target delete ratio | Mean achieved delete ratio | Mean fill | Protected overlap | Shadow overlap | Motif-contact overlap | Closure-unfriendly | Length success |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            handle.write(
                "| {method} | {n_proteins} | {mean_target_delete_ratio:.4f} | {mean_achieved_delete_ratio:.4f} | {mean_fill_ratio:.4f} | {protected_overlap_rate:.4f} | {shadow_overlap_rate:.4f} | {motif_contact_overlap_rate:.4f} | {closure_unfriendly_rate:.4f} | {length_success_rate:.4f} |\n".format(
                    **row
                )
            )
        handle.write("\n## Notes\n\n")
        handle.write("- BioDel-Cert defines the target deletion length; SCISOR and Raygun are evaluated at the same target.\n")
        handle.write("- SCISOR uses the strongest protection configuration available in the current repository: motif protection, motif shadow penalty, and closure protection.\n")
        handle.write("- Raygun uses the pretrained model selected by `--raygun_model` and picks the best of `--raygun_samples` candidates by adjusted PLL.\n")
        handle.write("\nMATCHED_LENGTH_DELETION_BENCHMARK_PASS\n")

    print("Wrote {}".format(table_path))
    print("Wrote {}".format(summary_path))
    print("Wrote {}".format(report_path))


def parse_args():
    parser = argparse.ArgumentParser(description="Run matched-length deletion benchmark.")
    parser.add_argument("--split_csv", default="data/processed/bioprior_10k_family_splits/test.csv")
    parser.add_argument("--selected_lengths_csv", default="results/biodel_planner/family_split/certified_frontier_test_protein_selection.csv")
    parser.add_argument("--selected_segments_csv", default="results/biodel_planner/family_split/certified_frontier_test_selected_segments.csv")
    parser.add_argument("--residue_priors_csv", default="data/features/bioprior_10k_family_test_residue_biopriors.csv")
    parser.add_argument("--structure_dir", default="data/structures/afdb_bioprior_10k")
    parser.add_argument("--segment_candidates_csv", default="data/features/bioprior_10k_bioprior_segments_with_stage1_utility_certified.csv")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--out_dir", default="results/biodel_planner/family_split/matched_length_comparison")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--scisor_modes", default="hardmask_shadow02")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed per-protein outputs and incremental metric rows.",
    )
    parser.add_argument("--raygun_model", choices=["2.2M", "4.4M", "8.8M"], default="8.8M")
    parser.add_argument("--raygun_device", type=int, default=0)
    parser.add_argument("--raygun_samples", type=int, default=4)
    parser.add_argument("--raygun_noise", type=float, default=0.1)
    parser.add_argument("--raygun_numcycles", type=int, default=1)
    parser.add_argument("--raygun_penalize_repeats", action="store_true", default=True)
    parser.add_argument("--raygun_randomize_noise", action="store_true", default=False)
    parser.add_argument(
        "--scisor_offline_hf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run SCISOR subprocesses with HuggingFace/Transformers offline mode.",
    )
    parser.add_argument(
        "--clear_proxy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove proxy and HF_ENDPOINT variables for SCISOR subprocesses.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
