#!/usr/bin/env python3
"""Evaluate BioDel-style deletion risk proxies on SKEMPI v2 binding mutations.

SKEMPI v2 is not a deletion benchmark.  It contains binding-affinity changes
for substitutions in protein complexes.  This script uses it as an external
binder-retention proxy: if a residue is experimentally sensitive to mutation,
then a deletion planner should treat nearby/interface-support residues as
risky.
"""

import argparse
import csv
import math
import os
import re
from collections import Counter, defaultdict


AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

MUT_RE = re.compile(r"^([A-Z])([A-Za-z0-9])(\d+)([A-Z])$")


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def safe_float(value):
    if value in ("", None):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def safe_log10(value):
    value = safe_float(value)
    if value is None or value <= 0.0:
        return None
    return math.log10(value)


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / float(len(values)) if values else None


def fmt(value, digits=6):
    if value is None:
        return ""
    return ("{:.%df}" % digits).format(float(value))


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
    if dx == 0.0 or dy == 0.0:
        return None
    return num / (dx * dy)


def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    xvals, yvals = zip(*pairs)
    return pearson(rankdata(list(xvals)), rankdata(list(yvals)))


def binary_metrics(labels, scores):
    pairs = [(int(label), float(score)) for label, score in zip(labels, scores) if label is not None and score is not None]
    if not pairs:
        return None, None
    positives = sum(label for label, _ in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None, None
    asc = sorted(pairs, key=lambda item: item[1])
    ranks = rankdata([score for _, score in asc])
    pos_rank_sum = sum(rank for rank, (label, _) in zip(ranks, asc) if label == 1)
    auroc = (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    desc = sorted(pairs, key=lambda item: item[1], reverse=True)
    tp = 0
    precision_sum = 0.0
    for idx, (label, _) in enumerate(desc, start=1):
        if label == 1:
            tp += 1
            precision_sum += tp / idx
    return auroc, precision_sum / positives


def topk_precision(labels, scores, frac):
    pairs = [(int(label), float(score)) for label, score in zip(labels, scores) if label is not None and score is not None]
    if not pairs:
        return None
    k = max(1, int(round(len(pairs) * frac)))
    top = sorted(pairs, key=lambda item: item[1], reverse=True)[:k]
    return sum(label for label, _ in top) / float(len(top))


def read_skempi(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def complex_id(pdb_field):
    return (pdb_field or "").split("_", 1)[0].upper()


def complex_chains(pdb_field):
    parts = (pdb_field or "").split("_")
    chains = set()
    for token in parts[1:]:
        for char in token:
            if char.strip():
                chains.add(char)
    return chains


def resolve_file(data_dir, pdb_id, suffix):
    names = [
        os.path.join(data_dir, "{}{}".format(pdb_id, suffix)),
        os.path.join(data_dir, "PDBs", "{}{}".format(pdb_id, suffix)),
        os.path.join(data_dir, pdb_id.lower() + suffix),
        os.path.join(data_dir, "PDBs", pdb_id.lower() + suffix),
    ]
    for path in names:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    return ""


def parse_mapping(path):
    mapping = {}
    if not path:
        return mapping
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 4:
                continue
            aa3, chain, pdb_resseq, cleaned_idx = parts[:4]
            aa1 = AA3_TO_AA1.get(aa3.upper(), "X")
            key = (chain, int(float(cleaned_idx)))
            mapping[key] = {
                "aa": aa1,
                "pdb_resseq": pdb_resseq,
                "cleaned_idx": int(float(cleaned_idx)),
                "chain": chain,
            }
    return mapping


def parse_pdb_residues(path):
    residues = {}
    if not path:
        return residues
    seen_atoms = set()
    with open(path, "rb") as handle:
        for raw in handle:
            if not raw.startswith(b"ATOM"):
                continue
            line = raw.decode("ascii", "ignore")
            altloc = line[16:17]
            if altloc not in (" ", "A"):
                continue
            atom = line[12:16].strip()
            chain = line[21:22]
            resseq = line[22:26].strip()
            icode = line[26:27].strip()
            key = (chain, resseq, icode)
            atom_key = key + (atom,)
            if atom_key in seen_atoms:
                continue
            seen_atoms.add(atom_key)
            try:
                coord = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            row = residues.setdefault(
                key,
                {
                    "chain": chain,
                    "resseq": resseq,
                    "icode": icode,
                    "aa": AA3_TO_AA1.get(line[17:20].strip().upper(), "X"),
                    "atoms": [],
                    "ca": None,
                },
            )
            row["atoms"].append(coord)
            if atom == "CA":
                row["ca"] = coord
    return residues


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def min_atom_distance(residue, other_residues):
    if not residue or not residue.get("atoms") or not other_residues:
        return None
    best = None
    atoms = residue["atoms"]
    for other in other_residues:
        for a in atoms:
            for b in other.get("atoms", []):
                value = dist(a, b)
                if best is None or value < best:
                    best = value
    return best


def ca_contact_density(residue, residues, cutoff):
    ca = residue.get("ca") if residue else None
    if ca is None:
        return None
    count = 0
    for other in residues.values():
        other_ca = other.get("ca")
        if other_ca is None or other is residue:
            continue
        if dist(ca, other_ca) <= cutoff:
            count += 1
    return count


def parse_mutations(text):
    mutations = []
    for token in (text or "").split(","):
        token = token.strip()
        if not token:
            continue
        match = MUT_RE.match(token)
        if not match:
            continue
        wt, chain, cleaned_idx, mut = match.groups()
        mutations.append(
            {
                "token": token,
                "wt": wt,
                "chain": chain,
                "cleaned_idx": int(cleaned_idx),
                "mut": mut,
            }
        )
    return mutations


def score_record(row, data_dir, caches, args):
    pdb_id = complex_id(row.get("#Pdb", ""))
    pdb_path = resolve_file(data_dir, pdb_id, ".pdb")
    mapping_path = resolve_file(data_dir, pdb_id, ".mapping")
    if pdb_id not in caches["mapping"]:
        caches["mapping"][pdb_id] = parse_mapping(mapping_path)
    if pdb_id not in caches["residues"]:
        caches["residues"][pdb_id] = parse_pdb_residues(pdb_path)

    mapping = caches["mapping"][pdb_id]
    residues = caches["residues"][pdb_id]
    complex_chain_set = complex_chains(row.get("#Pdb", ""))
    chain_residues = defaultdict(list)
    for residue in residues.values():
        chain_residues[residue["chain"]].append(residue)

    mutations = parse_mutations(row.get("Mutation(s)_cleaned"))
    affinity_mut = safe_float(row.get("Affinity_mut_parsed"))
    affinity_wt = safe_float(row.get("Affinity_wt_parsed"))
    log10_kd_ratio = None
    if affinity_mut is not None and affinity_wt is not None and affinity_mut > 0.0 and affinity_wt > 0.0:
        log10_kd_ratio = math.log10(affinity_mut / affinity_wt)
    damaging_label = None if log10_kd_ratio is None else int(log10_kd_ratio >= args.damage_log10_ratio)
    beneficial_label = None if log10_kd_ratio is None else int(log10_kd_ratio <= -args.damage_log10_ratio)

    per_mut = []
    for mutation in mutations:
        map_row = mapping.get((mutation["chain"], mutation["cleaned_idx"]))
        residue = None
        residue_key = ""
        aa_match = ""
        if map_row is not None:
            for key in (
                (mutation["chain"], map_row["pdb_resseq"], ""),
                (mutation["chain"], map_row["pdb_resseq"], "A"),
            ):
                if key in residues:
                    residue = residues[key]
                    residue_key = "{}{}{}".format(key[0], key[1], key[2])
                    break
            aa_match = str(map_row.get("aa") == mutation["wt"])
        other_chains = sorted(chain for chain in complex_chain_set if chain != mutation["chain"])
        other_residues = []
        for chain in other_chains:
            other_residues.extend(chain_residues.get(chain, []))
        interface_dist = min_atom_distance(residue, other_residues)
        contact_density = ca_contact_density(residue, residues, args.contact_cutoff)
        interface_score = 0.0
        if interface_dist is not None:
            interface_score = max(0.0, min(1.0, (args.interface_cutoff - interface_dist) / args.interface_cutoff))
        core_score = None if contact_density is None else max(0.0, min(1.0, contact_density / float(args.high_contact_count)))
        if core_score is None:
            local_risk = interface_score
        else:
            local_risk = 0.65 * interface_score + 0.35 * core_score
        per_mut.append(
            {
                "mutation_token": mutation["token"],
                "mutation_chain": mutation["chain"],
                "mutation_cleaned_idx": mutation["cleaned_idx"],
                "mutation_wt": mutation["wt"],
                "mutation_to": mutation["mut"],
                "mapping_found": map_row is not None,
                "pdb_residue_found": residue is not None,
                "aa_match": aa_match,
                "pdb_residue_key": residue_key,
                "interface_min_atom_distance": interface_dist,
                "interface_within_cutoff": interface_dist is not None and interface_dist <= args.interface_cutoff,
                "ca_contact_density": contact_density,
                "interface_risk_score": interface_score,
                "contact_core_score": core_score,
                "binder_proxy_risk_score": local_risk,
            }
        )

    risks = [item["binder_proxy_risk_score"] for item in per_mut if item["binder_proxy_risk_score"] is not None]
    interface_dists = [item["interface_min_atom_distance"] for item in per_mut if item["interface_min_atom_distance"] is not None]
    contacts = [item["ca_contact_density"] for item in per_mut if item["ca_contact_density"] is not None]
    return {
        "skempi_pdb": row.get("#Pdb", ""),
        "pdb_id": pdb_id,
        "mutation_cleaned": row.get("Mutation(s)_cleaned", ""),
        "mutation_pdb": row.get("Mutation(s)_PDB", ""),
        "protein_1": row.get("Protein 1", ""),
        "protein_2": row.get("Protein 2", ""),
        "hold_out_type": row.get("Hold_out_type", ""),
        "mutation_location": row.get("iMutation_Location(s)", ""),
        "affinity_wt_M": "" if affinity_wt is None else affinity_wt,
        "affinity_mut_M": "" if affinity_mut is None else affinity_mut,
        "log10_kd_ratio_mut_over_wt": log10_kd_ratio,
        "damaging_label": damaging_label,
        "beneficial_label": beneficial_label,
        "n_mutations": len(mutations),
        "n_mapped_mutations": sum(1 for item in per_mut if item["mapping_found"]),
        "n_pdb_residue_found": sum(1 for item in per_mut if item["pdb_residue_found"]),
        "n_interface_mutations": sum(1 for item in per_mut if item["interface_within_cutoff"]),
        "mean_interface_distance": mean(interface_dists),
        "min_interface_distance": min(interface_dists) if interface_dists else None,
        "mean_contact_density": mean(contacts),
        "max_contact_density": max(contacts) if contacts else None,
        "mean_binder_proxy_risk": mean(risks),
        "max_binder_proxy_risk": max(risks) if risks else None,
        "pdb_path": pdb_path,
        "mapping_path": mapping_path,
        "structure_status": "success" if pdb_path and residues else "missing_pdb",
        "mapping_status": "success" if mapping_path and mapping else "missing_mapping",
        "per_mutation_jsonish": "|".join(
            "{token}:{chain}{idx}:{dist}:{risk}".format(
                token=item["mutation_token"],
                chain=item["mutation_chain"],
                idx=item["mutation_cleaned_idx"],
                dist=fmt(item["interface_min_atom_distance"], 3),
                risk=fmt(item["binder_proxy_risk_score"], 3),
            )
            for item in per_mut
        ),
    }


def write_csv(path, rows, fields):
    ensure_parent(path)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(rows):
    score_defs = [
        ("binder_proxy_mean", "mean_binder_proxy_risk"),
        ("binder_proxy_max", "max_binder_proxy_risk"),
        ("interface_count", "n_interface_mutations"),
        ("mutation_count", "n_mutations"),
        ("contact_density_mean", "mean_contact_density"),
        ("contact_density_max", "max_contact_density"),
        ("location_COR_label", "location_COR_score"),
        ("location_RIM_label", "location_RIM_score"),
        ("location_SUR_label", "location_SUR_score"),
        ("location_SUP_label", "location_SUP_score"),
    ]
    for row in rows:
        locs = set((row.get("mutation_location") or "").split(","))
        row["location_COR_score"] = 1.0 if "COR" in locs else 0.0
        row["location_RIM_score"] = 1.0 if "RIM" in locs else 0.0
        row["location_SUR_score"] = 1.0 if "SUR" in locs else 0.0
        row["location_SUP_score"] = 1.0 if "SUP" in locs else 0.0

    metrics = []
    for subset_name, subset in [
        ("all_scored", rows),
        ("single_mutation_scored", [row for row in rows if int(row["n_mutations"]) == 1]),
        ("mapped_interface_scored", [row for row in rows if int(row["n_pdb_residue_found"]) > 0]),
    ]:
        targets = [safe_float(row.get("log10_kd_ratio_mut_over_wt")) for row in subset]
        labels = [safe_float(row.get("damaging_label")) for row in subset]
        for score_name, col in score_defs:
            scores = [safe_float(row.get(col)) for row in subset]
            auroc, auprc = binary_metrics(labels, scores)
            metrics.append(
                {
                    "subset": subset_name,
                    "score_name": score_name,
                    "score_column": col,
                    "n": len(subset),
                    "n_scored": sum(1 for value in scores if value is not None),
                    "spearman_log10_kd_ratio": fmt(spearman(scores, targets)),
                    "auroc_damaging": fmt(auroc),
                    "auprc_damaging": fmt(auprc),
                    "top10_damaging_precision": fmt(topk_precision(labels, scores, 0.10)),
                    "top20_damaging_precision": fmt(topk_precision(labels, scores, 0.20)),
                }
            )
    return metrics


def write_report(path, args, rows, metrics, stats):
    ensure_parent(path)
    metric_by_name = {(row["subset"], row["score_name"]): row for row in metrics}
    with open(path, "w") as handle:
        handle.write("# SKEMPI v2 Binder-Retention Proxy Benchmark\n\n")
        handle.write("SKEMPI v2 is used here as an external binding-sensitivity proxy, not as a deletion ground truth dataset.  The benchmark asks whether residues that are structurally close to a partner chain, or sit in dense local structure, are enriched for experimentally damaging binding mutations.\n\n")
        handle.write("## Dataset\n\n")
        handle.write("- SKEMPI CSV: `{}`\n".format(args.skempi_csv))
        handle.write("- SKEMPI structure dir: `{}`\n".format(args.skempi_dir))
        handle.write("- Input rows: {}\n".format(stats["input_rows"]))
        handle.write("- Scored rows: {}\n".format(len(rows)))
        handle.write("- Unique complexes: {}\n".format(len({row["pdb_id"] for row in rows})))
        handle.write("- Rows with local PDB: {}\n".format(sum(1 for row in rows if row["structure_status"] == "success")))
        handle.write("- Rows with local mapping: {}\n".format(sum(1 for row in rows if row["mapping_status"] == "success")))
        handle.write("- Damaging threshold: log10(Kd_mut/Kd_wt) >= {:.3f}\n\n".format(args.damage_log10_ratio))
        handle.write("## Main Metrics\n\n")
        handle.write("| Subset | Score | N scored | Spearman vs log10 Kd ratio | AUROC damaging | AUPRC damaging | Top10 damaging |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|\n")
        for key in [
            ("all_scored", "binder_proxy_max"),
            ("all_scored", "binder_proxy_mean"),
            ("all_scored", "interface_count"),
            ("all_scored", "contact_density_max"),
            ("single_mutation_scored", "binder_proxy_max"),
            ("mapped_interface_scored", "binder_proxy_max"),
            ("all_scored", "location_COR_label"),
            ("all_scored", "location_RIM_label"),
        ]:
            row = metric_by_name.get(key)
            if not row:
                continue
            handle.write(
                "| {subset} | {score_name} | {n_scored} | {spearman_log10_kd_ratio} | {auroc_damaging} | {auprc_damaging} | {top10_damaging_precision} |\n".format(**row)
            )
        handle.write("\n## Interpretation\n\n")
        handle.write("- `binder_proxy_max` combines partner-chain interface proximity and local CA contact density.  It is a structural proxy for deletion risk around binding interfaces.\n")
        handle.write("- The experiment does not claim that point-mutation affinity changes are equivalent to deletion fitness.  It tests whether the planner's interface/core caution is aligned with independent binding-sensitivity evidence.\n")
        handle.write("- A weak or mixed signal should be reported as a limitation: SKEMPI is dominated by substitutions and curated interface categories, so it is best used as supporting external validation alongside ProteinGym deletion assays.\n\n")
        handle.write("SKEMPI_BINDER_RETENTION_PROXY_PASS\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SKEMPI v2 as a binder-retention proxy benchmark.")
    parser.add_argument("--skempi_dir", default="/public/home/zhangyangroup/chengshiz/keyuan.zhou/data/raw/skempi_v2")
    parser.add_argument("--skempi_csv", default="/public/home/zhangyangroup/chengshiz/keyuan.zhou/data/raw/skempi_v2/skempi_v2.csv")
    parser.add_argument("--interface_cutoff", type=float, default=5.0)
    parser.add_argument("--contact_cutoff", type=float, default=8.0)
    parser.add_argument("--high_contact_count", type=float, default=20.0)
    parser.add_argument("--damage_log10_ratio", type=float, default=0.5)
    parser.add_argument("--limit_rows", type=int, default=None)
    parser.add_argument("--out_scored_csv", default="results/skempi_binder_retention/skempi_v2_binder_proxy_scored.csv")
    parser.add_argument("--out_metrics_csv", default="results/skempi_binder_retention/skempi_v2_binder_proxy_metrics.csv")
    parser.add_argument("--out_report", default="results/skempi_binder_retention/SKEMPI_BINDER_RETENTION_PROXY_REPORT.md")
    args = parser.parse_args()

    raw_rows = read_skempi(args.skempi_csv)
    if args.limit_rows is not None:
        raw_rows = raw_rows[: args.limit_rows]
    caches = {"mapping": {}, "residues": {}}
    scored = []
    stats = Counter(input_rows=len(raw_rows))
    for row in raw_rows:
        out = score_record(row, args.skempi_dir, caches, args)
        if out["log10_kd_ratio_mut_over_wt"] is None:
            stats["missing_affinity_ratio"] += 1
            continue
        scored.append(out)

    fields = [
        "skempi_pdb",
        "pdb_id",
        "mutation_cleaned",
        "mutation_pdb",
        "protein_1",
        "protein_2",
        "hold_out_type",
        "mutation_location",
        "affinity_wt_M",
        "affinity_mut_M",
        "log10_kd_ratio_mut_over_wt",
        "damaging_label",
        "beneficial_label",
        "n_mutations",
        "n_mapped_mutations",
        "n_pdb_residue_found",
        "n_interface_mutations",
        "mean_interface_distance",
        "min_interface_distance",
        "mean_contact_density",
        "max_contact_density",
        "mean_binder_proxy_risk",
        "max_binder_proxy_risk",
        "pdb_path",
        "mapping_path",
        "structure_status",
        "mapping_status",
        "per_mutation_jsonish",
    ]
    for row in scored:
        for field in fields:
            if isinstance(row.get(field), float):
                row[field] = fmt(row[field])
    metrics = evaluate(scored)
    metric_fields = [
        "subset",
        "score_name",
        "score_column",
        "n",
        "n_scored",
        "spearman_log10_kd_ratio",
        "auroc_damaging",
        "auprc_damaging",
        "top10_damaging_precision",
        "top20_damaging_precision",
    ]
    write_csv(args.out_scored_csv, scored, fields)
    write_csv(args.out_metrics_csv, metrics, metric_fields)
    write_report(args.out_report, args, scored, metrics, stats)
    print("Wrote {}".format(args.out_scored_csv))
    print("Wrote {}".format(args.out_metrics_csv))
    print("Wrote {}".format(args.out_report))


if __name__ == "__main__":
    main()
