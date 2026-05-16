import argparse
import csv
import json
import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from SCISOR.shortening_scud import ShorteningSCUD


DEFAULT_CKPT_URL = "https://huggingface.co/SCISOR/SCISOR/resolve/main/SCISOR_U90_S.ckpt"


def base_header(header):
    return header.split("|", 1)[0]


def read_fasta_to_df(fasta_file, max_sequences=100, max_length=1000):
    with open(fasta_file, "r") as file:
        content = file.read()

    sequences = []
    entries = content.strip().split(">")

    for entry in entries:
        if not entry:
            continue
        lines = entry.strip().split("\n")
        header = base_header(lines[0])
        sequence = "".join(lines[1:]).replace(" ", "").replace("\r", "")
        sequences.append(
            {"Header": header, "Sequence": sequence, "Length": len(sequence)}
        )

    df = pd.DataFrame(sequences).drop_duplicates()
    return df.head(max_sequences).query("Length <= @max_length").reset_index(drop=True)


def untokenize(seq, tokenizer):
    return (
        tokenizer.decode(seq)
        .replace(" ", "")
        .replace("<cls>", "")
        .replace("<eos>", "")
        .replace("<pad>", "")
    )


def save_sequences_to_fasta(fasta_file, seqs, headers):
    if os.path.dirname(fasta_file):
        os.makedirs(os.path.dirname(fasta_file), exist_ok=True)
    with open(fasta_file, "w") as f:
        for header, seq in zip(headers, seqs):
            f.write(f">{header}\n{seq}\n")


def read_motif_csv(path):
    motif_by_id = {}
    if path is None:
        return motif_by_id

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            positions = []
            raw_positions = row.get("protected_positions", "")
            if raw_positions:
                positions = [
                    int(token)
                    for token in raw_positions.split(";")
                    if token.strip() != ""
                ]
            motif_by_id[row["protein_id"]] = {
                "sequence": row.get("sequence", ""),
                "protected_positions": set(positions),
            }
    return motif_by_id


def read_structure_priors(path):
    priors_by_id = {}
    if path is None:
        return priors_by_id

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            protein_id = row.get("protein_id") or row.get("accession")
            position_value = row.get("position") or row.get("residue_index_0based")
            if protein_id is None or position_value is None:
                raise ValueError(
                    "--structure_priors must contain protein_id/position or "
                    "accession/residue_index_0based columns"
                )
            position = int(position_value)
            entry = priors_by_id.setdefault(
                protein_id,
                {
                    "motif_positions": set(),
                    "shadow_positions": set(),
                    "motif_contact_positions": set(),
                },
            )
            if row.get("is_motif") == "True" or row.get("is_protected") == "True":
                entry["motif_positions"].add(position)
            if row.get("is_motif_shadow_8A") == "True":
                entry["shadow_positions"].add(position)
            if int(float(row.get("motif_contact_count_8A", 0) or 0)) > 0:
                entry["motif_contact_positions"].add(position)
    return priors_by_id


def read_segment_candidates(path):
    candidates_by_id = {}
    if path is None:
        return candidates_by_id

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            protein_id = row["protein_id"]
            start = int(row["start"])
            end = int(row["end"])
            candidates_by_id.setdefault(protein_id, {})[(start, end)] = {
                "is_terminal_deletion": row.get("is_terminal_deletion") == "True",
                "closure_friendly": row.get("closure_friendly") == "True",
                "boundary_ca_distance": row.get("boundary_ca_distance", ""),
            }
    return candidates_by_id


def affected_segment_after_addition(deleted_positions, candidate_position):
    deleted = set(deleted_positions)
    deleted.add(candidate_position)
    start = candidate_position
    while start - 1 in deleted:
        start -= 1
    end = candidate_position
    while end + 1 in deleted:
        end += 1
    return start, end


def masked_shrink_sequence(
    model,
    x,
    num_deletions,
    original_lengths,
    protected_sets,
    row_ids=None,
    shadow_sets=None,
    protect_shadow=False,
    shadow_penalty=1.0,
    shadow_mode="multiply",
    closure_candidate_maps=None,
    protect_closure=False,
    closure_penalty=1.0,
    closure_mode="multiply",
    temperature=1,
):
    if shadow_mode != "multiply":
        raise ValueError(f"Unsupported shadow_mode: {shadow_mode}")
    if closure_mode != "multiply":
        raise ValueError(f"Unsupported closure_mode: {closure_mode}")
    if shadow_sets is None:
        shadow_sets = [set() for _ in protected_sets]
    if row_ids is None:
        row_ids = [f"row {idx}" for idx in range(x.shape[0])]
    if closure_candidate_maps is None:
        closure_candidate_maps = [{} for _ in protected_sets]

    original_index_maps = []
    for length in original_lengths:
        length = int(length)
        mapping = [None] + list(range(length)) + [None]
        mapping.extend([None] * (x.shape[1] - len(mapping)))
        original_index_maps.append(mapping)

    deleted_original_indices = [[] for _ in range(x.shape[0])]
    warnings = []
    closure_warning_keys = set()
    S = num_deletions.clone()
    with torch.no_grad():
        while S.sum() > 0:
            dels_this_interval = (S > 0).int()
            bsz, current_width = x.shape

            if model.window_size is not None and current_width > model.window_size:
                model_probs = model.predict_with_windows(x, None, S)
            else:
                model_logits = model.model_predict(x, None, None, S)
                model_probs = torch.softmax(model_logits, dim=-1)

            model_probs = (
                model_probs
                * (x != model.pad_token_id)
                * (x != model.tokenizer.cls_token_id)
                * (x != model.tokenizer.eos_token_id)
            )

            old_lengths = (x != model.pad_token_id).sum(dim=1)
            sampled_indices_by_row = []
            for row_idx in range(bsz):
                if dels_this_interval[row_idx] == 0:
                    sampled_indices_by_row.append([])
                    continue

                protected_mask = torch.zeros(
                    current_width, dtype=torch.bool, device=x.device
                )
                protected = protected_sets[row_idx]
                for current_idx, original_idx in enumerate(original_index_maps[row_idx]):
                    if original_idx in protected:
                        protected_mask[current_idx] = True

                row_probs = model_probs[row_idx].clone()
                row_probs[protected_mask] = 0
                if protect_shadow:
                    shadow = shadow_sets[row_idx]
                    if shadow:
                        shadow_mask = torch.zeros(
                            current_width, dtype=torch.bool, device=x.device
                        )
                        for current_idx, original_idx in enumerate(
                            original_index_maps[row_idx]
                        ):
                            if original_idx in shadow:
                                shadow_mask[current_idx] = True
                        row_probs[shadow_mask] *= shadow_penalty

                if protect_closure:
                    closure_candidates = closure_candidate_maps[row_idx]
                    if not closure_candidates:
                        warning_key = (row_ids[row_idx], "no_segment_candidates")
                        if warning_key not in closure_warning_keys:
                            warnings.append(
                                f"WARNING: {row_ids[row_idx]} has no_segment_candidates; "
                                "closure penalty disabled for this protein"
                            )
                            closure_warning_keys.add(warning_key)
                    else:
                        deleted_so_far = set(deleted_original_indices[row_idx])
                        for current_idx, original_idx in enumerate(
                            original_index_maps[row_idx]
                        ):
                            if (
                                original_idx is None
                                or float(row_probs[current_idx].item()) <= 0
                            ):
                                continue
                            segment = affected_segment_after_addition(
                                deleted_so_far, original_idx
                            )
                            candidate = closure_candidates.get(segment)
                            if candidate is None:
                                warning_key = (row_ids[row_idx], segment)
                                if warning_key not in closure_warning_keys:
                                    warnings.append(
                                        f"WARNING: {row_ids[row_idx]} missing segment_candidate "
                                        f"for [{segment[0]}, {segment[1]}]; no closure penalty"
                                    )
                                    closure_warning_keys.add(warning_key)
                                continue
                            if (
                                not candidate["is_terminal_deletion"]
                                and not candidate["closure_friendly"]
                            ):
                                row_probs[current_idx] *= closure_penalty
                candidate_indices = torch.nonzero(row_probs > 0, as_tuple=False).flatten()
                if candidate_indices.numel() == 0:
                    header_msg = f"row {row_idx}"
                    warnings.append(
                        f"WARNING: {header_msg} has no unprotected deletion candidates; "
                        f"remaining deletions skipped"
                    )
                    dels_this_interval[row_idx] = 0
                    S[row_idx] = 0
                    sampled_indices_by_row.append([])
                    continue

                if temperature == 0.0:
                    sampled_idx = candidate_indices[
                        torch.argmax(row_probs[candidate_indices])
                    ]
                else:
                    candidate_probs = row_probs[candidate_indices]
                    candidate_probs = torch.softmax(
                        torch.log(candidate_probs) / temperature, dim=-1
                    )
                    sampled_idx = candidate_indices[
                        torch.multinomial(candidate_probs, 1)[0]
                    ]
                sampled_indices_by_row.append([int(sampled_idx)])

            if dels_this_interval.sum() == 0:
                break

            new_lengths = x.shape[1] - dels_this_interval
            out = (
                torch.ones(
                    (bsz, int(new_lengths.max())),
                    dtype=x.dtype,
                    device=x.device,
                )
                * model.pad_token_id
            )
            new_maps = []
            for row_idx in range(bsz):
                sampled_indices = sampled_indices_by_row[row_idx]
                sampled_set = set(sampled_indices)
                if sampled_indices:
                    original_idx = original_index_maps[row_idx][sampled_indices[0]]
                    if original_idx is not None:
                        deleted_original_indices[row_idx].append(original_idx)

                keep_mask = ~torch.isin(
                    torch.arange(current_width, device=x.device),
                    torch.tensor(sampled_indices, device=x.device),
                )
                kept_tokens = x[row_idx, keep_mask]
                out[row_idx, : int(new_lengths[row_idx])] = kept_tokens
                new_maps.append(
                    [
                        original_idx
                        for current_idx, original_idx in enumerate(
                            original_index_maps[row_idx]
                        )
                        if current_idx not in sampled_set
                    ]
                )

            max_new_len = int(max(old_lengths - dels_this_interval))
            x = out[:, :max_new_len]
            original_index_maps = [mapping[:max_new_len] for mapping in new_maps]
            S = S - dels_this_interval

    deleted_original_indices = [sorted(indices) for indices in deleted_original_indices]
    return x, deleted_original_indices, warnings


def write_motif_eval(
    output_dir,
    headers,
    original_lengths,
    new_sequences,
    num_deletions,
    deleted_indices,
    motif_by_id,
):
    path = os.path.join(output_dir, "motif_eval.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "protein_id",
                "num_protected",
                "num_deleted_protected",
                "motif_deletion_rate",
                "compression_success",
                "orig_len",
                "new_len",
                "target_new_len",
                "deleted_protected_positions",
            ],
        )
        writer.writeheader()
        for header, orig_len, new_seq, target_dels, dels in zip(
            headers, original_lengths, new_sequences, num_deletions, deleted_indices
        ):
            protected = motif_by_id.get(header, {}).get("protected_positions", set())
            deleted_protected = sorted(set(dels) & protected)
            target_new_len = int(orig_len) - int(target_dels)
            num_protected = len(protected)
            rate = (
                len(deleted_protected) / num_protected
                if num_protected > 0
                else 0.0
            )
            writer.writerow(
                {
                    "protein_id": header,
                    "num_protected": num_protected,
                    "num_deleted_protected": len(deleted_protected),
                    "motif_deletion_rate": f"{rate:.6g}",
                    "compression_success": str(len(new_seq) == target_new_len),
                    "orig_len": int(orig_len),
                    "new_len": len(new_seq),
                    "target_new_len": target_new_len,
                    "deleted_protected_positions": ";".join(
                        str(pos) for pos in deleted_protected
                    ),
                }
            )
    return path


def write_shadow_eval(
    output_dir,
    headers,
    original_lengths,
    new_sequences,
    num_deletions,
    deleted_indices,
    motif_by_id,
    structure_priors_by_id,
):
    path = os.path.join(output_dir, "shadow_eval.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "protein_id",
                "has_structure_prior",
                "num_motif",
                "num_shadow",
                "num_deleted",
                "num_deleted_motif",
                "num_deleted_shadow",
                "num_deleted_motif_contact",
                "motif_deletion_rate",
                "shadow_deletion_rate_over_deleted",
                "motif_contact_deletion_rate_over_deleted",
                "compression_success",
                "orig_len",
                "new_len",
                "target_new_len",
            ],
        )
        writer.writeheader()
        for header, orig_len, new_seq, target_dels, dels in zip(
            headers, original_lengths, new_sequences, num_deletions, deleted_indices
        ):
            prior = structure_priors_by_id.get(header)
            has_prior = prior is not None
            motif_positions = motif_by_id.get(header, {}).get("protected_positions", set())
            shadow_positions = prior["shadow_positions"] if has_prior else set()
            motif_contact_positions = (
                prior["motif_contact_positions"] if has_prior else set()
            )
            deleted = set(dels)
            deleted_motif = deleted & motif_positions
            deleted_shadow = deleted & shadow_positions
            deleted_motif_contact = deleted & motif_contact_positions
            target_new_len = int(orig_len) - int(target_dels)
            num_deleted = len(dels)
            writer.writerow(
                {
                    "protein_id": header,
                    "has_structure_prior": str(has_prior),
                    "num_motif": len(motif_positions),
                    "num_shadow": len(shadow_positions),
                    "num_deleted": num_deleted,
                    "num_deleted_motif": len(deleted_motif),
                    "num_deleted_shadow": len(deleted_shadow),
                    "num_deleted_motif_contact": len(deleted_motif_contact),
                    "motif_deletion_rate": (
                        f"{len(deleted_motif) / len(motif_positions):.6g}"
                        if motif_positions
                        else "0"
                    ),
                    "shadow_deletion_rate_over_deleted": (
                        f"{len(deleted_shadow) / num_deleted:.6g}"
                        if num_deleted
                        else "0"
                    ),
                    "motif_contact_deletion_rate_over_deleted": (
                        f"{len(deleted_motif_contact) / num_deleted:.6g}"
                        if num_deleted
                        else "0"
                    ),
                    "compression_success": str(len(new_seq) == target_new_len),
                    "orig_len": int(orig_len),
                    "new_len": len(new_seq),
                    "target_new_len": target_new_len,
                }
            )
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Run SCISOR protein shrinking.")
    parser.add_argument("--input", default="data/examples/toy_input.fasta")
    parser.add_argument("--output-dir", default="results/scisor_toy/raw_run")
    parser.add_argument("--output-fasta", default="shrunk_sequences.fasta")
    parser.add_argument("--metadata-json", default="deletions.json")
    parser.add_argument("--p0", default="p0.pt")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT_URL)
    parser.add_argument("--shrink-pct", type=float, default=10.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-sequences", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=1000)
    parser.add_argument("--motif_csv", default=None)
    parser.add_argument("--protect_motif", action="store_true")
    parser.add_argument("--structure_priors", default=None)
    parser.add_argument("--protect_shadow", action="store_true")
    parser.add_argument("--shadow_penalty", type=float, default=1.0)
    parser.add_argument("--shadow_mode", default="multiply")
    parser.add_argument("--segment_candidates", default=None)
    parser.add_argument("--protect_closure", action="store_true")
    parser.add_argument("--closure_penalty", type=float, default=0.2)
    parser.add_argument("--closure_mode", default="multiply")
    parser.add_argument(
        "--disable-fa",
        action="store_true",
        help="Instantiate FAESM with PyTorch SDPA instead of flash-attention paths.",
    )
    return parser.parse_args()


class FAESMBaseNoFA(nn.Module):
    def __init__(self, hf_model_name="esm2_t6_8M_UR50D", **kwargs):
        super().__init__()
        from SCISOR.esm import FAEsmForMaskedLM

        print(f"Using FAESM model {hf_model_name} with use_fa=False")
        conditioning_dim = kwargs.get("d_embedding", 128)
        pretrained = kwargs.get("pretrained", True)

        self.faesm = FAEsmForMaskedLM.from_pretrained(
            pretrained_model_name_or_path=f"facebook/{hf_model_name}",
            use_fa=False,
            conditioning_dim=conditioning_dim,
            load_pretrained_weights=pretrained,
        )
        self.embed_dim = self.faesm.esm.embeddings.word_embeddings.embedding_dim
        self.proj = nn.Linear(self.embed_dim, 1)

    def forward(self, x, t, input_mask=None, S=None):
        cond = t if S is None else S
        embeddings = self.faesm(
            input_ids=x, attention_mask=input_mask, conditioning=cond
        )["last_hidden_state"]
        return self.proj(embeddings).squeeze()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_fasta = os.path.join(args.output_dir, args.output_fasta)
    metadata_json = os.path.join(args.output_dir, args.metadata_json)

    if args.disable_fa:
        import SCISOR.continuous_time_diffusion as ctd

        ctd.FAESM_Base = FAESMBaseNoFA

    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    original_protein_df = read_fasta_to_df(
        args.input, max_sequences=args.max_sequences, max_length=args.max_length
    )
    motif_by_id = read_motif_csv(args.motif_csv)
    structure_priors_by_id = read_structure_priors(args.structure_priors)
    segment_candidates_by_id = read_segment_candidates(args.segment_candidates)
    if args.protect_shadow and not args.structure_priors:
        raise ValueError("--protect_shadow requires --structure_priors")
    if args.protect_closure and not args.segment_candidates:
        raise ValueError("--protect_closure requires --segment_candidates")
    if args.shadow_mode != "multiply":
        raise ValueError(f"Unsupported --shadow_mode: {args.shadow_mode}")
    if args.closure_mode != "multiply":
        raise ValueError(f"Unsupported --closure_mode: {args.closure_mode}")
    if not 0 <= args.shadow_penalty <= 1:
        raise ValueError("--shadow_penalty must be between 0 and 1 for multiply mode")
    if not 0 <= args.closure_penalty <= 1:
        raise ValueError("--closure_penalty must be between 0 and 1 for multiply mode")
    if args.motif_csv:
        for row in original_protein_df.itertuples():
            motif = motif_by_id.get(row.Header)
            if motif is None:
                raise ValueError(f"Missing motif row for {row.Header}")
            if motif["sequence"] and motif["sequence"] != row.Sequence:
                raise ValueError(f"Motif CSV sequence mismatch for {row.Header}")
    seq_lengths = torch.tensor(original_protein_df.Length.values, device=model.device)
    num_deletions = torch.ceil(seq_lengths * args.shrink_pct / 100).int()
    print(
        f"Shrinking {len(original_protein_df)} sequences by {args.shrink_pct:g}% "
        f"on {device}"
    )

    input_ids = [model.tokenizer(s).input_ids for s in original_protein_df.Sequence]
    max_len = max(len(x) for x in input_ids)
    x = torch.vstack(
        [
            F.pad(
                torch.tensor(ids, device=model.device),
                (0, max_len - len(ids)),
                value=model.tokenizer.pad_token_id,
            )
            for ids in input_ids
        ]
    )

    sampled_sequences = []
    deleted_indices = []
    batch_size = model.hparams.batch_size
    for i in tqdm(range(0, len(x), batch_size)):
        batch_df = original_protein_df.iloc[i : i + batch_size]
        if args.protect_motif or args.protect_shadow or args.protect_closure:
            protected_sets = [
                motif_by_id.get(header, {}).get("protected_positions", set())
                for header in batch_df.Header
            ]
            shadow_sets = [
                structure_priors_by_id.get(header, {}).get("shadow_positions", set())
                for header in batch_df.Header
            ]
            closure_candidate_maps = [
                segment_candidates_by_id.get(header, {}) for header in batch_df.Header
            ]
            if args.protect_shadow:
                for header in batch_df.Header:
                    if header not in structure_priors_by_id:
                        print(f"WARNING: {header} has no_structure_prior")
            if args.protect_closure:
                for header in batch_df.Header:
                    if header not in segment_candidates_by_id:
                        print(f"WARNING: {header} has no_segment_candidates")
            sequences, del_idx, warnings = masked_shrink_sequence(
                model,
                x[i : i + batch_size],
                num_deletions[i : i + batch_size],
                batch_df.Length.values,
                protected_sets,
                row_ids=list(batch_df.Header),
                shadow_sets=shadow_sets,
                protect_shadow=args.protect_shadow,
                shadow_penalty=args.shadow_penalty,
                shadow_mode=args.shadow_mode,
                closure_candidate_maps=closure_candidate_maps,
                protect_closure=args.protect_closure,
                closure_penalty=args.closure_penalty,
                closure_mode=args.closure_mode,
                temperature=args.temperature,
            )
            for warning in warnings:
                print(warning)
        else:
            sequences, preserved_indices = model.shrink_sequence(
                x[i : i + batch_size],
                num_deletions[i : i + batch_size],
                temperature=args.temperature,
            )
            del_idx = [
                sorted(set(range(len(s))) - set(j - 1 for j in p))
                for s, p in zip(batch_df.Sequence, preserved_indices)
            ]
        decoded_seqs = [untokenize(s, model.tokenizer) for s in sequences]
        sampled_sequences.extend(decoded_seqs)
        deleted_indices.extend(del_idx)

    assert all(
        "".join(c for i, c in enumerate(s) if i not in d) == n
        for s, d, n in zip(
            original_protein_df.Sequence, deleted_indices, sampled_sequences
        )
    )

    del_str = [
        ",".join(f"{c}{i}" for i, c in enumerate(s) if i in d)
        for s, d in zip(original_protein_df.Sequence, deleted_indices)
    ]
    new_headers = (
        original_protein_df.Header
        + "|deletions "
        + pd.Series(del_str)
        + f"|percentage {args.shrink_pct:g}"
    )
    save_sequences_to_fasta(output_fasta, sampled_sequences, new_headers)

    metadata = []
    for header, old_seq, new_seq, del_idx, del_tokens in zip(
        original_protein_df.Header,
        original_protein_df.Sequence,
        sampled_sequences,
        deleted_indices,
        del_str,
    ):
        metadata.append(
            {
                "header": header,
                "original_length": len(old_seq),
                "new_length": len(new_seq),
                "deletion_positions_zero_based": del_idx,
                "deletions_header_field": del_tokens,
            }
        )
    with open(metadata_json, "w") as f:
        json.dump(metadata, f, indent=2)

    if args.motif_csv:
        motif_eval_path = write_motif_eval(
            args.output_dir,
            original_protein_df.Header,
            original_protein_df.Length,
            sampled_sequences,
            num_deletions.cpu().tolist(),
            deleted_indices,
            motif_by_id,
        )
        print(f"Saved motif eval to {motif_eval_path}")

    if args.structure_priors:
        shadow_eval_path = write_shadow_eval(
            args.output_dir,
            original_protein_df.Header,
            original_protein_df.Length,
            sampled_sequences,
            num_deletions.cpu().tolist(),
            deleted_indices,
            motif_by_id,
            structure_priors_by_id,
        )
        print(f"Saved shadow eval to {shadow_eval_path}")

    print(f"Saved shrunk sequences to {output_fasta}")
    print(f"Saved deletion metadata to {metadata_json}")


if __name__ == "__main__":
    main()
