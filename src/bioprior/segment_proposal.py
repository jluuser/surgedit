"""BioPrior v1 candidate segment proposal functions."""

from .prior_types import (
    DISORDER_LIKE,
    FUNCTIONAL_CORE_RISK,
    LINKER_COMPRESSIBLE,
    STRUCTURAL_CORE_RISK,
    SURFACE_FLEXIBLE_LOOP,
    TERMINAL_TAIL,
)
from .scoring import compute_geometric_closure_score
from .utils import (
    feature_bool,
    feature_float,
    make_segment,
    mean,
    quantile,
    unique_segments,
)


def _runs_from_mask(mask, min_len, max_len):
    runs = []
    start = None
    for idx, value in enumerate(mask + [False]):
        if value and start is None:
            start = idx
        if not value and start is not None:
            end = idx - 1
            length = end - start + 1
            if length >= min_len:
                if length <= max_len:
                    runs.append((start, end))
                else:
                    for chunk_start in range(start, end + 1, max_len):
                        chunk_end = min(end, chunk_start + max_len - 1)
                        if chunk_end - chunk_start + 1 >= min_len:
                            runs.append((chunk_start, chunk_end))
            start = None
    return runs


def propose_terminal_tail_segments(protein_length, config):
    segments = []
    min_len = int(config.get("min_segment_len", 3))
    max_len = int(config.get("max_segment_len", 30))
    for length in config.get("terminal_lengths", [5, 10, 15, 20, 30]):
        length = int(length)
        if length < min_len or length > max_len or length > protein_length:
            continue
        segments.append(
            make_segment(
                0,
                length - 1,
                TERMINAL_TAIL,
                "N-terminal tail truncation candidate",
            )
        )
        segments.append(
            make_segment(
                protein_length - length,
                protein_length - 1,
                TERMINAL_TAIL,
                "C-terminal tail truncation candidate",
            )
        )
    return unique_segments(segments)


def propose_surface_flexible_loop_segments(residue_features, structure_coords=None, config=None):
    config = config or {}
    min_len = int(config.get("min_segment_len", 3))
    max_len = int(config.get("max_segment_len", 30))
    contact_values = [
        feature_float(row, "contact_density_8A", "contact_density")
        for row in residue_features
    ]
    low_contact = quantile(contact_values, float(config.get("low_contact_quantile", 0.3)))
    motif_far = float(config.get("motif_far_cutoff", 10.0))
    mask = []
    for row in residue_features:
        mask.append(
            feature_float(row, "contact_density_8A", "contact_density") <= low_contact
            and feature_float(row, "distance_to_motif", default=999.0) >= motif_far
            and not feature_bool(row, "is_motif", "is_protected")
            and not feature_bool(row, "is_motif_shadow_8A", "is_shadow")
        )
    segments = []
    for start, end in _runs_from_mask(mask, min_len, max_len):
        segment = make_segment(
            start,
            end,
            SURFACE_FLEXIBLE_LOOP,
            "motif-far low-contact flexible loop-like candidate",
        )
        closure = compute_geometric_closure_score(segment, structure_coords, config)
        if closure["closure_friendly"]:
            segment.update(closure)
            segments.append(segment)
    return unique_segments(segments)


def propose_disorder_like_segments(residue_features, config=None):
    config = config or {}
    min_len = int(config.get("min_segment_len", 3))
    max_len = int(config.get("max_segment_len", 30))
    low_plddt = float(config.get("low_plddt_threshold", 70))
    mask = [
        feature_float(row, "plddt") <= low_plddt
        and not feature_bool(row, "is_motif", "is_protected")
        for row in residue_features
    ]
    return unique_segments(
        [
            make_segment(
                start,
                end,
                DISORDER_LIKE,
                "low-pLDDT disorder-like candidate",
            )
            for start, end in _runs_from_mask(mask, min_len, max_len)
        ]
    )


def propose_linker_compressible_segments(residue_features, structure_coords=None, config=None):
    config = config or {}
    min_len = int(config.get("min_segment_len", 3))
    max_len = int(config.get("max_segment_len", 30))
    contact_values = [
        feature_float(row, "contact_density_8A", "contact_density")
        for row in residue_features
    ]
    low_contact = quantile(contact_values, float(config.get("low_contact_quantile", 0.3)))
    motif_far = float(config.get("motif_far_cutoff", 10.0))
    mask = []
    for idx, row in enumerate(residue_features):
        is_terminal_near = idx < min_len or idx >= len(residue_features) - min_len
        mask.append(
            not is_terminal_near
            and feature_float(row, "contact_density_8A", "contact_density") <= low_contact
            and feature_float(row, "distance_to_motif", default=999.0) >= motif_far
            and not feature_bool(row, "is_motif", "is_protected")
        )
    segments = []
    for start, end in _runs_from_mask(mask, min_len, max_len):
        segment = make_segment(
            start,
            end,
            LINKER_COMPRESSIBLE,
            "low-contact motif-far linker-like compressibility candidate",
        )
        closure = compute_geometric_closure_score(segment, structure_coords, config)
        if closure["closure_friendly"]:
            segment.update(closure)
            segments.append(segment)
    return unique_segments(segments)


def propose_hard_negative_segments(residue_features, config=None):
    config = config or {}
    min_len = int(config.get("min_segment_len", 3))
    max_len = int(config.get("max_segment_len", 30))
    contact_values = [
        feature_float(row, "contact_density_8A", "contact_density")
        for row in residue_features
    ]
    high_contact = quantile(contact_values, float(config.get("high_contact_quantile", 0.8)))
    high_plddt = float(config.get("high_plddt_threshold", 90))

    protected_mask = [
        feature_bool(row, "is_motif", "is_protected")
        or feature_bool(row, "is_disulfide", "disulfide_overlap")
        or feature_bool(row, "is_metal_binding", "metal_binding_overlap")
        for row in residue_features
    ]
    core_mask = [
        feature_float(row, "contact_density_8A", "contact_density") >= high_contact
        and feature_float(row, "plddt") >= high_plddt
        for row in residue_features
    ]

    segments = []
    for start, end in _runs_from_mask(protected_mask, 1, max_len):
        pad_start = max(0, start - 1)
        pad_end = min(len(residue_features) - 1, end + 1)
        if pad_end - pad_start + 1 >= min_len:
            segments.append(
                make_segment(
                    pad_start,
                    pad_end,
                    FUNCTIONAL_CORE_RISK,
                    "functional protected-site hard negative",
                )
            )
    for start, end in _runs_from_mask(core_mask, min_len, max_len):
        segments.append(
            make_segment(
                start,
                end,
                STRUCTURAL_CORE_RISK,
                "high-contact structural-core hard negative",
            )
        )
    return unique_segments(segments)


def propose_bioprior_segments(residue_features, structure_coords=None, config=None, include_hard_negatives=True):
    config = config or {}
    protein_length = len(residue_features)
    segments = []
    segments.extend(propose_terminal_tail_segments(protein_length, config))
    segments.extend(propose_surface_flexible_loop_segments(residue_features, structure_coords, config))
    segments.extend(propose_disorder_like_segments(residue_features, config))
    segments.extend(propose_linker_compressible_segments(residue_features, structure_coords, config))
    if include_hard_negatives:
        segments.extend(propose_hard_negative_segments(residue_features, config))

    merged = {}
    for segment in segments:
        key = (segment["seg_start"], segment["seg_end"], segment["proposal_source"])
        merged[key] = segment
    return list(merged.values())
