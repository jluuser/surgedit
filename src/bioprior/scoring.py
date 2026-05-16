"""BioPrior v1 segment-level scoring functions."""

from .utils import (
    ca_distance,
    coord_at,
    feature_bool,
    feature_float,
    fraction,
    mean,
    normalize_segment,
    segment_len,
    segment_residue_features,
)


def compute_functional_core_risk(segment, residue_features):
    """Return hard functional overlap risk in [0, 1]."""

    rows = segment_residue_features(segment, residue_features)
    overlap = [
        feature_bool(row, "is_motif", "is_protected", "protected_overlap")
        or feature_bool(row, "is_disulfide", "disulfide_overlap")
        or feature_bool(row, "is_metal_binding", "metal_binding_overlap")
        or feature_bool(row, "is_ptm", "ptm_overlap")
        for row in rows
    ]
    return fraction(overlap)


def compute_structural_core_risk(segment, residue_features, config=None):
    """Return soft structural-core risk in [0, 1]."""

    config = config or {}
    high_plddt = float(config.get("high_plddt_threshold", 90))
    contact_values = [
        feature_float(row, "contact_density_8A", "contact_density")
        for row in segment_residue_features(segment, residue_features)
    ]
    if contact_values:
        sorted_contacts = sorted(contact_values)
        high_contact_threshold = sorted_contacts[int(0.8 * (len(sorted_contacts) - 1))]
    else:
        high_contact_threshold = float(config.get("contact_cutoff", 8.0))

    rows = segment_residue_features(segment, residue_features)
    high_contact_fraction = fraction(
        feature_float(row, "contact_density_8A", "contact_density")
        >= high_contact_threshold
        for row in rows
    )
    high_plddt_fraction = fraction(
        feature_float(row, "plddt") >= high_plddt for row in rows
    )
    motif_contact_fraction = fraction(
        feature_float(row, "motif_contact_count_8A", "motif_contact_count") > 0
        for row in rows
    )
    return min(1.0, mean([high_contact_fraction, high_plddt_fraction, motif_contact_fraction]))


def compute_motif_shadow_risk(segment, residue_features):
    rows = segment_residue_features(segment, residue_features)
    return fraction(feature_bool(row, "is_motif_shadow_8A", "is_shadow") for row in rows)


def compute_flexibility_score(segment, residue_features, config=None):
    """Proxy for surface/flexible loop: low contact, motif-far, not motif-shadow."""

    config = config or {}
    motif_far_cutoff = float(config.get("motif_far_cutoff", 10.0))
    rows = segment_residue_features(segment, residue_features)
    contact_values = [feature_float(row, "contact_density_8A", "contact_density") for row in rows]
    mean_contact = mean(contact_values)
    low_contact_ref = float(config.get("contact_cutoff", 8.0))
    low_contact_score = max(0.0, min(1.0, 1.0 - mean_contact / max(low_contact_ref, 1e-6)))
    motif_far_score = fraction(
        feature_float(row, "distance_to_motif") >= motif_far_cutoff for row in rows
    )
    shadow_free_score = 1.0 - compute_motif_shadow_risk(segment, residue_features)
    return mean([low_contact_score, motif_far_score, shadow_free_score])


def compute_terminal_tail_score(segment, protein_length):
    start, end = normalize_segment(segment)
    if start == 0 or end == int(protein_length) - 1:
        return 1.0
    return 0.0


def compute_disorder_like_score(segment, residue_features, config=None):
    config = config or {}
    low_plddt = float(config.get("low_plddt_threshold", 70))
    rows = segment_residue_features(segment, residue_features)
    return fraction(feature_float(row, "plddt") <= low_plddt for row in rows)


def compute_linker_compressibility_score(segment, residue_features, config=None):
    """Proxy linker score: low contact, motif-far, moderate length, low core risk."""

    config = config or {}
    flex = compute_flexibility_score(segment, residue_features, config)
    core_risk = compute_structural_core_risk(segment, residue_features, config)
    length = segment_len(segment)
    max_len = float(config.get("max_segment_len", 30))
    length_score = 1.0 - abs(length - max_len / 2.0) / max(max_len / 2.0, 1.0)
    length_score = max(0.0, min(1.0, length_score))
    return mean([flex, length_score, 1.0 - core_risk])


def compute_geometric_closure_score(segment, structure_coords, config=None):
    """Return closure score in [0, 1] and boundary metadata."""

    config = config or {}
    start, end = normalize_segment(segment)
    protein_length = len(structure_coords) if structure_coords is not None and not isinstance(structure_coords, dict) else None
    if start == 0 or (protein_length is not None and end == protein_length - 1):
        return {
            "score": 1.0,
            "is_terminal_deletion": True,
            "boundary_left": None,
            "boundary_right": None,
            "boundary_ca_distance": None,
            "closure_friendly": True,
        }

    left = start - 1
    right = end + 1
    coord_left = coord_at(structure_coords, left)
    coord_right = coord_at(structure_coords, right)
    dist = ca_distance(coord_left, coord_right)
    cutoff = float(config.get("closure_cutoff", 8.0))
    if dist is None:
        score = 0.0
        closure_friendly = False
    else:
        score = max(0.0, min(1.0, 1.0 - dist / max(cutoff * 2.0, 1e-6)))
        closure_friendly = dist <= cutoff
    return {
        "score": 1.0 if closure_friendly else score,
        "is_terminal_deletion": False,
        "boundary_left": left,
        "boundary_right": right,
        "boundary_ca_distance": dist,
        "closure_friendly": closure_friendly,
    }


def compute_bioprior_score(segment, residue_features, structure_coords=None, config=None):
    """Compute weighted BioPrior v1 score and component diagnostics."""

    config = config or {}
    weights = config.get("scoring_weights", {})
    protein_length = len(residue_features)

    functional_risk = compute_functional_core_risk(segment, residue_features)
    structural_risk = compute_structural_core_risk(segment, residue_features, config)
    shadow_risk = compute_motif_shadow_risk(segment, residue_features)
    terminal = compute_terminal_tail_score(segment, protein_length)
    flex = compute_flexibility_score(segment, residue_features, config)
    disorder = compute_disorder_like_score(segment, residue_features, config)
    linker = compute_linker_compressibility_score(segment, residue_features, config)
    closure = compute_geometric_closure_score(segment, structure_coords, config)

    score = (
        weights.get("terminal_tail", 1.0) * terminal
        + weights.get("surface_flexible_loop", 1.0) * flex
        + weights.get("disorder_like", 0.8) * disorder
        + weights.get("linker_compressibility", 0.6) * linker
        + weights.get("functional_core_risk", -5.0) * functional_risk
        + weights.get("structural_core_risk", -2.0) * structural_risk
        + weights.get("motif_shadow_risk", -2.0) * shadow_risk
        + weights.get("closure_bonus", 0.8) * closure["score"]
    )

    hard_reject_reasons = []
    if functional_risk > 0:
        hard_reject_reasons.append("protected_overlap")
    if any(
        feature_bool(row, "is_disulfide", "disulfide_overlap")
        for row in segment_residue_features(segment, residue_features)
    ):
        hard_reject_reasons.append("disulfide_overlap")
    if any(
        feature_bool(row, "is_metal_binding", "metal_binding_overlap")
        for row in segment_residue_features(segment, residue_features)
    ):
        hard_reject_reasons.append("metal_binding_overlap")

    return {
        "score": score,
        "hard_reject": bool(hard_reject_reasons),
        "hard_reject_reasons": hard_reject_reasons,
        "components": {
            "terminal_tail": terminal,
            "surface_flexible_loop": flex,
            "disorder_like": disorder,
            "linker_compressibility": linker,
            "functional_core_risk": functional_risk,
            "structural_core_risk": structural_risk,
            "motif_shadow_risk": shadow_risk,
            "geometric_closure": closure["score"],
        },
        "closure": closure,
    }
