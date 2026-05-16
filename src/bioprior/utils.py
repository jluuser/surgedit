"""Shared helpers for BioPrior v1 scoring and segment proposal."""

import math
from pathlib import Path


DEFAULT_CONFIG = {
    "terminal_lengths": [5, 10, 15, 20, 30],
    "max_segment_len": 30,
    "min_segment_len": 3,
    "protected_feature_types": [
        "active site",
        "binding site",
        "site",
        "short sequence motif",
    ],
    "shadow_cutoff": 8.0,
    "contact_cutoff": 8.0,
    "closure_cutoff": 8.0,
    "low_plddt_threshold": 70,
    "high_plddt_threshold": 90,
    "low_contact_quantile": 0.3,
    "high_contact_quantile": 0.8,
    "motif_far_cutoff": 10.0,
    "hard_reject": [
        "protected_overlap",
        "disulfide_overlap",
        "metal_binding_overlap",
    ],
    "scoring_weights": {
        "terminal_tail": 1.0,
        "surface_flexible_loop": 1.0,
        "disorder_like": 0.8,
        "linker_compressibility": 0.6,
        "functional_core_risk": -5.0,
        "structural_core_risk": -2.0,
        "motif_shadow_risk": -2.0,
        "closure_bonus": 0.8,
    },
}


def load_bioprior_config(path="configs/bioprior_v1.yaml"):
    """Load BioPrior YAML config, falling back to the in-code defaults."""

    config_path = Path(path)
    if not config_path.exists():
        return dict(DEFAULT_CONFIG)
    try:
        import yaml
    except ImportError:
        return dict(DEFAULT_CONFIG)
    with config_path.open() as handle:
        loaded = yaml.safe_load(handle) or {}
    config = dict(DEFAULT_CONFIG)
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            merged = dict(config[key])
            merged.update(value)
            config[key] = merged
        else:
            config[key] = value
    return config


def normalize_segment(segment):
    """Return (start, end) for common segment dictionary conventions."""

    if "seg_start" in segment:
        start = segment["seg_start"]
    else:
        start = segment["start"]
    if "seg_end" in segment:
        end = segment["seg_end"]
    else:
        end = segment["end"]
    start = int(start)
    end = int(end)
    if start > end:
        raise ValueError("segment start must be <= end")
    return start, end


def segment_positions(segment):
    start, end = normalize_segment(segment)
    return range(start, end + 1)


def segment_len(segment):
    start, end = normalize_segment(segment)
    return end - start + 1


def residue_at(residue_features, position):
    if isinstance(residue_features, dict):
        return residue_features[position]
    return residue_features[position]


def segment_residue_features(segment, residue_features):
    return [residue_at(residue_features, pos) for pos in segment_positions(segment)]


def feature_float(row, *names, default=0.0):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return float(row[name])
    return float(default)


def feature_bool(row, *names):
    for name in names:
        if name not in row:
            continue
        value = row[name]
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        return str(value).strip().lower() in ("true", "1", "yes", "y")
    return False


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def fraction(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(1 for value in values if value) / float(len(values))


def quantile(values, q):
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    if q <= 0:
        return values[0]
    if q >= 1:
        return values[-1]
    idx = (len(values) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return values[lo]
    weight = idx - lo
    return values[lo] * (1 - weight) + values[hi] * weight


def ca_distance(coord_a, coord_b):
    if coord_a is None or coord_b is None:
        return None
    return math.sqrt(
        (coord_a[0] - coord_b[0]) ** 2
        + (coord_a[1] - coord_b[1]) ** 2
        + (coord_a[2] - coord_b[2]) ** 2
    )


def coord_at(structure_coords, position):
    if structure_coords is None:
        return None
    if isinstance(structure_coords, dict):
        return structure_coords.get(position)
    if 0 <= position < len(structure_coords):
        return structure_coords[position]
    return None


def make_segment(start, end, proposal_source, biological_rationale, **extra):
    segment = {
        "seg_start": int(start),
        "seg_end": int(end),
        "seg_len": int(end) - int(start) + 1,
        "proposal_source": proposal_source,
        "biological_rationale": biological_rationale,
    }
    segment.update(extra)
    return segment


def unique_segments(segments):
    seen = set()
    unique = []
    for segment in segments:
        key = (segment["seg_start"], segment["seg_end"], segment["proposal_source"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(segment)
    return unique
