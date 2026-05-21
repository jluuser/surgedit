"""Trainable deletion-compatibility model utilities.

This module is intentionally lightweight.  It provides a tabular neural model
for the first BioDel-Cert learning core: predict whether a candidate deletion
interval is tolerated from ProteinGym DMS labels and existing BioPrior segment
features.  Larger PLM-embedding models can replace this module later while
keeping the same scored-column interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


DEFAULT_FEATURE_COLUMNS = [
    "deletion_fraction",
    "normalized_start",
    "normalized_end",
    "normalized_midpoint",
    "is_terminal_deletion",
    "terminal_proximity_score",
    "terminal_overlap_fraction",
    "protected_overlap_count",
    "protected_overlap_fraction_step3",
    "n_protected_overlap",
    "protected_overlap_fraction",
    "n_shadow_overlap",
    "shadow_overlap_fraction",
    "min_distance_to_protected",
    "mean_distance_to_protected",
    "mean_contact_density_8A",
    "max_contact_density_8A",
    "mean_motif_contact_count_8A",
    "max_motif_contact_count_8A",
    "mean_pLDDT",
    "min_pLDDT",
    "low_pLDDT_fraction",
    "high_pLDDT_fraction",
    "boundary_ca_distance",
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
    "stage1_mean_prob_b10",
    "stage1_sum_prob_b10",
    "stage1_max_prob_b10",
    "stage1_mean_logit_b10",
    "stage1_mean_prob_b20",
    "stage1_sum_prob_b20",
    "stage1_max_prob_b20",
    "stage1_mean_logit_b20",
    "stage1_mean_prob_b30",
    "stage1_sum_prob_b30",
    "stage1_max_prob_b30",
    "stage1_mean_logit_b30",
    "stage1_utility_score",
    "risk_point",
    "risk_upper",
    "risk_interval_width",
    "evidence_confidence",
    "has_annotation_evidence",
    "has_structure_evidence",
    "missing_annotation_penalty",
    "missing_structure_penalty",
    "low_plddt_uncertainty_penalty",
    "zero_shot_Tranception_S",
    "zero_shot_Tranception_M",
    "zero_shot_Tranception_L",
    "zero_shot_HMM",
    "zero_shot_Provean",
    "zero_shot_PoET",
]


CATEGORICAL_COLUMNS = [
    "proposal_source",
    "closure_type",
    "evidence_level",
    "structure_feature_status",
    "swissprot_feature_status",
    "risk_certificate_status",
]


def safe_float(value, default: Optional[float] = None) -> Optional[float]:
    if value in ("", None):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def safe_bool_float(value) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "y"):
        return 1.0
    if text in ("false", "0", "no", "n"):
        return 0.0
    return safe_float(value, 0.0) or 0.0


def selected_feature_columns(rows: Sequence[Dict[str, str]], requested: Optional[Sequence[str]] = None) -> List[str]:
    """Return numeric features that are present in at least one row."""

    candidates = list(requested or DEFAULT_FEATURE_COLUMNS)
    out = []
    for col in candidates:
        if any(col in row and row.get(col) not in ("", None) for row in rows):
            out.append(col)
    return out


def categorical_maps(rows: Sequence[Dict[str, str]], requested: Optional[Sequence[str]] = None, max_values: int = 32) -> Dict[str, Dict[str, int]]:
    maps: Dict[str, Dict[str, int]] = {}
    for col in requested or CATEGORICAL_COLUMNS:
        values = sorted({str(row.get(col, "") or "missing") for row in rows if col in row})
        if not values:
            continue
        values = values[:max_values]
        maps[col] = {value: idx for idx, value in enumerate(values)}
    return maps


@dataclass
class FeatureSpec:
    numeric_columns: List[str]
    categorical_value_maps: Dict[str, Dict[str, int]]
    means: List[float]
    stds: List[float]

    @property
    def input_dim(self) -> int:
        return len(self.means)

    def to_dict(self) -> Dict[str, object]:
        return {
            "numeric_columns": self.numeric_columns,
            "categorical_value_maps": self.categorical_value_maps,
            "means": self.means,
            "stds": self.stds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "FeatureSpec":
        return cls(
            numeric_columns=list(data["numeric_columns"]),
            categorical_value_maps={k: dict(v) for k, v in dict(data.get("categorical_value_maps", {})).items()},
            means=[float(x) for x in data["means"]],
            stds=[float(x) for x in data["stds"]],
        )


def raw_features(row: Dict[str, str], numeric_columns: Sequence[str], cat_maps: Dict[str, Dict[str, int]]) -> List[float]:
    values: List[float] = []
    for col in numeric_columns:
        if col in ("is_terminal_deletion", "closure_friendly_8A", "has_annotation_evidence", "has_structure_evidence"):
            values.append(safe_bool_float(row.get(col, "")))
        else:
            values.append(safe_float(row.get(col), 0.0) or 0.0)
    for col, value_map in cat_maps.items():
        value = str(row.get(col, "") or "missing")
        idx = value_map.get(value)
        for item_idx in range(len(value_map)):
            values.append(1.0 if idx == item_idx else 0.0)
    return values


def build_feature_spec(rows: Sequence[Dict[str, str]], numeric_columns: Optional[Sequence[str]] = None) -> FeatureSpec:
    numeric = selected_feature_columns(rows, numeric_columns)
    cat_maps = categorical_maps(rows)
    raw = [raw_features(row, numeric, cat_maps) for row in rows]
    if not raw:
        raise ValueError("Cannot build feature spec from zero rows")
    dim = len(raw[0])
    means = []
    stds = []
    for j in range(dim):
        vals = [item[j] for item in raw]
        mean = sum(vals) / float(len(vals))
        var = sum((value - mean) ** 2 for value in vals) / float(max(1, len(vals) - 1))
        std = math.sqrt(var)
        means.append(mean)
        stds.append(std if std > 1e-8 else 1.0)
    return FeatureSpec(numeric_columns=numeric, categorical_value_maps=cat_maps, means=means, stds=stds)


def featurize_row(row: Dict[str, str], spec: FeatureSpec) -> List[float]:
    raw = raw_features(row, spec.numeric_columns, spec.categorical_value_maps)
    return [(value - mean) / std for value, mean, std in zip(raw, spec.means, spec.stds)]


class TabularDeletionCompatibilityModel(nn.Module):
    """Small MLP that predicts a favorable deletion-compatibility score."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.10):
        super().__init__()
        layers: List[nn.Module] = []
        dim = int(input_dim)
        for _ in range(max(1, int(num_layers))):
            layers.append(nn.Linear(dim, int(hidden_dim)))
            layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(float(dropout)))
            dim = int(hidden_dim)
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def model_config(input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> Dict[str, object]:
    return {
        "input_dim": int(input_dim),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
    }
