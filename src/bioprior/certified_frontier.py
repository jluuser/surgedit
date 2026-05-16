"""Certified Pareto-frontier planning for protein deletion.

This module contains the algorithmic core for BioDel-style deletion planning.
It deliberately keeps the inputs simple: each deletion candidate is an interval
with utility, biological score, evidence confidence, and risk estimates.  The
planner then constructs a non-overlapping deletion-plan frontier and selects a
certified operating point from that frontier.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _as_float(value, default=0.0):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default=0):
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_bool(value):
    return str(value).strip().lower() in ("true", "1", "yes", "y")


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def conformal_quantile(scores, alpha):
    """Return the split-conformal upper quantile for nonconformity scores.

    The rank follows the usual finite-sample conformal choice
    ``ceil((n + 1) * (1 - alpha))`` with a conservative cap at the largest
    observed score when the requested rank exceeds the calibration set.
    """

    values = sorted(float(score) for score in scores)
    if not values:
        return 0.0
    rank = int(math.ceil((len(values) + 1) * (1.0 - float(alpha))))
    rank = max(1, min(rank, len(values)))
    return values[rank - 1]


def closure_risk_from_row(row):
    """Return deletion-boundary closure risk.

    Existing BioPrior rows store ``geometric_closure_score`` as a favorable
    score, where high means easy geometric closure.  The certified planner works
    in risk space, so the value must be inverted or derived from the explicit
    closure flags.
    """

    closure_type = row.get("closure_type", "")
    if closure_type == "terminal":
        return 0.0
    if closure_type and _as_bool(row.get("closure_friendly_8A")):
        return 0.0
    if closure_type:
        score = _as_float(row.get("geometric_closure_score"), None)
        if score is not None:
            return _clamp(1.0 - score)
        return 1.0
    score = _as_float(row.get("geometric_closure_score"), None)
    if score is None:
        return 0.0
    return _clamp(1.0 - score)


def component_risks_from_row(row):
    """Build conservative component risks from available annotations."""

    protected_count = _as_int(row.get("n_protected_overlap"), 0)
    protected_fraction = _as_float(row.get("protected_overlap_fraction"), 0.0)
    functional = max(
        _as_float(row.get("functional_core_risk_score"), 0.0),
        1.0 if protected_count > 0 else protected_fraction,
    )
    motif = max(
        _as_float(row.get("motif_shadow_risk_score"), 0.0),
        _as_float(row.get("shadow_overlap_fraction"), 0.0),
    )
    structural = _as_float(row.get("structural_core_risk_score"), 0.0)
    hard_reject = 1.0 if _as_bool(row.get("hard_reject")) else 0.0
    return {
        "functional": _clamp(functional),
        "motif_shadow": _clamp(motif),
        "closure": _clamp(closure_risk_from_row(row)),
        "structural": _clamp(structural),
        "hard_reject": _clamp(hard_reject),
    }


def risk_proxy_from_row(row):
    """Build a conservative observed-risk proxy from component risks."""

    risks = component_risks_from_row(row)
    return _clamp(max(risks.values()))


def predicted_risk_from_row(row):
    """Build an evidence-adjusted predicted risk used before calibration."""

    risks = component_risks_from_row(row)
    point = max(
        risks["functional"],
        risks["motif_shadow"],
        risks["closure"],
        risks["structural"],
        risks["hard_reject"],
    )
    uncertainty = 0.0
    uncertainty += _as_float(row.get("missing_annotation_penalty"), 0.0)
    uncertainty += _as_float(row.get("missing_structure_penalty"), 0.0)
    uncertainty += _as_float(row.get("low_plddt_uncertainty_penalty"), 0.0)
    return _clamp(point + uncertainty)


@dataclass(frozen=True)
class SegmentCandidate:
    accession: str
    start: int
    end: int
    length: int
    protein_length: int
    utility: float = 0.0
    bioprior: float = 0.0
    risk_point: float = 0.0
    risk_upper: float = 0.0
    evidence_confidence: float = 1.0
    evidence_level: str = "unknown"
    protected_overlap: int = 0
    shadow_overlap: int = 0
    closure_unfriendly_len: int = 0
    structural_risk_units: float = 0.0
    source_index: int = 0
    row: Dict[str, str] = field(default_factory=dict, compare=False)

    @property
    def key(self):
        return (self.start, self.end, self.source_index)

    @property
    def value_sum(self):
        return self.length * (self.utility + 0.20 * self.bioprior)

    @property
    def risk_upper_units(self):
        return self.length * self.risk_upper

    @property
    def risk_point_units(self):
        return self.length * self.risk_point

    @classmethod
    def from_row(cls, row, source_index=0, calibrator=None):
        start = _as_int(row.get("seg_start"))
        end = _as_int(row.get("seg_end"))
        length = _as_int(row.get("seg_len"), end - start + 1)
        protein_length = _as_int(row.get("protein_length"), 0)
        evidence_level = row.get("evidence_level") or "unknown"
        base_risk_upper = predicted_risk_from_row(row)
        if calibrator is not None:
            risk_upper = calibrator.calibrate(base_risk_upper, evidence_level)
        else:
            risk_upper = base_risk_upper
        closure_type = row.get("closure_type", "")
        closure_bad = closure_type != "terminal" and not _as_bool(row.get("closure_friendly_8A"))
        return cls(
            accession=row.get("accession") or row.get("protein_id") or "",
            start=start,
            end=end,
            length=length,
            protein_length=protein_length,
            utility=_as_float(row.get("stage1_utility_score"), 0.0),
            bioprior=_as_float(row.get("final_bioprior_score"), 0.0),
            risk_point=risk_proxy_from_row(row),
            risk_upper=_clamp(risk_upper),
            evidence_confidence=_clamp(_as_float(row.get("evidence_confidence"), 1.0)),
            evidence_level=evidence_level,
            protected_overlap=_as_int(row.get("n_protected_overlap"), 0),
            shadow_overlap=_as_int(row.get("n_shadow_overlap"), 0),
            closure_unfriendly_len=length if closure_bad else 0,
            structural_risk_units=length * max(0.0, _as_float(row.get("structural_core_risk_score"), 0.0)),
            source_index=int(source_index),
            row=dict(row),
        )


@dataclass(frozen=True)
class PlanState:
    accession: str
    protein_length: int
    selected: Tuple[SegmentCandidate, ...] = ()
    selected_len: int = 0
    value_sum: float = 0.0
    utility_sum: float = 0.0
    bioprior_sum: float = 0.0
    risk_upper_units: float = 0.0
    risk_point_units: float = 0.0
    confidence_units: float = 0.0
    protected_overlap: int = 0
    shadow_overlap: int = 0
    closure_unfriendly_len: int = 0
    structural_risk_units: float = 0.0
    last_end: int = -1

    @property
    def selected_segments(self):
        return len(self.selected)

    @property
    def delete_ratio(self):
        return self.selected_len / float(max(1, self.protein_length))

    @property
    def risk_upper_mean(self):
        return self.risk_upper_units / float(max(1, self.selected_len))

    @property
    def risk_point_mean(self):
        return self.risk_point_units / float(max(1, self.selected_len))

    @property
    def evidence_confidence_mean(self):
        return self.confidence_units / float(max(1, self.selected_len))

    @property
    def shadow_rate(self):
        return self.shadow_overlap / float(max(1, self.selected_len))

    @property
    def closure_rate(self):
        return self.closure_unfriendly_len / float(max(1, self.selected_len))

    @property
    def structural_risk_per_deleted(self):
        return self.structural_risk_units / float(max(1, self.selected_len))

    @property
    def value_norm(self):
        return self.value_sum / float(max(1, self.protein_length))

    def add(self, candidate):
        selected = self.selected + (candidate,)
        return PlanState(
            accession=self.accession,
            protein_length=self.protein_length,
            selected=selected,
            selected_len=self.selected_len + candidate.length,
            value_sum=self.value_sum + candidate.value_sum,
            utility_sum=self.utility_sum + candidate.length * candidate.utility,
            bioprior_sum=self.bioprior_sum + candidate.length * candidate.bioprior,
            risk_upper_units=self.risk_upper_units + candidate.risk_upper_units,
            risk_point_units=self.risk_point_units + candidate.risk_point_units,
            confidence_units=self.confidence_units + candidate.length * candidate.evidence_confidence,
            protected_overlap=self.protected_overlap + candidate.protected_overlap,
            shadow_overlap=self.shadow_overlap + candidate.shadow_overlap,
            closure_unfriendly_len=self.closure_unfriendly_len + candidate.closure_unfriendly_len,
            structural_risk_units=self.structural_risk_units + candidate.structural_risk_units,
            last_end=candidate.end,
        )


@dataclass(frozen=True)
class SelectionProfile:
    name: str
    delete_reward: float
    risk_weight: float
    length_penalty: float
    confidence_penalty: float = 0.5
    max_plan_risk: float = 0.35
    min_evidence_confidence: float = 0.0
    max_shadow_rate: float = 1.0
    max_closure_rate: float = 1.0
    max_structural_risk_per_deleted: float = 1.0


class RiskCalibrator:
    """Evidence-conditional conformal risk-upper calibration."""

    def __init__(self, group_offsets=None, global_offset=0.0):
        self.group_offsets = dict(group_offsets or {})
        self.global_offset = float(global_offset)

    @classmethod
    def from_rows(cls, rows, alpha=0.10, group_key="evidence_level"):
        grouped = {}
        all_scores = []
        for row in rows:
            predicted = predicted_risk_from_row(row)
            observed = risk_proxy_from_row(row)
            score = max(0.0, observed - predicted)
            group = row.get(group_key) or "unknown"
            grouped.setdefault(group, []).append(score)
            all_scores.append(score)
        global_offset = conformal_quantile(all_scores, alpha)
        group_offsets = {
            group: conformal_quantile(scores, alpha)
            for group, scores in grouped.items()
        }
        return cls(group_offsets=group_offsets, global_offset=global_offset)

    def calibrate(self, risk_upper, evidence_level):
        offset = self.group_offsets.get(evidence_level, self.global_offset)
        return _clamp(float(risk_upper) + offset)


class CertifiedFrontierPlanner:
    """Build and select from a certified non-overlapping deletion frontier."""

    def __init__(
        self,
        max_delete_ratio=0.30,
        max_plan_risk=0.35,
        frontier_size=128,
        max_candidates=220,
        risk_bin_width=0.005,
        length_bin_size=5,
        require_no_protected=True,
    ):
        self.max_delete_ratio = float(max_delete_ratio)
        self.max_plan_risk = float(max_plan_risk)
        self.frontier_size = int(frontier_size)
        self.max_candidates = int(max_candidates)
        self.risk_bin_width = float(risk_bin_width)
        self.length_bin_size = int(length_bin_size)
        self.require_no_protected = bool(require_no_protected)

    def _candidate_allowed(self, candidate):
        if self.require_no_protected and candidate.protected_overlap > 0:
            return False
        max_len = int(math.floor(candidate.protein_length * self.max_delete_ratio))
        if candidate.length > max(1, max_len):
            return False
        return True

    def _candidate_rank(self, candidate):
        return (
            candidate.value_sum
            + 0.10 * candidate.length
            - 0.50 * candidate.risk_upper_units
            + 0.05 * candidate.evidence_confidence * candidate.length
        )

    def _signature(self, state):
        risk_bin = int(math.floor(state.risk_upper_mean / max(self.risk_bin_width, 1e-9)))
        structural_bin = int(math.floor(state.structural_risk_per_deleted / max(self.risk_bin_width, 1e-9)))
        length_bin = int(math.floor(state.selected_len / float(max(1, self.length_bin_size))))
        return (
            length_bin,
            risk_bin,
            structural_bin,
            state.protected_overlap,
            state.shadow_overlap,
            state.closure_unfriendly_len,
        )

    def _dominates(self, a, b):
        if a.protected_overlap > b.protected_overlap:
            return False
        if a.selected_len < b.selected_len:
            return False
        if a.value_sum < b.value_sum:
            return False
        if a.risk_upper_units > b.risk_upper_units:
            return False
        if a.shadow_overlap > b.shadow_overlap:
            return False
        if a.closure_unfriendly_len > b.closure_unfriendly_len:
            return False
        if a.structural_risk_units > b.structural_risk_units:
            return False
        return (
            a.protected_overlap < b.protected_overlap
            or a.selected_len > b.selected_len
            or a.value_sum > b.value_sum
            or a.risk_upper_units < b.risk_upper_units
            or a.shadow_overlap < b.shadow_overlap
            or a.closure_unfriendly_len < b.closure_unfriendly_len
            or a.structural_risk_units < b.structural_risk_units
        )

    def _prune(self, states):
        states = [state for state in states if self._within_outer_caps(state)]
        nondominated = []
        for state in sorted(states, key=self._state_rank, reverse=True):
            if any(self._dominates(other, state) for other in nondominated):
                continue
            nondominated = [other for other in nondominated if not self._dominates(state, other)]
            nondominated.append(state)

        best_by_signature = {}
        for state in nondominated:
            signature = self._signature(state)
            current = best_by_signature.get(signature)
            if current is None or self._state_rank(state) > self._state_rank(current):
                best_by_signature[signature] = state
        compact = sorted(best_by_signature.values(), key=self._state_rank, reverse=True)
        return compact[: self.frontier_size]

    def _state_rank(self, state):
        return (
            state.value_norm
            + 0.40 * state.delete_ratio
            - 0.80 * state.risk_upper_mean * state.delete_ratio
            + 0.05 * state.evidence_confidence_mean
        )

    def _within_outer_caps(self, state):
        if state.selected_len > int(math.floor(state.protein_length * self.max_delete_ratio)):
            return False
        if state.risk_upper_mean > self.max_plan_risk and state.selected_len > 0:
            return False
        return True

    def build_frontier(self, candidates):
        candidates = [candidate for candidate in candidates if self._candidate_allowed(candidate)]
        if not candidates:
            return []
        accession = candidates[0].accession
        protein_length = candidates[0].protein_length
        candidates = sorted(candidates, key=self._candidate_rank, reverse=True)[: self.max_candidates]
        candidates = sorted(candidates, key=lambda item: (item.end, item.start, item.source_index))
        states = [PlanState(accession=accession, protein_length=protein_length)]
        for candidate in candidates:
            additions = []
            for state in states:
                if candidate.start <= state.last_end:
                    continue
                added = state.add(candidate)
                if self._within_outer_caps(added):
                    additions.append(added)
            if additions:
                states = self._prune(states + additions)
        frontier = [state for state in states if state.selected_len > 0]
        return self._prune(frontier)

    def score_plan(self, state, profile):
        uncertainty = 1.0 - state.evidence_confidence_mean
        return (
            state.value_norm
            + profile.delete_reward * state.delete_ratio
            - profile.risk_weight * state.risk_upper_mean * state.delete_ratio
            - profile.length_penalty * state.delete_ratio
            - profile.confidence_penalty * uncertainty * state.delete_ratio
        )

    def select(self, frontier, profile):
        feasible = [
            state for state in frontier
            if state.risk_upper_mean <= profile.max_plan_risk
            and state.evidence_confidence_mean >= profile.min_evidence_confidence
            and state.shadow_rate <= profile.max_shadow_rate
            and state.closure_rate <= profile.max_closure_rate
            and state.structural_risk_per_deleted <= profile.max_structural_risk_per_deleted
            and (not self.require_no_protected or state.protected_overlap == 0)
        ]
        if not feasible:
            return None, "abstain_no_certified_plan"
        chosen = max(
            feasible,
            key=lambda state: (
                self.score_plan(state, profile),
                -state.risk_upper_mean,
                state.delete_ratio,
            ),
        )
        return chosen, "certified"


def candidates_from_rows(rows, calibrator=None):
    return [
        SegmentCandidate.from_row(row, source_index=index, calibrator=calibrator)
        for index, row in enumerate(rows)
    ]


def group_rows_by_accession(rows):
    grouped = {}
    for row in rows:
        accession = row.get("accession") or row.get("protein_id") or ""
        grouped.setdefault(accession, []).append(row)
    return grouped


def plan_to_summary_row(state, profile_name, status):
    if state is None:
        return {
            "auto_profile": profile_name,
            "selection_status": status,
        }
    selected_len = float(max(1, state.selected_len))
    return {
        "accession": state.accession,
        "auto_profile": profile_name,
        "selection_status": status,
        "protein_length": state.protein_length,
        "selected_len": state.selected_len,
        "actual_delete_ratio": "{:.6f}".format(state.delete_ratio),
        "selected_segments": state.selected_segments,
        "value_norm": "{:.6f}".format(state.value_norm),
        "mean_risk_upper": "{:.6f}".format(state.risk_upper_mean),
        "mean_risk_point": "{:.6f}".format(state.risk_point_mean),
        "mean_evidence_confidence": "{:.6f}".format(state.evidence_confidence_mean),
        "protected_overlap_residues": state.protected_overlap,
        "shadow_overlap_residues": state.shadow_overlap,
        "shadow_overlap_rate": "{:.6f}".format(state.shadow_overlap / selected_len),
        "closure_unfriendly_len": state.closure_unfriendly_len,
        "closure_unfriendly_rate": "{:.6f}".format(state.closure_unfriendly_len / selected_len),
        "structural_risk_units": "{:.6f}".format(state.structural_risk_units),
        "structural_risk_per_deleted": "{:.6f}".format(state.structural_risk_units / selected_len),
    }
