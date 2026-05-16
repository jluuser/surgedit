#!/usr/bin/env python3
"""Smoke tests for certified frontier planning.

The test uses synthetic candidates so it does not depend on the large BioPrior
feature CSVs.  It checks the core algorithmic contract:

- closure scores are interpreted in risk space,
- conformal calibration is evidence-conditional,
- selected plans contain non-overlapping intervals,
- strict profiles can abstain while default profiles can return a plan.
"""

import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from bioprior.certified_frontier import (  # noqa: E402
    CertifiedFrontierPlanner,
    RiskCalibrator,
    SelectionProfile,
    candidates_from_rows,
    closure_risk_from_row,
)


def row(accession, start, end, utility, structural, closure_score=1.0, protected=0, evidence="annotation_structure"):
    length = end - start + 1
    closure_friendly = closure_score >= 0.75
    return {
        "accession": accession,
        "protein_length": "100",
        "seg_start": str(start),
        "seg_end": str(end),
        "seg_len": str(length),
        "stage1_utility_score": str(utility),
        "final_bioprior_score": "1.0",
        "n_protected_overlap": str(protected),
        "protected_overlap_fraction": "1.0" if protected else "0.0",
        "n_shadow_overlap": "0",
        "shadow_overlap_fraction": "0.0",
        "functional_core_risk_score": "1.0" if protected else "0.0",
        "structural_core_risk_score": str(structural),
        "motif_shadow_risk_score": "0.0",
        "geometric_closure_score": str(closure_score),
        "closure_type": "internal_friendly" if closure_friendly else "internal_unfriendly",
        "closure_friendly_8A": str(closure_friendly),
        "hard_reject": "True" if protected else "False",
        "evidence_level": evidence,
        "evidence_confidence": "1.0" if evidence == "annotation_structure" else "0.55",
        "missing_annotation_penalty": "0.0" if evidence == "annotation_structure" else "0.2",
        "missing_structure_penalty": "0.0",
        "low_plddt_uncertainty_penalty": "0.0",
    }


def assert_non_overlapping(plan):
    selected = sorted((candidate.start, candidate.end) for candidate in plan.selected)
    for (_, prev_end), (next_start, _) in zip(selected, selected[1:]):
        assert prev_end < next_start, selected


def main():
    calibration_rows = [
        row("CAL1", 0, 9, 0.2, 0.10),
        row("CAL1", 20, 29, 0.2, 0.20),
        row("CAL2", 0, 9, 0.2, 0.20, evidence="sequence_only"),
        row("CAL2", 20, 29, 0.2, 0.30, evidence="sequence_only"),
    ]
    assert closure_risk_from_row(row("X", 0, 9, 0.1, 0.1, closure_score=1.0)) == 0.0
    assert closure_risk_from_row(row("X", 0, 9, 0.1, 0.1, closure_score=0.2)) > 0.5

    calibrator = RiskCalibrator.from_rows(calibration_rows, alpha=0.2)
    assert "annotation_structure" in calibrator.group_offsets
    assert "sequence_only" in calibrator.group_offsets

    planning_rows = [
        row("P1", 0, 9, 0.8, 0.10),
        row("P1", 10, 19, 0.6, 0.15),
        row("P1", 15, 30, 1.0, 0.12),  # overlaps the previous interval
        row("P1", 40, 49, 0.2, 0.70),
        row("P1", 60, 69, 1.0, 0.05, protected=2),
    ]
    candidates = candidates_from_rows(planning_rows, calibrator=calibrator)
    planner = CertifiedFrontierPlanner(
        max_delete_ratio=0.30,
        max_plan_risk=0.35,
        frontier_size=32,
        max_candidates=16,
    )
    frontier = planner.build_frontier(candidates)
    assert frontier, "frontier should not be empty"

    default = SelectionProfile("default", delete_reward=0.5, risk_weight=1.0, length_penalty=0.05, max_plan_risk=0.35)
    chosen, status = planner.select(frontier, default)
    assert status == "certified"
    assert chosen is not None
    assert chosen.protected_overlap == 0
    assert_non_overlapping(chosen)

    strict = SelectionProfile("strict", delete_reward=0.1, risk_weight=5.0, length_penalty=0.5, max_plan_risk=0.05)
    strict_plan, strict_status = planner.select(frontier, strict)
    assert strict_plan is None
    assert strict_status == "abstain_no_certified_plan"

    print("CERTIFIED_FRONTIER_SMOKE_PASS")


if __name__ == "__main__":
    main()
