#!/usr/bin/env python3
"""Smoke test BioPrior v1 segment proposal and scoring on mock features."""

import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from bioprior.prior_types import (  # noqa: E402
    DISORDER_LIKE,
    FUNCTIONAL_CORE_RISK,
    LINKER_COMPRESSIBLE,
    STRUCTURAL_CORE_RISK,
    SURFACE_FLEXIBLE_LOOP,
    TERMINAL_TAIL,
)
from bioprior.scoring import compute_bioprior_score  # noqa: E402
from bioprior.segment_proposal import (  # noqa: E402
    propose_bioprior_segments,
    propose_disorder_like_segments,
    propose_hard_negative_segments,
    propose_linker_compressible_segments,
    propose_surface_flexible_loop_segments,
    propose_terminal_tail_segments,
)
from bioprior.utils import load_bioprior_config  # noqa: E402


def mock_residue_features(length=60):
    rows = []
    for pos in range(length):
        rows.append(
            {
                "position": pos,
                "aa": "A",
                "is_motif": False,
                "is_motif_shadow_8A": False,
                "distance_to_motif": 25.0,
                "contact_density_8A": 8,
                "motif_contact_count_8A": 0,
                "plddt": 92.0,
            }
        )

    for pos in range(5, 11):
        rows[pos]["plddt"] = 45.0
        rows[pos]["contact_density_8A"] = 2

    for pos in range(15, 18):
        rows[pos]["is_motif"] = True
        rows[pos]["distance_to_motif"] = 0.0

    for pos in range(19, 23):
        rows[pos]["is_motif_shadow_8A"] = True
        rows[pos]["distance_to_motif"] = 5.0
        rows[pos]["motif_contact_count_8A"] = 1

    for pos in range(25, 31):
        rows[pos]["contact_density_8A"] = 1
        rows[pos]["distance_to_motif"] = 20.0
        rows[pos]["plddt"] = 82.0

    for pos in range(35, 41):
        rows[pos]["contact_density_8A"] = 1
        rows[pos]["distance_to_motif"] = 22.0
        rows[pos]["plddt"] = 88.0

    for pos in range(45, 51):
        rows[pos]["contact_density_8A"] = 18
        rows[pos]["plddt"] = 98.0

    rows[52]["is_disulfide"] = True
    return rows


def mock_coords(length=60):
    coords = [(float(pos) * 3.8, 0.0, 0.0) for pos in range(length)]
    coords[24] = (0.0, 0.0, 0.0)
    coords[31] = (5.0, 0.0, 0.0)
    coords[34] = (1.0, 0.0, 0.0)
    coords[41] = (6.0, 0.0, 0.0)
    return coords


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    config = load_bioprior_config()
    features = mock_residue_features()
    coords = mock_coords()

    terminal = propose_terminal_tail_segments(len(features), config)
    surface = propose_surface_flexible_loop_segments(features, coords, config)
    disorder = propose_disorder_like_segments(features, config)
    linker = propose_linker_compressible_segments(features, coords, config)
    negatives = propose_hard_negative_segments(features, config)
    all_segments = propose_bioprior_segments(features, coords, config)

    require(any(seg["proposal_source"] == TERMINAL_TAIL for seg in terminal), "missing terminal proposal")
    require(any(seg["proposal_source"] == SURFACE_FLEXIBLE_LOOP for seg in surface), "missing surface flexible proposal")
    require(any(seg["proposal_source"] == DISORDER_LIKE for seg in disorder), "missing disorder proposal")
    require(any(seg["proposal_source"] == LINKER_COMPRESSIBLE for seg in linker), "missing linker proposal")
    require(any(seg["proposal_source"] == FUNCTIONAL_CORE_RISK for seg in negatives), "missing functional hard negative")
    require(any(seg["proposal_source"] == STRUCTURAL_CORE_RISK for seg in negatives), "missing structural hard negative")

    favorable = {"seg_start": 25, "seg_end": 30}
    functional_negative = {"seg_start": 14, "seg_end": 18}
    favorable_score = compute_bioprior_score(favorable, features, coords, config)
    negative_score = compute_bioprior_score(functional_negative, features, coords, config)

    require(not favorable_score["hard_reject"], "favorable segment should not hard reject")
    require(favorable_score["closure"]["closure_friendly"], "favorable segment should be closure-friendly")
    require(negative_score["hard_reject"], "functional negative should hard reject")
    require(negative_score["score"] < favorable_score["score"], "hard negative should score lower")

    print("terminal_segments={}".format(len(terminal)))
    print("surface_flexible_segments={}".format(len(surface)))
    print("disorder_like_segments={}".format(len(disorder)))
    print("linker_segments={}".format(len(linker)))
    print("hard_negative_segments={}".format(len(negatives)))
    print("all_bioprior_segments={}".format(len(all_segments)))
    print("favorable_score={:.4f}".format(favorable_score["score"]))
    print("functional_negative_score={:.4f}".format(negative_score["score"]))
    print("ALL_PASS")


if __name__ == "__main__":
    main()
