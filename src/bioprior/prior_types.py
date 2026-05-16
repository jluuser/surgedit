"""Canonical BioPrior v1 prior type constants."""

TERMINAL_TAIL = "terminal_tail_prior"
SURFACE_FLEXIBLE_LOOP = "surface_flexible_loop_prior"
DISORDER_LIKE = "disorder_like_prior"
LINKER_COMPRESSIBLE = "linker_compressibility_prior"
FUNCTIONAL_CORE_RISK = "functional_core_protection_prior"
STRUCTURAL_CORE_RISK = "structural_core_protection_prior"
MOTIF_SHADOW_RISK = "motif_neighborhood_support_prior"
GEOMETRIC_CLOSURE = "geometric_closure_prior"

DELETION_FAVORABLE_PRIORS = [
    TERMINAL_TAIL,
    SURFACE_FLEXIBLE_LOOP,
    DISORDER_LIKE,
    LINKER_COMPRESSIBLE,
]

DELETION_RISK_PRIORS = [
    FUNCTIONAL_CORE_RISK,
    STRUCTURAL_CORE_RISK,
    MOTIF_SHADOW_RISK,
    GEOMETRIC_CLOSURE,
]
