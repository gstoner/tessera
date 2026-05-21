"""Visual-complex-core — cross-lane (GA × EBM) library benchmark.

Composes the Clifford-core and Energy-core surfaces in one flow:
multivector state evolved by an annealed-Langevin chain whose energy
is a Clifford-norm objective.  Matches the M7 visual-complex milestone
shape (see ``docs/status/visual_complex_milestone.md``) but stays
domain-neutral.
"""
from .core import (
    VisualComplexCoreBenchmark,
    VisualComplexCoreConfig,
    VisualComplexCoreModel,
    VisualComplexCoreResult,
    clifford_energy,
    composition_oracle,
)

__all__ = [
    "VisualComplexCoreBenchmark",
    "VisualComplexCoreConfig",
    "VisualComplexCoreModel",
    "VisualComplexCoreResult",
    "clifford_energy",
    "composition_oracle",
]
