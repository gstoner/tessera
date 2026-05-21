"""Clifford-core — generic library benchmark for the GA / Clifford lane.

Sister surface to ``benchmarks/grid_ai_core``: domain-neutral, exercises the
compiler-visible substrate the GA lane provides (multivector tiling, rotor
sampling, sandwich-product chain, grade projection, norm) without baking in
any specific application.
"""
from .core import (
    CliffordCoreBenchmark,
    CliffordCoreConfig,
    CliffordCoreModel,
    CliffordCoreResult,
    RotorSampler,
    multivector_oracle_chain,
    tile_multivectors,
)

__all__ = [
    "CliffordCoreBenchmark",
    "CliffordCoreConfig",
    "CliffordCoreModel",
    "CliffordCoreResult",
    "RotorSampler",
    "multivector_oracle_chain",
    "tile_multivectors",
]
