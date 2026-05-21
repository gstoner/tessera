"""Energy-core — generic library benchmark for the EBM lane.

Sister surface to ``benchmarks/clifford_core`` and ``benchmarks/grid_ai_core``.
Domain-neutral; exercises the EBM primitives the Apple-GPU MSL kernels back:
quadratic energy, log-partition (stable logsumexp), Langevin step, and a
linear annealing schedule.
"""
from .core import (
    EnergyCoreBenchmark,
    EnergyCoreConfig,
    EnergyCoreModel,
    EnergyCoreResult,
    annealing_schedule,
    energy_grid_oracle,
    langevin_chain_oracle,
    quadratic_energy,
)

__all__ = [
    "EnergyCoreBenchmark",
    "EnergyCoreConfig",
    "EnergyCoreModel",
    "EnergyCoreResult",
    "annealing_schedule",
    "energy_grid_oracle",
    "langevin_chain_oracle",
    "quadratic_energy",
]
