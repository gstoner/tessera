"""Tessera energy-based models namespace.

This module is the entry point for the EBM-series primitive surface
sequenced in `docs/audit/ga_ebm_roadmap.md`. EBM0 (scope lock) ships
the namespace and adapts the archived EBT design into
`docs/spec/EBM_SPEC.md`; EBM1 onwards populates the namespace with the
energy primitive surface, samplers, partition function, and losses.

Scope-locked at EBM0:
    - Revive `examples/archive/advanced/EBT/Tessera_EBT_Package_v1/` as
      the seed for EBM5 (Graph IR dialect). Archived files stay archived;
      the live surface re-derives content from them.
    - Functional state + explicit RNGKey throughout (no mutable runner).
    - Broader namespace `tessera.ebm` (covers RBMs, EBTs, score-matching
      diffusion, geometric-Langevin demo) instead of narrow `tessera.ebt`.

See `docs/audit/ebm_scope_lock.md` for the locked decisions,
`docs/spec/EBM_SPEC.md` for the normative specification, and
`docs/audit/ga_ebm_roadmap.md` for the full sprint sequence.
"""

from tessera.ebm.energy import (
    decode_init,
    ebt_tiny,
    ebt_tiny_dispatched_on_gpu,
    ebt_tiny_last_route,
    energy,
    energy_quadratic,
    inner_step,
    langevin_step,
    refinement,
    self_verify,
)
from tessera.ebm.geo_sampling import (
    bivector_langevin_sample,
    bivector_langevin_step,
    sphere_langevin_sample,
    sphere_langevin_step,
    vmf_kappa_mle,
)
from tessera.ebm.partition import (
    partition_function,
    partition_function_ais,
    partition_function_exact,
    partition_function_monte_carlo,
)

__version__ = "0.0.0-ebm7"

__all__ = [
    # EBM1
    "decode_init",
    "energy",
    "ebt_tiny",
    "energy_quadratic",
    "inner_step",
    "langevin_step",
    "refinement",
    "self_verify",
    # EBM3
    "partition_function",
    "partition_function_ais",
    "partition_function_exact",
    "partition_function_monte_carlo",
    # EBM7 — manifold-aware integrators
    "bivector_langevin_sample",
    "bivector_langevin_step",
    "sphere_langevin_sample",
    "sphere_langevin_step",
    "vmf_kappa_mle",
    "__version__",
]
