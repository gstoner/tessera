"""Game theory — coalition-lattice operator family (G1, GAME_THEORY_PLAN.md).

The first slice of the plan's §5 phasing: the subset/superset zeta and Möbius
butterflies, the coalition marginal, and the weight-parameterized semivalue —
each a ``custom_primitive`` (tape-aware, catalog-visible), the linear ones with
declared transposes so VJP *and* JVP fall out of linearity (`custom.py`).

Index convention (G0, fail-closed): the trailing axis is the coalition lattice,
little-endian, bit ``i`` ↔ player ``i``; its extent must be a power of two and
``n`` is always derived from it, never defaulted.

Numerics (plan §6, measured): the lattice family computes and RETURNS float64.
For nonnegative games the zeta grows like ``2^n`` with no cancellation, so an
fp32-materialized intermediate is pure noise by ``n ≈ 20`` — fp64
materialization is a correctness requirement here, not a tuning knob.
"""

from .combinatorial import mex
from .lattice import (
    coalition_marginal,
    lattice_players,
    subset_mobius,
    subset_zeta,
    superset_mobius,
    superset_zeta,
)
from .values import (
    banzhaf_value,
    banzhaf_weights,
    boltzmann_value,
    coalition_excess,
    semivalue,
    shapley_value,
    shapley_weights,
)

__all__ = [
    "banzhaf_value",
    "banzhaf_weights",
    "boltzmann_value",
    "coalition_excess",
    "coalition_marginal",
    "lattice_players",
    "mex",
    "semivalue",
    "shapley_value",
    "shapley_weights",
    "subset_mobius",
    "subset_zeta",
    "superset_mobius",
    "superset_zeta",
]
