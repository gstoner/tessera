"""GA6 — Parallel autodiff registry for multivector ops.

Per the GA scope lock (Q3, 2026-05-17), multivector autodiff lives in
a parallel namespace `tessera.autodiff.geometric.*`. The existing
241+236 tensor VJP/JVP entries in `tessera.autodiff.vjp._VJPS` /
`tessera.autodiff.jvp._JVPS` are untouched.

Surface shipped in GA6:

    _VJPS_GEO, _JVPS_GEO        registries (op_name -> callable)
    register_vjp_geo / get_vjp_geo
    register_jvp_geo / get_jvp_geo
    check_grad_geo              central-difference verification utility
    tape_geo                    context manager (multivector tape)
    multivector_grad            scalar-loss gradient helper

The VJP / JVP implementations themselves live in
`tessera.autodiff.geometric.{vjp, jvp}`; importing this package
auto-registers every op.
"""

from __future__ import annotations

from tessera.autodiff.geometric.check_grad import (
    check_grad_geo,
)
from tessera.autodiff.geometric.registry import (
    _JVPS_GEO,
    _VJPS_GEO,
    get_jvp_geo,
    get_vjp_geo,
    register_jvp_geo,
    register_vjp_geo,
)
from tessera.autodiff.geometric.tape import (
    GeometricTape,
    multivector_grad,
    tape_geo,
)

# Importing vjp / jvp registers all the GA3 + GA5 entries.
from tessera.autodiff.geometric import jvp as _jvp_module  # noqa: F401
from tessera.autodiff.geometric import vjp as _vjp_module  # noqa: F401


__all__ = [
    "GeometricTape",
    "_JVPS_GEO",
    "_VJPS_GEO",
    "check_grad_geo",
    "get_jvp_geo",
    "get_vjp_geo",
    "multivector_grad",
    "register_jvp_geo",
    "register_vjp_geo",
    "tape_geo",
]
