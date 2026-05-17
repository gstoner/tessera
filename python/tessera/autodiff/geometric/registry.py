"""GA6 registry plumbing — parallel to `tessera.autodiff.vjp`.

Two dicts mapping op name -> callable: ``_VJPS_GEO`` for reverse-mode
VJPs, ``_JVPS_GEO`` for forward-mode JVPs.

Adding a new geometric op = register one VJP function and one JVP
function. The conventions:

    vjp_<op>(dout: Multivector, *primals, **kwargs)
        -> tuple[Multivector | None, ...]
            One entry per `primals` arg; ``None`` for non-multivector
            args (e.g. scalars, ints, grade selectors).

    jvp_<op>(tangents: tuple[Multivector | None, ...], primals: tuple, **kwargs)
        -> Multivector
            Forward pushforward at ``primals`` with input tangents
            ``tangents``. ``None`` in ``tangents`` means "this arg is
            non-differentiable / has zero tangent here".
"""

from __future__ import annotations

from typing import Callable, Optional


_VJPS_GEO: dict[str, Callable] = {}
_JVPS_GEO: dict[str, Callable] = {}


def register_vjp_geo(name: str, fn: Optional[Callable] = None) -> Callable:
    """Register a VJP for the geometric op ``name``.

    Usable as a decorator (``@register_vjp_geo("op")``) or as a function
    call (``register_vjp_geo("op", fn)``).
    """
    if fn is None:
        def decorator(f: Callable) -> Callable:
            _VJPS_GEO[name] = f
            return f
        return decorator
    _VJPS_GEO[name] = fn
    return fn


def register_jvp_geo(name: str, fn: Optional[Callable] = None) -> Callable:
    if fn is None:
        def decorator(f: Callable) -> Callable:
            _JVPS_GEO[name] = f
            return f
        return decorator
    _JVPS_GEO[name] = fn
    return fn


def get_vjp_geo(name: str) -> Optional[Callable]:
    return _VJPS_GEO.get(name)


def get_jvp_geo(name: str) -> Optional[Callable]:
    return _JVPS_GEO.get(name)


__all__ = [
    "_JVPS_GEO",
    "_VJPS_GEO",
    "get_jvp_geo",
    "get_vjp_geo",
    "register_jvp_geo",
    "register_vjp_geo",
]
