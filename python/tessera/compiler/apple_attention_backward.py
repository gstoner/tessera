"""APPLE-ATTN-BWD-1 route and workspace policy.

The policy is deliberately separate from the Metal implementation. It lets
portable tests reject impossible determinism/workspace requests before loading
the Apple runtime and records each candidate's execution contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import prod
from pathlib import Path
from typing import Sequence

from .apple_route_selector import AppleRouteContext, production_route_for


SERIAL_RECOMPUTE = "serial_recompute"
ATOMIC = "atomic"
SPLIT_REDUCED = "split_reduced"
ROUTES = (SERIAL_RECOMPUTE, ATOMIC, SPLIT_REDUCED)
ROUTE_IDS = {name: index for index, name in enumerate(ROUTES)}


def selector_shape_key(*, outer: int, q_heads: int, kv_heads: int, sq: int,
                       sk: int, dim: int, causal: bool, window: int,
                       bias: bool, softcap: float) -> str:
    cap = str(float(softcap)).replace(".", "p")
    return (f"b{outer}_hq{q_heads}_hkv{kv_heads}_sq{sq}_sk{sk}_d{dim}_"
            f"c{int(causal)}_w{window}_bias{int(bias)}_sc{cap}")


def resolve_variant_route(*, outer: int, q_heads: int, kv_heads: int,
                          sq: int, sk: int, dim: int, causal: bool,
                          window: int = 0, bias: bool = False,
                          softcap: float = 0.0, dtype: str = "f32",
                          route: str = "auto", deterministic: bool = False,
                          workspace_limit_bytes: int | None = None,
                          device: str | None = None,
                          timing_domain: str = "end_to_end",
                          ledger_path: str | Path | None = None,
                          context: AppleRouteContext | None = None,
                          now: datetime | None = None,
                          require_implemented: bool = True
                          ) -> BackwardRoutePolicy:
    """Resolve one complete MHA/GQA/MQA invocation against retained evidence."""
    if (outer <= 0 or q_heads <= 0 or kv_heads <= 0 or sq <= 0 or sk <= 0
            or dim <= 0 or q_heads % kv_heads != 0 or window < 0
            or softcap < 0.0):
        raise AppleAttentionBackwardPolicyError(
            "invalid Apple attention-backward variant geometry")
    key = selector_shape_key(
        outer=outer, q_heads=q_heads, kv_heads=kv_heads, sq=sq, sk=sk,
        dim=dim, causal=causal, window=window, bias=bias, softcap=softcap)
    kv_shape = (outer * kv_heads, sk, dim)
    return resolve_route(
        kv_shape, kv_shape, route=route, deterministic=deterministic,
        workspace_limit_bytes=workspace_limit_bytes,
        require_implemented=require_implemented, selector_key=key,
        dtype=dtype, device=device, timing_domain=timing_domain,
        ledger_path=ledger_path, context=context, now=now)


class AppleAttentionBackwardPolicyError(ValueError):
    """A route, determinism, or workspace request is invalid."""


class AppleAttentionBackwardUnavailable(RuntimeError):
    """A valid candidate has no native Apple implementation yet."""


@dataclass(frozen=True)
class BackwardRoutePolicy:
    route: str
    deterministic: bool
    workspace_bytes: int
    accumulation_dtype: str = "f32"
    implemented: bool = False


def _elements(shape: Sequence[int], name: str) -> int:
    dims = tuple(int(dim) for dim in shape)
    if not dims or any(dim <= 0 for dim in dims):
        raise AppleAttentionBackwardPolicyError(
            f"{name} shape must contain only positive dimensions")
    return prod(dims)


def workspace_bytes(k_shape: Sequence[int], v_shape: Sequence[int], *,
                    route: str) -> int:
    """Return extra f32 workspace, excluding final dK/dV outputs.

    The two-way split route stores one additional dK+dV partial; the
    output buffers hold the other partial before fixed-order reduction.
    """
    if route not in ROUTES:
        raise AppleAttentionBackwardPolicyError(
            f"unknown Apple attention-backward route {route!r}")
    if route != SPLIT_REDUCED:
        return 0
    return 4 * (_elements(k_shape, "K") + _elements(v_shape, "V"))


def resolve_route(k_shape: Sequence[int], v_shape: Sequence[int], *,
                  route: str = "auto", deterministic: bool = False,
                  workspace_limit_bytes: int | None = None,
                  require_implemented: bool = True,
                  selector_key: str | None = None, dtype: str = "f32",
                  device: str | None = None,
                  timing_domain: str = "end_to_end",
                  ledger_path: str | Path | None = None,
                  context: AppleRouteContext | None = None,
                  now: datetime | None = None) -> BackwardRoutePolicy:
    if workspace_limit_bytes is not None and workspace_limit_bytes < 0:
        raise AppleAttentionBackwardPolicyError(
            "workspace_limit_bytes must be non-negative")
    automatic = route == "auto"
    if automatic:
        route = (production_route_for(
            op="flash_attn_bwd", shape=selector_key, dtype=dtype,
            incumbent_route=SERIAL_RECOMPUTE, device=device,
            timing_domain=timing_domain, ledger_path=ledger_path,
            context=context, now=now) if selector_key
                 else SERIAL_RECOMPUTE)
    if route not in ROUTES:
        raise AppleAttentionBackwardPolicyError(
            f"unknown Apple attention-backward route {route!r}")
    if deterministic and route == ATOMIC:
        if automatic:
            route = SERIAL_RECOMPUTE
        else:
            raise AppleAttentionBackwardPolicyError(
                "deterministic attention backward cannot use the atomic route")
    required = workspace_bytes(k_shape, v_shape, route=route)
    if workspace_limit_bytes is not None and required > workspace_limit_bytes:
        if automatic:
            route = SERIAL_RECOMPUTE
            required = 0
        else:
            raise AppleAttentionBackwardPolicyError(
                f"{route} requires {required} workspace bytes, exceeding limit "
                f"{workspace_limit_bytes}")
    implemented = route in ROUTES
    policy = BackwardRoutePolicy(
        route=route,
        deterministic=(route != ATOMIC),
        workspace_bytes=required,
        implemented=implemented,
    )
    if require_implemented and not implemented:
        raise AppleAttentionBackwardUnavailable(
            f"Apple attention-backward route {route!r} is policy-defined but "
            "has no native Metal implementation")
    return policy
