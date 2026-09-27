"""Decision #12 benchmark-row route provenance (amended 2026-08-30).

A latency without its route is not comparable: the compiled
Graph -> Schedule -> Tile -> Target route, a Tier-3 delegate and a bootstrap
packager compete for the same ``(op, shape, dtype, target)`` under Decision #28,
so two rows that look identical can come from different compilers.

**The route is derived, never declared.** Every helper here reads it from the
artifact that actually executed -- the runtime artifact's compiler-stamped
``compiler_path``, or a native launch descriptor's ``provenance`` (the source
``benchmarks/e2e_spine/record_sm120_packet.py`` already uses). A row whose
latency did not come from an executed artifact (an analytical roofline, a mock
collective) has no route to derive, so it records :data:`UNKNOWN_ROUTE` and
says why in ``route_source`` -- it is never guessed from a benchmark's own
label. :func:`stable_row` only accepts a :class:`RouteProvenance` produced by
one of the helpers below: each helper marks what it returns as derived, and a
``RouteProvenance`` a caller constructs directly is refused -- so neither a
typed string nor a hand-built object can stand in for a derived route.

This module imports nothing from ``tessera`` so the analytical benchmarks stay
runnable without the package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

#: The stable Decision #12 fields. Never removed, renamed or repurposed.
STABLE_ROW_FIELDS: tuple[str, ...] = (
    "backend", "op", "shape", "dtype", "latency_ms", "tflops",
    "memory_bw_gb_s", "device", "tessera_version",
)

#: The additive Decision #12 amendment fields.
PROVENANCE_ROW_FIELDS: tuple[str, ...] = ("route", "route_source", "timing_source")

#: Recorded when no executed artifact exists to derive a route from.
UNKNOWN_ROUTE = "unknown"

#: Timing sources a row may name. ``analytical_model`` marks a latency that no
#: clock measured; ``host_wall_clock_first_call`` a single wall-clock interval
#: around a JIT function's first call, which includes compilation and is not a
#: steady-state latency; ``unknown`` is what a reader reports for a row that
#: predates the field.
TIMING_SOURCES: frozenset[str] = frozenset({
    "host_wall_clock", "host_wall_clock_first_call", "device_clock",
    "cuda_event", "hip_event", "metal4_timestamp_heap", "analytical_model",
    "unknown",
})

#: Marks a RouteProvenance built by a derivation helper in this module.
_DERIVED = object()


@dataclass(frozen=True)
class RouteProvenance:
    """Which lowering produced a number, and where that fact was read from."""

    route: str
    source: str
    #: Set only by this module's helpers; not part of equality or repr.
    _origin: object = field(default=None, repr=False, compare=False)

    @property
    def derived(self) -> bool:
        return self._origin is _DERIVED

    @property
    def known(self) -> bool:
        return self.route != UNKNOWN_ROUTE

    def as_fields(self) -> dict[str, str]:
        return {"route": self.route, "route_source": self.source}


def route_unavailable(reason: str) -> RouteProvenance:
    """No executed artifact: the route is unknown, and ``reason`` says why."""
    if not reason:
        raise ValueError("an unknown route must say why it is unknown")
    return _derived(UNKNOWN_ROUTE, f"unavailable: {reason}")


def _derived(route: str, source: str) -> RouteProvenance:
    return RouteProvenance(route, source, _DERIVED)


def route_from_runtime_artifact(artifact: Any) -> RouteProvenance:
    """Route from a ``tessera.runtime.RuntimeArtifact`` (or its metadata).

    Reads the ``compiler_path`` the JIT stamps when it builds the artifact
    (``jit_cpu_numpy``, ``rocm_compiled``, ``nvidia_mma``, ``apple_value_target_ir``
    ...). An artifact without one yields :data:`UNKNOWN_ROUTE`.
    """
    metadata = artifact if isinstance(artifact, Mapping) else getattr(artifact, "metadata", None)
    path = (metadata or {}).get("compiler_path")
    if not path:
        return route_unavailable("runtime artifact carries no compiler_path")
    return _derived(str(path), "runtime_artifact.metadata.compiler_path")


def route_from_descriptor(descriptor: Any) -> RouteProvenance:
    """Route from a native launch descriptor's ``provenance``.

    Same precedence as ``record_sm120_packet.py``: the scheduled route name
    (``schedule``) when present, else ``route``.
    """
    provenance = getattr(descriptor, "provenance", None) or {}
    for key in ("schedule", "route"):
        value = provenance.get(key)
        if value:
            return _derived(str(value), f"descriptor.provenance.{key}")
    return route_unavailable("launch descriptor provenance names no route")


def stable_row(
    *,
    backend: str,
    op: str,
    shape: Any,
    dtype: str,
    latency_ms: float,
    tflops: float | None,
    memory_bw_gb_s: float | None,
    device: str,
    tessera_version: str,
    route: RouteProvenance,
    timing_source: str,
    **extra: Any,
) -> dict[str, Any]:
    """One Decision #12 benchmark row: the stable fields plus route provenance.

    ``route`` must be a :class:`RouteProvenance` from one of the derivation
    helpers above -- a bare string is refused, because a label the caller
    typed is exactly what the amendment says a route is not.
    """
    if not isinstance(route, RouteProvenance) or not route.derived:
        raise TypeError(
            "route must be a RouteProvenance derived from the executed artifact "
            "(route_from_runtime_artifact / route_from_descriptor / "
            "route_unavailable), not a typed label")
    if timing_source not in TIMING_SOURCES:
        raise ValueError(f"unknown timing_source {timing_source!r}; one of {sorted(TIMING_SOURCES)}")
    clash = set(extra) & set(STABLE_ROW_FIELDS + PROVENANCE_ROW_FIELDS)
    if clash:
        raise ValueError(f"extra fields may not override Decision #12 fields: {sorted(clash)}")
    return {
        "backend": backend,
        "op": op,
        "shape": shape,
        "dtype": dtype,
        "latency_ms": latency_ms,
        "tflops": tflops,
        "memory_bw_gb_s": memory_bw_gb_s,
        "device": device,
        "tessera_version": tessera_version,
        **route.as_fields(),
        "timing_source": timing_source,
        **extra,
    }


__all__ = [
    "PROVENANCE_ROW_FIELDS",
    "RouteProvenance",
    "STABLE_ROW_FIELDS",
    "TIMING_SOURCES",
    "UNKNOWN_ROUTE",
    "route_from_descriptor",
    "route_from_runtime_artifact",
    "route_unavailable",
    "stable_row",
]
