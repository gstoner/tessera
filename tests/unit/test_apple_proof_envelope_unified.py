"""Apple proof-envelope unification lock
(Apple plan phase D + Test-tree review phase P2-10, 2026-05-20).

Phase D wired ``runtime.py::_apple_gpu_dispatch_matmul_softmax_matmul``
to emit a ``JitBridgeRoute`` via ``jit_bridge.record_driver_route``
on each successful native kernel dispatch.  Phase D's success
condition: a ``CompileReport`` from the generic-tensor canonical
carries a ``proof_routes`` tuple whose entries have **the same
6-field schema** as the GA/EBM canonical's ``proof_routes`` —
distinguishable only by ``context`` (``"driver"`` for the generic
tensor lane, ``"direct"``/``"test"`` for the GA/EBM manifest lane).

This test locks that invariant.  If the canonical or the bridge
stops populating any of the six fields, or the field shapes
diverge, the test fails fast.

Darwin-only.
"""
from __future__ import annotations

import sys
from dataclasses import fields

import pytest

pytestmark = pytest.mark.hardware_apple_gpu


# Fields we expect on every JitBridgeRoute, regardless of lane.
_EXPECTED_FIELDS = {
    "op_name", "target", "status", "symbol",
    "context", "latency_ms", "args_summary",
}


def test_jit_bridge_route_schema_is_stable() -> None:
    """The JitBridgeRoute dataclass carries the six fields the
    unified proof envelope contract depends on."""
    from tessera.compiler.jit_bridge import JitBridgeRoute
    actual = {f.name for f in fields(JitBridgeRoute)}
    missing = _EXPECTED_FIELDS - actual
    assert not missing, (
        f"JitBridgeRoute lost expected fields: {missing!r}.  "
        f"The unified proof envelope contract depends on this "
        f"schema staying stable across the manifest-dispatch "
        f"(GA/EBM) and driver-dispatch (generic tensor) lanes."
    )


def test_generic_tensor_canonical_emits_unified_proof_route() -> None:
    """The matmul_softmax_matmul canonical must emit a JitBridgeRoute
    with the same 6 fields the GA/EBM manifest-dispatch lane
    produces.  The only allowed difference is the ``context`` value
    (``"driver"`` for generic tensor vs ``"direct"``/``"test"`` for
    GA/EBM)."""
    from tessera.compiler.canonical import matmul_softmax_matmul

    report = matmul_softmax_matmul.run()
    # On Darwin within envelope, the canonical must emit at least
    # one proof route.
    assert report.fallback_reason is None, (
        "default-shape Darwin run should hit the fused MSL kernel; "
        f"got fallback_reason={report.fallback_reason!r}.  Without "
        "native dispatch there's no proof route to compare."
    )
    assert report.proof_routes, (
        "matmul_softmax_matmul canonical produced no proof routes "
        "on Darwin happy path — Phase D wiring regressed?"
    )
    route = report.proof_routes[0]
    # Verify all six fields are populated.
    for field_name in _EXPECTED_FIELDS:
        value = getattr(route, field_name, None)
        # ``args_summary`` and ``context`` are allowed to be empty
        # tuples / strings; the other four must be non-empty strings
        # or positive floats.
        if field_name in ("args_summary",):
            assert value is not None
        elif field_name == "context":
            assert value == "driver", (
                f"generic-tensor lane should mark proof routes with "
                f"context='driver'; got {value!r}"
            )
        elif field_name == "latency_ms":
            assert isinstance(value, float) and value > 0, (
                f"latency_ms must be a positive float; got {value!r}"
            )
        else:
            assert isinstance(value, str) and value, (
                f"{field_name} must be a non-empty string; got "
                f"{value!r}"
            )


def test_generic_lane_context_is_distinguishable_from_manifest_lane() -> None:
    """The unified-but-distinguishable contract: generic-tensor
    routes report ``context="driver"``; manifest-dispatch routes
    (GA/EBM/M7) report ``context="direct"`` or a JIT context name.
    A reader walking ``report.proof_routes`` can tell the lanes
    apart by context without losing the unified schema."""
    from tessera.compiler.canonical import matmul_softmax_matmul

    report = matmul_softmax_matmul.run()
    assert report.proof_routes
    for r in report.proof_routes:
        assert r.context == "driver", (
            f"generic-tensor proof route should carry "
            f"context='driver'; got context={r.context!r}"
        )
        # The lane-specific symbol prefix is also a structural hint;
        # confirm it matches the expected ``tessera_apple_gpu_`` shape
        # so any future renaming of the symbol surfaces here.
        assert r.symbol.startswith("tessera_apple_gpu_"), (
            f"generic-tensor proof route symbol must start with "
            f"``tessera_apple_gpu_``; got {r.symbol!r}"
        )
        assert r.target == "apple_gpu"
        assert r.status == "fused"


def test_proof_envelope_matches_ga_canonical_shape_when_available() -> None:
    """If the GA rotor_sandwich_norm canonical runs and emits at
    least one route, its route has the same dataclass type as the
    generic-tensor lane's route.  This is the structural-equivalence
    half of the Phase D contract: two different lanes, one schema.
    """
    from tessera.compiler.canonical import (
        matmul_softmax_matmul, rotor_sandwich_norm,
    )
    from tessera.compiler.jit_bridge import JitBridgeRoute

    gen_report = matmul_softmax_matmul.run()
    ga_report = rotor_sandwich_norm.run()

    if not gen_report.proof_routes or not ga_report.proof_routes:
        pytest.skip(
            "One of the canonicals produced no proof routes "
            "(non-default shape or platform); structural match "
            "can't be verified."
        )
    gen_route = gen_report.proof_routes[0]
    ga_route = ga_report.proof_routes[0]
    assert isinstance(gen_route, JitBridgeRoute)
    assert isinstance(ga_route, JitBridgeRoute)
    # Same dataclass type ⇒ same field set ⇒ unified envelope.
    gen_fields = {f.name for f in fields(gen_route)}
    ga_fields = {f.name for f in fields(ga_route)}
    assert gen_fields == ga_fields == _EXPECTED_FIELDS
