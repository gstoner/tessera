"""E2E-REAL-6F exact-device certificate packet for every Apple GPU VJP family.

Each launch goes through the public ``@jit(target="apple_gpu")`` reverse
product, so the certificate carries the tracer's frontend differential and the
Metal launch's runtime-origin physical attestation. The oracle is an
independent float64 VJP from the definitions, never another kernel. A newly
declared Apple GPU family fails the final set equality until it supplies an
exact-device row here.

The certificate registry models exact-device Apple evidence only for apple7
(M1 family); the attestation reads the live Metal device's GPU family, so a
host whose device reports another family skips rather than borrowing apple7.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts

pytestmark = pytest.mark.hardware_apple_gpu

_EPS = 1.0e-5


@ts.jit(target="apple_gpu", autodiff="reverse", wrt=("x", "gamma"))
def _apple_rmsnorm(x, gamma):
    return ts.ops.rmsnorm(x, gamma=gamma, eps=_EPS)


@ts.jit(target="apple_gpu", autodiff="reverse", wrt=("x",))
def _apple_rmsnorm_safe(x):
    return ts.ops.rmsnorm_safe(x, eps=_EPS)


@ts.jit(target="apple_gpu", autodiff="reverse", wrt=("x", "gamma", "beta"))
def _apple_layer_norm(x, gamma, beta):
    return ts.ops.layer_norm(x, gamma=gamma, beta=beta, eps=_EPS)


def _reference(x, gamma, dy, *, layer: bool):
    """Analytic normalization VJP in float64; returns (dx, dgamma, dbeta)."""
    x64 = np.asarray(x, np.float64)
    dy64 = np.asarray(dy, np.float64)
    n = x64.shape[-1]
    scale = np.ones(n) if gamma is None else np.asarray(gamma, np.float64)
    mean = x64.mean(-1, keepdims=True) if layer else 0.0
    inv = 1.0 / np.sqrt(((x64 - mean) ** 2).mean(-1, keepdims=True) + _EPS)
    y_hat = (x64 - mean) * inv
    scaled = dy64 * scale
    centred = scaled.mean(-1, keepdims=True) if layer else 0.0
    dx = inv * (scaled - centred - y_hat * (scaled * y_hat).mean(-1, keepdims=True))
    return dx, (dy64 * y_hat).reshape(-1, n).sum(0), dy64.reshape(-1, n).sum(0)


def _run(compiled, args, dy, expected, op_name):
    gradients = compiled.native_backward(*args, out_cotangents=dy)
    assert len(gradients) == len(expected)
    for actual, reference in zip(gradients, expected):
        np.testing.assert_allclose(
            np.asarray(actual, np.float64), reference, rtol=2e-4, atol=2e-5
        )
    execution = compiled.last_backward_execution
    assert execution["execution_kind"] == "native_gpu"
    assert execution["execution_mode"] == "metal_runtime"
    assert execution["target_consumer"] == "apple.metal_normalization"
    certificate = execution["execution_certificate"]
    assert certificate["graph_consumer"] == op_name
    assert certificate["evidence_scope"] == "exact_device"
    # Public entry: the tracer's frontend differential must be bound.
    assert len(certificate["frontend_certificate_digest"]) == 64
    return certificate["digest"]


def test_every_declared_apple_gpu_vjp_family_records_an_exact_certificate() -> None:
    from tessera import runtime
    from tessera.compiler.frontend_authority_audit import collect_rows
    from tessera.compiler.native_vjp_plugins import (
        native_vjp_exact_execution_coverage,
        native_vjp_execution_certificates,
        validate_native_vjp_execution_certificate,
    )

    # The Metal runtime/device boundary is the `hardware_apple_gpu` marker
    # (tests/conftest.py); only the certificate model's arch is checked here.
    arch = runtime._apple_gpu_live_arch()
    if arch != "apple7":
        pytest.skip(f"no exact-device VJP certificate model for {arch!r}")

    rng = np.random.default_rng(6107)
    digests = set()
    # A rank-3 input with a ragged width: the affine reductions run over
    # every leading axis and one thread per channel.
    x = rng.standard_normal((3, 5, 37)).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, 37).astype(np.float32)
    beta = rng.standard_normal(37).astype(np.float32)
    dy = rng.standard_normal(x.shape).astype(np.float32)

    dx, dg, db = _reference(x, gamma, dy, layer=False)
    digests.add(_run(_apple_rmsnorm, (x, gamma), dy, (dx, dg), "tessera.rmsnorm"))
    dx, _, _ = _reference(x, None, dy, layer=False)
    digests.add(_run(_apple_rmsnorm_safe, (x,), dy, (dx,), "tessera.rmsnorm_safe"))
    dx, dg, db = _reference(x, gamma, dy, layer=True)
    digests.add(
        _run(_apple_layer_norm, (x, gamma, beta), dy, (dx, dg, db), "tessera.layer_norm")
    )

    required = {
        (row.family, "apple_gpu") for row in collect_rows() if "apple_gpu" in row.targets
    }
    observed = {
        pair
        for pair in native_vjp_exact_execution_coverage()
        if pair[1] == "apple_gpu"
    }
    assert observed == required

    certificates = native_vjp_execution_certificates()
    for family, target in sorted(required):
        rows = [
            row
            for row in certificates[family]
            if row["target"] == target and row["evidence_scope"] == "exact_device"
        ]
        assert rows, f"{family}/{target} has no exact-device certificate"
        for row in rows:
            validate_native_vjp_execution_certificate(row)
            assert row["physical_attestation"]["device_arch"] == "apple7"
            assert row["physical_attestation"]["execution_mode"] == "metal_runtime"
    # Every certificate this packet produced is in the process registry.
    recorded = {row["digest"] for rows in certificates.values() for row in rows}
    assert digests <= recorded
