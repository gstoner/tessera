"""E2E-REAL-6F remaining optimizer reverse-authority migration."""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.compiler.jit import JitFn
from tessera.compiler.native_stateful_vjp import (
    validate_native_stateful_vjp_runtime_metadata,
)
from tessera.compiler.scheduled_matmul import find_tessera_opt


@ts.jit(target="x86", autodiff="reverse", wrt=("p", "g"))
def _x86_sgd(p, g):
    return ts.ops.sgd(p, g, lr=0.05)


@ts.jit(target="rocm", autodiff="reverse", wrt=("p", "g"))
def _rocm_sgd(p, g):
    return ts.ops.sgd(p, g, lr=0.05)


@ts.jit(target="x86", autodiff="reverse", wrt=("p", "g", "velocity"))
def _x86_nesterov(p, g, velocity):
    return ts.ops.nesterov(p, g, velocity, lr=0.05, momentum=0.8)


@ts.jit(target="rocm", autodiff="reverse", wrt=("p", "g", "velocity"))
def _rocm_momentum(p, g, velocity):
    return ts.ops.momentum(p, g, velocity, lr=0.05, momentum=0.8)


@ts.jit(target="rocm", autodiff="reverse", wrt=("p", "g", "m1", "m2"))
def _rocm_adamw(p, g, m1, m2):
    return ts.ops.adamw(
        p, g, m1, m2, lr=0.01, beta1=0.8, beta2=0.9,
        eps=1.0e-6, weight_decay=0.02, step=3,
    )


def test_optimizer_plugins_declare_exact_target_owners() -> None:
    from tessera.compiler.native_vjp_plugins import native_vjp_plugin_declarations

    declarations = native_vjp_plugin_declarations()
    for name in ("sgd", "momentum", "nesterov"):
        declaration = declarations[name]
        assert declaration.family == "optimizer_vjp"
        assert declaration.schedule_consumer == "schedule.optimizer_vjp"
        assert declaration.tile_consumer == "tile.training_kernel"
        assert declaration.differential_policy == "non_reexecuting_state_lineage"
        assert set(declaration.target_consumers) == {
            "x86", "rocm", "nvidia_sm120"
        }
    for name in ("adam", "adamw"):
        assert set(declarations[name].target_consumers) == {
            "rocm", "nvidia_sm120"
        }


def test_jitfn_optimizer_compatibility_helpers_are_retired() -> None:
    assert not hasattr(JitFn, "_native_sgd_backward")
    assert not hasattr(JitFn, "_native_momentum_backward")
    assert not hasattr(JitFn, "_native_rocm_adam_backward")


@pytest.mark.parametrize(
    ("compiled", "kind"),
    [(_rocm_sgd, "sgd"), (_rocm_momentum, "momentum")],
)
def test_rocm_sgd_momentum_variants_record_exact_gfx1151_certificates(
    compiled, kind: str
) -> None:
    from tessera import runtime as rt
    from tessera.compiler.native_vjp_plugins import (
        _canonical_digest,
        native_vjp_exact_execution_coverage,
        validate_native_vjp_execution_certificate,
    )

    if rt._tessera_opt_path() is None or not rt._rocm_wmma_runtime_available():
        pytest.skip("ROCm compiler/gfx1151 runtime unavailable")
    rng = np.random.default_rng(20260828 + int(kind == "momentum"))
    shape = (5, 17)
    p = rng.normal(size=shape).astype(np.float32)
    g = rng.normal(size=shape).astype(np.float32)
    dp = rng.normal(size=shape).astype(np.float32)
    if kind == "sgd":
        actual = compiled.native_backward(p, g, out_cotangents=dp)
        expected = (dp, np.float32(-0.05) * dp)
    else:
        velocity = rng.normal(size=shape).astype(np.float32)
        dv = rng.normal(size=shape).astype(np.float32)
        actual = compiled.native_backward(
            p, g, velocity, out_cotangents=(dp, dv)
        )
        expected = (
            dp,
            np.float32(-0.05) * dp + dv,
            np.float32(0.8) * (np.float32(-0.05) * dp + dv),
        )
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference, rtol=2e-6, atol=2e-6)
    certificate = compiled.last_backward_execution["execution_certificate"]
    validate_native_vjp_execution_certificate(certificate)
    assert certificate["graph_consumer"] == f"tessera.{kind}"
    assert certificate["evidence_scope"] == "exact_device"
    assert certificate["physical_attestation"]["device_arch"] == "gfx1151"
    assert ("optimizer_vjp", "rocm") in native_vjp_exact_execution_coverage()
    if kind == "sgd":
        stale = dict(certificate)
        stale["physical_attestation"] = {
            **certificate["physical_attestation"],
            "device_arch": "gfx1100",
        }
        stale_body = dict(stale)
        stale_body.pop("digest")
        stale["digest"] = _canonical_digest(stale_body)
        with pytest.raises(ValueError, match="stale physical identity"):
            validate_native_vjp_execution_certificate(stale)


@pytest.mark.parametrize(
    ("function", "values", "cotangents", "mode"),
    [
        (_x86_sgd, 2, 1, "cpu_avx512"),
        (_x86_nesterov, 3, 2, "cpu_avx512"),
        (_rocm_adamw, 4, 3, "hip_runtime"),
    ],
)
def test_optimizer_source_executes_once_and_runtime_receives_no_graph(
    monkeypatch: pytest.MonkeyPatch,
    function,
    values: int,
    cotangents: int,
    mode: str,
) -> None:
    if find_tessera_opt() is None:
        pytest.skip("production tessera-opt is required")
    from tessera import runtime as rt

    shape = (7,)
    operands = tuple(np.full(shape, index + 1, np.float32) for index in range(values))
    cots = tuple(np.ones(shape, np.float32) for _ in range(cotangents))
    captured: dict = {}

    def fake_launch(artifact, launch_values):
        captured.update(artifact.metadata or {})
        validate_native_stateful_vjp_runtime_metadata(captured)
        return {
            "ok": True,
            "execution_mode": mode,
            "output": tuple(np.ones_like(value) for value in operands),
        }

    monkeypatch.setattr(rt, "launch", fake_launch)
    actual = function.native_backward(
        *operands, out_cotangents=cots[0] if cotangents == 1 else cots
    )
    assert len(actual) == values
    assert "source_graph_ir" not in captured
    assert "ops" not in captured
    assert captured["scheduled_training"]["family"] == "optimizer_vjp"
    assert function.last_backward_execution["implementation"] == "family_plugin"
    assert function.last_backward_execution["proof_mode"] == (
        "structural_non_reexecuting"
    )
    assert function.last_backward_execution["execution_certificate"][
        "evidence_scope"
    ] == "runtime_unattested"
