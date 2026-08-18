"""E2E-REAL-6D non-reexecuting Lion family authority."""

from __future__ import annotations

import copy

import numpy as np
import pytest

import tessera as ts
from tessera.compiler.native_lion_vjp import (
    validate_native_lion_vjp_runtime_metadata,
)
from tessera.compiler.scheduled_matmul import find_tessera_opt


@ts.jit(target="x86", autodiff="reverse", wrt=("param", "grad", "moment"))
def _x86_lion(param, grad, moment):
    return ts.ops.lion(
        param,
        grad,
        moment,
        lr=0.004,
        beta1=0.8,
        beta2=0.93,
        weight_decay=0.025,
    )


def test_lion_plugin_executes_source_once_and_runtime_receives_no_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tessera import runtime as rt

    if find_tessera_opt() is None or not rt._x86_elementwise_available():
        pytest.skip("production tessera-opt and AVX-512 image are required")
    calls = 0
    original_lion = ts.ops.lion

    def counted_lion(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_lion(*args, **kwargs)

    captured: dict = {}
    original_launch = rt.launch

    def capture_launch(artifact, args):
        captured.update(artifact.metadata or {})
        return original_launch(artifact, args)

    monkeypatch.setattr(ts.ops, "lion", counted_lion)
    monkeypatch.setattr(rt, "launch", capture_launch)
    rng = np.random.default_rng(20260817)
    shape = (5, 19)
    values = tuple(rng.normal(size=shape).astype(np.float32) for _ in range(3))
    cotangents = tuple(
        rng.normal(size=shape).astype(np.float32) for _ in range(2)
    )
    actual = _x86_lion.native_backward(
        *values, out_cotangents=cotangents
    )

    expected = (
        cotangents[0] * (1.0 - 0.004 * 0.025),
        cotangents[1] * (1.0 - 0.93),
        cotangents[1] * 0.93,
    )
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, reference, atol=2e-6, rtol=2e-6)
    assert calls == 1
    assert "source_graph_ir" not in captured
    assert "ops" not in captured
    validate_native_lion_vjp_runtime_metadata(captured)

    execution = _x86_lion.last_backward_execution
    assert execution["implementation"] == "family_plugin"
    assert execution["frontend_authority"] == "tracer"
    assert execution["proof_mode"] == "structural_non_reexecuting"
    assert execution["schedule_consumer"] == "schedule.lion_vjp"
    assert execution["tile_consumer"] == "tile.training_kernel"
    assert execution["target_consumer"] == "x86.avx512_lion_backward"
    for field in (
        "source_graph_ir_digest",
        "frontend_certificate_digest",
        "schedule_artifact_hash",
        "state_lineage_digest",
        "tile_program_digest",
        "artifact_hash",
    ):
        assert len(execution[field]) == 64
    certificate = _x86_lion.last_frontend_differential.contract
    assert certificate["concrete_executions"] == 1
    assert certificate["graph_consumers"] == ["tessera.lion"]
    assert certificate["numerical_authority"] == "physical_package_required"

    stale = copy.deepcopy(captured)
    stale["native_vjp_package"]["tile_program_digest"] = "0" * 64
    with pytest.raises(ValueError, match="package identity is stale"):
        validate_native_lion_vjp_runtime_metadata(stale)


def test_lion_plugin_declares_non_reexecuting_state_lineage_policy() -> None:
    from tessera.compiler.native_vjp_plugins import (
        native_vjp_plugin_declarations,
    )

    declaration = native_vjp_plugin_declarations()["lion"]
    assert declaration.family == "lion_vjp"
    assert declaration.differential_policy == "non_reexecuting_state_lineage"
    assert declaration.target_consumers == {
        "x86": "x86.avx512_lion_backward",
        "rocm": "rocm.gfx1151_lion_backward",
        "nvidia_sm120": "nvidia.sm120_lion_backward",
    }
