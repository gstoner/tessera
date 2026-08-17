"""E2E-REAL-6E Adafactor and sequence-mixer authority migration."""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.compiler.jit import JitFn
from tessera.compiler.native_stateful_vjp import (
    validate_native_stateful_vjp_runtime_metadata,
)
from tessera.compiler.scheduled_matmul import find_tessera_opt


@ts.jit(target="x86", autodiff="reverse", wrt=("p", "g", "state"))
def _x86_adafactor_full(p, g, state):
    updated, _new_state = ts.ops.adafactor(
        p, g, state, lr=0.003, beta2=0.91, eps=1.0e-7
    )
    return updated


@ts.jit(
    target="x86",
    autodiff="reverse",
    wrt=("q", "k", "v", "gate", "beta", "decay"),
)
def _x86_sequence(q, k, v, gate, beta, decay):
    return ts.ops.gated_deltanet(
        q,
        k,
        v,
        gate,
        beta,
        decay,
        causal=True,
    )


def test_stateful_plugins_declare_complete_owned_spines() -> None:
    from tessera.compiler.native_vjp_plugins import native_vjp_plugin_declarations

    declarations = native_vjp_plugin_declarations()
    adafactor = declarations["adafactor"]
    assert adafactor.family == "adafactor_vjp"
    assert adafactor.schedule_consumer == "schedule.adafactor_vjp"
    assert adafactor.tile_consumer == "tile.training_kernel"
    assert adafactor.differential_policy == "non_reexecuting_state_lineage"
    assert set(adafactor.target_consumers) == {"x86", "rocm"}
    for name in (
        "gated_deltanet",
        "kimi_delta_attention",
        "modified_delta_attention",
    ):
        sequence = declarations[name]
        assert sequence.family == "sequence_mixer_backward"
        assert sequence.schedule_consumer == "schedule.sequence_mixer_backward"
        assert sequence.tile_consumer == "tile.training_kernel"
        assert sequence.differential_policy == "non_reexecuting_state_lineage"
        assert set(sequence.target_consumers) == {"x86", "rocm"}


def test_jitfn_no_longer_constructs_stateful_family_packages() -> None:
    assert not hasattr(JitFn, "_native_rocm_adafactor_backward")
    assert not hasattr(JitFn, "_native_sequence_mixer_backward")


def test_flat_adafactor_full_and_factored_match_tree_reference() -> None:
    rng = np.random.default_rng(20260817)
    p = rng.normal(size=(3, 5)).astype(np.float32)
    g = rng.normal(size=(3, 5)).astype(np.float32)
    row = np.abs(rng.normal(size=(3,))).astype(np.float32)
    col = np.abs(rng.normal(size=(5,))).astype(np.float32)
    kwargs = {"lr": 0.003, "beta2": 0.91, "eps": 1.0e-7}
    actual_p, actual_row, actual_col = ts.ops.adafactor(
        p, g, row, col, **kwargs
    )
    expected_p, expected_state = ts.optim.adafactor(
        p,
        g,
        {"v": {"row": row, "col": col, "factored": True}, "step": 0},
        **kwargs,
    )
    np.testing.assert_allclose(actual_p, expected_p, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual_row, expected_state["v"]["row"], rtol=2e-6)
    np.testing.assert_allclose(actual_col, expected_state["v"]["col"], rtol=2e-6)

    vector_p = p[0]
    vector_g = g[0]
    full = np.abs(rng.normal(size=(5,))).astype(np.float32)
    actual_p, actual_full = ts.ops.adafactor(vector_p, vector_g, full, **kwargs)
    expected_p, expected_state = ts.optim.adafactor(
        vector_p,
        vector_g,
        {"v": {"v": full, "factored": False}, "step": 0},
        **kwargs,
    )
    np.testing.assert_allclose(actual_p, expected_p, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual_full, expected_state["v"]["v"], rtol=2e-6)


@pytest.mark.parametrize("which", ["adafactor", "sequence"])
def test_stateful_source_executes_once_and_runtime_receives_no_graph(
    monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    if find_tessera_opt() is None:
        pytest.skip("production tessera-opt is required")
    from tessera import runtime as rt

    captured: dict = {}
    if which == "adafactor":
        shape = (7,)
        values = (
            np.ones(shape, np.float32),
            np.full(shape, 0.25, np.float32),
            np.full(shape, 0.5, np.float32),
        )
        cotangent = np.ones(shape, np.float32)
        function = _x86_adafactor_full
        op_name = "adafactor"
        outputs = tuple(np.ones_like(value) for value in values)
    else:
        q_shape, v_shape = (1, 1, 4, 3), (1, 1, 4, 2)
        values = (
            np.ones(q_shape, np.float32),
            np.ones(q_shape, np.float32),
            np.ones(v_shape, np.float32),
            np.ones(v_shape, np.float32),
            np.ones(q_shape[:3], np.float32),
            np.ones(q_shape[:3], np.float32),
        )
        cotangent = np.ones(v_shape, np.float32)
        function = _x86_sequence
        op_name = "gated_deltanet"
        outputs = tuple(np.ones_like(value) for value in values)

    original = getattr(ts.ops, op_name)
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    def fake_launch(artifact, launch_values):
        captured.update(artifact.metadata or {})
        validate_native_stateful_vjp_runtime_metadata(captured)
        return {
            "ok": True,
            "execution_mode": "cpu_avx512",
            "output": outputs,
        }

    monkeypatch.setattr(ts.ops, op_name, counted)
    monkeypatch.setattr(rt, "launch", fake_launch)
    actual = function.native_backward(*values, out_cotangents=cotangent)
    assert len(actual) == len(values)
    assert calls == 1
    assert "source_graph_ir" not in captured
    assert "ops" not in captured
    assert function.last_backward_execution["implementation"] == "family_plugin"
    assert function.last_backward_execution["proof_mode"] == (
        "structural_non_reexecuting"
    )
    assert len(function.last_backward_execution["artifact_hash"]) == 64
