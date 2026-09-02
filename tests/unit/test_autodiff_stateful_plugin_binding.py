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
        p, g, state, lr=0.003, beta2=0.91, eps=1.0e-7, step=3
    )
    return updated


@ts.jit(target="x86", autodiff="reverse", wrt=("p", "g", "row", "col"))
def _x86_adafactor_factored(p, g, row, col):
    updated, _new_row, _new_col = ts.ops.adafactor(
        p, g, row, col, lr=0.003, beta2=0.91, eps=1.0e-7, step=3
    )
    return updated


@ts.jit(target="rocm", autodiff="reverse", wrt=("p", "g", "state"))
def _rocm_adafactor_full(p, g, state):
    updated, _new_state = ts.ops.adafactor(
        p, g, state, lr=0.003, beta2=0.91, eps=1.0e-7, step=3
    )
    return updated


@ts.jit(target="rocm", autodiff="reverse", wrt=("p", "g", "row", "col"))
def _rocm_adafactor_factored(p, g, row, col):
    updated, _new_row, _new_col = ts.ops.adafactor(
        p, g, row, col, lr=0.003, beta2=0.91, eps=1.0e-7, step=3
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


@ts.jit(
    target="rocm",
    autodiff="reverse",
    wrt=("q", "k", "v", "gate", "beta", "decay"),
)
def _rocm_sequence(q, k, v, gate, beta, decay):
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
    assert set(adafactor.target_consumers) == {"x86", "rocm", "nvidia_sm120"}
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
        assert set(sequence.target_consumers) == {"x86", "rocm", "nvidia_sm120"}


def test_jitfn_no_longer_constructs_stateful_family_packages() -> None:
    assert not hasattr(JitFn, "_native_rocm_adafactor_backward")
    assert not hasattr(JitFn, "_native_sequence_mixer_backward")


def test_flat_adafactor_full_and_factored_match_tree_reference() -> None:
    rng = np.random.default_rng(20260817)
    p = rng.normal(size=(3, 5)).astype(np.float32)
    g = rng.normal(size=(3, 5)).astype(np.float32)
    row = np.abs(rng.normal(size=(3,))).astype(np.float32)
    col = np.abs(rng.normal(size=(5,))).astype(np.float32)
    # Step 3, not the default 1: at step 1 the bias-corrected decay is exactly
    # 0, so the carried row/col are discarded and the comparison would not
    # exercise the state at all.
    kwargs = {"lr": 0.003, "beta2": 0.91, "eps": 1.0e-7}
    actual_p, actual_row, actual_col = ts.ops.adafactor(
        p, g, row, col, step=3, **kwargs
    )
    expected_p, expected_state = ts.optim.adafactor(
        p,
        g,
        {"v": {"row": row, "col": col, "factored": True}, "step": 2},
        **kwargs,
    )
    np.testing.assert_allclose(actual_p, expected_p, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual_row, expected_state["v"]["row"], rtol=2e-6)
    np.testing.assert_allclose(actual_col, expected_state["v"]["col"], rtol=2e-6)

    vector_p = p[0]
    vector_g = g[0]
    full = np.abs(rng.normal(size=(5,))).astype(np.float32)
    actual_p, actual_full = ts.ops.adafactor(
        vector_p, vector_g, full, step=3, **kwargs
    )
    expected_p, expected_state = ts.optim.adafactor(
        vector_p,
        vector_g,
        {"v": {"v": full, "factored": False}, "step": 2},
        **kwargs,
    )
    np.testing.assert_allclose(actual_p, expected_p, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual_full, expected_state["v"]["v"], rtol=2e-6)


def test_factored_adafactor_plugin_executes_and_records_topology_certificate() -> None:
    if find_tessera_opt() is None:
        pytest.skip("production tessera-opt is required")
    from tessera import runtime as rt
    from tessera.autodiff.vjp import get_vjp
    from tessera.compiler.native_vjp_plugins import (
        native_vjp_execution_certificates,
        validate_native_vjp_execution_certificate,
    )

    if rt._load_x86_elementwise() is None:
        pytest.skip("AVX-512 runtime unavailable")
    rng = np.random.default_rng(20260827)
    p = rng.normal(size=(3, 5)).astype(np.float32)
    g = rng.normal(scale=0.2, size=p.shape).astype(np.float32)
    row = rng.uniform(0.1, 0.3, size=p.shape[:-1]).astype(np.float32)
    col = rng.uniform(0.1, 0.3, size=p.shape[-1]).astype(np.float32)
    dy = rng.normal(size=p.shape).astype(np.float32)
    actual = _x86_adafactor_factored.native_backward(
        p, g, row, col, out_cotangents=dy
    )
    expected = get_vjp("adafactor")(
        dy,
        p,
        g,
        {"v": {"row": row, "col": col, "factored": True}, "step": 2},
        lr=0.003,
        beta2=0.91,
        eps=1.0e-7,
    )
    references = (
        expected[0],
        expected[1],
        expected[2]["v"]["row"],
        expected[2]["v"]["col"],
    )
    for value, reference in zip(actual, references, strict=True):
        np.testing.assert_allclose(value, reference, rtol=4e-4, atol=4e-5)

    execution = _x86_adafactor_factored.last_backward_execution
    certificate = execution["execution_certificate"]
    assert certificate["schema"] == "tessera.native_vjp_execution.v1"
    assert certificate["status"] == "executed"
    assert certificate["family"] == "adafactor_vjp"
    assert certificate["topology"] == "factored"
    assert certificate["source_reexecution"] == "prohibited"
    assert len(certificate["digest"]) == 64
    records = native_vjp_execution_certificates()["adafactor_vjp"]
    assert any(row["digest"] == certificate["digest"] for row in records)
    validate_native_vjp_execution_certificate(certificate)
    stale = dict(certificate)
    stale["topology"] = "full"
    with pytest.raises(ValueError, match="stale identity"):
        validate_native_vjp_execution_certificate(stale)


@pytest.mark.parametrize("topology", ["full", "factored"])
def test_rocm_adafactor_topologies_record_exact_gfx1151_certificates(
    topology: str,
) -> None:
    from tessera import runtime as rt
    from tessera.autodiff.vjp import get_vjp
    from tessera.compiler.native_vjp_plugins import (
        native_vjp_exact_execution_coverage,
        validate_native_vjp_execution_certificate,
    )

    if find_tessera_opt() is None or not rt._rocm_wmma_runtime_available():
        pytest.skip("ROCm compiler/gfx1151 runtime unavailable")
    rng = np.random.default_rng(20260830 + int(topology == "factored"))
    shape = (3, 5) if topology == "factored" else (17,)
    p = rng.normal(size=shape).astype(np.float32)
    g = rng.normal(scale=0.2, size=shape).astype(np.float32)
    dy = rng.normal(size=shape).astype(np.float32)
    if topology == "factored":
        row = rng.uniform(0.1, 0.3, size=shape[:-1]).astype(np.float32)
        col = rng.uniform(0.1, 0.3, size=shape[-1]).astype(np.float32)
        actual = _rocm_adafactor_factored.native_backward(
            p, g, row, col, out_cotangents=dy
        )
        expected = get_vjp("adafactor")(
            dy,
            p,
            g,
            {"v": {"row": row, "col": col, "factored": True}, "step": 2},
            lr=0.003,
            beta2=0.91,
            eps=1.0e-7,
        )
        references = (
            expected[0],
            expected[1],
            expected[2]["v"]["row"],
            expected[2]["v"]["col"],
        )
        compiled = _rocm_adafactor_factored
    else:
        state = rng.uniform(0.1, 0.3, size=shape).astype(np.float32)
        actual = _rocm_adafactor_full.native_backward(
            p, g, state, out_cotangents=dy
        )
        expected = get_vjp("adafactor")(
            dy,
            p,
            g,
            {"v": {"v": state, "factored": False}, "step": 2},
            lr=0.003,
            beta2=0.91,
            eps=1.0e-7,
        )
        references = (expected[0], expected[1], expected[2]["v"]["v"])
        compiled = _rocm_adafactor_full
    for value, reference in zip(actual, references, strict=True):
        np.testing.assert_allclose(value, reference, rtol=4e-4, atol=4e-5)

    certificate = compiled.last_backward_execution["execution_certificate"]
    validate_native_vjp_execution_certificate(certificate)
    assert certificate["family"] == "adafactor_vjp"
    assert certificate["topology"] == topology
    assert certificate["evidence_scope"] == "exact_device"
    assert certificate["physical_attestation"]["device_arch"] == "gfx1151"
    assert ("adafactor_vjp", "rocm") in native_vjp_exact_execution_coverage()


def test_rocm_sequence_mixer_records_exact_gfx1151_certificate() -> None:
    from tessera import runtime as rt
    from tessera.autodiff.vjp import get_vjp
    from tessera.compiler.native_vjp_plugins import (
        native_vjp_exact_execution_coverage,
        validate_native_vjp_execution_certificate,
    )

    if find_tessera_opt() is None or not rt._rocm_wmma_runtime_available():
        pytest.skip("ROCm compiler/gfx1151 runtime unavailable")
    rng = np.random.default_rng(20260831)
    q_shape, v_shape = (1, 1, 5, 3), (1, 1, 5, 2)
    q = rng.normal(scale=0.3, size=q_shape).astype(np.float32)
    k = rng.normal(scale=0.3, size=q_shape).astype(np.float32)
    k /= np.maximum(np.linalg.norm(k, axis=-1, keepdims=True), 1.0e-6)
    v = rng.normal(scale=0.3, size=v_shape).astype(np.float32)
    gate = rng.normal(scale=0.2, size=v_shape).astype(np.float32)
    beta = rng.uniform(0.2, 0.8, q_shape[:-1]).astype(np.float32)
    decay = rng.uniform(0.7, 0.95, q_shape[:-1]).astype(np.float32)
    dy = rng.normal(scale=0.3, size=v_shape).astype(np.float32)
    actual = _rocm_sequence.native_backward(
        q, k, v, gate, beta, decay, out_cotangents=dy
    )
    expected = get_vjp("gated_deltanet")(
        dy, q, k, v, gate, beta, decay, erase=False
    )
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference, rtol=2e-3, atol=2e-3)
    certificate = _rocm_sequence.last_backward_execution["execution_certificate"]
    validate_native_vjp_execution_certificate(certificate)
    assert certificate["family"] == "sequence_mixer_backward"
    assert certificate["evidence_scope"] == "exact_device"
    assert certificate["physical_attestation"]["device_arch"] == "gfx1151"
    assert (
        "sequence_mixer_backward",
        "rocm",
    ) in native_vjp_exact_execution_coverage()


def test_x86_sequence_mixer_records_exact_avx512_certificate() -> None:
    from tessera import runtime as rt
    from tessera.autodiff.vjp import get_vjp
    from tessera.compiler.native_vjp_plugins import (
        native_vjp_exact_execution_coverage,
        validate_native_vjp_execution_certificate,
    )

    if find_tessera_opt() is None or not rt._x86_elementwise_available():
        pytest.skip("production tessera-opt and AVX-512 runtime are required")
    rng = np.random.default_rng(20260902)
    q_shape, v_shape = (1, 1, 5, 3), (1, 1, 5, 2)
    q = rng.normal(scale=0.3, size=q_shape).astype(np.float32)
    k = rng.normal(scale=0.3, size=q_shape).astype(np.float32)
    k /= np.maximum(np.linalg.norm(k, axis=-1, keepdims=True), 1.0e-6)
    v = rng.normal(scale=0.3, size=v_shape).astype(np.float32)
    gate = rng.normal(scale=0.2, size=v_shape).astype(np.float32)
    beta = rng.uniform(0.2, 0.8, q_shape[:-1]).astype(np.float32)
    decay = rng.uniform(0.7, 0.95, q_shape[:-1]).astype(np.float32)
    dy = rng.normal(scale=0.3, size=v_shape).astype(np.float32)
    actual = _x86_sequence.native_backward(
        q, k, v, gate, beta, decay, out_cotangents=dy
    )
    expected = get_vjp("gated_deltanet")(
        dy, q, k, v, gate, beta, decay, erase=False
    )
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference, rtol=2e-4, atol=2e-4)
    certificate = _x86_sequence.last_backward_execution["execution_certificate"]
    validate_native_vjp_execution_certificate(certificate)
    assert certificate["family"] == "sequence_mixer_backward"
    assert certificate["evidence_scope"] == "exact_device"
    assert certificate["physical_attestation"]["device_arch"] == "x86_avx512"
    assert (
        "sequence_mixer_backward",
        "x86",
    ) in native_vjp_exact_execution_coverage()


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
