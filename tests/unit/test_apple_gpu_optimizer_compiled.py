"""Apple GPU optimizer lane — sgd / momentum / adam / adamw / lion.

A single per-parameter update parameterized by optimizer kind, with state
(m/v) in/out. Dense f32 dispatches through the Apple Metal kernel; unavailable
hardware and unsupported contracts truthfully fall back to the reference path.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera import optim
from tessera import runtime as rt
from tests._support.apple import assert_native_apple_gpu, assert_reference_cpu

SHAPE = (4, 5)


@ts.jit(target="apple_gpu")
def _jit_sgd(p, g):
    return ts.ops.sgd(p, g, lr=0.1)


def _art(op, operands, extras, kw):
    names = [f"a{i}" for i in range(len(operands))]
    return rt.RuntimeArtifact(metadata={
        "target": "apple_gpu", "compiler_path": "apple_gpu_optimizer_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": {**kw, "extras": extras}}]})


def _launch(op, operands, extras, kw):
    res = rt.launch(_art(op, operands, extras, kw), tuple(operands))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "apple_gpu_optimizer_compiled"
    expected_kind = "native_gpu" if rt.DeviceTensor.is_metal() else "reference_cpu"
    assert res["execution_kind"] == expected_kind
    return res["output"]


def test_optimizer_is_in_the_apple_gpu_compiler_envelope():
    from tessera.compiler.apple_gpu_envelope import runtime_ops
    for op in ("tessera.sgd", "tessera.momentum", "tessera.adam",
               "tessera.adamw", "tessera.lion"):
        assert op in runtime_ops()


def test_optimizer_execution_matrix_declares_native_base_lane():
    from tessera.compiler.execution_matrix import lookup
    row = lookup("apple_gpu", "apple_gpu_optimizer_compiled")
    assert row is not None
    assert row.execution_kind == "native_gpu"


def test_sgd_jit_routes_through_the_apple_gpu_envelope():
    p = np.ones(SHAPE, np.float32)
    g = np.full(SHAPE, 0.25, np.float32)
    np.testing.assert_allclose(np.asarray(_jit_sgd(p, g)), p - 0.1 * g)
    metadata = _jit_sgd.runtime_artifact().metadata
    assert metadata["compiler_path"] == "apple_gpu_mps"
    assert metadata["execution_mode"] == "metal_runtime"
    assert metadata["capability_reason"] == (
        "Apple GPU fused Metal f32 optimizer ABI; shares p/g/m/v semantics "
        "with x86 and ROCm"
    )
    assert "JIT_COMPILED_TARGET_RUNTIME" in metadata["diagnostics"][0]


def test_sgd():
    rng = np.random.default_rng(2)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    g = rng.standard_normal(SHAPE).astype(np.float32)
    np.testing.assert_allclose(np.asarray(_launch("tessera.sgd", [p, g], [], {"lr": 0.1})),
                               np.asarray(optim.sgd(p, g, lr=0.1)), atol=1e-6)


def test_momentum():
    rng = np.random.default_rng(3)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    g = rng.standard_normal(SHAPE).astype(np.float32)
    v0 = np.zeros(SHAPE, np.float32)
    pn, vn = (np.asarray(x) for x in _launch(
        "tessera.momentum", [p, g, v0], ["v"], {"lr": 0.1, "momentum": 0.9}))
    rp, rst = optim.momentum(p, g, None, lr=0.1, momentum=0.9)
    np.testing.assert_allclose(pn, np.asarray(rp), atol=1e-6)
    np.testing.assert_allclose(vn, np.asarray(rst["velocity"]), atol=1e-6)


def test_adamw_multistep():
    rng = np.random.default_rng(1)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    m = np.zeros(SHAPE, np.float32)
    v = np.zeros(SHAPE, np.float32)
    state = None
    for step in range(1, 6):
        g = rng.standard_normal(SHAPE).astype(np.float32)
        pn, m, v = (np.asarray(x) for x in _launch(
            "tessera.adamw", [p, g, m, v], ["m", "v"],
            {"lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
             "weight_decay": 0.01, "step": step}))
        ref_p, state = optim.adamw(p, g, state, lr=1e-3, beta1=0.9, beta2=0.999,
                                   eps=1e-8, weight_decay=0.01)
        np.testing.assert_allclose(pn, np.asarray(ref_p), atol=2e-5)
        np.testing.assert_allclose(m, np.asarray(state["m"]), atol=2e-5)
        np.testing.assert_allclose(v, np.asarray(state["v"]), atol=2e-6)
        p = pn


def test_adam_multistep():
    rng = np.random.default_rng(4)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    m = np.zeros(SHAPE, np.float32)
    v = np.zeros(SHAPE, np.float32)
    state = None
    for step in range(1, 4):
        g = rng.standard_normal(SHAPE).astype(np.float32)
        pn, m, v = (np.asarray(x) for x in _launch(
            "tessera.adam", [p, g, m, v], ["m", "v"], {"lr": 1e-3, "step": step}))
        ref_p, state = optim.adam(p, g, state, lr=1e-3)
        np.testing.assert_allclose(pn, np.asarray(ref_p), atol=2e-5)
        p = pn


def test_generic_envelope_preserves_positional_adam_state(monkeypatch):
    """Graph-IR envelope operands carry m/v positionally, without ``extras``."""
    rng = np.random.default_rng(41)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    g = rng.standard_normal(SHAPE).astype(np.float32)
    m = rng.standard_normal(SHAPE).astype(np.float32)
    v = np.abs(rng.standard_normal(SHAPE)).astype(np.float32)
    monkeypatch.setattr(rt, "_apple_optimizer_metal_kernel", rt._apple_optimizer_kernel)

    out = rt._apple_gpu_dispatch_optimizer(
        "tessera.adam", [p, g, m, v], {"lr": 1e-3, "step": 3}, np)
    ref_p, ref_state = optim.adam(
        p, g, {"m": m, "v": v, "step": 2}, lr=1e-3)

    assert out is not None
    pn, mn, vn = (np.asarray(value) for value in out)
    np.testing.assert_allclose(pn, np.asarray(ref_p), atol=2e-5)
    np.testing.assert_allclose(mn, np.asarray(ref_state["m"]), atol=2e-5)
    np.testing.assert_allclose(vn, np.asarray(ref_state["v"]), atol=2e-6)


def test_lion():
    rng = np.random.default_rng(5)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    g = rng.standard_normal(SHAPE).astype(np.float32)
    m = np.zeros(SHAPE, np.float32)
    pn, mn = (np.asarray(x) for x in _launch(
        "tessera.lion", [p, g, m], ["m"],
        {"lr": 1e-4, "beta1": 0.9, "beta2": 0.99, "weight_decay": 0.01}))
    rp, rst = optim.lion(p, g, None, lr=1e-4, beta1=0.9, beta2=0.99,
                         weight_decay=0.01)
    np.testing.assert_allclose(pn, np.asarray(rp), atol=2e-6)
    np.testing.assert_allclose(mn, np.asarray(rst["m"]), atol=2e-6)


@pytest.mark.hardware_apple_gpu
def test_f32_optimizer_ops_report_native_gpu_on_metal():
    p = np.ones(SHAPE, np.float32)
    g = np.full(SHAPE, 0.25, np.float32)
    m = np.zeros(SHAPE, np.float32)
    v = np.zeros(SHAPE, np.float32)
    res = rt.launch(_art("tessera.adamw", [p, g, m, v], ["m", "v"],
                         {"lr": 1e-3, "step": 1}), (p, g, m, v))
    assert_native_apple_gpu(res, compiler_path="apple_gpu_optimizer_compiled")


def test_unsupported_optimizer_dtype_uses_reference_cpu_override():
    p = np.ones(SHAPE, np.float64)
    g = np.full(SHAPE, 0.25, np.float64)
    res = rt.launch(_art("tessera.sgd", [p, g], [], {"lr": 0.1}), (p, g))
    assert_reference_cpu(res)
    np.testing.assert_allclose(np.asarray(res["output"]), p - 0.1 * g)
