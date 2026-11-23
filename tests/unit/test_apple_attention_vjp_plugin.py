"""APPLE-NTBI-1: the attention VJP plugin has an Apple consumer.

``native_backward`` on an ``apple_gpu`` attention program reaches the
``attention_backward`` native-VJP plugin, which lowers the traced reverse
carrier through Schedule and Tile and packages it on the existing Apple7 split
Metal backward (APPLE-ATTN-BWD-1). The runtime executes that package; it never
receives the forward Graph.
"""
from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera._jit_boundary import TesseraJitError
from tessera.compiler.attention_contract import reference_attention_backward_split_reduced
from tessera.compiler.native_vjp_plugins import native_vjp_plugin_available


@ts.jit(target="apple_gpu", autodiff="reverse", wrt=("q", "key", "value"))
def _gqa(q, key, value):
    return ts.ops.gqa_attention(q, key, value, num_query_heads=4, num_kv_heads=2,
                                scale=0.25, causal=True)


@ts.jit(target="apple_gpu", autodiff="reverse", wrt=("q", "key", "value"))
def _biased(q, key, value, bias):
    return ts.ops.flash_attn(q, key, value, scale=0.25, causal=False, attn_bias=bias)


@ts.jit(target="apple_gpu", autodiff="reverse", wrt=("q", "key", "value"))
def _dropout(q, key, value):
    return ts.ops.flash_attn(q, key, value, scale=0.25, dropout_p=0.1, seed=3)


@ts.jit(target="apple_gpu", autodiff="reverse", wrt=("q", "key", "value"))
def _saved_lse(q, key, value):
    return ts.ops.flash_attn(q, key, value, scale=0.25, lse_checkpoint="saved")


def _storage(name):
    if name == "bf16":
        from ml_dtypes import bfloat16
        return bfloat16
    return {"fp32": np.float32, "fp16": np.float16}[name]


def _gqa_inputs(dtype, seed=20261122):
    rng = np.random.default_rng(seed)
    q = rng.normal(0.0, 0.2, (1, 4, 9, 16)).astype(dtype)
    key = rng.normal(0.0, 0.2, (1, 2, 11, 16)).astype(dtype)
    value = rng.normal(0.0, 0.2, key.shape).astype(dtype)
    dout = rng.normal(0.0, 0.2, q.shape).astype(dtype)
    return q, key, value, dout


def _require_metal():
    from tessera.compiler import apple_native
    from tessera.compiler.scheduled_matmul import find_tessera_opt

    if find_tessera_opt() is None or not apple_native.tools_available():
        pytest.skip("production tessera-opt and the Tessera Apple GPU runtime dylib are required")


def test_the_attention_plugin_declares_an_apple_consumer():
    for op in ("tessera.flash_attn", "tessera.gqa_attention", "tessera.mqa_attention"):
        assert native_vjp_plugin_available(op, "apple_gpu")


@pytest.mark.parametrize("fn", [_dropout, _saved_lse])
def test_contracts_the_metal_backward_cannot_carry_refuse(fn):
    # Dropout and a saved LSE are not expressible on the Apple7 backward
    # (it recomputes the row statistics); both fail closed, never silently
    # computing another program. Dropout's ``random`` effect is refused one
    # gate earlier, by the frontend differential certificate.
    _require_metal()
    q, key, value, dout = _gqa_inputs(np.float32)
    q = q[:, :2]
    with pytest.raises(TesseraJitError, match="dropout|LSE|effect envelope"):
        fn.native_backward(q, key, value, out_cotangents=dout[:, :2])


def _assert_metal_authority(proof):
    assert proof["implementation"] == "family_plugin"
    assert proof["frontend_authority"] == "tracer"
    assert proof["family"] == "attention_backward"
    assert proof["target_consumer"] == "apple.metal_attention_backward"
    assert proof["compiler_path"] == "apple_gpu_flash_attn_bwd_compiled"
    assert proof["execution_kind"] == "native_gpu"
    assert proof["execution_mode"] == "metal_runtime"
    assert proof["residual_policy"] == "recompute"
    for field in ("source_graph_ir_digest", "schedule_artifact_hash",
                  "tile_program_digest", "native_image_digest", "artifact_hash"):
        assert len(proof[field]) == 64
    attestation = proof["physical_attestation"]
    assert attestation["device_arch"] == "apple7"
    assert attestation["artifact_hash"] == proof["artifact_hash"]
    assert proof["execution_certificate"]["evidence_scope"] == "exact_device"


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("storage", ["fp32", "fp16", "bf16"])
def test_gqa_attention_vjp_runs_on_the_metal_backward(storage):
    _require_metal()
    q, key, value, dout = _gqa_inputs(_storage(storage))
    actual = _gqa.native_backward(q, key, value, out_cotangents=dout)
    expected = reference_attention_backward_split_reduced(
        *(array.astype(np.float32) for array in (dout, q, key, value)),
        split_count=2, scale=0.25, causal=True)
    for observed, reference in zip(actual, expected, strict=True):
        assert observed.dtype == np.float32
        np.testing.assert_allclose(observed, reference, rtol=5e-3, atol=1e-4)
    _assert_metal_authority(_gqa.last_backward_execution)


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("storage", ["fp32", "fp16"])
def test_biased_attention_vjp_binds_an_fp32_bias(storage):
    _require_metal()
    dtype = _storage(storage)
    rng = np.random.default_rng(7)
    q = rng.normal(0.0, 0.2, (1, 2, 7, 16)).astype(dtype)
    key = rng.normal(0.0, 0.2, (1, 2, 5, 16)).astype(dtype)
    value = rng.normal(0.0, 0.2, key.shape).astype(dtype)
    bias = rng.normal(0.0, 0.2, (1, 2, 7, 5)).astype(np.float32)
    dout = rng.normal(0.0, 0.2, q.shape).astype(dtype)
    actual = _biased.native_backward(q, key, value, bias, out_cotangents=dout)
    expected = reference_attention_backward_split_reduced(
        *(array.astype(np.float32) for array in (dout, q, key, value)),
        split_count=2, scale=0.25, bias=bias, causal=False)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, reference, rtol=5e-3, atol=1e-4)
    _assert_metal_authority(_biased.last_backward_execution)


@pytest.mark.hardware_apple_gpu
def test_a_launch_without_its_own_attestation_is_not_exact_device(monkeypatch):
    # A ``launch`` test double reports ok/native_gpu but ran no Metal kernel
    # and carries no launch-origin attestation: the package proof must stay
    # runtime-unattested rather than be manufactured from device availability.
    _require_metal()
    from tessera import runtime

    def fake_launch(artifact, args, stream=None):
        return {"ok": True, "execution_kind": "native_gpu"}

    monkeypatch.setattr(runtime, "launch", fake_launch)
    q, key, value, dout = _gqa_inputs(np.float32)
    _gqa.native_backward(q, key, value, out_cotangents=dout)
    proof = _gqa.last_backward_execution
    assert proof["physical_attestation"] is None
    assert proof["execution_certificate"]["evidence_scope"] == "runtime_unattested"
