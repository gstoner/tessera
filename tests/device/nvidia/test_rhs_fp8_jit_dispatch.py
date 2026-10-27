"""SM120-W11-FP8-RHS-20261026: ordinary JIT through the scaled FP8 RHS edge.

``matmul(lhs, dequantize_scaled(quantize_scaled(rmsnorm(source), s), s))``
executes as two native packages sharing one FP8 edge and one scale."""
import json

import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tests._support.nvidia import nvidia_cuda_host_ready

pytestmark = pytest.mark.skipif(not nvidia_cuda_host_ready(), reason="owning NVIDIA host required")


@ts.jit(target="nvidia_sm120")
def fp8_e4m3_rhs(lhs, source, scale):
    codes = ts.ops.quantize_scaled(ts.ops.rmsnorm(source, eps=1e-5), scale, format="fp8_e4m3")
    return ts.ops.matmul(lhs, ts.ops.dequantize_scaled(codes, scale), output_dtype="fp32")


@ts.jit(target="nvidia_sm120")
def fp8_e5m2_rhs(scale, source, lhs):
    codes = ts.ops.quantize_scaled(ts.ops.rmsnorm(source, eps=1e-5), scale, format="fp8_e5m2")
    return ts.ops.matmul(lhs, ts.ops.dequantize_scaled(codes, scale), output_dtype="fp32")


def call(function, a, x, s):
    return function(a, x, s) if function is fp8_e4m3_rhs else function(s, x, a)


@pytest.mark.parametrize("function", [fp8_e4m3_rhs, fp8_e5m2_rhs])
@pytest.mark.parametrize("dtype", ["fp16"])
@pytest.mark.parametrize("shape", [(16, 16, 8), (17, 35, 19), (64, 256, 64)])
def test_ordinary_jit_fp8_rhs_native_cached_and_portable(function, dtype, shape, monkeypatch):
    storage = np.dtype(np.float16 if dtype == "fp16" else ml_dtypes.bfloat16)
    m, k, n = shape
    rng = np.random.default_rng(20261026)
    a = (rng.normal(size=(m, k)) * .2).astype(storage)
    x = (rng.normal(size=(k, n)) * .2).astype(storage)
    s = np.full((1, 1), 0.0625, np.float32)
    actual = call(function, a, x, s)
    assert actual.dtype == np.float32 and actual.shape == (m, n)
    assert function.execution_kind == "native_gpu"
    assert all(r["execution_kind"] == "native_gpu" and r["ok"] for r in function._nvidia_rhs_last_receipts)
    producer, consumer = function.native_rhs_packages()
    fmt = "fp8_e4m3" if function is fp8_e4m3_rhs else "fp8_e5m2"
    assert f"_to_{fmt}_" in producer.descriptor.entry_symbol
    assert consumer.descriptor.provenance["b_codes"] == fmt
    # The program's own FP8 edge, re-produced by its producer package.
    codes = np.empty((k, n), ml_dtypes.float8_e4m3fn if fmt == "fp8_e4m3" else ml_dtypes.float8_e5m2)
    receipt = rt.launch(rt.RuntimeArtifact(metadata={"target": "nvidia_sm120"}, native_image=producer.image,
                                           launch_descriptor=producer.descriptor, tile_ir=producer.tile_ir,
                                           target_ir=producer.target_ir),
                        {producer.descriptor.buffers[0].name: x, producer.descriptor.buffers[1].name: s,
                         producer.descriptor.buffers[2].name: codes, "Rows": k, "Columns": n})
    assert receipt["ok"], receipt
    decoded = ts.ops.dequantize_scaled(codes, s).astype(np.float64)
    expected = a.astype(np.float64) @ decoded
    bound = 1e-5 * (np.abs(a.astype(np.float64)) @ np.abs(decoded)) + 1e-30
    assert np.all(np.abs(actual - expected) <= bound)
    # Against the eager composition the FP8 codes agree except at rare rounding
    # ties moved by the norm's f32 reduction order.
    xf = x.astype(np.float64)
    norm = (xf / np.sqrt(np.mean(xf * xf, axis=1, keepdims=True) + 1e-5)).astype(storage)
    eager = ts.ops.quantize_scaled(norm, s, format=fmt)
    assert np.mean(eager.view(np.uint8) == codes.view(np.uint8)) >= 0.995
    artifact = function.runtime_artifact()
    assert artifact.metadata["compiler_path"] == "canonical_nvidia_rhs_program"
    assert artifact.metadata["native_program"]["schema"] == "tessera.nvidia.norm_rhs_program.v3"
    replay = rt.launch(rt.RuntimeArtifact.from_dict(json.loads(json.dumps(artifact.to_dict()))),
                       dict(zip(function.arg_names, (a, x, s) if function is fp8_e4m3_rhs else (s, x, a))))
    assert replay["ok"] and replay["execution_kind"] == "native_gpu", replay
    np.testing.assert_array_equal(replay["output"], actual)

    def unavailable(*args, **kwargs):
        raise AssertionError("eager body or recompilation was invoked")
    monkeypatch.setattr(function, "_fn", unavailable)
    monkeypatch.setattr(function, "compile_native_rhs_matmul", unavailable)
    np.testing.assert_array_equal(call(function, a, x, s), actual)


def test_fp8_rhs_scale_must_be_the_shared_f32_per_tensor_scale():
    a = np.zeros((16, 16), np.float16)
    x = np.ones((16, 8), np.float16)
    # A non-f32 scale is not the native edge; the reference refuses it.
    from tessera._jit_boundary import TesseraJitError
    with pytest.raises((ValueError, TesseraJitError), match="f32 scale"):
        fp8_e4m3_rhs(a, x, np.ones((1, 1), np.float16))


def test_bf16_fp8_edge_runs_the_eager_composition():
    """The fused FP8 consumer admits f16 only (its factored scale is
    overflow-safe for f16); a bf16 FP8 edge falls back to the reference."""
    rng = np.random.default_rng(3)
    a = (rng.normal(size=(16, 32)) * .2).astype(ml_dtypes.bfloat16)
    x = (rng.normal(size=(32, 8)) * .2).astype(ml_dtypes.bfloat16)
    s = np.full((1, 1), 0.0625, np.float32)
    actual = fp8_e4m3_rhs(a, x, s)
    assert fp8_e4m3_rhs.execution_kind != "native_gpu"
    assert fp8_e4m3_rhs.native_rhs_packages() == ()
    codes = ts.ops.quantize_scaled(ts.ops.rmsnorm(x, eps=1e-5), s, format="fp8_e4m3")
    expected = ts.ops.matmul(a, ts.ops.dequantize_scaled(codes, s), output_dtype="fp32")
    np.testing.assert_allclose(np.asarray(actual, np.float64), np.asarray(expected, np.float64), rtol=1e-5, atol=1e-6)

