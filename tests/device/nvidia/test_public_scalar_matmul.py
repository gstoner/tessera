"""Original public trace -> native Schedule/Tile -> checked scalar owner."""
import subprocess

import numpy as np
import pytest
import tessera as ts

from tessera import runtime as rt

pytestmark = pytest.mark.hardware_nvidia


def fp32_product(a, b):
    return ts.ops.matmul(a, b, output_dtype="fp32")


def fp16_product(a, b):
    return ts.ops.matmul(a, b, output_dtype="fp16")


def fp32_reordered(b, a):
    return ts.ops.matmul(a, b, output_dtype="fp32")


def fp16_reordered(b, a):
    return ts.ops.matmul(a, b, output_dtype="fp16")


@pytest.mark.parametrize("half", ["fp16", "bf16"])
@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("output", ["fp32", "fp16"])
@pytest.mark.parametrize("permuted", [False, True])
def test_public_scalar_matmul_reuses_native_owner(half, side, output, permuted, monkeypatch):
    import ml_dtypes
    assert rt._nvidia_device_name() == "sm_120"
    dtype = np.float16 if half == "fp16" else ml_dtypes.bfloat16
    a_dtype, b_dtype = (np.float32, dtype) if side == "a" else (dtype, np.float32)
    rng = np.random.default_rng(931)
    a = rng.normal(0, .25, (17, 35)).astype(a_dtype)
    b = rng.normal(0, .25, (35, 19)).astype(b_dtype)
    source = {(False, "fp32"): fp32_product, (False, "fp16"): fp16_product,
              (True, "fp32"): fp32_reordered, (True, "fp16"): fp16_reordered}[permuted, output]
    fn = ts.jit(target="nvidia_sm120")(source)
    try:
        for index in range(2):
            aa = (a.astype(np.float32) + index * .0625).astype(a_dtype)
            bb = (b.astype(np.float32) - index * .03125).astype(b_dtype)
            roots = (bb, aa) if permuted else (aa, bb)
            expected = aa.astype(np.float64) @ bb.astype(np.float64)
            out_dtype = np.float32 if output == "fp32" else np.float16
            expected = expected.astype(out_dtype)
            if index:
                with monkeypatch.context() as warm:
                    warm.setattr(subprocess, "run", lambda *a, **k: pytest.fail("warm call compiled"))
                    warm.setattr(fn, "_trace_frontend_capture", lambda *a, **k: pytest.fail("warm call retraced"))
                    actual = fn(*roots)
            else:
                actual = fn(*roots)
            assert fn.last_fallback_reason is None
            receipt = fn._native_descriptor_last_receipt
            assert receipt["execution_kind"] == "native_gpu"
            assert receipt["native_call_binding"] == "prepared_cpp_scalar_matmul"
            magnitude = np.abs(aa.astype(np.float64)) @ np.abs(bb.astype(np.float64))
            gamma = 35 * np.finfo(np.float32).eps / (1 - 35 * np.finfo(np.float32).eps)
            upper = np.nextafter(expected, out_dtype(np.inf)).astype(np.float64)
            bound = gamma * magnitude + np.abs(upper - expected.astype(np.float64)) + 1e-8
            assert actual.dtype == out_dtype and np.isfinite(actual).all()
            assert np.all(np.abs(actual.astype(np.float64) - expected.astype(np.float64)) <= bound)
        artifact = rt.RuntimeArtifact.from_json(fn._cached_artifact.to_json())
        descriptor = artifact.launch_descriptor
        out = np.empty((17, 19), out_dtype)
        by_frontend = dict(zip((arg.name for arg in fn.graph_ir.functions[0].args), roots, strict=True))
        arguments = {"buffers": {**by_frontend, descriptor.buffers[-1].name: out},
                     "scalars": {"M": 17, "N": 19, "K": 35}}
        fn.close_native_storage()
        with monkeypatch.context() as replay:
            replay.setattr(subprocess, "run", lambda *a, **k: pytest.fail("portable replay compiled"))
            result = rt.launch(artifact, arguments)
        assert result["ok"], result
        assert np.all(np.abs(out.astype(np.float64) - expected.astype(np.float64)) <= bound)
    finally:
        fn.close_native_storage()


def default_product(a, b):
    return ts.ops.matmul(a, b)


@pytest.mark.parametrize("half", ["fp16", "bf16"])
@pytest.mark.parametrize("side", ["a", "b"])
def test_public_scalar_default_preserves_wide_dtype_and_values(half, side):
    import ml_dtypes
    narrow = np.float16 if half == "fp16" else ml_dtypes.bfloat16
    a_dtype, b_dtype = (np.float32, narrow) if side == "a" else (narrow, np.float32)
    a, b = np.zeros((17, 35), a_dtype), np.zeros((35, 19), b_dtype)
    delta = 2.0 ** (-12 if half == "fp16" else -9)
    if side == "a":
        a[:, 0], a[:, 1], b[:2, :] = 1 + delta, -1, 1
    else:
        a[:, :2], b[0, :], b[1, :] = 1, 1 + delta, -1
    fn = ts.jit(target="nvidia_sm120")(default_product)
    try:
        actual = fn(a, b)
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, np.full((17, 19), delta, np.float32))
        assert fn._native_descriptor_last_receipt["native_call_binding"] == "prepared_cpp_scalar_matmul"
    finally:
        fn.close_native_storage()
