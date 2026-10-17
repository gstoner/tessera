"""Owning native half-JVP member proof; public program integration is separate."""
import inspect
import os
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.native_attention_jvp import _lower_graph
from tessera.compiler.native_gpu_storage import build_native_gpu_storage
from tessera.compiler.native_storage_contract import generate_tensor_binding
from benchmarks.nvidia.benchmark_jit_attention_vjp import download
from tests.device.nvidia.test_saved_lse_half_backward_kernel import compare

pytestmark = [pytest.mark.hardware_nvidia, pytest.mark.compiler_route]


def graph(storage, output_storage, causal):
    q = f"tensor<1x2x3x16x{storage}>"
    kv = f"tensor<1x1x5x16x{storage}>"
    o = f"tensor<1x2x3x16x{output_storage}>"
    lse = "tensor<1x2x3xf32>"
    policy = f"scale = 2.5e-1 : f32, causal = {str(causal).lower()}"
    return f"""module attributes {{tessera.target = "nvidia_sm120", tessera.arch = "sm_120"}} {{
      func.func @half_jvp(%q: {q}, %k: {kv}, %v: {kv}, %dq: {q}, %dk: {kv}, %dv: {kv}) -> {o} {{
        %o, %lse = "tessera_attn.checkpoint_forward"(%q,%k,%v) {{{policy}}}
          : ({q},{kv},{kv}) -> ({o},{lse})
        %t = "tessera_attn.checkpoint_jvp"(%q,%k,%v,%o,%lse,%dq,%dk,%dv) {{{policy}}}
          : ({q},{kv},{kv},{o},{lse},{q},{kv},{kv}) -> {o}
        return %t : {o}
      }}
    }}"""


def inputs(storage, output, causal):
    rng = np.random.default_rng(20261017)
    roots = [rng.normal(0, 1.2, shape).astype(storage) for shape in
             ((1, 2, 3, 16), (1, 1, 5, 16), (1, 1, 5, 16))]
    directions = [rng.normal(0, .4, x.shape).astype(storage) for x in roots]
    q, k, v = (x.astype(np.float64) for x in roots)
    dq, dk, dv = (x.astype(np.float64) for x in directions)
    k, v, dk, dv = (np.repeat(x, 2, axis=1) for x in (k, v, dk, dv))
    scores = .25 * (q @ k.swapaxes(-1, -2))
    if causal:
        allowed = np.arange(5)[None, :] <= np.arange(3)[:, None] + 2
        scores = np.where(allowed, scores, -np.inf)
    maximum = scores.max(axis=-1, keepdims=True)
    exponent = np.exp(scores-maximum)
    denominator = exponent.sum(axis=-1, keepdims=True)
    probability = exponent / denominator
    primal = probability @ v
    lse = (maximum+np.log(denominator))[..., 0]
    ds = .25 * (dq @ k.swapaxes(-1, -2) + q @ dk.swapaxes(-1, -2))
    moment = np.sum(probability*ds, axis=-1, keepdims=True)
    product = (probability*ds) @ v + probability @ dv
    expected = product-primal*moment
    wrong = product-primal.astype(output).astype(np.float64)*moment
    return (*roots, primal.astype(output), lse.astype(np.float32), *directions), expected, wrong


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("wide_output", [False, True])
def test_native_typed_jvp_uses_correct_storage_and_unrounded_value_moment(
    storage, causal, wide_output, compiler_toolchain,
):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    compiler = compiler_toolchain.require_tessera_opt("tessera-graph-to-schedule", "tessera-schedule-to-tile")
    llvm_bin = Path(os.environ.get("TESSERA_LLVM_BIN", ""))
    if not (llvm_bin / "mlir-opt").is_file():
        pytest.skip("matched LLVM/MLIR tools required")
    dtype = np.dtype(np.float16 if storage == "f16" else ml_dtypes.bfloat16)
    output = np.dtype("float32") if wide_output else dtype
    lowered = _lower_graph(graph(storage, "f32" if wide_output else storage, causal), compiler=compiler)
    assert lowered.count("tile.alloc_shared") == (2 if wide_output else 3)
    if not wide_output:
        assert "cooperative_saved_lse_unrounded_value_moments_v1" in lowered
    package = build_native_gpu_storage(lowered, compiler=compiler, llvm_bin=llvm_bin,
        backend="nvidia", chip="sm_120", toolkit=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")))
    call = generate_tensor_binding(package, inspect.signature(
        lambda q, k, v, primal, lse, dq, dk, dv, tangent, scratch: None))
    host, expected, wrong = inputs(dtype, output, causal)
    if not wide_output:
        rounded = expected.astype(output)
        center = rounded.astype(np.float64)
        upper = np.nextafter(rounded, output.type(np.inf)).astype(np.float64)
        lower = np.nextafter(rounded, output.type(-np.inf)).astype(np.float64)
        budget = np.maximum(upper-center, center-lower)+3e-6
        assert np.any(np.abs(wrong.astype(output).astype(np.float64)-center) > budget)
    with NvidiaDeviceSession() as session:
        roots = tuple(session.upload(x) for x in host)
        result = session.empty(expected.shape, output)
        call(*roots, result, 128)
        actual = download(session, result)
        assert actual.dtype == output
        compare(actual, expected)
        altered = session.upload((host[3].astype(np.float32)+.5).astype(output))
        call(*roots[:3], altered, *roots[4:], result, 128)
        changed = download(session, result)
        if wide_output:
            assert not np.array_equal(actual, changed)
        else:
            np.testing.assert_array_equal(actual.view(np.uint8), changed.view(np.uint8))
