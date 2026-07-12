"""Compiler-generated WMMA GEMM with a FUSED EPILOGUE on gfx1151.

The `tessera_rocm.wmma_gemm` directive gains two epilogue knobs that the
`generate-wmma-gemm-kernel` pass fuses onto the f32 accumulator *before the
store* — no intermediate D round-trip:

  * ``bias = true``     — add a per-output-column bias (a trailing memref<?>
                          operand of length N, accumulator dtype).
  * ``activation = ...``— a pointwise activation: ``relu`` (max(x,0)),
                          ``gelu`` (tanh approximation), or ``silu`` (x·σ(x)).

The activation transcendentals (exp/tanh) lower through the same `math` → ROCDL
(`__ocml_*`) path the flash_attn softmax uses, so they execute natively on the
APU. This is the `fused_epilogue` Target-IR node: matmul + bias + activation in
one kernel. Here we build the kernel in-process (tessera-opt, no mlir-opt),
launch the hsaco via HIP, and compare to a numpy gemm+bias+activation oracle.

Skip-clean: tessera-opt not built, or no usable AMD GPU.
"""

from __future__ import annotations

import ctypes

import pytest

np = pytest.importorskip("numpy")


def _rt_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _act_ref(x, activation):
    if activation == "none":
        return x
    if activation == "relu":
        return np.maximum(x, 0.0)
    if activation == "silu":
        return x / (1.0 + np.exp(-x))
    if activation == "gelu":  # tanh approximation (matches the kernel)
        c = np.sqrt(2.0 / np.pi)
        return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))
    raise AssertionError(activation)


def _launch_epilogue(rt, a, b, bias, activation):
    """Build + launch the fused-epilogue GEMM (f16 storage, f32 accumulate),
    returning the MxN f32 result. Mirrors the device_verified_jit-GEMM HIP launch, with the
    optional per-column bias appended as the trailing memref argument."""
    m, k = a.shape
    n = b.shape[1]
    mt, nt = rt._rocm_prod_tile(m, n, k)
    has_bias = bias is not None
    hsaco = rt._build_compiled_gemm_hsaco(
        mt, nt, "f16", bias=has_bias, activation=activation)

    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        pytest.skip("libamdhip64 not loadable")
    mod = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(mod), hsaco) != 0:
        pytest.skip("no usable AMD GPU (module load failed)")
    fn = ctypes.c_void_p()
    assert hip.hipModuleGetFunction(ctypes.byref(fn), mod, b"gemm") == 0

    a_c = np.ascontiguousarray(a, dtype=np.float16)
    b_c = np.ascontiguousarray(b, dtype=np.float16)
    d = np.zeros((m, n), dtype=np.float32)
    da, db, dd = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    nbytes = (2 * m * k, 2 * k * n, 4 * m * n)
    for dev, size in ((da, nbytes[0]), (db, nbytes[1]), (dd, nbytes[2])):
        assert hip.hipMalloc(ctypes.byref(dev), size) == 0
    hip.hipMemcpy(da, a_c.ctypes.data_as(ctypes.c_void_p), nbytes[0], 1)
    hip.hipMemcpy(db, b_c.ctypes.data_as(ctypes.c_void_p), nbytes[1], 1)

    dbias = ctypes.c_void_p()
    bias_c = None
    if has_bias:
        bias_c = np.ascontiguousarray(bias, dtype=np.float32)
        assert hip.hipMalloc(ctypes.byref(dbias), 4 * n) == 0
        hip.hipMemcpy(dbias, bias_c.ctypes.data_as(ctypes.c_void_p), 4 * n, 1)

    def _mr(p, size):
        return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
                ctypes.c_int64(0), ctypes.c_int64(size), ctypes.c_int64(1)]

    launch_args = (_mr(da, m * k) + _mr(db, k * n) + _mr(dd, m * n)
                   + [ctypes.c_int64(m), ctypes.c_int64(n), ctypes.c_int64(k)])
    if has_bias:
        launch_args += _mr(dbias, n)
    arr = (ctypes.c_void_p * len(launch_args))()
    for i, val in enumerate(launch_args):
        arr[i] = ctypes.cast(ctypes.byref(val), ctypes.c_void_p)
    gx = (n + 16 * nt - 1) // (16 * nt)
    gy = (m + 16 * mt - 1) // (16 * mt)
    rc = hip.hipModuleLaunchKernel(fn, gx, gy, 1, 32, 1, 1, 0, None, arr, None)
    assert rc == 0, f"launch failed rc={rc}"
    hip.hipDeviceSynchronize()
    hip.hipMemcpy(d.ctypes.data_as(ctypes.c_void_p), dd, nbytes[2], 2)
    for dev in (da, db, dd):
        hip.hipFree(dev)
    if has_bias:
        hip.hipFree(dbias)
    return d


@pytest.mark.parametrize("activation", ["none", "relu", "gelu", "silu"])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("m,n,k", [(16, 16, 16), (64, 48, 32), (96, 80, 64)])
def test_fused_epilogue_matches_numpy(activation, with_bias, m, n, k):
    if activation == "none" and not with_bias:
        pytest.skip("plain GEMM is covered by the matmul lane test")
    rt = _rt_or_skip()
    rng = np.random.default_rng(7 + m + n + k + len(activation) + int(with_bias))
    a = (rng.standard_normal((m, k)) * 0.4).astype(np.float16)
    b = (rng.standard_normal((k, n)) * 0.4).astype(np.float16)
    bias = ((rng.standard_normal((n,)) * 0.5).astype(np.float32)
            if with_bias else None)

    out = _launch_epilogue(rt, a, b, bias, activation)
    assert out.shape == (m, n)

    ref = a.astype(np.float32) @ b.astype(np.float32)
    if with_bias:
        ref = ref + bias[None, :]
    ref = _act_ref(ref, activation)
    maxerr = float(np.max(np.abs(out - ref)))
    assert maxerr < 5e-2, (
        f"fused epilogue act={activation} bias={with_bias} "
        f"{m}x{n}x{k} maxerr={maxerr}")
