"""The @jit front door for 8/4-bit storage tensors on Apple GPU.

Before 2026-09-15 an FP8 or FP4 program traced as an f32 program (the tracer
named every unknown numpy dtype "f32"), the Graph IR spelling of the
low-precision types was unparseable (`xxf8E4M3FN`, `!tessera.fp4_e2m1`), and
the MPS matmul dispatcher, finding no lane, computed the product on the host
with numpy while the artifact reported native_gpu. This file pins the three
fixes: trace-time dtype naming, MLIR-builtin spellings with an fp32 matmul
result (storage-only dtypes, Decision #15a), and a real Metal lane -- the
strided-view MPP matmul2d the compiled route also uses -- for every
low-precision pair. Execution rows run on the owning Mac and compare against
a float64 oracle built from the exact quantized bytes.
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.graph_ir import _shape_matmul_2d, tensor_ir_type
from tessera.compiler.trace import _np_dtype_to_elem

ml_dtypes = pytest.importorskip("ml_dtypes")

LOWP = {"fp8_e4m3": ml_dtypes.float8_e4m3fn, "fp8_e5m2": ml_dtypes.float8_e5m2,
        "fp4_e2m1": ml_dtypes.float4_e2m1fn}


# ---------------------------------------------------------------- host-free

@pytest.mark.parametrize("canonical,dt", sorted(LOWP.items()))
def test_tracer_names_low_precision_arrays_by_their_storage_dtype(canonical, dt):
    assert _np_dtype_to_elem(np.dtype(dt)) == canonical


def test_tracer_still_names_the_16_and_32_bit_dtypes():
    assert _np_dtype_to_elem(np.dtype(np.float16)) == "f16"
    assert _np_dtype_to_elem(np.dtype(ml_dtypes.bfloat16)) == "bf16"
    assert _np_dtype_to_elem(np.dtype(np.float32)) == "f32"


@pytest.mark.parametrize("canonical,spelling", [("fp8_e4m3", "f8E4M3FN"), ("fp8_e5m2", "f8E5M2"),
                                                ("fp4_e2m1", "f4E2M1FN")])
def test_graph_ir_spells_low_precision_types_as_mlir_builtins(canonical, spelling):
    from tessera.compiler.graph_ir import _parse_mlir_tensor_type
    text = str(tensor_ir_type((64, 256), canonical))
    assert text == f"tensor<64x256x{spelling}>"
    assert _parse_mlir_tensor_type(text).dtype == canonical  # round trip through the reverse table


@pytest.mark.parametrize("a,b", [("fp8_e4m3", "fp8_e4m3"), ("fp4_e2m1", "fp4_e2m1"), ("fp16", "fp8_e4m3"),
                                 ("fp16", "fp4_e2m1")])
def test_matmul_over_low_precision_storage_results_in_fp32(a, b):
    out = _shape_matmul_2d([tensor_ir_type((64, 256), a), tensor_ir_type((256, 128), b)])
    assert out.dtype == "fp32" and str(out) == "tensor<64x128xf32>"


def test_matmul_over_16_bit_storage_keeps_its_result_dtype():
    out = _shape_matmul_2d([tensor_ir_type((64, 256), "fp16"), tensor_ir_type((256, 128), "fp16")])
    assert out.dtype == "fp16"


def test_low_precision_program_lowers_to_the_matmul2d_view_call(compiler_toolchain):
    """A traced FP8 program reaches the default Apple pipeline as an executable
    mtl4_matmul2d value call (no MPSGraph claim). Host-free: tessera-opt only."""
    import subprocess
    from tessera import ops
    from tessera.compiler.trace import to_graph_ir_module, trace
    a = np.zeros((64, 256), ml_dtypes.float8_e4m3fn)
    b = np.zeros((256, 128), ml_dtypes.float8_e4m3fn)
    traced = trace(lambda x, y: ops.matmul(x, y), a, b, arg_names=("a", "b"))
    text = to_graph_ir_module(traced, name="f", target="apple_gpu").to_mlir(canonical=True, target="apple_gpu")
    assert "tensor<64x256xf8E4M3FN>" in text and "tensor<64x128xf32>" in text
    opt = compiler_toolchain.require_tessera_opt("tessera-tiling", "tessera-lower-to-apple_gpu")
    proc = subprocess.run([str(opt), "-", "--tessera-tiling", "--tessera-lower-to-apple_gpu", "--allow-unregistered-dialect"],
                          input=text, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert 'op_kind = "mtl4_matmul2d"' in proc.stdout and "matmul_contract" not in proc.stdout


# ---------------------------------------------------------------- owning-Mac execution

def _operand(rng, dt, shape):
    if dt is np.float16:
        return (rng.standard_normal(shape) * 0.25).astype(np.float16)
    if dt is ml_dtypes.float4_e2m1fn:
        return rng.integers(0, 16, size=shape, dtype=np.uint8).view(dt)
    c = rng.integers(0, 256, size=shape, dtype=np.uint8)
    c = ((c & 0x87) | (((c >> 3) & 0xF) % 12) << 3) if dt is ml_dtypes.float8_e4m3fn \
        else ((c & 0x83) | (((c >> 2) & 0x1F) % 24) << 2)
    return c.astype(np.uint8).view(dt)


@pytest.mark.metal4
@pytest.mark.parametrize("a_dt,b_dt", [(ml_dtypes.float8_e4m3fn, ml_dtypes.float8_e4m3fn),
                                       (ml_dtypes.float8_e5m2, ml_dtypes.float8_e5m2),
                                       (ml_dtypes.float4_e2m1fn, ml_dtypes.float4_e2m1fn),
                                       (np.float16, ml_dtypes.float8_e4m3fn),
                                       (np.float16, ml_dtypes.float4_e2m1fn)])
def test_jit_low_precision_matmul_runs_on_the_metal_view_lane(monkeypatch, a_dt, b_dt):
    import tessera
    from tessera import ops, runtime as R
    if not R._apple_gpu_mtl4_matmul2d_lane_available():
        pytest.fail("Metal 4 is up but the runtime lacks the strided-view matmul2d symbols: rebuild the dylib")
    calls: list[int] = []
    real = R.apple_gpu_mtl4_matmul2d_view

    def spy(*args, **kwargs):
        calls.append(int(kwargs["pair"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(R, "apple_gpu_mtl4_matmul2d_view", spy)

    @tessera.jit(target="apple_gpu")
    def f(a, b):
        return ops.matmul(a, b)

    rng = np.random.default_rng(3)
    K, N = 256, 256
    A, B = _operand(rng, a_dt, (100, K)), _operand(rng, b_dt, (K, N))
    out = np.asarray(f(A, B))
    assert calls, "the product did not go through the Metal strided-view lane"
    assert out.dtype == np.float32 and out.shape == (100, N)
    ref = A.astype(np.float64) @ B.astype(np.float64)
    assert np.abs(out.astype(np.float64) - ref).max() <= 4e-6 * (np.abs(ref).max() + 1.0)
    md = f.runtime_artifact().metadata or {}
    assert md.get("execution_kind") == "native_gpu"


@pytest.mark.metal4
def test_jit_f16_matmul_is_unchanged_by_the_low_precision_lane(monkeypatch):
    import tessera
    from tessera import ops, runtime as R
    calls: list[int] = []
    real = R.apple_gpu_mtl4_matmul2d_view
    monkeypatch.setattr(R, "apple_gpu_mtl4_matmul2d_view",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    @tessera.jit(target="apple_gpu")
    def f(a, b):
        return ops.matmul(a, b)

    rng = np.random.default_rng(4)
    A, B = _operand(rng, np.float16, (64, 256)), _operand(rng, np.float16, (256, 128))
    out = np.asarray(f(A, B))
    assert not calls and out.dtype == np.float16
    ref = A.astype(np.float64) @ B.astype(np.float64)
    assert np.abs(out.astype(np.float64) - ref).max() <= 3e-2 * (np.abs(ref).max() + 1.0)


@pytest.mark.metal4
def test_dispatcher_refuses_a_dtype_without_a_lane_under_strict_dispatch(monkeypatch):
    from tessera import runtime as R
    monkeypatch.setenv("TESSERA_STRICT_DISPATCH", "1")
    a = np.ones((8, 8), np.float64)
    with pytest.raises(Exception, match="no Metal matmul lane"):
        R._apple_gpu_dispatch_matmul("tessera.matmul", [a, a], np)
