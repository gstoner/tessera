"""APPLE-MATMUL2D-1: the canonical GEMM reduction on the Metal 4 cooperative-tensor lane.

Graph IR matmul -> shared TilingPass -> `tessera_apple.gpu.tensor_view` +
`gpu.matmul2d` (storage pair, fp32 accumulator, MTLTensor layout quantum in
verified IR) -> optional `gpu.matmul2d_epilogue` fusion -> `gpu.kernel_call`
on the runtime's strided-view matmul2d symbol -> the value-lane dispatcher,
which consumes the projected view ABI (pair code, extents, row strides, byte
offsets, activation) and never re-derives layout. Lowering tests are
host-free; execution rows run on the owning Mac and compare against a float64
oracle built from the exact quantized bytes. Recognition is not promotion:
nothing here changes route selection.
"""
from __future__ import annotations

import re
import subprocess

import numpy as np
import pytest

from tests._support.environment import CompilerToolchain

PASSES = ("tessera-tiling", "tessera-apple-canonical-gemm-matmul2d",
          "tessera-apple-matmul2d-fuse-epilogue", "tessera-apple-matmul2d-to-call")
VIEW = "tessera_apple_gpu_mtl4_matmul2d_view"
VIEW_EPI = VIEW + "_epilogue"
PAIR = {("f8E4M3FN", "f8E4M3FN"): 0, ("f8E5M2", "f8E5M2"): 1, ("f4E2M1FN", "f4E2M1FN"): 2,
        ("f16", "f8E4M3FN"): 3, ("f16", "f8E5M2"): 4, ("f16", "f4E2M1FN"): 5,
        ("f16", "f16"): 10, ("bf16", "bf16"): 11}


def _gemm_module(m: int, k: int, n: int, a_storage: str, b_storage: str) -> str:
    return f"""module {{
  func.func @gemm(%a: tensor<{m}x{k}x{a_storage}>, %b: tensor<{k}x{n}x{b_storage}>) -> tensor<{m}x{n}xf32> {{
    %0 = "tessera.matmul"(%a, %b) : (tensor<{m}x{k}x{a_storage}>, tensor<{k}x{n}x{b_storage}>) -> tensor<{m}x{n}xf32>
    return %0 : tensor<{m}x{n}xf32>
  }}
}}"""


def _sub_block_module(pa: tuple[int, int], pb: tuple[int, int], a_org: tuple[int, int], b_org: tuple[int, int],
                      m: int, k: int, n: int, a_storage: str, b_storage: str) -> str:
    """A GEMM over static unit-stride windows of larger matrices."""
    return f"""module {{
  func.func @gemm(%pa: tensor<{pa[0]}x{pa[1]}x{a_storage}>, %pb: tensor<{pb[0]}x{pb[1]}x{b_storage}>) -> tensor<{m}x{n}xf32> {{
    %a = tensor.extract_slice %pa[{a_org[0]}, {a_org[1]}] [{m}, {k}] [1, 1] : tensor<{pa[0]}x{pa[1]}x{a_storage}> to tensor<{m}x{k}x{a_storage}>
    %b = tensor.extract_slice %pb[{b_org[0]}, {b_org[1]}] [{k}, {n}] [1, 1] : tensor<{pb[0]}x{pb[1]}x{b_storage}> to tensor<{k}x{n}x{b_storage}>
    %0 = "tessera.matmul"(%a, %b) : (tensor<{m}x{k}x{a_storage}>, tensor<{k}x{n}x{b_storage}>) -> tensor<{m}x{n}xf32>
    return %0 : tensor<{m}x{n}xf32>
  }}
}}"""


def _epilogue_module(m: int, k: int, n: int, storage: str, *, bias: bool, act: str) -> str:
    """The tracer's epilogue: per-column bias via broadcast + add, then act."""
    args = f"%a: tensor<{m}x{k}x{storage}>, %b: tensor<{k}x{n}x{storage}>" + (f", %bias: tensor<{n}xf32>" if bias else "")
    body = [f'%0 = "tessera.matmul"(%a, %b) : (tensor<{m}x{k}x{storage}>, tensor<{k}x{n}x{storage}>) -> tensor<{m}x{n}xf32>']
    cur = "%0"
    if bias:
        body.append(f'%bb = "tessera.broadcast"(%bias) {{shape = [{m}, {n}]}} : (tensor<{n}xf32>) -> tensor<{m}x{n}xf32>')
        body.append(f'%1 = "tessera.add"({cur}, %bb) : (tensor<{m}x{n}xf32>, tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>')
        cur = "%1"
    if act != "none":
        body.append(f'%2 = "tessera.{act}"({cur}) : (tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>')
        cur = "%2"
    return "module {\n  func.func @mlp(" + args + f") -> tensor<{m}x{n}xf32> {{\n    " + "\n    ".join(body) + \
        f"\n    return {cur} : tensor<{m}x{n}xf32>\n  }}\n}}"


def _lower(toolchain: CompilerToolchain, module: str, *passes: str) -> subprocess.CompletedProcess[str]:
    tessera_opt = toolchain.require_tessera_opt(*passes)
    return subprocess.run([str(tessera_opt), "-", *(f"--{p}" for p in passes), "--allow-unregistered-dialect"],
                          input=module, capture_output=True, text=True)


def _call(stdout: str) -> dict[str, str]:
    lines = [line for line in stdout.splitlines() if "tessera_apple.gpu.kernel_call" in line]
    assert lines, f"no kernel_call in:\n{stdout}"
    return {key: quoted or bare for key, quoted, bare in re.findall(r'([\w.]+) = (?:"([^"]+)"|([^,\s}]+))', lines[0])}


def _lowered_call(toolchain, module) -> tuple[str, dict[str, str]]:
    proc = _lower(toolchain, module, *PASSES)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout, _call(proc.stdout)


# ---------------------------------------------------------------- host-free lowering

def test_f16_canonical_loop_becomes_tensor_views_and_matmul2d(compiler_toolchain: CompilerToolchain):
    proc = _lower(compiler_toolchain, _gemm_module(64, 128, 96, "f16", "f16"), *PASSES[:2])
    assert proc.returncode == 0, proc.stderr
    assert "scf.for" not in proc.stdout
    assert "tessera_apple.gpu.tensor_view %arg0 {byte_offset = 0 : i64, extents = array<i64: 128, 64>, strides = array<i64: 1, 128>}" in proc.stdout
    assert "tessera_apple.gpu.tensor_view %arg1 {byte_offset = 0 : i64, extents = array<i64: 96, 128>, strides = array<i64: 1, 96>}" in proc.stdout
    assert 'tessera_apple.gpu.matmul2d %0, %1 {accumulate = "f32", simdgroups = 4 : i64, tessera_apple.canonical_k_loop = true' in proc.stdout


@pytest.mark.parametrize("a,b", sorted(PAIR, key=PAIR.get))
def test_call_projects_one_symbol_and_the_pair_code_from_ir(compiler_toolchain, a, b):
    """Every admitted pair lowers to the same strided-view symbol; the pair
    code and the storage pair are stated, and a low-precision pair keeps the
    legacy format code for readers of the first slice."""
    n = 256 if "f4" in b else 128
    _, call = _lowered_call(compiler_toolchain, _gemm_module(64, 256, n, a, b))
    assert call["symbol"] == VIEW and call["op_kind"] == "mtl4_matmul2d"
    assert call["abi"] == "mtl4_matmul2d_view" and call["dtype"] == f"{a}x{b}"
    assert int(call["tessera_apple.pair"]) == PAIR[(a, b)]
    assert call["tessera_apple.accumulate"] == "fp32"
    assert (int(call["tessera_apple.a_inner"]), int(call["tessera_apple.a_outer"]), int(call["tessera_apple.a_stride"])) == (256, 64, 256)
    assert (int(call["tessera_apple.b_inner"]), int(call["tessera_apple.b_outer"]), int(call["tessera_apple.b_stride"])) == (n, 256, n)
    assert (int(call["tessera_apple.a_byte_offset"]), int(call["tessera_apple.b_byte_offset"])) == (0, 0)
    if PAIR[(a, b)] < 10:
        assert int(call["tessera_apple.lowp_format"]) == PAIR[(a, b)]
    else:
        assert "tessera_apple.lowp_format" not in call


def test_ragged_m_is_bound_at_its_true_extent_without_host_padding(compiler_toolchain):
    """The tiling pass pads M = 100 to 112; the lowering looks through the
    zero-pad, binds the original operand, and states the runtime's partial-tile
    contract (`ragged_tail`) instead of a padded product plus a slice."""
    proc = _lower(compiler_toolchain, _gemm_module(100, 256, 128, "f8E4M3FN", "f8E4M3FN"), *PASSES[:2])
    assert proc.returncode == 0, proc.stderr
    assert "tensor.insert_slice" not in proc.stdout and "tensor.extract_slice" not in proc.stdout
    assert "extents = array<i64: 256, 100>, strides = array<i64: 1, 256>} : tensor<100x256xf8E4M3FN>" in proc.stdout
    assert "tessera_apple.ragged_tail = true" in proc.stdout and "ragged_zero_pad" not in proc.stdout
    assert "-> tensor<100x128xf32>" in proc.stdout
    stdout, call = _lowered_call(compiler_toolchain, _gemm_module(100, 256, 128, "f8E4M3FN", "f8E4M3FN"))
    assert int(call["tessera_apple.a_outer"]) == 100 and call["tessera_apple.ragged_tail"] == "true"


def test_ragged_n_binds_the_true_column_extent(compiler_toolchain):
    _, call = _lowered_call(compiler_toolchain, _gemm_module(64, 128, 100, "f16", "f16"))
    assert (int(call["tessera_apple.b_inner"]), int(call["tessera_apple.b_stride"])) == (100, 100)
    assert call["tessera_apple.ragged_tail"] == "true"


def test_sub_block_origins_are_views_into_the_parents(compiler_toolchain):
    """A GEMM on windows of larger matrices binds views INTO the parents:
    nonzero byte offsets and the parents' row strides, nothing copied."""
    stdout, call = _lowered_call(compiler_toolchain, _sub_block_module(
        (200, 512), (512, 512), (8, 64), (0, 128), 100, 256, 128, "f16", "f16"))
    assert "tensor.extract_slice" not in stdout
    assert "kernel_call %arg0, %arg1" in stdout  # the parents, not the slices
    assert int(call["tessera_apple.a_byte_offset"]) == (8 * 512 + 64) * 2
    assert int(call["tessera_apple.a_stride"]) == 512 and int(call["tessera_apple.a_outer"]) == 100
    assert int(call["tessera_apple.b_byte_offset"]) == 128 * 2 and int(call["tessera_apple.b_stride"]) == 512


def test_misaligned_fp8_sub_block_origin_is_refused(compiler_toolchain):
    proc = _lower(compiler_toolchain, _sub_block_module(
        (200, 512), (512, 512), (8, 64), (0, 128), 64, 256, 128, "f8E4M3FN", "f8E4M3FN"), *PASSES[:2])
    assert proc.returncode != 0
    assert "APPLE_MATMUL2D_LAYOUT_UNSUPPORTED: operand a" in proc.stderr
    assert "128-byte aligned" in proc.stderr


@pytest.mark.parametrize("bias,act", [(True, "gelu"), (True, "none"), (False, "relu"), (True, "silu")])
def test_bias_and_activation_fuse_into_matmul2d_epilogue(compiler_toolchain, bias, act):
    proc = _lower(compiler_toolchain, _epilogue_module(64, 128, 96, "f16", bias=bias, act=act), *PASSES[:3])
    assert proc.returncode == 0, proc.stderr
    assert "tessera_apple.gpu.matmul2d_epilogue" in proc.stdout
    assert "tessera.add" not in proc.stdout and "tessera.broadcast" not in proc.stdout
    assert f"tessera.{act}" not in proc.stdout
    assert f'act = "{act}"' in proc.stdout
    assert ("bias %arg2 : tensor<96xf32>" in proc.stdout) is bias
    stdout, call = _lowered_call(compiler_toolchain, _epilogue_module(64, 128, 96, "f16", bias=bias, act=act))
    assert call["symbol"] == VIEW_EPI and call["op_kind"] == "mtl4_matmul2d_epilogue"
    assert call["tessera_apple.act"] == act and call["tessera_apple.has_bias"] == str(bias).lower()
    assert ("kernel_call %arg0, %arg1, %arg2" in stdout) is bias


def test_product_with_two_consumers_is_not_fused(compiler_toolchain):
    module = """module {
  func.func @two(%a: tensor<64x128xf16>, %b: tensor<128x96xf16>) -> (tensor<64x96xf32>, tensor<64x96xf32>) {
    %0 = "tessera.matmul"(%a, %b) : (tensor<64x128xf16>, tensor<128x96xf16>) -> tensor<64x96xf32>
    %1 = "tessera.gelu"(%0) : (tensor<64x96xf32>) -> tensor<64x96xf32>
    return %0, %1 : tensor<64x96xf32>, tensor<64x96xf32>
  }
}"""
    proc = _lower(compiler_toolchain, module, *PASSES[:3])
    assert proc.returncode == 0, proc.stderr
    assert "matmul2d_epilogue" not in proc.stdout and "tessera.gelu" in proc.stdout


def test_packed_fp8_off_the_quantum_is_refused_not_reformed(compiler_toolchain):
    proc = _lower(compiler_toolchain, _gemm_module(64, 64, 128, "f8E4M3FN", "f8E4M3FN"), *PASSES[:2])
    assert proc.returncode != 0
    assert "APPLE_MATMUL2D_LAYOUT_UNSUPPORTED: operand a" in proc.stderr
    assert "multiple of 128 bytes" in proc.stderr


def test_mixed_bf16_f16_pair_is_refused(compiler_toolchain):
    proc = _lower(compiler_toolchain, _gemm_module(64, 128, 128, "bf16", "f16"), *PASSES[:2])
    assert proc.returncode != 0 and "APPLE_MATMUL2D_PAIR_UNSUPPORTED" in proc.stderr


def test_f32_storage_is_not_an_mpp_pair(compiler_toolchain):
    proc = _lower(compiler_toolchain, _gemm_module(64, 128, 128, "f32", "f32"), *PASSES[:2])
    assert proc.returncode != 0 and "APPLE_MATMUL2D_PAIR_UNSUPPORTED" in proc.stderr


# ---------------------------------------------------------------- owning-Mac execution

def _decode(codes, elem, np):
    from tessera import runtime as R
    fmt = {"f8E4M3FN": "fp8_e4m3", "f8E5M2": "fp8_e5m2", "f4E2M1FN": "fp4_e2m1"}[elem]
    return R.apple_lowp_decode(codes, fmt, np)


def _operand(rng, elem, shape):
    ml = pytest.importorskip("ml_dtypes")
    kinds = {"f16": np.float16, "bf16": ml.bfloat16, "f8E4M3FN": ml.float8_e4m3fn,
             "f8E5M2": ml.float8_e5m2, "f4E2M1FN": ml.float4_e2m1fn}
    if elem in ("f16", "bf16"):
        return (rng.standard_normal(shape) * 0.25).astype(kinds[elem])
    # exact quantized bytes: draw codes with bounded exponents so K-long fp32 dot products stay exact
    if elem == "f4E2M1FN":
        return rng.integers(0, 16, size=shape, dtype=np.uint8).view(ml.float4_e2m1fn)
    c = rng.integers(0, 256, size=shape, dtype=np.uint8)
    c = ((c & 0x87) | (((c >> 3) & 0xF) % 12) << 3) if elem == "f8E4M3FN" else ((c & 0x83) | (((c >> 2) & 0x1F) % 24) << 2)
    return c.astype(np.uint8).view(kinds[elem])


def _require_lane():
    from tessera import runtime as R
    if not R._apple_gpu_mtl4_matmul2d_lane_available():
        pytest.fail("Metal 4 is up but the runtime lacks the strided-view matmul2d symbols: rebuild the dylib")
    return R


def _check(out, ref, a, tol_lowp=4e-6, tol_wide=3e-2):
    assert out.dtype == np.float32 and out.shape == ref.shape
    scale = np.abs(ref).max() + 1.0
    tol = tol_lowp if a not in ("f16", "bf16") else tol_wide
    assert np.abs(out.astype(np.float64) - ref).max() <= tol * scale


@pytest.mark.metal4
@pytest.mark.parametrize("a,b,m", [("f16", "f16", 64), ("bf16", "bf16", 100), ("f8E4M3FN", "f8E4M3FN", 64),
                                   ("f8E5M2", "f8E5M2", 100), ("f4E2M1FN", "f4E2M1FN", 64),
                                   ("f16", "f8E4M3FN", 64), ("f16", "f4E2M1FN", 100)])
def test_lowered_call_executes_through_the_value_lane_and_matches_the_oracle(compiler_toolchain, a, b, m):
    """Ragged M (100) executes at its true extent: the operand is NOT padded on
    the host; the runtime's partial-tile store covers the tail."""
    R = _require_lane()
    K, N = 256, 256
    _, call = _lowered_call(compiler_toolchain, _gemm_module(m, K, N, a, b))
    assert int(call["tessera_apple.a_outer"]) == m
    rng = np.random.default_rng(m + K)
    A, B = _operand(rng, a, (m, K)), _operand(rng, b, (K, N))
    out = R._dispatch_gpu_mtl4_matmul2d([A, B], call, np)
    _check(out, A.astype(np.float64) @ B.astype(np.float64), a)


@pytest.mark.metal4
def test_ragged_n_f16_executes_with_a_200_byte_row_stride(compiler_toolchain):
    """N = 100 f16 gives a 200-byte row stride, off every 16-byte multiple. The
    verifier admits it for 16-bit views; this row is the device evidence that
    Apple does too (if it ever refuses, tighten the verifier, not this test)."""
    R = _require_lane()
    _, call = _lowered_call(compiler_toolchain, _gemm_module(64, 128, 100, "f16", "f16"))
    rng = np.random.default_rng(7)
    A, B = _operand(rng, "f16", (64, 128)), _operand(rng, "f16", (128, 100))
    out = R._dispatch_gpu_mtl4_matmul2d([A, B], call, np)
    _check(out, A.astype(np.float64) @ B.astype(np.float64), "f16")


@pytest.mark.metal4
@pytest.mark.parametrize("a,b,a_org,b_org", [("f16", "f16", (8, 64), (0, 128)),
                                             ("bf16", "bf16", (3, 8), (5, 16)),
                                             ("f8E4M3FN", "f8E4M3FN", (8, 128), (0, 128)),
                                             ("f16", "f4E2M1FN", (1, 0), (0, 512))])
def test_sub_block_origins_execute_through_the_dispatcher(compiler_toolchain, a, b, a_org, b_org):
    """Nonzero origins and padded strides projected from IR: the dispatcher
    receives the PARENT arrays and binds the window the verifier admitted."""
    R = _require_lane()
    m, k, n = 100, 256, 128
    pa, pb = (200, 512), (512, 1024)
    _, call = _lowered_call(compiler_toolchain, _sub_block_module(pa, pb, a_org, b_org, m, k, n, a, b))
    assert int(call["tessera_apple.a_byte_offset"]) > 0 or int(call["tessera_apple.b_byte_offset"]) > 0
    rng = np.random.default_rng(11)
    PA, PB = _operand(rng, a, pa), _operand(rng, b, pb)
    out = R._dispatch_gpu_mtl4_matmul2d([PA, PB], call, np)
    ref = PA[a_org[0]:a_org[0] + m, a_org[1]:a_org[1] + k].astype(np.float64) @ \
        PB[b_org[0]:b_org[0] + k, b_org[1]:b_org[1] + n].astype(np.float64)
    _check(out, ref, a)


def _epilogue_ref(x, bias, act):
    y = x + (bias[None, :] if bias is not None else 0.0)
    if act == "relu":
        return np.maximum(y, 0.0)
    if act == "gelu":  # Tessera's gelu is the tanh form
        return 0.5 * y * (1.0 + np.tanh(0.7978845608028654 * (y + 0.044715 * y ** 3)))
    if act == "silu":
        return y / (1.0 + np.exp(-y))
    return y


@pytest.mark.metal4
@pytest.mark.parametrize("storage,bias,act", [("f16", True, "gelu"), ("bf16", True, "relu"),
                                              ("f16", False, "silu"), ("f8E4M3FN", True, "none")])
def test_fused_epilogue_executes_and_matches_the_decomposed_oracle(compiler_toolchain, storage, bias, act):
    R = _require_lane()
    m, k, n = 100, 256, 128
    module = _epilogue_module(m, k, n, storage, bias=bias, act=act)
    _, call = _lowered_call(compiler_toolchain, module)
    assert call["op_kind"] == "mtl4_matmul2d_epilogue"
    rng = np.random.default_rng(23)
    A, B = _operand(rng, storage, (m, k)), _operand(rng, storage, (k, n))
    b_vec = rng.standard_normal(n).astype(np.float32) if bias else None
    out = R._dispatch_gpu_mtl4_matmul2d([A, B] + ([b_vec] if bias else []), call, np)
    ref = _epilogue_ref(A.astype(np.float64) @ B.astype(np.float64), b_vec, act)
    assert np.isfinite(out).all()
    _check(out, ref, storage, tol_lowp=2e-5)


@pytest.mark.metal4
def test_launch_executes_the_lowered_gemm_through_the_value_artifact(compiler_toolchain):
    """End to end through `runtime.launch`: the artifact carries the real
    lowered Target IR and its extracted value call, and the GPU value executor
    admits the op kind (it used to reject it before its own branch)."""
    from tessera.compiler import driver
    from tessera.runtime import RuntimeArtifact, launch
    _require_lane()
    stdout, call = _lowered_call(compiler_toolchain, _gemm_module(100, 256, 128, "f16", "f16"))
    calls = driver.extract_apple_value_calls(stdout)
    assert len(calls) == 1 and calls[0]["symbol"] == VIEW
    assert driver.apple_value_call_is_executable(calls[0])
    art = RuntimeArtifact(target_ir=stdout, metadata={
        "target": "apple_gpu", "compiler_path": "apple_value_target_ir", "executable": True,
        "apple_target_ir_kind": "value_target_ir", "apple_value_calls": calls, "arg_names": ["a", "b"],
    })
    rng = np.random.default_rng(5)
    A, B = _operand(rng, "f16", (100, 256)), _operand(rng, "f16", (256, 128))
    res = launch(art, [A, B])
    assert res["ok"] is True, res
    assert res["compiler_path"] == "apple_value_target_ir" and res["execution_kind"] == "native_gpu"
    _check(res["output"], A.astype(np.float64) @ B.astype(np.float64), "f16")


@pytest.mark.metal4
def test_value_lane_refuses_a_call_without_the_view_abi():
    from tessera import runtime as R
    call = {"symbol": VIEW, "dtype": "f16xf16", "op_kind": "mtl4_matmul2d"}
    with pytest.raises(ValueError, match="view ABI"):
        R._dispatch_gpu_mtl4_matmul2d([np.zeros((8, 8), np.float16)] * 2, call, np)


@pytest.mark.metal4
def test_value_lane_refuses_storage_whose_rows_disagree_with_the_stated_stride(compiler_toolchain):
    """A storage row that is not the IR's row stride would place every row of
    the view at the wrong address; the dispatcher refuses instead of binding."""
    R = _require_lane()
    _, call = _lowered_call(compiler_toolchain, _gemm_module(64, 256, 128, "f16", "f16"))
    rng = np.random.default_rng(1)
    A, B = _operand(rng, "f16", (64, 320)), _operand(rng, "f16", (256, 128))
    with pytest.raises(ValueError, match="do not match the projected view strides"):
        R._dispatch_gpu_mtl4_matmul2d([A, B], call, np)
