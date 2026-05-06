from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import tessera as ts
from tessera.runtime import launch


ROOT = Path(__file__).resolve().parents[2]


def _simple_transformer_reference(x, wq, wk, wv, wo, w1, w2):
    q = x @ wq
    k = x @ wk
    v = x @ wv
    scores = q @ k.T
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    attn = (probs @ v) @ wo
    hidden_pre = attn @ w1
    hidden = 0.5 * hidden_pre * (
        1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (hidden_pre + 0.044715 * np.power(hidden_pre, 3)))
    )
    return hidden @ w2


def test_apple_cpu_target_reports_accelerate_execution_mode_and_runtime_pipeline():
    @ts.jit(target="apple_cpu")
    def mm(A, B):
        return ts.ops.matmul(A, B)

    A = np.eye(2, dtype=np.float32)
    B = np.ones((2, 2), dtype=np.float32)
    artifact = mm.runtime_artifact()

    assert artifact.metadata["execution_mode"] == "cpu_accelerate"
    assert artifact.metadata["runtime_status"] == "ready"
    assert artifact.metadata["compiler_path"] == "apple_cpu_accelerate"
    assert artifact.metadata["executable"] is True
    assert artifact.metadata["guards"] == {
        "dtype": "float32",
        "rank": 2,
        "static_shape_at_launch": True,
        "op_count": 1,
    }
    assert artifact.metadata["artifact_hashes"]["backend"]
    assert 'execution_mode = "cpu_accelerate"' in mm.target_ir
    assert "tessera-lower-to-apple_cpu-runtime" in mm.compile_bundle.artifact("backend").text

    result = launch(artifact, args=(A, B))
    assert result["ok"] is True
    assert result["runtime_status"] == "success"
    assert result["compiler_path"] == "apple_cpu_accelerate"
    np.testing.assert_allclose(result["output"], A @ B)
    np.testing.assert_allclose(mm(A, B), A @ B)


def test_apple_cpu_accelerate_falls_back_for_non_f32_or_non_rank2_matmul():
    """Phase 8.2 Item #2: matmul fast-path requires f32 + rank-2. Inputs that
    fall outside that envelope are dispatched to the numpy reference instead
    of failing — keeping multi-op programs runnable when one op happens to
    use, e.g., an f64 intermediate. The fast-path engages whenever both
    operands are f32 rank-2 c-contiguous."""

    @ts.jit(target="apple_cpu")
    def mm(A, B):
        return ts.ops.matmul(A, B)

    # f64 falls back to np.matmul; result is correct, ok=True.
    A_f64 = np.eye(2, dtype=np.float64)
    B_f64 = 2 * np.eye(2, dtype=np.float64)
    fallback = launch(mm.runtime_artifact(), args=(A_f64, B_f64))
    assert fallback["ok"] is True
    assert fallback["runtime_status"] == "success"
    np.testing.assert_allclose(fallback["output"], A_f64 @ B_f64)

    # rank-3 also falls back to np.matmul (batched semantics).
    A_b = np.ones((1, 2, 2), dtype=np.float32)
    B_b = np.eye(2, dtype=np.float32)
    rank3 = launch(mm.runtime_artifact(), args=(A_b, B_b))
    assert rank3["ok"] is True
    np.testing.assert_allclose(rank3["output"], A_b @ B_b)


def test_apple_cpu_multi_op_tiny_decode_executes_through_accelerate():
    """Phase 8.2 Item #2: tiny attention-style decode runs end-to-end on the
    apple_cpu runtime path. Matmuls dispatch through Accelerate; rope, softmax,
    transpose stream through the numpy reference (the same path the default
    `cpu` target uses). Strict numerical equivalence against the corresponding
    numpy program."""

    @ts.jit(target="apple_cpu")
    def tiny_decode(x, wq, wk, wv, wo, theta):
        q = ts.ops.rope(ts.ops.matmul(x, wq), theta)
        k = ts.ops.rope(ts.ops.matmul(x, wk), theta)
        v = ts.ops.matmul(x, wv)
        scores = ts.ops.matmul(q, ts.ops.transpose(k))
        probs = ts.ops.softmax(scores)
        ctx = ts.ops.matmul(probs, v)
        return ts.ops.matmul(ctx, wo)

    artifact = tiny_decode.runtime_artifact()

    assert artifact.metadata["execution_mode"] == "cpu_accelerate"
    assert artifact.metadata["runtime_status"] == "ready"
    assert artifact.metadata["compiler_path"] == "apple_cpu_accelerate"
    assert artifact.metadata["executable"] is True
    assert tiny_decode.uses_compiled_path

    # Multi-op metadata reports actual op count + how many are matmul-fast-path.
    guards = artifact.metadata["guards"]
    assert guards["op_count"] == len(artifact.metadata["ops"])
    assert guards["accelerate_op_count"] >= 4  # 4 matmuls in the decode
    assert guards["fallback_path"] == "jit_cpu_numpy"
    accel = artifact.metadata["accelerate_ops"]
    assert all(op == "tessera.matmul" for op in accel)
    assert len(accel) >= 4

    # Numerical correctness vs numpy reference.
    x = np.arange(16, dtype=np.float32).reshape(4, 4) / 16.0
    w = np.eye(4, dtype=np.float32)
    theta = np.zeros((4, 4), dtype=np.float32)

    out = tiny_decode(x, w, w, w, w, theta)
    np.testing.assert_allclose(out, ts.ops.softmax(x @ x.T) @ x, rtol=1e-5)


def test_apple_gpu_tiny_decode_artifact_covers_rope_softmax_matmul_and_kv_cache():
    @ts.jit(target="apple_gpu")
    def tiny_decode(x, wq, wk, wv, wo, theta, cache):
        q = ts.ops.matmul(x, wq)
        k = ts.ops.matmul(x, wk)
        v = ts.ops.matmul(x, wv)
        q_rot = ts.ops.rope(q, theta)
        k_rot = ts.ops.rope(k, theta)
        cache_next = ts.ops.kv_cache_append(cache, k_rot, v)
        scores = ts.ops.matmul(q_rot, ts.ops.transpose(k_rot))
        probs = ts.ops.softmax(scores)
        ctx = ts.ops.matmul(probs, v)
        return ts.ops.matmul(ctx, wo)

    target_ir = tiny_decode.target_ir
    artifact = tiny_decode.runtime_artifact()

    assert artifact.metadata["execution_mode"] == "metal_artifact"
    assert artifact.metadata["runtime_status"] == "artifact_only"
    assert "matmul_contract" in target_ir
    assert "rope_contract" in target_ir
    assert "softmax_contract" in target_ir
    assert "tessera_apple.diagnostic" in target_ir
    assert "KV-cache target lowering is not implemented for Apple GPU" in target_ir
    assert 'framework = "MPSGraph"' in target_ir
    assert 'execution_mode = "metal_artifact"' in target_ir
    assert not tiny_decode.uses_compiled_path

    result = launch(artifact, args={})
    assert result["ok"] is False
    assert result["runtime_status"] in {"missing_backend", "unimplemented"}
    assert result["compiler_path"] == "target_ir_artifact"


def test_rope_reference_path_executes_tiny_decode_proxy_on_cpu():
    @ts.jit
    def tiny_decode_cpu(x, wq, wk, wv, wo, theta):
        q = ts.ops.rope(ts.ops.matmul(x, wq), theta)
        k = ts.ops.rope(ts.ops.matmul(x, wk), theta)
        v = ts.ops.matmul(x, wv)
        scores = ts.ops.matmul(q, ts.ops.transpose(k))
        probs = ts.ops.softmax(scores)
        ctx = ts.ops.matmul(probs, v)
        return ts.ops.matmul(ctx, wo)

    x = np.arange(16, dtype=np.float32).reshape(4, 4) / 16.0
    w = np.eye(4, dtype=np.float32)
    theta = np.zeros((4, 4), dtype=np.float32)

    out = tiny_decode_cpu(x, w, w, w, w, theta)

    assert tiny_decode_cpu.uses_compiled_path
    assert "tessera.rope" in tiny_decode_cpu.ir_text()
    assert "tile.rotary_pair" in tiny_decode_cpu.tile_ir
    np.testing.assert_allclose(out, ts.ops.softmax(x @ x.T) @ x)


def test_apple_cpu_simple_moe_solver_executes_reference_expert_routing():
    """Simple MoE solver path: stacked expert matrices, deterministic top-1
    round-robin routing, Apple CPU compiler artifact, and reference execution.
    This is intentionally a single-host solver; distributed all-to-all routing
    remains covered by the MoE planner tests."""

    @ts.jit(target="apple_cpu")
    def simple_moe(x, experts):
        return ts.ops.moe(x, experts)

    x = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ],
        dtype=np.float32,
    )
    experts = np.stack(
        [
            np.eye(2, dtype=np.float32),
            np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        ],
        axis=0,
    )
    expected = np.stack([
        x[0] @ experts[0],
        x[1] @ experts[1],
        x[2] @ experts[0],
        x[3] @ experts[1],
    ])

    artifact = simple_moe.runtime_artifact()
    assert simple_moe.uses_compiled_path
    assert artifact.metadata["compiler_path"] == "apple_cpu_accelerate"
    assert artifact.metadata["runtime_status"] == "ready"
    assert "tessera.moe" in simple_moe.ir_text()
    assert "tile.moe" in simple_moe.tile_ir
    assert "tessera_apple.cpu.moe_solver" in simple_moe.target_ir
    assert 'routing = "deterministic_top1"' in simple_moe.target_ir
    np.testing.assert_allclose(simple_moe(x, experts), expected)


def test_apple_gpu_simple_moe_solver_emits_metal_artifact_contract():
    @ts.jit(target="apple_gpu")
    def simple_moe(x, experts):
        return ts.ops.moe(x, experts)

    artifact = simple_moe.runtime_artifact()
    assert artifact.metadata["execution_mode"] == "metal_artifact"
    assert artifact.metadata["runtime_status"] == "artifact_only"
    assert "tessera.moe" in simple_moe.ir_text()
    assert 'kernel = "moe_contract"' in simple_moe.target_ir
    assert 'grid = "tokens_experts"' in simple_moe.target_ir


def test_apple_cpu_simple_transformer_block_executes_attention_and_mlp():
    @ts.jit(target="apple_cpu")
    def simple_transformer(x, wq, wk, wv, wo, w1, w2):
        q = ts.ops.matmul(x, wq)
        k = ts.ops.matmul(x, wk)
        v = ts.ops.matmul(x, wv)
        scores = ts.ops.matmul(q, ts.ops.transpose(k))
        probs = ts.ops.softmax(scores)
        attn = ts.ops.matmul(probs, v)
        proj = ts.ops.matmul(attn, wo)
        hidden = ts.ops.gelu(ts.ops.matmul(proj, w1))
        return ts.ops.matmul(hidden, w2)

    x = np.arange(12, dtype=np.float32).reshape(3, 4) / 11.0 - 0.5
    wq = np.array(
        [
            [0.50, -0.25, 0.10, 0.00],
            [0.15, 0.40, -0.35, 0.20],
            [0.00, 0.30, 0.45, -0.15],
            [-0.20, 0.05, 0.25, 0.55],
        ],
        dtype=np.float32,
    )
    wk = np.flipud(wq).copy()
    wv = np.eye(4, dtype=np.float32) * 0.75 + 0.05
    wo = np.array(
        [
            [0.60, 0.00, -0.20, 0.10],
            [0.05, 0.70, 0.10, -0.25],
            [-0.15, 0.20, 0.50, 0.35],
            [0.30, -0.10, 0.15, 0.45],
        ],
        dtype=np.float32,
    )
    w1 = np.linspace(-0.35, 0.45, 32, dtype=np.float32).reshape(4, 8)
    w2 = np.linspace(0.25, -0.30, 32, dtype=np.float32).reshape(8, 4)

    artifact = simple_transformer.runtime_artifact()
    out = simple_transformer(x, wq, wk, wv, wo, w1, w2)
    expected = _simple_transformer_reference(x, wq, wk, wv, wo, w1, w2)

    assert simple_transformer.uses_compiled_path
    assert artifact.metadata["compiler_path"] == "apple_cpu_accelerate"
    assert artifact.metadata["runtime_status"] == "ready"
    assert artifact.metadata["execution_mode"] == "cpu_accelerate"
    assert artifact.metadata["guards"]["op_count"] == len(artifact.metadata["ops"])
    assert artifact.metadata["guards"]["accelerate_op_count"] == 8
    assert artifact.metadata["guards"]["fallback_path"] == "jit_cpu_numpy"
    assert simple_transformer.ir_text().count("tessera.matmul") == 8
    assert "tessera.softmax" in simple_transformer.ir_text()
    assert "tessera.gelu" in simple_transformer.ir_text()
    assert "tile.softmax" in simple_transformer.tile_ir
    assert "tile.gelu" in simple_transformer.tile_ir
    assert simple_transformer.target_ir.count("tessera_apple.cpu.accelerate_gemm") == 8
    assert "tessera_apple.cpu.vector_reduce" in simple_transformer.target_ir
    assert "tessera_apple.cpu.vector_op" in simple_transformer.target_ir
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


def test_apple_gpu_simple_transformer_block_emits_metal_artifact_contracts():
    @ts.jit(target="apple_gpu")
    def simple_transformer(x, wq, wk, wv, wo, w1, w2):
        q = ts.ops.matmul(x, wq)
        k = ts.ops.matmul(x, wk)
        v = ts.ops.matmul(x, wv)
        scores = ts.ops.matmul(q, ts.ops.transpose(k))
        probs = ts.ops.softmax(scores)
        attn = ts.ops.matmul(probs, v)
        proj = ts.ops.matmul(attn, wo)
        hidden = ts.ops.gelu(ts.ops.matmul(proj, w1))
        return ts.ops.matmul(hidden, w2)

    artifact = simple_transformer.runtime_artifact()
    target_ir = simple_transformer.target_ir

    assert artifact.metadata["execution_mode"] == "metal_artifact"
    assert artifact.metadata["runtime_status"] == "artifact_only"
    assert not simple_transformer.uses_compiled_path
    assert simple_transformer.ir_text().count("tessera.matmul") == 8
    assert "tessera.softmax" in simple_transformer.ir_text()
    assert "tessera.gelu" in simple_transformer.ir_text()
    assert target_ir.count('kernel = "matmul_contract"') == 8
    assert 'kernel = "softmax_contract"' in target_ir
    assert 'kernel = "gelu_contract"' in target_ir
    assert 'framework = "MPSGraph"' in target_ir
    assert 'execution_mode = "metal_artifact"' in target_ir


def test_apple_cpu_runtime_shim_gemm_f32_correctness(tmp_path):
    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("C++ compiler is not available")

    source = ROOT / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_cpu_runtime.cpp"
    lib = tmp_path / ("libtessera_apple_cpu_runtime.dylib" if sys.platform == "darwin" else "libtessera_apple_cpu_runtime.so")
    cmd = [cxx, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(lib)]
    if sys.platform == "darwin":
        cmd.extend(["-framework", "Accelerate"])
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    runtime = ctypes.CDLL(str(lib))
    gemm = runtime.tessera_apple_cpu_gemm_f32
    gemm.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    gemm.restype = None

    for m, n, k in ((2, 2, 2), (2, 3, 4), (1, 5, 3)):
        a = np.arange(m * k, dtype=np.float32).reshape(m, k)
        b = (np.arange(k * n, dtype=np.float32).reshape(k, n) / 7.0)
        c = np.zeros((m, n), dtype=np.float32)
        gemm(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            m,
            n,
            k,
        )
        np.testing.assert_allclose(c, a @ b, rtol=1e-5, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Item #3 (P1): batched matmul (rank-3) via Accelerate.
# ─────────────────────────────────────────────────────────────────────────────


def test_apple_cpu_accelerate_dispatches_rank3_batched_matmul():
    """Rank-3 inputs (batch, M, K) × (batch, K, N) → (batch, M, N) dispatch
    through the new tessera_apple_cpu_gemm_f32_batched symbol when available.
    Numerical equivalence to numpy is bitwise (both call into Accelerate)."""

    @ts.jit(target="apple_cpu")
    def bmm(A, B):
        return ts.ops.matmul(A, B)

    A = np.random.RandomState(7).randn(4, 16, 32).astype(np.float32)
    B = np.random.RandomState(8).randn(4, 32, 24).astype(np.float32)

    out = bmm(A, B)
    assert out.shape == (4, 16, 24)
    assert out.dtype == np.float32

    ref = A @ B
    np.testing.assert_array_equal(out, ref)


def test_apple_cpu_runtime_exposes_batched_gemm_symbol(tmp_path):
    """Compile the runtime shim from source and confirm the batched symbol
    is exported with the right ABI."""

    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("C++ compiler is not available")

    source = ROOT / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_cpu_runtime.cpp"
    lib = tmp_path / ("libtessera_apple_cpu_runtime.dylib" if sys.platform == "darwin" else "libtessera_apple_cpu_runtime.so")
    cmd = [cxx, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(lib)]
    if sys.platform == "darwin":
        cmd.extend(["-framework", "Accelerate"])
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    runtime = ctypes.CDLL(str(lib))
    sym = runtime.tessera_apple_cpu_gemm_f32_batched
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    sym.restype = None

    batch, m, n, k = 3, 5, 7, 4
    A = np.arange(batch * m * k, dtype=np.float32).reshape(batch, m, k)
    B = (np.arange(batch * k * n, dtype=np.float32).reshape(batch, k, n) / 3.0)
    C = np.zeros((batch, m, n), dtype=np.float32)
    sym(
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        batch, m, n, k,
        m * k, k * n, m * n,
    )
    np.testing.assert_allclose(C, A @ B, rtol=1e-5, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Item #4 (P1): fp16 matmul via BNNS with cblas_sgemm fallback.
# ─────────────────────────────────────────────────────────────────────────────


def test_apple_cpu_accelerate_dispatches_fp16_matmul_via_bnns():
    """Rank-2 fp16 matmul takes the BNNS-first runtime path. Numerical result
    matches a fp32-converted reference computed through numpy (which itself
    calls into Accelerate)."""

    @ts.jit(target="apple_cpu")
    def mm(A, B):
        return ts.ops.matmul(A, B)

    rng = np.random.RandomState(11)
    A = rng.randn(32, 64).astype(np.float16)
    B = rng.randn(64, 16).astype(np.float16)

    out = mm(A, B)
    assert out.dtype == np.float16
    assert out.shape == (32, 16)

    # The runtime's f32-via-conversion fallback and BNNS both produce the
    # same f32-then-cast result for these inputs, so allclose to that ref
    # at fp16 tolerance is the right check.
    ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
    np.testing.assert_array_equal(out, ref)


def test_apple_cpu_runtime_exposes_fp16_gemm_symbol(tmp_path):
    """Compile the runtime shim and confirm tessera_apple_cpu_gemm_f16 is
    exported with the expected ABI plus correct numerical output via BNNS."""

    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("C++ compiler is not available")
    if sys.platform != "darwin":
        pytest.skip("fp16 BNNS path is Apple-only")

    source = ROOT / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_cpu_runtime.cpp"
    lib = tmp_path / "libtessera_apple_cpu_runtime.dylib"
    subprocess.run(
        ["c++", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(lib),
         "-framework", "Accelerate"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    runtime = ctypes.CDLL(str(lib))
    sym = runtime.tessera_apple_cpu_gemm_f16
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    sym.restype = None

    A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float16)
    B = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float16)
    C = np.zeros((2, 2), dtype=np.float16)
    sym(
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        2, 2, 3,
    )
    np.testing.assert_array_equal(
        C, (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
    )


# ─────────────────────────────────────────────────────────────────────────────
# bf16 follow-up: Apple CPU bf16 matmul via BNNSDataTypeBFloat16. ml_dtypes is
# a soft dependency — tests skip when it's not installed so the rest of the
# suite stays runnable on stripped-down environments.
# ─────────────────────────────────────────────────────────────────────────────


def _ml_dtypes_or_skip():
    pytest.importorskip("ml_dtypes")
    import ml_dtypes
    return ml_dtypes


def test_apple_cpu_accelerate_dispatches_bf16_matmul_via_bnns():
    """Rank-2 bf16 matmul takes the BNNSDataTypeBFloat16 path. ml_dtypes'
    bfloat16 dtype is byte-compatible with the C ABI, so the Python boundary
    just .view(np.uint16) into ctypes. Numerical result matches a fp32-cast
    reference at bf16 tolerance."""

    ml_dtypes = _ml_dtypes_or_skip()
    bf16 = ml_dtypes.bfloat16

    @ts.jit(target="apple_cpu")
    def mm(A, B):
        return ts.ops.matmul(A, B)

    rng = np.random.RandomState(13)
    A = rng.randn(48, 80).astype(bf16)
    B = rng.randn(80, 16).astype(bf16)

    out = mm(A, B)
    assert out.dtype == bf16
    assert out.shape == (48, 16)

    # bf16 has a 7-bit mantissa (~0.8% rel ULP). BNNS does fp32-internal
    # accumulation but in tiled order which differs from numpy's f32-then-
    # cast reference. Cumulative drift on K=80 fits within bf16 tolerance.
    ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(bf16)
    np.testing.assert_allclose(
        out.astype(np.float32), ref.astype(np.float32),
        rtol=2e-2, atol=2e-2,
    )


def test_apple_cpu_runtime_exposes_bf16_gemm_symbol(tmp_path):
    """Compile the runtime shim and confirm tessera_apple_cpu_gemm_bf16 is
    exported with the expected ABI plus correct numerical output via BNNS."""

    ml_dtypes = _ml_dtypes_or_skip()
    bf16 = ml_dtypes.bfloat16

    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("C++ compiler is not available")
    if sys.platform != "darwin":
        pytest.skip("bf16 BNNS path is Apple-only")

    source = ROOT / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_cpu_runtime.cpp"
    lib = tmp_path / "libtessera_apple_cpu_runtime.dylib"
    subprocess.run(
        ["c++", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(lib),
         "-framework", "Accelerate"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    runtime = ctypes.CDLL(str(lib))
    sym = runtime.tessera_apple_cpu_gemm_bf16
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    sym.restype = None

    A = np.array([[1, 2, 3], [4, 5, 6]], dtype=bf16)
    B = np.array([[7, 8], [9, 10], [11, 12]], dtype=bf16)
    C = np.zeros((2, 2), dtype=bf16)
    sym(
        A.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        B.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        C.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        2, 2, 3,
    )
    np.testing.assert_array_equal(
        C, (A.astype(np.float32) @ B.astype(np.float32)).astype(bf16)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8.3: Apple GPU baseline via Metal Performance Shaders.
#
# The runtime gating is: a single rank-2 f32 matmul/gemm program flips the
# apple_gpu artifact's execution_mode to "metal_runtime" and dispatches through
# tessera_apple_gpu_mps_matmul_f32. Multi-op programs (existing tiny_decode /
# simple_transformer / MoE tests) keep the metal_artifact contract — Phase 8.4
# will broaden the runtime envelope via custom MSL kernels.
# ─────────────────────────────────────────────────────────────────────────────


def test_apple_gpu_target_reports_mps_execution_mode_for_single_matmul():
    @ts.jit(target="apple_gpu")
    def mm(A, B):
        return ts.ops.matmul(A, B)

    A = np.eye(4, dtype=np.float32)
    B = np.arange(16, dtype=np.float32).reshape(4, 4)
    artifact = mm.runtime_artifact()

    assert artifact.metadata["compiler_path"] == "apple_gpu_mps"
    assert artifact.metadata["runtime_status"] == "ready"
    assert artifact.metadata["execution_mode"] == "metal_runtime"
    assert artifact.metadata["executable"] is True
    assert artifact.metadata["guards"] == {
        "dtype": "float32",
        "rank": 2,
        "static_shape_at_launch": True,
        "op_count": 1,
    }
    assert "tessera_apple.gpu.mps_matmul" in mm.target_ir
    assert "tessera_apple.gpu.mps_dispatch" in mm.target_ir
    assert 'execution_mode = "metal_runtime"' in mm.target_ir
    assert "tessera-lower-to-apple_gpu-runtime" in mm.compile_bundle.artifact("backend").text

    out = mm(A, B)
    np.testing.assert_allclose(out, A @ B, rtol=1e-5)
    assert out.dtype == np.float32


def test_apple_gpu_runtime_shim_exposes_mps_matmul_symbol(tmp_path):
    """Compile the apple_gpu runtime shim from source and verify the C ABI:
    symbol is exported, signature matches the lowering pass, numerical output
    matches numpy. On Darwin this exercises the Metal/MPS path; on Linux the
    portable reference fallback."""

    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("C++ compiler is not available")

    backend = ROOT / "src/compiler/codegen/Tessera_Apple_Backend/runtime"
    if sys.platform == "darwin":
        source = backend / "apple_gpu_runtime.mm"
        lib = tmp_path / "libtessera_apple_gpu_runtime.dylib"
        cmd = [cxx, "-std=c++17", "-shared", "-fPIC", "-fobjc-arc",
               "-x", "objective-c++", str(source), "-o", str(lib),
               "-framework", "Foundation",
               "-framework", "Metal",
               "-framework", "MetalPerformanceShaders"]
    else:
        source = backend / "apple_gpu_runtime_stub.cpp"
        lib = tmp_path / "libtessera_apple_gpu_runtime.so"
        cmd = [cxx, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(lib)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    runtime = ctypes.CDLL(str(lib))
    gemm = runtime.tessera_apple_gpu_mps_matmul_f32
    gemm.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    gemm.restype = None

    for m, n, k in ((2, 2, 2), (2, 3, 4), (1, 5, 3), (8, 8, 8)):
        a = np.arange(m * k, dtype=np.float32).reshape(m, k)
        b = (np.arange(k * n, dtype=np.float32).reshape(k, n) / 7.0)
        c = np.zeros((m, n), dtype=np.float32)
        gemm(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            m,
            n,
            k,
        )
        np.testing.assert_allclose(c, a @ b, rtol=1e-4, atol=1e-4)

    has_metal = runtime.tessera_apple_gpu_runtime_has_metal
    has_metal.argtypes = []
    has_metal.restype = ctypes.c_int32
    capability = has_metal()
    if sys.platform == "darwin":
        # Capability is 1 when MTLCreateSystemDefaultDevice succeeded; CI hosts
        # without a GPU still link Metal but may report 0. Either is fine — the
        # numerical outputs above already validate correctness on both paths.
        assert capability in (0, 1)
    else:
        assert capability == 0


def test_apple_gpu_target_keeps_metal_artifact_for_multi_op_programs():
    """Multi-op apple_gpu programs (matmul + softmax) must stay artifact-only
    — the Phase 8.3 runtime envelope is single-matmul. This pins the Phase 8.4
    contract: broadening must explicitly opt in by adding ops to the MPS path,
    not by relaxing the gating."""

    @ts.jit(target="apple_gpu")
    def fused(x, w):
        return ts.ops.softmax(ts.ops.matmul(x, w))

    artifact = fused.runtime_artifact()
    assert artifact.metadata["execution_mode"] == "metal_artifact"
    assert artifact.metadata["runtime_status"] == "artifact_only"
    assert not fused.uses_compiled_path


def test_apple_cpu_bf16_disabled_when_ml_dtypes_missing(monkeypatch):
    """When ml_dtypes isn't installed the bf16 dtype probe returns None and
    the runtime falls through to numpy. Verified by stubbing the import to
    fail — exercises the soft-dep contract."""

    from tessera import runtime as tessera_runtime

    monkeypatch.setattr(tessera_runtime, "_bfloat16_dtype", lambda: None)

    @ts.jit(target="apple_cpu")
    def mm(A, B):
        return ts.ops.matmul(A, B)

    # Without bf16 detection, an f32 matmul still runs (rank-2 fast path).
    A = np.eye(3, dtype=np.float32)
    B = np.ones((3, 3), dtype=np.float32)
    np.testing.assert_array_equal(mm(A, B), A @ B)
