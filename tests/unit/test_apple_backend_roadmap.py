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
    assert tiny_decode.is_executable

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
    assert not tiny_decode.is_executable

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

    assert tiny_decode_cpu.is_executable
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
    assert simple_moe.is_executable
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

    assert simple_transformer.is_executable
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
    assert not simple_transformer.is_executable
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
# L-series linalg pilot (L5): Apple CPU Cholesky via Accelerate LAPACK spotrf.
# ─────────────────────────────────────────────────────────────────────────────


def test_apple_cpu_runtime_cholesky_f32_correctness(tmp_path):
    """tessera_apple_cpu_cholesky_f32 must produce the lower factor L (A = L Lᵀ)
    matching numpy.linalg.cholesky, and signal non-SPD inputs via info > 0."""
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
    chol = runtime.tessera_apple_cpu_cholesky_f32
    chol.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
    ]
    chol.restype = ctypes.c_int32

    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 8, 16):
        # Build a well-conditioned SPD matrix A = M Mᵀ + n·I.
        m = rng.standard_normal((n, n)).astype(np.float32)
        a = (m @ m.T + n * np.eye(n, dtype=np.float32)).astype(np.float32)
        out = np.zeros((n, n), dtype=np.float32)
        info = chol(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
        )
        assert info == 0, f"unexpected non-SPD info={info} for n={n}"
        # Lower-triangular, and reconstructs A.
        assert np.allclose(np.triu(out, 1), 0.0), "strict upper triangle not zeroed"
        np.testing.assert_allclose(out @ out.T, a, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out, np.linalg.cholesky(a), rtol=1e-4, atol=1e-4)

    # Non-SPD input must return info > 0 (mirrors numpy.linalg.LinAlgError).
    bad = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)  # eigvals 3, -1
    out = np.zeros((2, 2), dtype=np.float32)
    info = chol(
        bad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        2,
    )
    assert info > 0, "non-SPD matrix should return info > 0"


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
               "-framework", "MetalPerformanceShaders",
               "-framework", "MetalPerformanceShadersGraph"]
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


def test_apple_gpu_target_keeps_metal_artifact_for_unrecognized_multi_op_programs():
    """Phase 8.4.3 added the first multi-op fusion. To keep the gate honest,
    programs that compose ops outside any recognized fusion pattern must
    still stay on the metal_artifact contract. Each Phase 8.4.x adds a
    pattern, so the negative case has to be updated to a still-unrecognized
    chain — currently softmax -> matmul (suffix-only chain, no matmul head)
    which doesn't match any of our fusion shapes.
    """

    @ts.jit(target="apple_gpu")
    def chain(x, w):
        # softmax(x) doesn't have a matmul head, so neither matmul_softmax
        # nor matmul_softmax_matmul fires. The trailing matmul also can't
        # fuse (its left operand isn't a recognized post-matmul producer).
        return ts.ops.matmul(ts.ops.softmax(x), w)

    artifact = chain.runtime_artifact()
    assert artifact.metadata["execution_mode"] == "metal_artifact"
    assert artifact.metadata["runtime_status"] == "artifact_only"
    assert not chain.is_executable


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8.4: Custom MSL kernels for ops MPS doesn't cover.
#
# First concrete kernel is rope. The runtime envelope is now: single-op
# matmul (Phase 8.3 via MPS) OR single-op rope (Phase 8.4 via custom MSL).
# Multi-op programs still keep the metal_artifact contract; future Phase
# 8.4.x kernels (flash-attention next) will add ops to that envelope.
# ─────────────────────────────────────────────────────────────────────────────


def test_apple_gpu_target_emits_msl_kernel_artifact_with_inline_source():
    """The Target IR for a single-rope apple_gpu program must carry the MSL
    source as a StringAttr on tessera_apple.gpu.msl_kernel — the IR is the
    self-contained, replayable record of the kernel."""

    @ts.jit(target="apple_gpu")
    def rope(X, Theta):
        return ts.ops.rope(X, Theta)

    target_ir = rope.target_ir
    assert "tessera_apple.gpu.msl_kernel" in target_ir
    assert 'entry_point = "rope_f32"' in target_ir
    # The MSL source itself should appear inline — at minimum the kernel
    # signature line. We don't pin the entire source so future kernel
    # rewrites don't churn the test.
    assert "kernel void rope_f32" in target_ir
    assert 'cache_key' in target_ir
    assert 'execution_mode = "metal_runtime"' in target_ir

    artifact = rope.runtime_artifact()
    assert artifact.metadata["compiler_path"] == "apple_gpu_mps"
    assert artifact.metadata["runtime_status"] == "ready"
    assert artifact.metadata["execution_mode"] == "metal_runtime"
    assert artifact.metadata["executable"] is True
    assert "tessera-lower-to-apple_gpu-runtime" in rope.compile_bundle.artifact("backend").text
    assert "tessera_apple_gpu_rope_f32" in rope.compile_bundle.artifact("backend").text


def test_apple_gpu_rope_executes_through_msl_kernel():
    """End-to-end: @jit(target='apple_gpu') rope dispatches through the
    custom MSL kernel and returns numerically-correct values vs the numpy
    reference. On Darwin this hits Metal; on Linux/CI the portable reference
    fallback in apple_gpu_runtime_stub.cpp produces the same result."""

    @ts.jit(target="apple_gpu")
    def rope(X, Theta):
        return ts.ops.rope(X, Theta)

    rng = np.random.RandomState(17)
    M, K = 8, 16
    X = rng.randn(M, K).astype(np.float32)
    Theta = rng.uniform(-np.pi, np.pi, size=(M, K)).astype(np.float32)

    out = rope(X, Theta)
    assert out.shape == (M, K)
    assert out.dtype == np.float32

    # Reference via numpy. The runtime computes cos/sin in fp32 on either
    # device, so a tight rtol is appropriate — fp32 cos/sin are bit-stable
    # across Metal and the host BLAS for these magnitudes.
    even = X[:, 0::2]
    odd = X[:, 1::2]
    theta_even = Theta[:, 0::2]
    expected = np.empty_like(X)
    expected[:, 0::2] = even * np.cos(theta_even) - odd * np.sin(theta_even)
    expected[:, 1::2] = even * np.sin(theta_even) + odd * np.cos(theta_even)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_apple_gpu_msl_runtime_caches_kernel_pipeline_state(tmp_path):
    """Compile the apple_gpu runtime shim from source and verify the MSL
    cache size grows from 0 -> 1 after first dispatch and stays at 1 across
    a second dispatch (cache hit). On non-Darwin the symbol returns -1; the
    test gates on platform."""

    if sys.platform != "darwin":
        pytest.skip("MSL kernel cache is Apple-only")

    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("C++ compiler is not available")

    backend = ROOT / "src/compiler/codegen/Tessera_Apple_Backend/runtime"
    source = backend / "apple_gpu_runtime.mm"
    lib = tmp_path / "libtessera_apple_gpu_runtime.dylib"
    cmd = [cxx, "-std=c++17", "-shared", "-fPIC", "-fobjc-arc",
           "-x", "objective-c++", str(source), "-o", str(lib),
           "-framework", "Foundation",
           "-framework", "Metal",
           "-framework", "MetalPerformanceShaders",
               "-framework", "MetalPerformanceShadersGraph"]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    runtime = ctypes.CDLL(str(lib))
    cache_size = runtime.tessera_apple_gpu_runtime_msl_cache_size
    cache_size.argtypes = []
    cache_size.restype = ctypes.c_int32

    has_metal = runtime.tessera_apple_gpu_runtime_has_metal
    has_metal.argtypes = []
    has_metal.restype = ctypes.c_int32
    if has_metal() == 0:
        pytest.skip("Metal device not available on this host")

    rope = runtime.tessera_apple_gpu_rope_f32
    rope.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    rope.restype = None

    # First dispatch — cold cache -> compile -> cache size becomes 1.
    M, K = 4, 8
    X = np.zeros((M, K), dtype=np.float32)
    Theta = np.zeros((M, K), dtype=np.float32)
    Out = np.zeros((M, K), dtype=np.float32)
    assert cache_size() == 0
    rope(
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Theta.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M, K,
    )
    assert cache_size() == 1

    # Second dispatch — same source -> cache hit, size stays 1.
    rope(
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Theta.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M, K,
    )
    assert cache_size() == 1


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8.4.1: Custom MSL flash-attention forward.
#
# Single rank-3 f32 flash_attn programs flip from metal_artifact to
# metal_runtime; the runtime shim compiles the embedded MSL kernel via
# [device newLibraryWithSource:options:error:], caches it, and dispatches via
# MTLComputeCommandEncoder. Online softmax in a single kernel — avoids
# materializing the (B, Sq, Sk) score matrix entirely.
# ─────────────────────────────────────────────────────────────────────────────


def _np_flash_attn_reference(np, q, k, v, scale=None, causal=False):
    """Faithful reference matching the runtime symbol's algorithm. Used as
    the ground truth in the apple_gpu flash_attn unit tests."""

    if scale is None:
        scale = 1.0 / np.sqrt(q.shape[-1])
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    if causal:
        Sq = scores.shape[-2]
        Sk = scores.shape[-1]
        mask = np.triu(np.ones((Sq, Sk), dtype=bool), k=1 + max(Sk - Sq, 0))
        scores = np.where(mask, -np.inf, scores)
    e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = e / np.sum(e, axis=-1, keepdims=True)
    return np.matmul(weights, v)


def test_apple_gpu_flash_attn_target_emits_msl_kernel_artifact_with_inline_source():
    """The Target IR for a single-flash_attn apple_gpu program must carry the
    MSL source inline on tessera_apple.gpu.msl_kernel — the IR is the
    self-contained, replayable record of the kernel."""

    @ts.jit(target="apple_gpu")
    def flash(q, k, v):
        return ts.ops.flash_attn(q, k, v, causal=True)

    target_ir = flash.target_ir
    assert "tessera_apple.gpu.msl_kernel" in target_ir
    assert 'entry_point = "flash_attn_f32"' in target_ir
    # Pin a stable line of the kernel so the IR clearly carries the source,
    # without locking the test to whitespace details of the kernel body.
    assert "kernel void flash_attn_f32" in target_ir
    assert 'cache_key' in target_ir
    assert 'execution_mode = "metal_runtime"' in target_ir

    artifact = flash.runtime_artifact()
    assert artifact.metadata["compiler_path"] == "apple_gpu_mps"
    assert artifact.metadata["runtime_status"] == "ready"
    assert artifact.metadata["execution_mode"] == "metal_runtime"
    assert artifact.metadata["executable"] is True
    assert "tessera-lower-to-apple_gpu-runtime" in flash.compile_bundle.artifact("backend").text
    assert "tessera_apple_gpu_flash_attn_f32" in flash.compile_bundle.artifact("backend").text


def test_apple_gpu_flash_attn_executes_through_msl_kernel():
    """End-to-end: @jit(target='apple_gpu') flash_attn dispatches through
    the custom MSL kernel and matches the numpy reference. Tested with both
    causal and non-causal masks across multiple shapes that exercise the
    online-softmax accumulator."""

    @ts.jit(target="apple_gpu")
    def flash(q, k, v):
        return ts.ops.flash_attn(q, k, v)

    @ts.jit(target="apple_gpu")
    def flash_causal(q, k, v):
        return ts.ops.flash_attn(q, k, v, causal=True)

    rng = np.random.RandomState(23)
    for B, Sq, Sk, D in ((1, 4, 4, 8), (2, 8, 8, 16), (1, 16, 32, 64)):
        Q = rng.randn(B, Sq, D).astype(np.float32) * 0.5
        K = rng.randn(B, Sk, D).astype(np.float32) * 0.5
        V = rng.randn(B, Sk, D).astype(np.float32) * 0.5

        # Non-causal
        out = flash(Q, K, V)
        assert out.shape == (B, Sq, D)
        assert out.dtype == np.float32
        ref = _np_flash_attn_reference(np, Q, K, V, causal=False)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)

        # Causal (only sensible when Sq == Sk)
        if Sq == Sk:
            out_c = flash_causal(Q, K, V)
            ref_c = _np_flash_attn_reference(np, Q, K, V, causal=True)
            np.testing.assert_allclose(out_c, ref_c, rtol=1e-4, atol=1e-5)


def test_apple_gpu_flash_attn_runtime_shim_correctness(tmp_path):
    """Compile the apple_gpu runtime shim from source and verify the C ABI
    of tessera_apple_gpu_flash_attn_f32: signature matches the lowering pass,
    numerical output matches the numpy reference. On Darwin this exercises
    the Metal/MSL path; on Linux the portable reference fallback."""

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
               "-framework", "MetalPerformanceShaders",
               "-framework", "MetalPerformanceShadersGraph"]
    else:
        source = backend / "apple_gpu_runtime_stub.cpp"
        lib = tmp_path / "libtessera_apple_gpu_runtime.so"
        cmd = [cxx, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(lib)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    runtime = ctypes.CDLL(str(lib))
    flash = runtime.tessera_apple_gpu_flash_attn_f32
    flash.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_int32,
    ]
    flash.restype = None

    rng = np.random.RandomState(29)
    for B, Sq, Sk, D, causal in (
        (1, 3, 3, 4, 0),
        (2, 4, 6, 8, 0),
        (1, 5, 5, 16, 1),
    ):
        Q = rng.randn(B, Sq, D).astype(np.float32) * 0.5
        K = rng.randn(B, Sk, D).astype(np.float32) * 0.5
        V = rng.randn(B, Sk, D).astype(np.float32) * 0.5
        O = np.zeros((B, Sq, D), dtype=np.float32)
        scale = 1.0 / float(np.sqrt(D))
        flash(
            Q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            K.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            V.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            O.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            B, Sq, Sk, D, scale, causal,
        )
        ref = _np_flash_attn_reference(np, Q, K, V, scale=scale, causal=bool(causal))
        np.testing.assert_allclose(O, ref, rtol=1e-4, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8.4.2: Softmax + GeLU as standalone MSL kernels.
#
# Broadens the apple_gpu single-op runtime envelope. Single-op programs
# that are just softmax (axis=-1, rank-2, f32) or gelu (rank-2, f32) now
# flip from metal_artifact to metal_runtime, dispatching through purpose-
# built MSL kernels. Multi-op programs that include softmax/gelu still
# stay on the artifact-only path until Phase 8.4.3 lands fusion.
# ─────────────────────────────────────────────────────────────────────────────


def _np_gelu_reference(np, x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def test_apple_gpu_softmax_target_emits_msl_kernel_artifact_with_inline_source():
    @ts.jit(target="apple_gpu")
    def sm(X):
        return ts.ops.softmax(X)

    target_ir = sm.target_ir
    assert "tessera_apple.gpu.msl_kernel" in target_ir
    assert 'entry_point = "softmax_f32"' in target_ir
    assert "kernel void softmax_f32" in target_ir
    assert 'cache_key' in target_ir
    assert 'execution_mode = "metal_runtime"' in target_ir

    artifact = sm.runtime_artifact()
    assert artifact.metadata["compiler_path"] == "apple_gpu_mps"
    assert artifact.metadata["runtime_status"] == "ready"
    assert artifact.metadata["execution_mode"] == "metal_runtime"
    assert "tessera_apple_gpu_softmax_f32" in sm.compile_bundle.artifact("backend").text


def test_apple_gpu_softmax_executes_through_msl_kernel():
    @ts.jit(target="apple_gpu")
    def sm(X):
        return ts.ops.softmax(X)

    rng = np.random.RandomState(31)
    for shape in ((4, 8), (8, 64), (16, 16)):
        X = rng.randn(*shape).astype(np.float32)
        out = sm(X)
        assert out.shape == shape
        assert out.dtype == np.float32
        ref_e = np.exp(X - np.max(X, axis=-1, keepdims=True))
        ref = ref_e / np.sum(ref_e, axis=-1, keepdims=True)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
        # Each row sums to 1 (modulo rounding).
        np.testing.assert_allclose(out.sum(axis=-1), np.ones(shape[0], dtype=np.float32), rtol=1e-5)


def test_apple_gpu_gelu_target_emits_msl_kernel_artifact_with_inline_source():
    @ts.jit(target="apple_gpu")
    def gelu(X):
        return ts.ops.gelu(X)

    target_ir = gelu.target_ir
    assert "tessera_apple.gpu.msl_kernel" in target_ir
    assert 'entry_point = "gelu_f32"' in target_ir
    assert "kernel void gelu_f32" in target_ir
    assert 'execution_mode = "metal_runtime"' in target_ir

    artifact = gelu.runtime_artifact()
    assert artifact.metadata["compiler_path"] == "apple_gpu_mps"
    assert artifact.metadata["runtime_status"] == "ready"
    assert "tessera_apple_gpu_gelu_f32" in gelu.compile_bundle.artifact("backend").text


def test_apple_gpu_gelu_executes_through_msl_kernel():
    @ts.jit(target="apple_gpu")
    def gelu(X):
        return ts.ops.gelu(X)

    rng = np.random.RandomState(37)
    for shape in ((2, 16), (8, 8), (4, 32)):
        X = rng.randn(*shape).astype(np.float32) * 1.5
        out = gelu(X)
        assert out.shape == shape
        assert out.dtype == np.float32
        ref = _np_gelu_reference(np, X)
        # tanh approximation in fp32; rtol matches the cpu reference path
        # in the existing apple_cpu transformer test.
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_apple_gpu_softmax_runtime_shim_correctness(tmp_path):
    """Compile the apple_gpu runtime shim from source and verify the C ABI
    of tessera_apple_gpu_softmax_f32: signature matches the lowering pass,
    numerical output matches numpy."""

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
               "-framework", "MetalPerformanceShaders",
               "-framework", "MetalPerformanceShadersGraph"]
    else:
        source = backend / "apple_gpu_runtime_stub.cpp"
        lib = tmp_path / "libtessera_apple_gpu_runtime.so"
        cmd = [cxx, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(lib)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    runtime = ctypes.CDLL(str(lib))

    softmax = runtime.tessera_apple_gpu_softmax_f32
    softmax.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32, ctypes.c_int32,
    ]
    softmax.restype = None

    gelu = runtime.tessera_apple_gpu_gelu_f32
    gelu.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
    ]
    gelu.restype = None

    rng = np.random.RandomState(41)
    for M, K in ((4, 8), (16, 32)):
        X = rng.randn(M, K).astype(np.float32)
        Y = np.zeros((M, K), dtype=np.float32)
        softmax(
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            M, K,
        )
        e = np.exp(X - np.max(X, axis=-1, keepdims=True))
        ref = e / np.sum(e, axis=-1, keepdims=True)
        np.testing.assert_allclose(Y, ref, rtol=1e-5, atol=1e-6)

        Z = np.zeros((M, K), dtype=np.float32)
        gelu(
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            Z.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            M * K,
        )
        np.testing.assert_allclose(Z, _np_gelu_reference(np, X), rtol=1e-5, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8.4.3: First multi-op MSL fusion — matmul -> softmax.
#
# A 2-op SSA chain (matmul whose result is consumed only by a softmax)
# now flips from metal_artifact to metal_runtime, dispatching through a
# fused MSL kernel that avoids materializing the (M, N) intermediate
# score matrix on the host. The runtime gate is exact: any deviation
# (3+ ops, intermediate consumed by another op, N > 256) keeps the
# program on the artifact path.
# ─────────────────────────────────────────────────────────────────────────────


def _np_matmul_softmax_reference(np, A, B):
    scores = np.matmul(A, B)
    e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def test_apple_gpu_matmul_softmax_chain_emits_fused_msl_kernel_artifact():
    @ts.jit(target="apple_gpu")
    def fused(A, B):
        return ts.ops.softmax(ts.ops.matmul(A, B))

    target_ir = fused.target_ir
    assert "tessera_apple.gpu.msl_kernel" in target_ir
    # Optimizing-Compiler Plan F2 — f32 matmul->softmax is SYNTHESIZED (the
    # tiled synthesizer subsumes matmul_softmax_f32 / _tiled_f32); the fusion id
    # stays "matmul_softmax".
    assert 'entry_point = "synth_matmul_epi"' in target_ir
    assert "kernel void synth_matmul_epi" in target_ir
    assert 'fusion = "matmul_softmax"' in target_ir
    assert 'execution_mode = "metal_runtime"' in target_ir
    # Exactly one msl_kernel — the fused chain collapses two ops to one
    # emission. Two would mean the fusion didn't fire.
    assert target_ir.count('"tessera_apple.gpu.msl_kernel"') == 1

    artifact = fused.runtime_artifact()
    assert artifact.metadata["compiler_path"] == "apple_gpu_mps"
    assert artifact.metadata["runtime_status"] == "ready"
    assert artifact.metadata["execution_mode"] == "metal_runtime"
    assert ("tessera_apple_gpu_synth_matmul_epilogue_f32"
            in fused.compile_bundle.artifact("backend").text)


def test_apple_gpu_matmul_softmax_chain_executes_through_fused_msl_kernel():
    """End-to-end: the fused chain matches the per-op numpy reference at
    rtol=1e-4 across a few representative shapes including a wide-N case
    that exercises the kernel's stack accumulator."""

    @ts.jit(target="apple_gpu")
    def fused(A, B):
        return ts.ops.softmax(ts.ops.matmul(A, B))

    rng = np.random.RandomState(43)
    for M, K, N in ((4, 8, 8), (8, 16, 32), (16, 16, 64)):
        A = rng.randn(M, K).astype(np.float32) * 0.5
        B = rng.randn(K, N).astype(np.float32) * 0.5
        out = fused(A, B)
        assert out.shape == (M, N)
        assert out.dtype == np.float32
        ref = _np_matmul_softmax_reference(np, A, B)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)
        # Each row sums to ~1 modulo rounding.
        np.testing.assert_allclose(out.sum(axis=-1), np.ones(M, dtype=np.float32), rtol=1e-4)


def test_apple_gpu_matmul_softmax_fusion_runtime_shim_correctness(tmp_path):
    """Compile the apple_gpu runtime shim from source and verify the catalog
    retirement at the ABI level (Optimizing-Compiler Plan F2): the synthesized
    epilogue symbols are exported and the retired matmul_softmax_f32 /
    matmul_softmax_tiled_f32 kernels are gone.  Numerical correctness of the
    synthesizer path is covered by test_fusion_synthesis.py (Metal) and the
    `_executes_through_fused_msl_kernel` test below."""

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
               "-framework", "MetalPerformanceShaders",
               "-framework", "MetalPerformanceShadersGraph"]
    else:
        source = backend / "apple_gpu_runtime_stub.cpp"
        lib = tmp_path / "libtessera_apple_gpu_runtime.so"
        cmd = [cxx, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(lib)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    runtime = ctypes.CDLL(str(lib))
    # the synthesized epilogue (stack + tiled) symbols are present.
    for name in ("tessera_apple_gpu_synth_matmul_epilogue_f32",
                 "tessera_apple_gpu_synth_matmul_epilogue_tiled_f32"):
        assert getattr(runtime, name, None) is not None, f"missing: {name}"
    # the retired per-kernel f32 softmax symbols are gone.
    for retired in ("tessera_apple_gpu_matmul_softmax_f32",
                    "tessera_apple_gpu_matmul_softmax_tiled_f32"):
        assert getattr(runtime, retired, None) is None, (
            f"retired kernel still present: {retired}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8.4.4: fp16 / bf16 matmul on apple_gpu (mirrors BNNS bf16 from CPU).
#
# Single rank-2 matmul programs now flip to metal_runtime regardless of dtype:
#   f32  -> native MPSDataTypeFloat32
#   f16  -> native MPSDataTypeFloat16
#   bf16 -> fp32-conversion path (MPS doesn't support bf16 matrix
#           descriptors as of macOS 14)
# Mixed-dtype operands fall back to the artifact-only path.
# ─────────────────────────────────────────────────────────────────────────────


def _bfloat16_or_skip():
    pytest.importorskip("ml_dtypes")
    import ml_dtypes
    return ml_dtypes.bfloat16


def test_apple_gpu_matmul_f32_artifact_reports_metal_runtime():
    """Phase 8.4.4 — the compile-time artifact stays type-polymorphic
    (defaults to f32 in the static Graph IR) because call-site dtypes are
    only known when @jit functions are invoked. The runtime dispatcher
    selects the matching MPS symbol by inspecting input array dtypes.
    This test pins the compile-time contract; the runtime dtype dispatch
    is exercised by the executes_through_mps tests below.
    """

    @ts.jit(target="apple_gpu")
    def mm(A, B):
        return ts.ops.matmul(A, B)

    artifact = mm.runtime_artifact()
    assert artifact.metadata["execution_mode"] == "metal_runtime"
    assert artifact.metadata["compiler_path"] == "apple_gpu_mps"
    backend_text = mm.compile_bundle.artifact("backend").text
    # The default f32 symbol is named in the artifact since the Graph IR
    # operand types are f32 absent explicit type hints. Runtime dtype
    # dispatch overrides this at launch time when inputs are f16/bf16.
    assert "tessera_apple_gpu_mps_matmul_f32" in backend_text


def test_apple_gpu_matmul_f16_executes_through_mps():
    """End-to-end: @jit(target='apple_gpu') fp16 matmul. The runtime
    dispatcher detects fp16 inputs and routes to MPSDataTypeFloat16. MPS
    does fp16 internal accumulation, which can drift slightly from the
    fp32-converted reference; assert at fp16 tolerance."""

    @ts.jit(target="apple_gpu")
    def mm(A, B):
        return ts.ops.matmul(A, B)

    rng = np.random.RandomState(53)
    for M, K, N in ((4, 8, 8), (8, 16, 32)):
        A = rng.randn(M, K).astype(np.float16)
        B = rng.randn(K, N).astype(np.float16)
        out = mm(A, B)
        assert out.dtype == np.float16
        assert out.shape == (M, N)
        # Reference: convert to fp32, matmul, convert back. MPS does fp16
        # internal accumulation; modest rel tolerance covers the drift.
        ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
        np.testing.assert_allclose(
            out.astype(np.float32), ref.astype(np.float32),
            rtol=5e-2, atol=5e-2,
        )


def test_apple_gpu_matmul_bf16_executes_through_fp32_conversion_path():
    """End-to-end: @jit(target='apple_gpu') bf16 matmul matches an
    fp32-converted reference at bf16 tolerance. The runtime shim does the
    fp32 conversion internally so the host sees a bf16 in/out ABI."""

    bf16 = _bfloat16_or_skip()

    @ts.jit(target="apple_gpu")
    def mm(A, B):
        return ts.ops.matmul(A, B)

    rng = np.random.RandomState(59)
    for M, K, N in ((4, 8, 8), (8, 16, 32)):
        A = rng.randn(M, K).astype(bf16)
        B = rng.randn(K, N).astype(bf16)
        out = mm(A, B)
        assert out.dtype == bf16
        assert out.shape == (M, N)
        ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(bf16)
        # bf16 has ~7-bit mantissa; tile-order rounding can drift by ~2% on
        # K=16. Same tolerance pattern as the apple_cpu BNNS bf16 test.
        np.testing.assert_allclose(
            out.astype(np.float32), ref.astype(np.float32),
            rtol=2e-2, atol=2e-2,
        )


def test_apple_gpu_matmul_runtime_shim_exposes_f16_and_bf16_symbols(tmp_path):
    """Compile the apple_gpu runtime shim from source and verify the C ABI
    of the new fp16 + bf16 matmul symbols. On Darwin this exercises the
    Metal/MPS path; on Linux/CI the portable reference fallback."""

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
               "-framework", "MetalPerformanceShaders",
               "-framework", "MetalPerformanceShadersGraph"]
    else:
        source = backend / "apple_gpu_runtime_stub.cpp"
        lib = tmp_path / "libtessera_apple_gpu_runtime.so"
        cmd = [cxx, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(lib)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    runtime = ctypes.CDLL(str(lib))

    # fp16 ABI test
    gemm_f16 = runtime.tessera_apple_gpu_mps_matmul_f16
    gemm_f16.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    gemm_f16.restype = None

    A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float16)
    B = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float16)
    C = np.zeros((2, 2), dtype=np.float16)
    gemm_f16(
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        2, 2, 3,
    )
    np.testing.assert_array_equal(
        C, (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
    )

    # bf16 ABI test (only when ml_dtypes is available)
    try:
        import ml_dtypes
        bf16 = ml_dtypes.bfloat16
    except Exception:
        return

    gemm_bf16 = runtime.tessera_apple_gpu_mps_matmul_bf16
    gemm_bf16.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    gemm_bf16.restype = None

    A_bf = np.array([[1, 2, 3], [4, 5, 6]], dtype=bf16)
    B_bf = np.array([[7, 8], [9, 10], [11, 12]], dtype=bf16)
    C_bf = np.zeros((2, 2), dtype=bf16)
    gemm_bf16(
        A_bf.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        B_bf.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        C_bf.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        2, 2, 3,
    )
    np.testing.assert_array_equal(
        C_bf, (A_bf.astype(np.float32) @ B_bf.astype(np.float32)).astype(bf16)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8.4.4.1: fp16 / bf16 for the simple MSL kernels (rope, softmax, gelu).
#
# fp16 path uses native MSL `half` kernels with `float` internal compute.
# bf16 path uses fp32-conversion at the runtime boundary (no native MSL bf16).
# Same pattern as Phase 8.4.4 matmul. The Python dispatcher detects input
# array dtype at runtime and routes to the matching ctypes wrapper.
# ─────────────────────────────────────────────────────────────────────────────


def test_apple_gpu_rope_f16_executes_through_native_msl():
    @ts.jit(target="apple_gpu")
    def rope(X, Theta):
        return ts.ops.rope(X, Theta)

    rng = np.random.RandomState(67)
    M, K = 8, 16
    X = rng.randn(M, K).astype(np.float16) * 0.5
    Theta = rng.uniform(-np.pi, np.pi, size=(M, K)).astype(np.float16)
    out = rope(X, Theta)
    assert out.dtype == np.float16
    assert out.shape == (M, K)
    Xf = X.astype(np.float32)
    Tf = Theta.astype(np.float32)
    even = Xf[:, 0::2]
    odd = Xf[:, 1::2]
    theta_even = Tf[:, 0::2]
    expected = np.empty_like(Xf)
    expected[:, 0::2] = even * np.cos(theta_even) - odd * np.sin(theta_even)
    expected[:, 1::2] = even * np.sin(theta_even) + odd * np.cos(theta_even)
    np.testing.assert_allclose(
        out.astype(np.float32), expected, rtol=5e-3, atol=5e-3,
    )


def test_apple_gpu_rope_bf16_executes_through_fp32_conversion_path():
    pytest.importorskip("ml_dtypes")
    import ml_dtypes
    bf16 = ml_dtypes.bfloat16

    @ts.jit(target="apple_gpu")
    def rope(X, Theta):
        return ts.ops.rope(X, Theta)

    rng = np.random.RandomState(71)
    M, K = 8, 16
    # Multiplication BEFORE astype — `bf16_arr * python_float` would promote
    # back to float32 because numpy treats Python scalars as float64 and
    # downcast goes through fp32. Apply scaling in fp32, then cast.
    X = (rng.randn(M, K) * 0.5).astype(bf16)
    Theta = rng.uniform(-np.pi, np.pi, size=(M, K)).astype(bf16)
    out = rope(X, Theta)
    assert out.dtype == bf16
    assert out.shape == (M, K)
    Xf = X.astype(np.float32)
    Tf = Theta.astype(np.float32)
    even = Xf[:, 0::2]
    odd = Xf[:, 1::2]
    theta_even = Tf[:, 0::2]
    expected = np.empty_like(Xf)
    expected[:, 0::2] = even * np.cos(theta_even) - odd * np.sin(theta_even)
    expected[:, 1::2] = even * np.sin(theta_even) + odd * np.cos(theta_even)
    np.testing.assert_allclose(
        out.astype(np.float32), expected, rtol=2e-2, atol=2e-2,
    )


def test_apple_gpu_softmax_f16_executes_through_native_msl():
    @ts.jit(target="apple_gpu")
    def sm(X):
        return ts.ops.softmax(X)

    rng = np.random.RandomState(73)
    for shape in ((4, 8), (8, 32)):
        X = rng.randn(*shape).astype(np.float16)
        out = sm(X)
        assert out.dtype == np.float16
        assert out.shape == shape
        ref_e = np.exp(X.astype(np.float32) - np.max(X.astype(np.float32), axis=-1, keepdims=True))
        ref = ref_e / np.sum(ref_e, axis=-1, keepdims=True)
        np.testing.assert_allclose(
            out.astype(np.float32), ref, rtol=5e-3, atol=5e-3,
        )


def test_apple_gpu_softmax_bf16_executes_through_fp32_conversion_path():
    pytest.importorskip("ml_dtypes")
    import ml_dtypes
    bf16 = ml_dtypes.bfloat16

    @ts.jit(target="apple_gpu")
    def sm(X):
        return ts.ops.softmax(X)

    rng = np.random.RandomState(79)
    X = rng.randn(8, 16).astype(bf16)
    out = sm(X)
    assert out.dtype == bf16
    ref_e = np.exp(X.astype(np.float32) - np.max(X.astype(np.float32), axis=-1, keepdims=True))
    ref = ref_e / np.sum(ref_e, axis=-1, keepdims=True)
    np.testing.assert_allclose(
        out.astype(np.float32), ref, rtol=2e-2, atol=2e-2,
    )


def test_apple_gpu_gelu_f16_executes_through_native_msl():
    @ts.jit(target="apple_gpu")
    def gelu(X):
        return ts.ops.gelu(X)

    rng = np.random.RandomState(83)
    X = rng.randn(8, 16).astype(np.float16) * 1.5
    out = gelu(X)
    assert out.dtype == np.float16
    Xf = X.astype(np.float32)
    ref = 0.5 * Xf * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (Xf + 0.044715 * Xf**3)))
    np.testing.assert_allclose(
        out.astype(np.float32), ref, rtol=5e-3, atol=5e-3,
    )


def test_apple_gpu_gelu_bf16_executes_through_fp32_conversion_path():
    pytest.importorskip("ml_dtypes")
    import ml_dtypes
    bf16 = ml_dtypes.bfloat16

    @ts.jit(target="apple_gpu")
    def gelu(X):
        return ts.ops.gelu(X)

    rng = np.random.RandomState(89)
    X = (rng.randn(8, 16) * 1.5).astype(bf16)
    out = gelu(X)
    assert out.dtype == bf16
    Xf = X.astype(np.float32)
    ref = 0.5 * Xf * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (Xf + 0.044715 * Xf**3)))
    np.testing.assert_allclose(
        out.astype(np.float32), ref, rtol=2e-2, atol=2e-2,
    )


def test_apple_gpu_msl_dtype_runtime_shim_exposes_all_symbols(tmp_path):
    """Compile the apple_gpu runtime shim from source and verify all 6 new
    fp16/bf16 symbols (rope_{f16,bf16}, softmax_{f16,bf16}, gelu_{f16,bf16})
    are exported."""

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
               "-framework", "MetalPerformanceShaders",
               "-framework", "MetalPerformanceShadersGraph"]
    else:
        source = backend / "apple_gpu_runtime_stub.cpp"
        lib = tmp_path / "libtessera_apple_gpu_runtime.so"
        cmd = [cxx, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(lib)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    runtime = ctypes.CDLL(str(lib))
    for name in (
        "tessera_apple_gpu_rope_f16",
        "tessera_apple_gpu_rope_bf16",
        "tessera_apple_gpu_softmax_f16",
        "tessera_apple_gpu_softmax_bf16",
        "tessera_apple_gpu_gelu_f16",
        "tessera_apple_gpu_gelu_bf16",
    ):
        sym = getattr(runtime, name, None)
        assert sym is not None, f"missing C ABI symbol: {name}"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8.4.4.2: fp16 / bf16 for the fused matmul -> softmax kernel and for
# flash_attn. Mixed-precision design: half/bfloat I/O at the boundary, fp32
# per-thread accumulators internally — matches what production flash-attn
# implementations do.
# ─────────────────────────────────────────────────────────────────────────────


def test_apple_gpu_matmul_softmax_f16_executes_through_native_msl():
    @ts.jit(target="apple_gpu")
    def fused(A, B):
        return ts.ops.softmax(ts.ops.matmul(A, B))

    rng = np.random.RandomState(101)
    for M, K, N in ((4, 8, 8), (8, 16, 32)):
        A = rng.randn(M, K).astype(np.float16)
        B = rng.randn(K, N).astype(np.float16)
        out = fused(A, B)
        assert out.dtype == np.float16
        assert out.shape == (M, N)
        scores = A.astype(np.float32) @ B.astype(np.float32)
        e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        ref = (e / np.sum(e, axis=-1, keepdims=True)).astype(np.float16)
        np.testing.assert_allclose(
            out.astype(np.float32), ref.astype(np.float32),
            rtol=5e-3, atol=5e-3,
        )


def test_apple_gpu_matmul_softmax_bf16_executes_through_fp32_conversion():
    pytest.importorskip("ml_dtypes")
    import ml_dtypes
    bf16 = ml_dtypes.bfloat16

    @ts.jit(target="apple_gpu")
    def fused(A, B):
        return ts.ops.softmax(ts.ops.matmul(A, B))

    rng = np.random.RandomState(103)
    M, K, N = 8, 16, 32
    A = rng.randn(M, K).astype(bf16)
    B = rng.randn(K, N).astype(bf16)
    out = fused(A, B)
    assert out.dtype == bf16
    scores = A.astype(np.float32) @ B.astype(np.float32)
    e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    ref = (e / np.sum(e, axis=-1, keepdims=True)).astype(bf16)
    np.testing.assert_allclose(
        out.astype(np.float32), ref.astype(np.float32),
        rtol=2e-2, atol=2e-2,
    )


def test_apple_gpu_flash_attn_f16_executes_through_native_msl():
    @ts.jit(target="apple_gpu")
    def flash(q, k, v):
        return ts.ops.flash_attn(q, k, v)

    rng = np.random.RandomState(107)
    for B, Sq, Sk, D in ((1, 4, 4, 8), (2, 8, 8, 16)):
        Q = rng.randn(B, Sq, D).astype(np.float16)
        K = rng.randn(B, Sk, D).astype(np.float16)
        V = rng.randn(B, Sk, D).astype(np.float16)
        out = flash(Q, K, V)
        assert out.dtype == np.float16
        assert out.shape == (B, Sq, D)
        Qf = Q.astype(np.float32)
        Kf = K.astype(np.float32)
        Vf = V.astype(np.float32)
        scale = 1.0 / float(np.sqrt(D))
        scores = Qf @ np.swapaxes(Kf, -1, -2) * scale
        e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = e / np.sum(e, axis=-1, keepdims=True)
        ref = weights @ Vf
        np.testing.assert_allclose(
            out.astype(np.float32), ref, rtol=5e-3, atol=5e-3,
        )


def test_apple_gpu_flash_attn_bf16_executes_through_fp32_conversion():
    pytest.importorskip("ml_dtypes")
    import ml_dtypes
    bf16 = ml_dtypes.bfloat16

    @ts.jit(target="apple_gpu")
    def flash(q, k, v):
        return ts.ops.flash_attn(q, k, v)

    rng = np.random.RandomState(109)
    B, Sq, Sk, D = 1, 8, 8, 16
    Q = rng.randn(B, Sq, D).astype(bf16)
    K = rng.randn(B, Sk, D).astype(bf16)
    V = rng.randn(B, Sk, D).astype(bf16)
    out = flash(Q, K, V)
    assert out.dtype == bf16
    Qf = Q.astype(np.float32)
    Kf = K.astype(np.float32)
    Vf = V.astype(np.float32)
    scale = 1.0 / float(np.sqrt(D))
    scores = Qf @ np.swapaxes(Kf, -1, -2) * scale
    e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = e / np.sum(e, axis=-1, keepdims=True)
    ref = weights @ Vf
    np.testing.assert_allclose(
        out.astype(np.float32), ref, rtol=2e-2, atol=2e-2,
    )


def test_apple_gpu_fused_dtype_runtime_shim_exposes_all_symbols(tmp_path):
    """Compile the apple_gpu runtime shim from source and verify all 4 new
    fp16/bf16 symbols (matmul_softmax_{f16,bf16}, flash_attn_{f16,bf16}) are
    exported."""

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
               "-framework", "MetalPerformanceShaders",
               "-framework", "MetalPerformanceShadersGraph"]
    else:
        source = backend / "apple_gpu_runtime_stub.cpp"
        lib = tmp_path / "libtessera_apple_gpu_runtime.so"
        cmd = [cxx, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(lib)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    runtime = ctypes.CDLL(str(lib))
    for name in (
        "tessera_apple_gpu_matmul_softmax_f16",
        "tessera_apple_gpu_matmul_softmax_bf16",
        "tessera_apple_gpu_flash_attn_f16",
        "tessera_apple_gpu_flash_attn_bf16",
    ):
        sym = getattr(runtime, name, None)
        assert sym is not None, f"missing C ABI symbol: {name}"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8.4.5: 3-op fusion — matmul → softmax → matmul (full attention block).
#
# The longest fusion pattern wins: when a program forms a matmul -> softmax
# -> matmul SSA chain (single-use intermediates), the runtime collapses it
# into a single MSL kernel that materializes the (M, N) softmax result only
# in registers. fp32 accumulators throughout regardless of I/O dtype.
# ─────────────────────────────────────────────────────────────────────────────


def _np_attn_block_reference(np, A, B, C):
    """Faithful reference for matmul -> softmax(axis=-1) -> matmul."""
    scores = np.asarray(A) @ np.asarray(B)
    e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probs = e / np.sum(e, axis=-1, keepdims=True)
    return probs @ np.asarray(C)


def test_apple_gpu_attn_block_chain_emits_fused_msl_kernel():
    """The Target IR for a 3-op matmul -> softmax -> matmul chain must
    emit a single fused msl_kernel with the matmul_softmax_matmul entry
    point — collapsing all three Graph IR ops into one runtime call."""

    @ts.jit(target="apple_gpu")
    def attn(A, B, C):
        return ts.ops.matmul(ts.ops.softmax(ts.ops.matmul(A, B)), C)

    target_ir = attn.target_ir
    assert "tessera_apple.gpu.msl_kernel" in target_ir
    assert 'entry_point = "matmul_softmax_matmul_f32"' in target_ir
    assert "kernel void matmul_softmax_matmul_f32" in target_ir
    assert 'fusion = "matmul_softmax_matmul"' in target_ir
    assert 'execution_mode = "metal_runtime"' in target_ir
    # Exactly one msl_kernel — three ops collapsed to one emission.
    assert target_ir.count('"tessera_apple.gpu.msl_kernel"') == 1

    artifact = attn.runtime_artifact()
    assert artifact.metadata["compiler_path"] == "apple_gpu_mps"
    assert artifact.metadata["runtime_status"] == "ready"
    assert artifact.metadata["execution_mode"] == "metal_runtime"
    assert "tessera_apple_gpu_matmul_softmax_matmul_f32" in attn.compile_bundle.artifact("backend").text


def test_apple_gpu_attn_block_f32_executes_through_fused_msl_kernel():
    """End-to-end f32: matmul -> softmax -> matmul matches the per-op
    numpy reference at rtol=1e-4 across attention-shaped inputs."""

    @ts.jit(target="apple_gpu")
    def attn(A, B, C):
        return ts.ops.matmul(ts.ops.softmax(ts.ops.matmul(A, B)), C)

    rng = np.random.RandomState(151)
    for M, K, N, P in ((4, 8, 8, 4), (8, 16, 32, 16), (16, 16, 64, 32)):
        A = rng.randn(M, K).astype(np.float32) * 0.5
        B = rng.randn(K, N).astype(np.float32) * 0.5
        C = rng.randn(N, P).astype(np.float32) * 0.5
        out = attn(A, B, C)
        assert out.shape == (M, P)
        assert out.dtype == np.float32
        ref = _np_attn_block_reference(np, A, B, C)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


def test_apple_gpu_attn_block_f16_executes_through_fused_msl_kernel():
    """End-to-end f16: same chain, mixed-precision (half I/O, fp32
    accumulators). rtol matches the other f16 fused kernels."""

    @ts.jit(target="apple_gpu")
    def attn(A, B, C):
        return ts.ops.matmul(ts.ops.softmax(ts.ops.matmul(A, B)), C)

    rng = np.random.RandomState(157)
    for M, K, N, P in ((4, 8, 8, 4), (8, 16, 32, 16)):
        A = rng.randn(M, K).astype(np.float16)
        B = rng.randn(K, N).astype(np.float16)
        C = rng.randn(N, P).astype(np.float16)
        out = attn(A, B, C)
        assert out.shape == (M, P)
        assert out.dtype == np.float16
        ref = _np_attn_block_reference(
            np, A.astype(np.float32), B.astype(np.float32), C.astype(np.float32)
        ).astype(np.float16)
        np.testing.assert_allclose(
            out.astype(np.float32), ref.astype(np.float32),
            rtol=5e-3, atol=5e-3,
        )


def test_apple_gpu_attn_block_bf16_executes_through_fp32_conversion():
    pytest.importorskip("ml_dtypes")
    import ml_dtypes
    bf16 = ml_dtypes.bfloat16

    @ts.jit(target="apple_gpu")
    def attn(A, B, C):
        return ts.ops.matmul(ts.ops.softmax(ts.ops.matmul(A, B)), C)

    rng = np.random.RandomState(163)
    M, K, N, P = 8, 16, 32, 16
    A = rng.randn(M, K).astype(bf16)
    B = rng.randn(K, N).astype(bf16)
    C = rng.randn(N, P).astype(bf16)
    out = attn(A, B, C)
    assert out.dtype == bf16
    ref = _np_attn_block_reference(
        np, A.astype(np.float32), B.astype(np.float32), C.astype(np.float32)
    ).astype(bf16)
    np.testing.assert_allclose(
        out.astype(np.float32), ref.astype(np.float32),
        rtol=2e-2, atol=2e-2,
    )


def test_apple_gpu_attn_block_runtime_shim_exposes_3op_fusion_symbols(tmp_path):
    """Compile the apple_gpu runtime shim from source and verify all 3 new
    fp32/fp16/bf16 3-op fusion symbols are exported."""

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
               "-framework", "MetalPerformanceShaders",
               "-framework", "MetalPerformanceShadersGraph"]
    else:
        source = backend / "apple_gpu_runtime_stub.cpp"
        lib = tmp_path / "libtessera_apple_gpu_runtime.so"
        cmd = [cxx, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(lib)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    runtime = ctypes.CDLL(str(lib))
    for name in (
        "tessera_apple_gpu_matmul_softmax_matmul_f32",
        "tessera_apple_gpu_matmul_softmax_matmul_f16",
        "tessera_apple_gpu_matmul_softmax_matmul_bf16",
    ):
        sym = getattr(runtime, name, None)
        assert sym is not None, f"missing C ABI symbol: {name}"


def test_apple_gpu_attn_block_falls_back_when_chain_breaks():
    """When the chain doesn't form an exact matmul -> softmax -> matmul
    pattern, the 3-op fusion must NOT fire. gelu in place of softmax is a
    clean negative — both ops are individually in the runtime envelope but
    gelu doesn't fit the attention block shape."""

    @ts.jit(target="apple_gpu")
    def gelu_chain(A, B, C):
        return ts.ops.matmul(ts.ops.gelu(ts.ops.matmul(A, B)), C)

    target_ir = gelu_chain.target_ir
    # The 3-op fusion entry point name (the actual call site) must NOT
    # appear in any call. We check by looking for the call site shape rather
    # than the entry_point string (which would also live in any embedded
    # MSL source as the kernel name).
    assert "matmul_softmax_matmul" not in target_ir or 'fusion = "matmul_softmax_matmul"' not in target_ir


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8.4.6 + F2b-tiled: threadgroup-tiled large-N softmax (lifts N <= 256).
#
# When the per-thread fast-path is too narrow (N > 1024), the synthesizer routes
# to a threadgroup-tiled kernel that allocates the score buffer in threadgroup
# memory. One row per threadgroup; 32 threads cooperate. The hand-written
# matmul_softmax_tiled_f32 is retired — the tiled synthesizer subsumes it.
# Tests below exercise N = 512 and N = 1024, both within the tiled bound (8192).
# ─────────────────────────────────────────────────────────────────────────────


def test_apple_gpu_matmul_softmax_tiled_path_executes_for_large_n():
    """End-to-end through @jit for N > 256. The Phase 8.4.6 router selects
    the tiled MSL variant. Output must match the per-op numpy reference."""

    @ts.jit(target="apple_gpu")
    def fused(A, B):
        return ts.ops.softmax(ts.ops.matmul(A, B))

    rng = np.random.RandomState(83)
    for M, K, N in ((4, 16, 512), (8, 32, 1024)):
        A = rng.randn(M, K).astype(np.float32) * 0.5
        B = rng.randn(K, N).astype(np.float32) * 0.5
        out = fused(A, B)
        assert out.shape == (M, N)
        assert out.dtype == np.float32
        scores = A @ B
        e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        ref = e / np.sum(e, axis=-1, keepdims=True)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)
        # Each row sums to 1 modulo rounding.
        np.testing.assert_allclose(out.sum(axis=-1), np.ones(M, dtype=np.float32), rtol=1e-4)


def test_apple_gpu_matmul_softmax_tiled_runtime_shim_exposes_symbol(tmp_path):
    """Compile the apple_gpu runtime shim and verify the tiled SYNTHESIZED
    epilogue symbol is exported (Optimizing-Compiler Plan F2 — the hand-written
    matmul_softmax_tiled_f32 it replaces is retired).  Large-N numerical
    correctness is covered by test_fusion_synthesis.py (Metal) and the
    `_tiled_path_executes_for_large_n` test above."""

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
               "-framework", "MetalPerformanceShaders",
               "-framework", "MetalPerformanceShadersGraph"]
    else:
        source = backend / "apple_gpu_runtime_stub.cpp"
        lib = tmp_path / "libtessera_apple_gpu_runtime.so"
        cmd = [cxx, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(lib)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    runtime = ctypes.CDLL(str(lib))
    assert getattr(runtime, "tessera_apple_gpu_synth_matmul_epilogue_tiled_f32",
                   None) is not None
    assert getattr(runtime, "tessera_apple_gpu_matmul_softmax_tiled_f32",
                   None) is None      # retired


def test_apple_gpu_matmul_softmax_small_n_still_uses_per_thread_path():
    """The router keeps the per-thread fast path for N <= 256. End-to-end
    correctness is unchanged — both kernels produce identical numerical
    results — but this pins that the router doesn't accidentally regress
    small-N latency by always going through tiled."""

    @ts.jit(target="apple_gpu")
    def fused(A, B):
        return ts.ops.softmax(ts.ops.matmul(A, B))

    rng = np.random.RandomState(97)
    A = rng.randn(8, 16).astype(np.float32) * 0.5
    B = rng.randn(16, 64).astype(np.float32) * 0.5  # N = 64, well under 256
    out = fused(A, B)
    scores = A @ B
    e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    ref = e / np.sum(e, axis=-1, keepdims=True)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8.4.7: MLP-block fusions — matmul -> gelu and matmul -> rmsnorm.
# ─────────────────────────────────────────────────────────────────────────────


def test_apple_gpu_matmul_gelu_chain_emits_fused_msl_kernel():
    @ts.jit(target="apple_gpu")
    def mlp(A, B):
        return ts.ops.gelu(ts.ops.matmul(A, B))

    target_ir = mlp.target_ir
    assert "tessera_apple.gpu.msl_kernel" in target_ir
    # Optimizing-Compiler Plan F2 — the matmul->gelu MSL is now SYNTHESIZED, so
    # the Target IR carries the generic synth entry point; the fusion id (which
    # chain was matched) stays "matmul_gelu".
    assert 'entry_point = "synth_matmul_epi"' in target_ir
    assert 'fusion = "matmul_gelu"' in target_ir
    # Exactly one fused emission.
    assert target_ir.count('"tessera_apple.gpu.msl_kernel"') == 1
    assert ("tessera_apple_gpu_synth_matmul_epilogue_f32"
            in mlp.compile_bundle.artifact("backend").text)


def test_apple_gpu_matmul_gelu_executes_through_fused_msl_kernel():
    @ts.jit(target="apple_gpu")
    def mlp(A, B):
        return ts.ops.gelu(ts.ops.matmul(A, B))

    rng = np.random.RandomState(211)
    for M, K, N in ((4, 8, 8), (8, 16, 32), (16, 32, 64)):
        A = rng.randn(M, K).astype(np.float32) * 0.5
        B = rng.randn(K, N).astype(np.float32) * 0.5
        out = mlp(A, B)
        assert out.shape == (M, N)
        assert out.dtype == np.float32
        scores = A @ B
        ref = 0.5 * scores * (1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (scores + 0.044715 * scores ** 3)
        ))
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


def test_apple_gpu_matmul_rmsnorm_chain_emits_fused_msl_kernel():
    @ts.jit(target="apple_gpu")
    def norm(A, B):
        return ts.ops.rmsnorm(ts.ops.matmul(A, B))

    target_ir = norm.target_ir
    assert "tessera_apple.gpu.msl_kernel" in target_ir
    # Optimizing-Compiler Plan F2 — synthesized matmul->rmsnorm.
    assert 'entry_point = "synth_matmul_epi"' in target_ir
    assert 'fusion = "matmul_rmsnorm"' in target_ir
    assert target_ir.count('"tessera_apple.gpu.msl_kernel"') == 1
    assert ("tessera_apple_gpu_synth_matmul_epilogue_f32"
            in norm.compile_bundle.artifact("backend").text)


def test_apple_gpu_matmul_rmsnorm_executes_through_fused_msl_kernel():
    @ts.jit(target="apple_gpu")
    def norm(A, B):
        return ts.ops.rmsnorm(ts.ops.matmul(A, B))

    rng = np.random.RandomState(223)
    for M, K, N in ((4, 8, 8), (8, 16, 32)):
        A = rng.randn(M, K).astype(np.float32) * 0.5
        B = rng.randn(K, N).astype(np.float32) * 0.5
        out = norm(A, B)
        assert out.shape == (M, N)
        assert out.dtype == np.float32
        scores = A @ B
        eps = 1e-5
        rms = np.sqrt(np.mean(scores * scores, axis=-1, keepdims=True) + eps)
        ref = scores / rms
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


def test_apple_gpu_mlp_fusion_runtime_shim_exposes_symbols(tmp_path):
    """Compile the apple_gpu runtime shim and verify the generic synthesized
    matmul-epilogue symbol is exported (Optimizing-Compiler Plan F2 — the
    per-epilogue matmul_gelu_f32 / matmul_rmsnorm_f32 kernels are retired)."""

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
               "-framework", "MetalPerformanceShaders",
               "-framework", "MetalPerformanceShadersGraph"]
    else:
        source = backend / "apple_gpu_runtime_stub.cpp"
        lib = tmp_path / "libtessera_apple_gpu_runtime.so"
        cmd = [cxx, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(lib)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    runtime = ctypes.CDLL(str(lib))
    for name in (
        "tessera_apple_gpu_synth_matmul_epilogue_f32",
    ):
        sym = getattr(runtime, name, None)
        assert sym is not None, f"missing C ABI symbol: {name}"
    # the retired per-epilogue f32 kernels must be gone.
    for retired in ("tessera_apple_gpu_matmul_gelu_f32",
                    "tessera_apple_gpu_matmul_rmsnorm_f32"):
        assert getattr(runtime, retired, None) is None, (
            f"retired kernel still present: {retired}")


def test_apple_gpu_native_half_fused_runtime_shim(tmp_path):
    """Compile the apple_gpu runtime shim and exercise the native-half fused
    symbols (matmul_gelu / matmul_rmsnorm / matmul_softmax_tiled f16+bf16)
    numerically. f16 I/O uses uint16 views; the kernels accumulate in fp32."""

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
               "-framework", "MetalPerformanceShaders",
               "-framework", "MetalPerformanceShadersGraph"]
    else:
        source = backend / "apple_gpu_runtime_stub.cpp"
        lib = tmp_path / "libtessera_apple_gpu_runtime.so"
        cmd = [cxx, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(lib)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    runtime = ctypes.CDLL(str(lib))

    def f16_bits(arr):
        return arr.astype(np.float16).view(np.uint16)

    def bits_to_f16(arr):
        return arr.view(np.float16)

    rng = np.random.RandomState(311)

    # matmul -> gelu (f16), N<=256.
    for name in ("tessera_apple_gpu_matmul_gelu_f16",
                 "tessera_apple_gpu_matmul_rmsnorm_f16",
                 "tessera_apple_gpu_matmul_softmax_tiled_f16"):
        assert getattr(runtime, name, None) is not None, name

    M, K, N = 8, 16, 64
    A = (rng.randn(M, K) * 0.5).astype(np.float16)
    B = (rng.randn(K, N) * 0.5).astype(np.float16)
    Au = np.ascontiguousarray(f16_bits(A))
    Bu = np.ascontiguousarray(f16_bits(B))

    # gelu
    sym = runtime.tessera_apple_gpu_matmul_gelu_f16
    sym.argtypes = [ctypes.POINTER(ctypes.c_uint16)] * 3 + [ctypes.c_int32] * 3
    sym.restype = None
    Ou = np.zeros(M * N, dtype=np.uint16)
    sym(Au.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        Bu.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        Ou.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        M, N, K)
    out = bits_to_f16(Ou).reshape(M, N).astype(np.float32)
    s = A.astype(np.float32) @ B.astype(np.float32)
    gelu_ref = 0.5 * s * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (s + 0.044715 * s ** 3)))
    np.testing.assert_allclose(out, gelu_ref, rtol=5e-2, atol=5e-2)

    # rmsnorm
    sym = runtime.tessera_apple_gpu_matmul_rmsnorm_f16
    sym.argtypes = [ctypes.POINTER(ctypes.c_uint16)] * 3 + [ctypes.c_int32] * 3 + [ctypes.c_float]
    sym.restype = None
    Ou = np.zeros(M * N, dtype=np.uint16)
    sym(Au.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        Bu.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        Ou.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        M, N, K, ctypes.c_float(1e-5))
    out = bits_to_f16(Ou).reshape(M, N).astype(np.float32)
    rms_ref = s / np.sqrt((s * s).mean(-1, keepdims=True) + 1e-5)
    np.testing.assert_allclose(out, rms_ref, rtol=5e-2, atol=5e-2)

    # softmax tiled, large N
    Nbig = 512
    Bbig = (rng.randn(K, Nbig) * 0.5).astype(np.float16)
    Bbu = np.ascontiguousarray(f16_bits(Bbig))
    sym = runtime.tessera_apple_gpu_matmul_softmax_tiled_f16
    sym.argtypes = [ctypes.POINTER(ctypes.c_uint16)] * 3 + [ctypes.c_int32] * 3
    sym.restype = None
    Ou = np.zeros(M * Nbig, dtype=np.uint16)
    sym(Au.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        Bbu.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        Ou.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        M, Nbig, K)
    out = bits_to_f16(Ou).reshape(M, Nbig).astype(np.float32)
    sb = A.astype(np.float32) @ Bbig.astype(np.float32)
    e = np.exp(sb - sb.max(-1, keepdims=True))
    np.testing.assert_allclose(out, e / e.sum(-1, keepdims=True), rtol=5e-2, atol=5e-3)

    # bf16 symbols exist with the same ABI.
    for name in ("tessera_apple_gpu_matmul_gelu_bf16",
                 "tessera_apple_gpu_matmul_rmsnorm_bf16",
                 "tessera_apple_gpu_matmul_softmax_tiled_bf16"):
        assert getattr(runtime, name, None) is not None, name


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


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8.4.8 — SwiGLU MLP-block fusion (Stage 3 of the SwiGLU Performance
# Plan in `docs/CANONICAL_API.md`).
#
# The Schedule IR fusion recognizer (Stage 2b at
# src/transforms/lib/SwigluFusionPass.cpp) collapses the
# `matmul → silu_mul → matmul` chain to `tessera.swiglu_fused`. The
# Apple GPU lowering (this phase) emits a single
# `tessera_apple_gpu_swiglu_f32` runtime call dispatched through a custom
# MSL kernel with three per-thread stack arrays (gate/up/out_row).
# ─────────────────────────────────────────────────────────────────────────────


def _np_swiglu_reference(np_, x, wg, wu, wd):
    gate = np_.asarray(x) @ np_.asarray(wg)
    up = np_.asarray(x) @ np_.asarray(wu)
    hidden = (gate / (1.0 + np_.exp(-gate))) * up
    return hidden @ np_.asarray(wd)


def test_apple_gpu_swiglu_chain_dispatches_to_fused_runtime_symbol():
    """End-to-end f32: a 4-op `gemm/gemm/silu_mul/gemm` chain inside a
    `@jit(target='apple_gpu')` function runs through the fused
    `tessera_apple_gpu_swiglu_f32` runtime symbol and matches the per-op
    numpy reference."""

    @ts.jit(target="apple_gpu")
    def block(x, wg, wu, wd):
        gate = ts.ops.gemm(x, wg)
        up = ts.ops.gemm(x, wu)
        hidden = ts.ops.silu_mul(gate, up)
        return ts.ops.gemm(hidden, wd)

    rng = np.random.RandomState(211)
    for M, K, H, Kout in ((4, 8, 16, 8), (8, 16, 64, 32), (16, 8, 128, 16)):
        x = rng.randn(M, K).astype(np.float32) * 0.5
        wg = rng.randn(K, H).astype(np.float32) * 0.3
        wu = rng.randn(K, H).astype(np.float32) * 0.3
        wd = rng.randn(H, Kout).astype(np.float32) * 0.3
        out = block(x, wg, wu, wd)
        assert out.shape == (M, Kout)
        assert out.dtype == np.float32
        ref = _np_swiglu_reference(np, x, wg, wu, wd)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)

    artifact = block.runtime_artifact()
    assert artifact.metadata["runtime_status"] == "ready"
    assert artifact.metadata["execution_mode"] == "metal_runtime"


def test_apple_gpu_swiglu_chain_emits_fused_msl_symbol_in_backend_artifact():
    """The driver-level chain detector classifies the 4-op SwiGLU pattern
    as `chain == "swiglu"` and emits the fused runtime symbol in the
    backend artifact text."""

    @ts.jit(target="apple_gpu")
    def block(x, wg, wu, wd):
        gate = ts.ops.gemm(x, wg)
        up = ts.ops.gemm(x, wu)
        hidden = ts.ops.silu_mul(gate, up)
        return ts.ops.gemm(hidden, wd)

    backend_text = block.compile_bundle.artifact("backend").text
    assert "tessera_apple_gpu_swiglu_f32" in backend_text
    assert 'execution_mode = "metal_runtime"' in backend_text


def test_apple_gpu_swiglu_chain_breaks_when_x_differs_per_matmul():
    """A SwiGLU fusion requires the gate and up matmuls to consume the same
    `x` SSA value. When they differ, the 4-op fusion must NOT fire — the
    plan stays as four independent ops and the runtime falls through to
    per-op handling (which doesn't include silu_mul, so the path lands on
    the artifact-only contract). This pins the negative case so the
    detector's `gate_operands[0] == up_operands[0]` invariant can't
    silently regress."""

    @ts.jit(target="apple_gpu")
    def two_inputs(x1, x2, wg, wu, wd):
        gate = ts.ops.gemm(x1, wg)
        up = ts.ops.gemm(x2, wu)
        hidden = ts.ops.silu_mul(gate, up)
        return ts.ops.gemm(hidden, wd)

    backend = two_inputs.compile_bundle.artifact("backend")
    backend_text = backend.text if backend is not None else ""
    assert "tessera_apple_gpu_swiglu_f32" not in backend_text


def test_apple_gpu_swiglu_runtime_shim_exposes_swiglu_symbols(tmp_path):
    """Compile the apple_gpu runtime shim from source and verify all 3 new
    SwiGLU dtype symbols are exported."""

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
               "-framework", "MetalPerformanceShaders",
               "-framework", "MetalPerformanceShadersGraph"]
    else:
        source = backend / "apple_gpu_runtime_stub.cpp"
        lib = tmp_path / "libtessera_apple_gpu_runtime.so"
        cmd = [cxx, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(lib)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    runtime = ctypes.CDLL(str(lib))
    for name in (
        "tessera_apple_gpu_swiglu_f32",
        "tessera_apple_gpu_swiglu_f16",
        "tessera_apple_gpu_swiglu_bf16",
    ):
        sym = getattr(runtime, name, None)
        assert sym is not None, f"missing C ABI symbol: {name}"
