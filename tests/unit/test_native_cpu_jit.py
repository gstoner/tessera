"""Phase 4 — the tessera_jit MLIR→LLVM lane is the executed CPU path.

`@jit(target="cpu")` now runs the whole graph through tessera_jit (tessera-to-
linalg → bufferize → loops → LLVM, optLevel=2) for the covered f32 op set,
instead of the numpy reference interpreter. This is the keystone that makes the
real compiler the executed path — so C++ IR optimizations reach execution.

The proof-of-execution counter (`_jit_boundary.invocation_count`) distinguishes
"the JIT actually ran" from a silent numpy fallback. Correctness is proven by
equivalence to numpy across the covered ops — a fallback handles "couldn't run",
never "ran wrong". See docs/audit/compiler/COMPILER_AUDIT.md (Phase 4).

NB: the functions below are real ``def``s, not lambdas — ``@jit`` source-inspects
the body to build the graph, and a lambda would fall through to eager Python.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

import tessera as ts
from tessera import _jit_boundary as jb

# The CPU JIT lane is the *executed* path here, so every test needs the compiled
# `libtessera_jit` dylib. CI's unit lane is Python-only (no C++ build), so skip
# the whole module when the lib isn't built rather than hard-failing — exactly
# the pattern the Apple/lit suites use when their backend is unavailable. Locally
# (where `ninja -C build tessera_jit` has run) every test stays live.
pytestmark = pytest.mark.skipif(
    not jb.is_available(),
    reason="libtessera_jit not built; run `ninja -C build tessera_jit` "
           "(or set TESSERA_JIT_LIB)")

_RNG = np.random.default_rng(20260615)


def _f32(*shape):
    return _RNG.standard_normal(shape).astype(np.float32)


def _ran_through_jit(fn, *arrays):
    n0 = jb.invocation_count()
    out = np.asarray(fn(*arrays))
    return out, (jb.invocation_count() - n0) == 1


# ── numpy references ────────────────────────────────────────────────────────
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _gelu(x):  # tanh approximation, matching the linalg/MSL kernel
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _rmsnorm(x, eps=1e-5):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)


def _layer_norm(x, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + eps)


# ── jitted ops (real defs) ──────────────────────────────────────────────────
def _relu(x):
    return ts.ops.relu(x)


def _sig(x):
    return ts.ops.sigmoid(x)


def _tanh(x):
    return ts.ops.tanh(x)


def _silu(x):
    return ts.ops.silu(x)


def _gel(x):
    return ts.ops.gelu(x)


def _sm(x):
    return ts.ops.softmax(x)


def _rms(x):
    return ts.ops.rmsnorm(x)


def _ln(x):
    return ts.ops.layer_norm(x)


_UNARY_CASES = {
    "relu": (_relu, lambda x: np.maximum(0.0, x)),
    "sigmoid": (_sig, _sigmoid),
    "tanh": (_tanh, np.tanh),
    "silu": (_silu, lambda x: x * _sigmoid(x)),
    "gelu": (_gel, _gelu),
    "softmax": (_sm, _softmax),
    "rmsnorm": (_rms, _rmsnorm),
    "layer_norm": (_ln, _layer_norm),
}


@pytest.mark.parametrize("name", list(_UNARY_CASES))
def test_unary_op_runs_through_jit_and_matches_numpy(name):
    build, ref = _UNARY_CASES[name]
    fn = ts.jit(target="cpu")(build)
    x = _f32(4, 16)
    got, used_jit = _ran_through_jit(fn, x)
    assert used_jit, f"{name}: did not route through the tessera_jit lane"
    np.testing.assert_allclose(got, ref(x), rtol=1e-4, atol=1e-5)


def _add(a, b):
    return ts.ops.add(a, b)


def _sub(a, b):
    return ts.ops.sub(a, b)


def _mul(a, b):
    return ts.ops.mul(a, b)


def _div(a, b):
    return ts.ops.div(a, b)


_BINARY_CASES = {
    "add": (_add, lambda a, b: a + b),
    "sub": (_sub, lambda a, b: a - b),
    "mul": (_mul, lambda a, b: a * b),
    "div": (_div, lambda a, b: a / b),
}


@pytest.mark.parametrize("name", list(_BINARY_CASES))
def test_binary_op_runs_through_jit_and_matches_numpy(name):
    build, ref = _BINARY_CASES[name]
    fn = ts.jit(target="cpu")(build)
    a, b = _f32(4, 16), _f32(4, 16)
    if name == "div":
        b = b + 2.0
    got, used_jit = _ran_through_jit(fn, a, b)
    assert used_jit, f"{name}: did not route through the tessera_jit lane"
    np.testing.assert_allclose(got, ref(a, b), rtol=1e-5, atol=1e-5)


def _trans(x):
    return ts.ops.transpose(x)


def test_transpose_runs_through_jit():
    fn = ts.jit(target="cpu")(_trans)
    x = _f32(4, 7)
    got, used_jit = _ran_through_jit(fn, x)
    assert used_jit
    np.testing.assert_allclose(got, x.T, rtol=1e-6, atol=1e-6)


def _attn(a, b, c):
    return ts.ops.matmul(ts.ops.softmax(ts.ops.matmul(a, b)), c)


def test_multi_op_attention_block_matches_numpy():
    fn = ts.jit(target="cpu")(_attn)
    a, b, c = _f32(4, 8), _f32(8, 6), _f32(6, 5)
    got, used_jit = _ran_through_jit(fn, a, b, c)
    assert used_jit
    np.testing.assert_allclose(got, _softmax(a @ b) @ c, rtol=1e-4, atol=1e-5)


def _mlp(x, w, b):
    return ts.ops.gelu(ts.ops.add(ts.ops.matmul(x, w), b))


def test_multi_op_mlp_chain_matches_numpy():
    fn = ts.jit(target="cpu")(_mlp)
    x, w, b = _f32(4, 8), _f32(8, 6), _f32(4, 6)
    got, used_jit = _ran_through_jit(fn, x, w, b)
    assert used_jit
    np.testing.assert_allclose(got, _gelu(x @ w + b), rtol=1e-4, atol=1e-5)


def test_f64_runs_through_jit_at_exact_precision():
    # f64 is wired into the GraphFn boundary table (2026-06-16): it accumulates
    # in f64 throughout (the matmul/reduce low-precision-→f32 rule never fires),
    # so it is the exact-precision lane for gradient-checking / validation.
    fn = ts.jit(target="cpu")(_gel)
    x = _f32(4, 8).astype(np.float64)
    got, used_jit = _ran_through_jit(fn, x)
    assert used_jit, "f64 must route through the jit lane"
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, _gelu(x), rtol=1e-6, atol=1e-12)


def test_f64_gemm_is_exact_over_k():
    # The f64 lane's K-reduction accumulates in f64 (not f32), so a GEMM matches
    # numpy to ~machine epsilon — far tighter than the f32 lane's ~1e-6. A 2-op
    # program (matmul + add) routes through the general tessera_jit lane (a lone
    # rank-2 GEMM takes the f32-only Accelerate fast path, so we pair it with an
    # add). If the K-reduction were f32-emulated the error would be ~1e-6 even
    # after the f64 add — this distinguishes real f64 codegen from emulation.
    def prog(a, b, c):
        return ts.ops.add(ts.ops.matmul(a, b), c)

    fn = ts.jit(target="cpu")(prog)
    rng = np.random.default_rng(7)
    a = rng.standard_normal((6, 32)).astype(np.float64)
    b = rng.standard_normal((32, 5)).astype(np.float64)
    c = rng.standard_normal((6, 5)).astype(np.float64)
    got, used_jit = _ran_through_jit(fn, a, b, c)
    assert used_jit and got.dtype == np.float64
    np.testing.assert_allclose(got, a @ b + c, rtol=0, atol=1e-12)


# Apple M1 Max native NEON dtypes for the lane: f32 + f16 (ARMv8.2-A FP16).
# bf16 is correct but emulated via f32 in-kernel (M1 predates ARMv8.6 BF16).
_DTYPE_TOL = {"f16": (3e-2, 3e-2), "bf16": (2e-1, 2e-1)}


def _as_elem(arr, elem):
    if elem == "f16":
        return arr.astype(np.float16)
    if elem == "bf16":
        ml = pytest.importorskip("ml_dtypes")
        return arr.astype(ml.bfloat16)
    return arr


@pytest.mark.parametrize("elem", ["f16", "bf16"])
def test_low_precision_attention_runs_through_jit(elem):
    rtol, atol = _DTYPE_TOL[elem]
    fn = ts.jit(target="cpu")(_attn)
    a, b, c = _f32(4, 8), _f32(8, 6), _f32(6, 5)
    aa, bb, cc = (_as_elem(a, elem), _as_elem(b, elem), _as_elem(c, elem))
    got, used_jit = _ran_through_jit(fn, aa, bb, cc)
    assert used_jit, f"{elem}: did not route through the tessera_jit lane"
    # f32 reference (the kernel accumulates matmul in f32, ABI §12.5).
    ref = _softmax(a @ b) @ c
    np.testing.assert_allclose(np.asarray(got).astype(np.float32), ref,
                               rtol=rtol, atol=atol)


def test_mixed_dtype_falls_back_to_numpy():
    fn = ts.jit(target="cpu")(_mlp)
    x, w = _f32(4, 8).astype(np.float16), _f32(8, 6)  # f16 + f32 mix
    b = _f32(4, 6)
    n0 = jb.invocation_count()
    np.asarray(fn(x, w, b))
    assert jb.invocation_count() == n0, "mixed dtypes must not route through jit"


# ── Phase 1: Graph-IR folders/canonicalizers, observable end-to-end ─────────
def _transpose_sq(x):
    return ts.ops.transpose(ts.ops.transpose(x))


def test_transpose_of_transpose_folds_and_is_identity():
    # The tessera_jit pipeline runs createCanonicalizerPass before lowering, so
    # transpose(transpose(x)) folds to x (TransposeOp::getCanonicalizationPatterns)
    # on the executed CPU path. It must run through the jit and return x exactly.
    fn = ts.jit(target="cpu")(_transpose_sq)
    x = _f32(4, 7)
    got, used_jit = _ran_through_jit(fn, x)
    assert used_jit
    np.testing.assert_array_equal(got, x)


def _matmul_transposed_a(a, b):
    return ts.ops.matmul(ts.ops.transpose(a), b)


def test_transpose_into_matmul_folds_on_executed_path():
    # MatmulOp::getCanonicalizationPatterns (the per-op-hook twin of
    # CanonicalizeTesseraIR's TransposeIntoMatmul) now fires under the generic
    # --canonicalize the tessera_jit CPU lane runs, so matmul(transpose(A), B)
    # folds the transpose into the transposeA flag on the executed path and
    # equals A.T @ B numerically.
    fn = ts.jit(target="cpu")(_matmul_transposed_a)
    a = _f32(8, 4)
    b = _f32(8, 16)
    got, used_jit = _ran_through_jit(fn, a, b)
    assert used_jit
    np.testing.assert_allclose(got, a.T @ b, rtol=1e-5, atol=1e-5)


# ── Phase 2: the executed GEMM uses the linalg K-reduction loop ─────────────
def test_gemm_multi_op_is_exact_over_k():
    # A multi-op graph routes matmul through tessera_jit (tessera.matmul →
    # linalg.matmul → ConvertLinalgToLoops M×N×K nest). Exactly-representable
    # integer values prove the K-reduction accumulates every term.
    def chain(a, b):
        return ts.ops.relu(ts.ops.matmul(a, b))

    fn = ts.jit(target="cpu")(chain)
    a = np.arange(4 * 8, dtype=np.float32).reshape(4, 8) / 64.0
    b = np.arange(8 * 6, dtype=np.float32).reshape(8, 6) / 64.0
    got, used_jit = _ran_through_jit(fn, a, b)
    assert used_jit
    np.testing.assert_allclose(got, np.maximum(0.0, a @ b), rtol=1e-5, atol=1e-5)


def test_disable_env_forces_numpy_fallback(monkeypatch):
    monkeypatch.setenv("TESSERA_DISABLE_CPU_JIT", "1")
    fn = ts.jit(target="cpu")(_gel)
    x = _f32(4, 8)
    got, used_jit = _ran_through_jit(fn, x)
    assert not used_jit, "TESSERA_DISABLE_CPU_JIT must disable the jit lane"
    np.testing.assert_allclose(got, _gelu(x), rtol=1e-4, atol=1e-5)


def _mean_last(x):
    return ts.ops.mean(x, axis=-1)


def test_unsupported_op_falls_back_without_jit():
    # A reduction (mean) is not in the GraphFn lane today → numpy fallback.
    fn = ts.jit(target="cpu")(_mean_last)
    x = _f32(4, 8)
    n0 = jb.invocation_count()
    got = np.asarray(fn(x))
    assert jb.invocation_count() == n0, "unsupported op must not invoke the jit"
    np.testing.assert_allclose(got, x.mean(-1), rtol=1e-5, atol=1e-5)


# ── opt-in linalg→vector GEMM lane (TESSERA_JIT_VECTORIZE) ────────────────────
_RUNNER_UTILS = os.environ.get(
    "TESSERA_MLIR_C_RUNNER_UTILS",
    "/opt/homebrew/opt/llvm/lib/libmlir_c_runner_utils.dylib",
)


@pytest.mark.integration
@pytest.mark.compiler_tool
@pytest.mark.compiler_cpu
def test_vectorize_lane_correct_in_and_out_of_envelope(monkeypatch):
    from tests._support.apple import require_darwin_host

    require_darwin_host()
    assert os.path.exists(_RUNNER_UTILS), "MLIR C runner utils dylib is unavailable"
    # The opt-in transform-interpreter tile+vectorize lane: within the size
    # envelope (default <=2048) a supported MLIR toolchain register-tiles and
    # vectorizes the matmul. MLIR 23 deliberately uses the scalar JIT pipeline
    # because its one-shot bufferizer aborts on the transformed tensor IR.
    # Either way, in- and out-of-envelope requests must be correct and
    # crash-free. To exercise the out-of-envelope guard without
    # compiling a giant matmul, we pin a tiny MAXDIM and check a size just past it.
    monkeypatch.setenv("TESSERA_JIT_VECTORIZE", "1")

    def prog(a, b, c):
        return ts.ops.add(ts.ops.matmul(a, b), c)

    fn = ts.jit(target="cpu")(prog)
    rng = np.random.default_rng(0)
    # In-envelope (vectorized): a spread of sizes incl. non-square / remainder peel.
    for S in (64, 96, 384):
        a = rng.standard_normal((S, S)).astype(np.float32)
        b = rng.standard_normal((S, S)).astype(np.float32)
        c = rng.standard_normal((S, S)).astype(np.float32)
        n0 = jb.invocation_count()
        out = np.asarray(fn(a, b, c))
        assert jb.invocation_count() == n0 + 1, f"S={S} must run through the jit"
        np.testing.assert_allclose(out, a @ b + c, rtol=1e-3, atol=1e-3)

    # Out-of-envelope (scalar JIT fallback): pin MAXDIM=128 and run a 256 program.
    # Still routes through the jit, still correct — just no vectorize transform.
    monkeypatch.setenv("TESSERA_JIT_VECTORIZE_MAXDIM", "128")
    fn2 = ts.jit(target="cpu")(prog)
    S = 256
    a = rng.standard_normal((S, S)).astype(np.float32)
    b = rng.standard_normal((S, S)).astype(np.float32)
    c = rng.standard_normal((S, S)).astype(np.float32)
    n0 = jb.invocation_count()
    out = np.asarray(fn2(a, b, c))
    assert jb.invocation_count() == n0 + 1, "out-of-envelope must still run the jit"
    np.testing.assert_allclose(out, a @ b + c, rtol=1e-3, atol=1e-3)
