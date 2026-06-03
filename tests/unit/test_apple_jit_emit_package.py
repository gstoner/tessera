"""PK8a wiring — ``@jit(target="apple_gpu")`` → ``.mtlpackage`` emission.

This proves the recognizer is wired into the *actual* compile path: a
decorated Apple-GPU function exposes ``.recognized_package`` (the shape-free
op/chain identity, recognized from the live Graph IR's op names), and
``.emit_package(example_args=...)`` authors a real packaged kernel that
loads + dispatches through PK1-PK7.

Recognition is pure/device-free, so those assertions run on any host; the
emit-and-dispatch assertions gate on packaged-ML availability.
"""

from __future__ import annotations

import numpy as np
import pytest
import tessera
from tessera import Tensor

from tessera.apple_mlpkg import (
    compile_mlpackage,
    first_function_name,
    packaged_ml_available,
    packaged_ml_skip_reason,
)


# Module-level so @jit can inspect source (file-based).
@tessera.jit(target="apple_gpu")
def _mm(A: Tensor[8, 6], B: Tensor[6, 5]) -> Tensor[8, 5]:
    return tessera.ops.matmul(A, B)


@tessera.jit(target="apple_gpu")
def _silu(X: Tensor[4, 8]) -> Tensor[4, 8]:
    return tessera.ops.silu(X)


@tessera.jit(target="apple_gpu")
def _attn_scores(A: Tensor[4, 6], B: Tensor[6, 5]) -> Tensor[4, 5]:
    return tessera.ops.softmax(tessera.ops.matmul(A, B))


@tessera.jit(target="apple_cpu")
def _mm_cpu(A: Tensor[8, 6], B: Tensor[6, 5]) -> Tensor[8, 5]:
    return tessera.ops.matmul(A, B)


# PK8e — dispatch_via_package: execution routes through the authored package.
@tessera.jit(target="apple_gpu", dispatch_via_package=True)
def _mm_pkg(A: Tensor[8, 6], B: Tensor[6, 5]) -> Tensor[8, 5]:
    return tessera.ops.matmul(A, B)


@tessera.jit(target="apple_gpu", dispatch_via_package=True)
def _attn_pkg(A: Tensor[4, 6], B: Tensor[6, 5]) -> Tensor[4, 5]:
    return tessera.ops.softmax(tessera.ops.matmul(A, B))


# PK8g — "auto": route only fused chains through the package, single ops live.
@tessera.jit(target="apple_gpu", dispatch_via_package="auto")
def _mm_auto(A: Tensor[8, 6], B: Tensor[6, 5]) -> Tensor[8, 5]:
    return tessera.ops.matmul(A, B)


@tessera.jit(target="apple_gpu", dispatch_via_package="auto")
def _attn_auto(A: Tensor[4, 6], B: Tensor[6, 5]) -> Tensor[4, 5]:
    return tessera.ops.softmax(tessera.ops.matmul(A, B))


# ── recognition wired into compile (pure, no GPU) ────────────────────────


def test_jit_apple_gpu_matmul_is_recognized():
    rec = _mm.recognized_package
    assert rec is not None
    assert rec.kind == "matmul" and rec.name == "matmul" and rec.arity == 2


def test_jit_apple_gpu_single_op_is_recognized():
    rec = _silu.recognized_package
    assert rec is not None
    assert rec.kind == "op" and rec.name == "silu"


def test_jit_apple_gpu_chain_is_recognized():
    rec = _attn_scores.recognized_package
    assert rec is not None
    assert rec.kind == "chain" and rec.name == "matmul_softmax"


def test_jit_non_apple_gpu_target_has_no_recognized_package():
    """Recognition only runs for target='apple_gpu' — a CPU-target jit must
    not carry a package plan."""
    assert _mm_cpu.recognized_package is None


def test_emit_package_without_examples_symbolic_returns_none():
    """A symbolic-shape function can't be authored without concrete example
    shapes — emit returns None rather than guessing. (Defined inside the test
    so the symbolic Tensor types are bound to local names — the string dim
    labels aren't seen as forward refs by the linter.)"""
    MK = Tensor["M", "K"]
    KN = Tensor["K", "N"]
    MN = Tensor["M", "N"]

    @tessera.jit(target="apple_gpu")
    def _sym(A: MK, B: KN) -> MN:
        return tessera.ops.matmul(A, B)

    assert _sym.recognized_package is not None  # op recognized
    assert _sym._static_input_shapes() is None  # but no concrete shapes
    assert _sym.emit_package() is None  # so no authoring without examples


def test_static_input_shapes_from_integer_annotations():
    """Integer annotations (``Tensor[8, 6]``) yield concrete shapes via arg
    dim_names — the source for compile-time auto-emit. Pure / no GPU."""
    assert _mm._static_input_shapes() == [(8, 6), (6, 5)]
    assert _silu._static_input_shapes() == [(4, 8)]


# ── emit + dispatch from @jit (gated on packaged ML) ─────────────────────


def _require_packaged_ml():
    if not packaged_ml_available():
        pytest.skip(packaged_ml_skip_reason() or "packaged ML unavailable")


def _dispatch(pkg, inputs, out_shape):
    fn = first_function_name(pkg) or "main"
    pipe = compile_mlpackage(pkg, function_name=fn)
    assert pipe is not None
    try:
        assert pipe.prepare_tensors()
        for i, arr in enumerate(inputs):
            assert pipe.fill_input_at(i, arr.astype(np.float32).tobytes())
        assert pipe.dispatch(timeout_ms=30_000)
        r, c = out_shape
        raw = pipe.read_output_at(len(inputs), r * c * 4)
        return np.frombuffer(raw, dtype=np.float32).reshape(r, c)
    finally:
        pipe.destroy()


def test_jit_emit_matmul_package_dispatches(tmp_path):
    _require_packaged_ml()
    rng = np.random.default_rng(50)
    a = rng.standard_normal((8, 6)).astype(np.float32)
    b = rng.standard_normal((6, 5)).astype(np.float32)
    pkg = _mm.emit_package(tmp_path / "mm.mtlpackage", example_args=[a, b])
    assert pkg is not None
    out = _dispatch(pkg, [a, b], (8, 5))
    assert np.allclose(out, a @ b, rtol=1e-4, atol=2e-4)


def test_jit_emit_chain_package_dispatches(tmp_path):
    _require_packaged_ml()
    rng = np.random.default_rng(51)
    M, K, N = 4, 6, 5
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    pkg = _attn_scores.emit_package(
        tmp_path / "ms.mtlpackage", example_args=[a, b])
    assert pkg is not None
    out = _dispatch(pkg, [a, b], (M, N))
    ab = a @ b
    e = np.exp(ab - ab.max(axis=1, keepdims=True))
    assert np.allclose(out, e / e.sum(axis=1, keepdims=True),
                       rtol=1e-4, atol=2e-4)


def test_jit_emit_single_op_package_dispatches(tmp_path):
    _require_packaged_ml()
    rng = np.random.default_rng(52)
    x = rng.standard_normal((4, 8)).astype(np.float32)
    pkg = _silu.emit_package(tmp_path / "silu.mtlpackage", example_args=[x])
    assert pkg is not None
    out = _dispatch(pkg, [x], (4, 8))
    assert np.allclose(out, x * (1.0 / (1.0 + np.exp(-x))),
                       rtol=1e-4, atol=2e-4)


def test_jit_emit_rejects_non_fp32_examples(tmp_path):
    """Authoring is fp32-only — a non-fp32 example arg yields None, not a
    wrong-dtype package."""
    a = np.ones((8, 6), dtype=np.float64)
    b = np.ones((6, 5), dtype=np.float64)
    assert _mm.emit_package(tmp_path / "x.mtlpackage", example_args=[a, b]) is None


def test_emit_package_from_static_annotations_no_examples(tmp_path):
    """PK8d — a statically-annotated fn authors with NO example args, deriving
    shapes from the integer annotations (Tensor[8,6])."""
    _require_packaged_ml()
    pkg = _mm.emit_package(tmp_path / "auto.mtlpackage")
    assert pkg is not None
    rng = np.random.default_rng(60)
    a = rng.standard_normal((8, 6)).astype(np.float32)
    b = rng.standard_normal((6, 5)).astype(np.float32)
    out = _dispatch(pkg, [a, b], (8, 5))
    assert np.allclose(out, a @ b, rtol=1e-4, atol=2e-4)


# ── compile-time auto-emit (@jit(emit_package=...)) ──────────────────────


def test_jit_emit_package_flag_auto_emits_at_compile():
    """PK8d — ``@jit(target="apple_gpu", emit_package=True)`` authors the
    package at decoration when annotations are static; no manual call."""
    _require_packaged_ml()

    @tessera.jit(target="apple_gpu", emit_package=True)
    def _auto(A: Tensor[8, 6], B: Tensor[6, 5]) -> Tensor[8, 5]:
        return tessera.ops.matmul(A, B)

    assert _auto._emitted_package_path is not None
    rng = np.random.default_rng(61)
    a = rng.standard_normal((8, 6)).astype(np.float32)
    b = rng.standard_normal((6, 5)).astype(np.float32)
    out = _dispatch(_auto._emitted_package_path, [a, b], (8, 5))
    assert np.allclose(out, a @ b, rtol=1e-4, atol=2e-4)


def test_jit_emit_package_flag_to_explicit_path(tmp_path):
    """``emit_package="<path>"`` authors to that path at compile."""
    _require_packaged_ml()
    target = tmp_path / "explicit.mtlpackage"

    @tessera.jit(target="apple_gpu", emit_package=str(target))
    def _autop(A: Tensor[4, 6], B: Tensor[6, 5]) -> Tensor[4, 5]:
        return tessera.ops.softmax(tessera.ops.matmul(A, B))

    assert _autop._emitted_package_path == str(target)
    assert target.is_dir()


def test_jit_emit_package_flag_rejects_non_apple_gpu():
    """The flag is apple_gpu-only — a CPU target raises at decoration."""
    from tessera.compiler.jit import TesseraJitError
    with pytest.raises(TesseraJitError):
        @tessera.jit(target="apple_cpu", emit_package=True)
        def _bad(A: Tensor[8, 6], B: Tensor[6, 5]) -> Tensor[8, 5]:
            return tessera.ops.matmul(A, B)


def test_jit_emit_package_flag_symbolic_is_silent_noop():
    """emit_package=True on a symbolic fn doesn't error — it just leaves the
    emitted path None (best-effort AOT convenience). Pure / no GPU."""
    MK = Tensor["M", "K"]
    KN = Tensor["K", "N"]
    MN = Tensor["M", "N"]

    @tessera.jit(target="apple_gpu", emit_package=True)
    def _sym(A: MK, B: KN) -> MN:
        return tessera.ops.matmul(A, B)

    assert _sym._emitted_package_path is None


# ── PK8e — dispatch through the authored package (per-shape cache) ────────


def test_dispatch_via_package_flag_set_and_caches_empty_before_call():
    """The flag is recorded; caches start empty (pure, no GPU)."""
    assert _mm_pkg.dispatch_via_package is True
    assert _mm_pkg._package_path_cache == {}
    assert _mm_pkg._package_pipeline_cache == {}


def test_dispatch_via_package_rejects_non_apple_gpu():
    from tessera.compiler.jit import TesseraJitError
    with pytest.raises(TesseraJitError):
        @tessera.jit(target="apple_cpu", dispatch_via_package=True)
        def _bad(A: Tensor[8, 6], B: Tensor[6, 5]) -> Tensor[8, 5]:
            return tessera.ops.matmul(A, B)


def test_dispatch_via_package_matmul_matches_numpy():
    """Calling the jitted fn executes *through the authored package* and
    matches numpy — the package is now the execution path."""
    _require_packaged_ml()
    rng = np.random.default_rng(70)
    a = rng.standard_normal((8, 6)).astype(np.float32)
    b = rng.standard_normal((6, 5)).astype(np.float32)
    out = _mm_pkg(a, b)
    assert np.allclose(out, a @ b, rtol=1e-4, atol=2e-4)
    # A package + a prepared pipeline were cached for this shape.
    assert ("matmul", (8, 6, 5)) in _mm_pkg._package_path_cache
    assert ("matmul", (8, 6, 5)) in _mm_pkg._package_pipeline_cache


def test_dispatch_via_package_caches_per_shape():
    """Same shape reuses the cache; a new shape adds one entry."""
    _require_packaged_ml()
    rng = np.random.default_rng(71)
    a = rng.standard_normal((8, 6)).astype(np.float32)
    b = rng.standard_normal((6, 5)).astype(np.float32)
    _mm_pkg(a, b)
    _mm_pkg(a, b)  # repeat — must not author again
    n_after_same = len(_mm_pkg._package_path_cache)
    a2 = rng.standard_normal((4, 4)).astype(np.float32)
    b2 = rng.standard_normal((4, 4)).astype(np.float32)
    out = _mm_pkg(a2, b2)
    assert np.allclose(out, a2 @ b2, rtol=1e-4, atol=2e-4)
    assert len(_mm_pkg._package_path_cache) == n_after_same + 1


def test_dispatch_via_package_chain_matches_numpy():
    _require_packaged_ml()
    rng = np.random.default_rng(72)
    a = rng.standard_normal((4, 6)).astype(np.float32)
    b = rng.standard_normal((6, 5)).astype(np.float32)
    out = _attn_pkg(a, b)
    ab = a @ b
    e = np.exp(ab - ab.max(axis=1, keepdims=True))
    assert np.allclose(out, e / e.sum(axis=1, keepdims=True),
                       rtol=1e-4, atol=2e-4)


def test_dispatch_via_package_falls_back_for_non_fp32():
    """A non-fp32 call can't be packaged (fp32-only authoring) — it falls
    back to the normal path and authors NO package for that shape."""
    _require_packaged_ml()
    a = np.ones((8, 6), dtype=np.float64)
    b = np.ones((6, 5), dtype=np.float64)
    before = len(_mm_pkg._package_path_cache)
    out = _mm_pkg(a, b)  # falls through to the live MPS path
    assert np.allclose(out, a @ b, rtol=1e-4, atol=1e-3)
    # No fp64 package was authored — proves the fallback routed away.
    assert len(_mm_pkg._package_path_cache) == before


# ── PK8g — "auto" heuristic: chains → package, single ops → live ─────────


def test_resolve_policy_default_is_opt_in_not_auto(monkeypatch):
    """PK8h deliberate decision — auto is NOT the unconditional default
    (it SIGABRTs the suite under MTL4-ML-compile volume). Default (None) on
    apple_gpu resolves to False; the env switch opts into auto globally.
    Pure / no GPU."""
    from tessera.compiler.jit import _resolve_dispatch_via_package as _r
    monkeypatch.delenv("TESSERA_APPLE_GPU_PACKAGE_AUTOROUTE", raising=False)
    assert _r(None, "apple_gpu") is False           # default = live lane
    assert _r(None, "apple_cpu") is False            # non-apple never routes
    assert _r(True, "apple_gpu") is True             # explicit wins
    assert _r("auto", "apple_gpu") == "auto"
    assert _r(False, "apple_gpu") is False
    # Global opt-in via env.
    monkeypatch.setenv("TESSERA_APPLE_GPU_PACKAGE_AUTOROUTE", "1")
    assert _r(None, "apple_gpu") == "auto"
    monkeypatch.setenv("TESSERA_APPLE_GPU_PACKAGE_AUTOROUTE", "off")
    assert _r(None, "apple_gpu") is False
    # Explicit per-fn choice still overrides the env switch.
    monkeypatch.setenv("TESSERA_APPLE_GPU_PACKAGE_AUTOROUTE", "1")
    assert _r(False, "apple_gpu") is False


def test_default_apple_gpu_jit_is_opt_in_live():
    """A plain ``@jit(target="apple_gpu")`` (no flag) keeps the live lane —
    the default is unchanged. Pure / no GPU."""
    assert _mm.dispatch_via_package is False
    assert _attn_scores.dispatch_via_package is False


def test_auto_flag_value_recorded():
    """``dispatch_via_package="auto"`` is preserved (not coerced to bool).
    Pure / no GPU."""
    assert _mm_auto.dispatch_via_package == "auto"
    assert _attn_auto.dispatch_via_package == "auto"


def test_auto_routes_single_matmul_to_live_lane():
    """In auto mode a single matmul stays on the live lane — no package is
    authored (benchmark: package is 1.3–2.8× slower for plain GEMM)."""
    _require_packaged_ml()
    rng = np.random.default_rng(80)
    a = rng.standard_normal((8, 6)).astype(np.float32)
    b = rng.standard_normal((6, 5)).astype(np.float32)
    out = _mm_auto(a, b)
    assert np.allclose(out, a @ b, rtol=1e-4, atol=2e-4)
    assert _mm_auto._package_path_cache == {}  # never authored


def test_auto_routes_fused_chain_to_package_lane():
    """In auto mode a fused chain DOES route through the package (benchmark:
    up to ~14× faster) — a package is authored + cached."""
    _require_packaged_ml()
    rng = np.random.default_rng(81)
    a = rng.standard_normal((4, 6)).astype(np.float32)
    b = rng.standard_normal((6, 5)).astype(np.float32)
    out = _attn_auto(a, b)
    ab = a @ b
    e = np.exp(ab - ab.max(axis=1, keepdims=True))
    assert np.allclose(out, e / e.sum(axis=1, keepdims=True),
                       rtol=1e-4, atol=2e-4)
    assert ("matmul_softmax", (4, 6, 5)) in _attn_auto._package_path_cache
