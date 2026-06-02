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


def test_emit_package_without_examples_or_static_dims_returns_none():
    """Symbolic-shape IR (the common case) can't be authored without concrete
    example shapes — emit returns None rather than guessing."""
    assert _mm.emit_package() is None


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
