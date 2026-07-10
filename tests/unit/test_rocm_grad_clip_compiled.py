"""grad_clip_norm lane on AMD ROCm gfx1151 (§5.5) — global gradient-norm
clipping ``g * min(1, max_norm/||g||)`` composed on the device reduce kernel:
the L2 norm's global sum-of-squares runs on the gfx1151 row-reduction kernel
(the FLOP-heavy O(n) part), host does sqrt + the clip scale. Reachable via
``compiler_path="rocm_grad_clip_compiled"``. Validated vs optim.clip_grad_norm
(single-tensor form) on gfx1151. Skip-clean: tessera-opt not built / no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import optim


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _art(rt, max_norm, norm_type=2.0):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_grad_clip_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["g"], "output_name": "o",
        "ops": [{"op_name": "tessera.grad_clip_norm", "result": "o",
                 "operands": ["g"],
                 "kwargs": {"max_norm": max_norm, "norm_type": norm_type}}]})


def _run(rt, g, max_norm, norm_type=2.0):
    res = rt.launch(_art(rt, max_norm, norm_type), (g,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_grad_clip_compiled"
    assert res["execution_kind"] == "native_gpu"
    return np.asarray(res["output"], np.float32)


def _ref(g, max_norm, norm_type=2.0):
    """Single-tensor optim.clip_grad_norm — total in f64 (tree_l2_norm)."""
    g64 = g.astype(np.float64)
    if norm_type == float("inf"):
        total = float(np.max(np.abs(g64))) if g.size else 0.0
    else:
        total = float(np.sqrt(np.sum(g64 * g64)))
    scale = min(1.0, max_norm / (total + 1e-12))
    return (g * np.float32(scale)).astype(np.float32)


_RNG = np.random.default_rng(19)


@pytest.mark.parametrize("shape,max_norm", [
    ((256,), 1.0),
    ((16, 32), 0.5),
    ((4, 8, 8), 5.0),
    ((1024,), 2.0),
])
def test_grad_clip_l2_matches_reference(shape, max_norm):
    rt = _rocm_or_skip()
    g = (_RNG.standard_normal(shape) * 2.0).astype(np.float32)
    out = _run(rt, g, max_norm)
    np.testing.assert_allclose(out, _ref(g, max_norm), rtol=1e-5, atol=1e-5)
    # cross-check against the canonical optim.clip_grad_norm (single-leaf tree).
    ref_opt = np.asarray(optim.clip_grad_norm(g, max_norm)[0], np.float32)
    np.testing.assert_allclose(out, ref_opt, rtol=1e-5, atol=1e-5)


def test_grad_clip_noop_when_under_max():
    # ||g|| < max_norm → scale == 1 → g returned unchanged.
    rt = _rocm_or_skip()
    g = (_RNG.standard_normal((64,)) * 0.01).astype(np.float32)
    np.testing.assert_allclose(_run(rt, g, 100.0), g, rtol=1e-5, atol=1e-6)


def test_grad_clip_inf_norm():
    rt = _rocm_or_skip()
    g = (_RNG.standard_normal((100,)) * 3.0).astype(np.float32)
    out = _run(rt, g, 1.0, norm_type=float("inf"))
    np.testing.assert_allclose(out, _ref(g, 1.0, float("inf")),
                               rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("bad", [1.0, 3.0, 0.5])
def test_grad_clip_rejects_unsupported_finite_pnorm(bad):
    # Only L2 (2.0) and inf are implemented; a finite p-norm != 2 must be
    # rejected, not silently clipped by the L2 norm (Decision #21).
    rt = _rocm_or_skip()
    g = _RNG.standard_normal((32,)).astype(np.float32)
    art = rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_grad_clip_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["g"], "output_name": "o",
        "ops": [{"op_name": "tessera.grad_clip_norm", "result": "o",
                 "operands": ["g"],
                 "kwargs": {"max_norm": 1.0, "norm_type": bad}}]})
    assert rt.launch(art, (g,))["ok"] is False


def test_grad_clip_rejects_missing_max_norm():
    rt = _rocm_or_skip()
    g = np.ones((8,), np.float32)
    art = rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_grad_clip_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["g"], "output_name": "o",
        "ops": [{"op_name": "tessera.grad_clip_norm", "result": "o",
                 "operands": ["g"], "kwargs": {}}]})
    assert rt.launch(art, (g,))["ok"] is False
