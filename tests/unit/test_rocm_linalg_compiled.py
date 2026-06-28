"""Compiler-generated dense linear algebra on gfx1151 (linalg PR-A) — cholesky /
tri_solve / cholesky_solve. The Tessera compiler GENERATES the kernels (generate-
rocm-cholesky-kernel / generate-rocm-tri-solve-kernel → ROCDL → hsaco), then HIP
launches them. Reachable via `compiler_path="rocm_linalg_compiled"`. Validated vs
numpy on gfx1151. Skip-clean: tessera-opt not built / no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _art(rt, op, operands, kwargs=None):
    names = [f"a{i}" for i in range(len(operands))]
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_linalg_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": kwargs or {}}],
    })


def _spd(rng, n):
    m = rng.standard_normal((n, n)).astype(np.float32)
    return (m @ m.T + n * np.eye(n)).astype(np.float32)


def test_cholesky():
    rt = _rocm_or_skip()
    a = _spd(np.random.default_rng(1), 5)
    res = rt.launch(_art(rt, "tessera.cholesky", [a]), (a,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_linalg_compiled"
    np.testing.assert_allclose(np.asarray(res["output"]),
                               np.linalg.cholesky(a), atol=1e-3)


def test_cholesky_batched():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(2)
    a = np.stack([_spd(rng, 5), _spd(rng, 5)])
    res = rt.launch(_art(rt, "tessera.cholesky", [a]), (a,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]),
                               np.linalg.cholesky(a), atol=1e-3)


def test_tri_solve_lower_vector():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(3)
    n = 5
    a = np.tril(rng.standard_normal((n, n)).astype(np.float32)) + n * np.eye(n, dtype=np.float32)
    b = rng.standard_normal((n,)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.tri_solve", [a, b], {"lower": True}), (a, b))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]),
                               np.linalg.solve(np.tril(a), b), atol=1e-3)


def test_tri_solve_upper_matrix():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(4)
    n = 6
    a = np.triu(rng.standard_normal((n, n)).astype(np.float32)) + n * np.eye(n, dtype=np.float32)
    b = rng.standard_normal((n, 3)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.tri_solve", [a, b], {"lower": False}), (a, b))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]),
                               np.linalg.solve(np.triu(a), b), atol=1e-3)


def test_cholesky_solve():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(5)
    a = _spd(rng, 5)
    ell = np.linalg.cholesky(a).astype(np.float32)
    b = rng.standard_normal((5,)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.cholesky_solve", [ell, b]), (ell, b))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]),
                               np.linalg.solve(a, b), atol=1e-3)


@pytest.mark.parametrize("op,attr", [("cholesky", ""),
                                     ("tri-solve", ", lower = true"),
                                     ("tri-solve", ", lower = false")])
def test_linalg_codegen_lowers(op, attr):
    import subprocess
    from pathlib import Path
    opt = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"
    if not opt.is_file():
        pytest.skip("build tessera-opt")
    od = op.replace("-", "_")
    d = f'module {{\n  "tessera_rocm.{od}"() {{name = "k"{attr}}} : () -> ()\n}}\n'
    low = subprocess.run(
        [str(opt), "-",
         f"--pass-pipeline=builtin.module(generate-rocm-{op}-kernel,"
         "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
         "reconcile-unrealized-casts))"],
        input=d, capture_output=True, text=True)
    assert low.returncode == 0 and "llvm." in low.stdout
