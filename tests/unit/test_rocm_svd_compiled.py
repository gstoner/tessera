"""Compiler-generated SVD on gfx1151 (linalg PR-C) — one-sided Jacobi. The
Tessera compiler GENERATES the kernel (generate-rocm-svd-kernel → ROCDL →
hsaco), then HIP launches it. Reachable via `compiler_path="rocm_linalg_compiled"`.
Validated by invariants on gfx1151. Skip-clean: tessera-opt not built / no GPU.
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


def _art(rt, operands):
    names = [f"a{i}" for i in range(len(operands))]
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_linalg_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": "tessera.svd", "result": "o", "operands": names,
                 "kwargs": {}}],
    })


def _check(a, u, s, vh):
    m, n = a.shape[-2], a.shape[-1]
    k = min(m, n)
    assert u.shape == a.shape[:-2] + (m, k)
    assert s.shape == a.shape[:-2] + (k,)
    assert vh.shape == a.shape[:-2] + (k, n)
    np.testing.assert_allclose(u @ (s[..., None] * vh), a, atol=1e-3)
    ref_s = np.linalg.svd(a, full_matrices=False)[1]
    np.testing.assert_allclose(np.sort(s, axis=-1)[..., ::-1], ref_s, atol=1e-3)
    eye = np.broadcast_to(np.eye(k, dtype=np.float32), a.shape[:-2] + (k, k))
    np.testing.assert_allclose(np.swapaxes(u, -1, -2) @ u, eye, atol=1e-3)
    np.testing.assert_allclose(vh @ np.swapaxes(vh, -1, -2), eye, atol=1e-3)


@pytest.mark.parametrize("m,n", [(6, 4), (5, 5), (4, 6), (8, 3)])
def test_svd(m, n):
    rt = _rocm_or_skip()
    a = np.random.default_rng(1 + m + n).standard_normal((m, n)).astype(np.float32)
    res = rt.launch(_art(rt, [a]), (a,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_linalg_compiled"
    u, s, vh = (np.asarray(x) for x in res["output"])
    _check(a, u, s, vh)


def test_svd_batched():
    rt = _rocm_or_skip()
    a = np.random.default_rng(9).standard_normal((3, 6, 4)).astype(np.float32)
    res = rt.launch(_art(rt, [a]), (a,))
    assert res["ok"] is True, res.get("reason")
    u, s, vh = (np.asarray(x) for x in res["output"])
    _check(a, u, s, vh)


def test_svd_codegen_lowers():
    import subprocess
    from pathlib import Path
    opt = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"
    if not opt.is_file():
        pytest.skip("build tessera-opt")
    d = 'module {\n  "tessera_rocm.svd"() {name = "sv"} : () -> ()\n}\n'
    low = subprocess.run(
        [str(opt), "-",
         "--pass-pipeline=builtin.module(generate-rocm-svd-kernel,"
         "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
         "reconcile-unrealized-casts))"],
        input=d, capture_output=True, text=True)
    assert low.returncode == 0 and "llvm." in low.stdout
