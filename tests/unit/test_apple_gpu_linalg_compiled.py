"""Apple GPU linalg lane — cholesky_solve / lu / qr / svd.

Apple ships no MPS lu/qr/svd primitive, so the decompositions resolve on the
numpy reference (np.linalg + a standalone partial-pivot LU) the x86/ROCm device
kernels are matched against. Reachable via
`compiler_path="apple_gpu_linalg_compiled"`. Validated by reconstruction /
np.linalg — parity with test_x86_linalg_compiled. Numpy path always runs (no
skip).
"""

from __future__ import annotations

import numpy as np

from tessera import runtime as rt


def _launch(op, operands, kwargs=None):
    names = [f"a{i}" for i in range(len(operands))]
    art = rt.RuntimeArtifact(metadata={
        "target": "apple_gpu", "compiler_path": "apple_gpu_linalg_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": dict(kwargs or {})}]})
    res = rt.launch(art, tuple(operands))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "apple_gpu_linalg_compiled"
    # Apple has no MPS lu/qr/svd; the lane runs on the CPU numpy reference.
    assert res["execution_kind"] == "reference_cpu"
    return res["output"]


def _spd(rng, n):
    a = rng.standard_normal((n, n)).astype(np.float32)
    return (a @ a.T + n * np.eye(n)).astype(np.float32)


def test_qr_reconstructs():
    rng = np.random.default_rng(1)
    for shape in [(5, 4), (2, 6, 3)]:
        a = rng.standard_normal(shape).astype(np.float32)
        q, r = _launch("tessera.qr", [a])
        q, r = np.asarray(q), np.asarray(r)
        np.testing.assert_allclose(q @ r, a, atol=1e-4)
        # R upper-triangular
        np.testing.assert_allclose(np.tril(r, -1), 0.0, atol=1e-4)


def test_svd_reconstructs():
    rng = np.random.default_rng(2)
    for shape in [(5, 4), (4, 6), (2, 5, 5)]:
        a = rng.standard_normal(shape).astype(np.float32)
        u, s, vh = _launch("tessera.svd", [a])
        u, s, vh = np.asarray(u), np.asarray(s), np.asarray(vh)
        recon = (u * s[..., None, :]) @ vh
        np.testing.assert_allclose(recon, a, atol=1e-4)
        np.testing.assert_array_equal(s, np.sort(s, axis=-1)[..., ::-1])  # descending


def test_lu_reconstructs():
    rng = np.random.default_rng(3)
    for shape in [(5, 5), (2, 4, 4)]:
        a = rng.standard_normal(shape).astype(np.float32)
        packed, piv = _launch("tessera.lu", [a])
        packed, piv = np.asarray(packed), np.asarray(piv)
        n = shape[-1]
        lo = np.tril(packed, -1) + np.eye(n, dtype=np.float32)
        up = np.triu(packed)
        a_perm = np.take_along_axis(
            a, np.broadcast_to(piv[..., None], (*piv.shape, n)).astype(np.intp),
            axis=-2)
        np.testing.assert_allclose(lo @ up, a_perm, atol=1e-4)


def test_cholesky_solve_matches_reference():
    rng = np.random.default_rng(5)
    a = _spd(rng, 5)
    ell = np.linalg.cholesky(a).astype(np.float32)
    b = rng.standard_normal((5,)).astype(np.float32)
    np.testing.assert_allclose(np.asarray(_launch("tessera.cholesky_solve", [ell, b])),
                               np.linalg.solve(a, b), atol=1e-3)
    # matrix RHS
    bm = rng.standard_normal((5, 3)).astype(np.float32)
    np.testing.assert_allclose(np.asarray(_launch("tessera.cholesky_solve", [ell, bm])),
                               np.linalg.solve(a, bm), atol=1e-3)


def test_cholesky_and_tri_solve_match_reference():
    rng = np.random.default_rng(7)
    a = _spd(rng, 4)
    np.testing.assert_allclose(np.asarray(_launch("tessera.cholesky", [a])),
                               np.linalg.cholesky(a), atol=1e-4)
    lo = np.linalg.cholesky(a).astype(np.float32)
    b = rng.standard_normal((4,)).astype(np.float32)
    np.testing.assert_allclose(
        np.asarray(_launch("tessera.tri_solve", [lo, b], {"lower": True})),
        np.linalg.solve(np.tril(lo), b), atol=1e-3)
