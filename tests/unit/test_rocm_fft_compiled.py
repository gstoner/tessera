"""Canonical spectral FFT on gfx1151 — fft / ifft / rfft / irfft through the
shipping mixed-radix Stockham/Bluestein package. Reachable via
``compiler_path="rocm_fft_compiled"``. Validated vs NumPy on gfx1151.

Skip-clean: tessera-opt not built / no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._support.compiler_tool import run_tessera_opt


def _rocm_or_skip():
    from tessera import runtime as rt
    from tessera.compiler.emit import spectral_candidates as sc
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    if sc._amd_lib() is None:
        pytest.skip("prebuilt ROCm spectral package not built")
    return rt


def _art(rt, op_name, kwargs, x):
    from tessera.compiler.scheduled_fft import lower_scheduled_fft

    scheduled = lower_scheduled_fft(
        target="rocm",
        op_name=op_name,
        input_shape=tuple(int(dim) for dim in x.shape),
        axis=int(kwargs.get("axis", -1)),
        n=kwargs.get("n"),
    ).to_metadata()
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_fft_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["x"], "output_name": "o",
        "scheduled_fft": scheduled,
    })


_TOL = dict(atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("n", [4, 8, 16, 100, 257])  # radix + Bluestein
@pytest.mark.parametrize("shape_pre", [(), (3,)])
def test_fft_ifft(n, shape_pre):
    rt = _rocm_or_skip()
    rng = np.random.default_rng(1 + n + len(shape_pre))
    x = (rng.standard_normal(shape_pre + (n,))
         + 1j * rng.standard_normal(shape_pre + (n,))).astype(np.complex64)
    rf = rt.launch(_art(rt, "tessera.fft", {"axis": -1}, x), (x,))
    assert rf["ok"] is True, rf.get("reason")
    assert rf["compiler_path"] == "rocm_fft_compiled"
    np.testing.assert_allclose(np.asarray(rf["output"]).astype(np.complex64),
                               np.fft.fft(x, axis=-1).astype(np.complex64), **_TOL)
    ri = rt.launch(_art(rt, "tessera.ifft", {"axis": -1}, x), (x,))
    assert ri["ok"] is True, ri.get("reason")
    np.testing.assert_allclose(np.asarray(ri["output"]).astype(np.complex64),
                               np.fft.ifft(x, axis=-1).astype(np.complex64), **_TOL)


def test_fft_inner_axis():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((4, 16, 3))
         + 1j * rng.standard_normal((4, 16, 3))).astype(np.complex64)
    res = rt.launch(_art(rt, "tessera.fft", {"axis": 1}, x), (x,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.complex64),
                               np.fft.fft(x, axis=1).astype(np.complex64), **_TOL)


@pytest.mark.parametrize("n", [8, 16])
def test_rfft_irfft(n):
    rt = _rocm_or_skip()
    rng = np.random.default_rng(3 + n)
    x = rng.standard_normal((2, n)).astype(np.float32)
    rf = rt.launch(_art(rt, "tessera.rfft", {"axis": -1}, x), (x,))
    assert rf["ok"] is True, rf.get("reason")
    ref = np.fft.rfft(x, axis=-1).astype(np.complex64)
    np.testing.assert_allclose(np.asarray(rf["output"]).astype(np.complex64),
                               ref, **_TOL)
    ir = rt.launch(_art(rt, "tessera.irfft", {"axis": -1, "n": n}, ref), (ref,))
    assert ir["ok"] is True, ir.get("reason")
    np.testing.assert_allclose(np.asarray(ir["output"]).astype(np.float32), x,
                               **_TOL)


def test_fft_codegen_lowers():
    # The direct DFT remains buildable as a diagnostic/oracle, but is no
    # longer the public rocm_fft_compiled execution package.
    d = ('module {\n  "tessera_rocm.dft"() {name = "dt", inverse = false} '
         ': () -> ()\n}\n')
    low = run_tessera_opt(
        d, "--pass-pipeline=builtin.module(generate-rocm-dft-kernel,"
        "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
        "reconcile-unrealized-casts))")
    assert low.returncode == 0 and "llvm." in low.stdout


def test_fft_runtime_does_not_reenter_direct_dft(monkeypatch):
    rt = _rocm_or_skip()
    monkeypatch.setattr(
        rt,
        "_rocm_dft_rows",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("public FFT lane re-entered the direct DFT")
        ),
    )
    rng = np.random.default_rng(19)
    x = (rng.standard_normal((3, 256)) + 1j * rng.standard_normal((3, 256))).astype(
        np.complex64
    )
    result = rt.launch(_art(rt, "tessera.fft", {"axis": -1}, x), (x,))
    assert result["ok"] is True, result.get("reason")
    np.testing.assert_allclose(result["output"], np.fft.fft(x, axis=-1), **_TOL)


def test_bluestein_plan_is_cached_and_bound_to_schedule_digest():
    rt = _rocm_or_skip()
    from tessera.compiler.emit import spectral_candidates as sc

    rng = np.random.default_rng(31)
    x = (rng.standard_normal((2, 257)) + 1j * rng.standard_normal((2, 257))).astype(
        np.complex64
    )
    artifact = _art(rt, "tessera.fft", {"axis": -1}, x)
    digest = artifact.metadata["scheduled_fft"]["schedule_digest"]
    sc._clear_rocm_plan_cache()

    first = rt.launch(artifact, (x,))
    second = rt.launch(artifact, (x,))
    assert first["ok"] is True, first.get("reason")
    assert second["ok"] is True, second.get("reason")
    assert list(sc._rocm_plan_cache) == [(257, -1, digest)]
    lib, handle = sc._rocm_plan_cache[(257, -1, digest)]
    assert lib.ts_fft_plan_workspace_elems_amd(handle) == 4 * 1024
    assert lib.ts_fft_plan_artifact_digest_amd(handle).decode() == digest
    np.testing.assert_allclose(second["output"], np.fft.fft(x, axis=-1), **_TOL)
