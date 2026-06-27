"""Compiler-generated inclusive prefix scan (cumsum/cumprod/cummax/cummin) on
gfx1151 — the CUB `BlockScan` technique reimplemented in Tessera codegen.

The `tessera_rocm.scan` directive expands (via `generate-rocm-scan-kernel`) into
a block-scan kernel: per BD-element tile a `gpu.shuffle up` (Kogge-Stone) warp
scan + per-subgroup exclusive offset + cross-tile carry. Reachable through
`runtime.launch()` via `compiler_path="rocm_scan_compiled"`; op names
tessera.cumsum / cumprod / cummax / cummin; f16/bf16/f32; same-shape output.

Validated vs numpy. Skip-clean: tessera-opt not built / no AMD GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


def _scan_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, op_name, axis):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_scan_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["x"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": ["x"],
                 "kwargs": {"axis": axis}}],
    })


def _np_cumprod(x, axis):
    return np.cumprod(x, axis=axis)


_CASES = {
    "tessera.cumsum": (np.cumsum, (-2.0, 2.0)),
    # cumprod over a long row of values >1 overflows; keep |v| ~ 1.
    "tessera.cumprod": (_np_cumprod, (0.85, 1.15)),
    "tessera.cummax": (np.maximum.accumulate, (-3.0, 3.0)),
    "tessera.cummin": (np.minimum.accumulate, (-3.0, 3.0)),
}


@pytest.mark.parametrize("op_name", list(_CASES))
@pytest.mark.parametrize("dtype,tol", [(np.float32, 2e-4), (np.float16, 6e-2)])
@pytest.mark.parametrize("shape,axis", [
    ((4, 50), -1), ((4, 50), 1), ((8, 300), -1), ((130,), 0), ((3, 5, 7), -1),
])
def test_scan_matches_numpy(op_name, dtype, tol, shape, axis):
    rt = _scan_or_skip()
    ref, (lo, hi) = _CASES[op_name]
    rng = np.random.default_rng(83 + len(shape) + int(np.prod(shape)))
    x = (rng.random(shape) * (hi - lo) + lo).astype(dtype)
    res = rt.launch(_artifact(rt, op_name, axis), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_scan_compiled"
    out = np.asarray(res["output"]).astype(np.float32)
    # numpy's accumulate has no `axis` kw via ufunc.accumulate -> use axis arg
    if op_name in ("tessera.cummax", "tessera.cummin"):
        expect = ref(x.astype(np.float32), axis=axis)
    else:
        expect = ref(x.astype(np.float32), axis=axis)
    assert out.shape == expect.shape
    np.testing.assert_allclose(out, expect, atol=tol, rtol=tol)


def test_scan_long_row_crosses_tiles():
    # K > blockDim (256) exercises the cross-tile carry.
    rt = _scan_or_skip()
    x = np.ones((2, 600), np.float32)
    res = rt.launch(_artifact(rt, "tessera.cumsum", -1), (x,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(
        np.asarray(res["output"]).astype(np.float32),
        np.cumsum(x, axis=-1), atol=1e-3, rtol=1e-3)


def test_scan_unknown_op_rejected():
    from tessera import runtime as rt
    x = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="rocm_scan_compiled executor"):
        rt._execute_rocm_compiled_scan(_artifact(rt, "tessera.softmax", -1), (x,))


# ── GPU-free codegen gate ────────────────────────────────────────────────────
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

_OPT = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"


def _opt(directive, *passes):
    if not _OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    return subprocess.run([str(_OPT), "-", *passes], input=directive,
                          capture_output=True, text=True)


@pytest.mark.parametrize("kind", ["cumsum", "cumprod", "cummax", "cummin"])
def test_scan_codegen_and_lowers(kind):
    d = ('module {\n  "tessera_rocm.scan"() {name = "sc", '
         f'kind = "{kind}", dtype = "f32"}} : () -> ()\n}}\n')
    ir = _opt(d, "--generate-rocm-scan-kernel")
    assert ir.returncode == 0, ir.stderr
    assert "gpu.shuffle  up" in ir.stdout
    low = _opt(d, "--pass-pipeline=builtin.module(generate-rocm-scan-kernel,"
               "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
               "reconcile-unrealized-casts))")
    assert low.returncode == 0 and "llvm." in low.stdout


def test_scan_codegen_bad_kind_rejected():
    d = ('module {\n  "tessera_rocm.scan"() {name = "sc", kind = "zz"} '
         ': () -> ()\n}\n')
    r = _opt(d, "--generate-rocm-scan-kernel")
    assert r.returncode != 0 and "cumsum" in r.stderr
