"""Compiler-generated row reduction (sum/mean/max/min) on gfx1151 — the ROCm
analog of the x86 AVX-512 reduction lane.

The `tessera_rocm.reduce` directive expands (via `generate-rocm-reduce-kernel`)
into a row-reduction kernel; the runtime folds an arbitrary `axis` to a
[outer, inner] last-axis reduction. Reachable through `runtime.launch()` via
`compiler_path="rocm_reduce_compiled"`; op names tessera.sum / mean / max / min
(+ amax/amin); f16/bf16/f32 storage, f32 reduce.

Validated vs numpy (np.sum/mean/amax/amin), incl. axis variants + keepdims.

Skip-clean: tessera-opt not built, or no usable AMD GPU.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")


def _reduce_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, op_name, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_reduce_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["x"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o",
                 "operands": ["x"], "kwargs": kwargs}],
    })


_NP = {"tessera.sum": np.sum, "tessera.mean": np.mean,
       "tessera.max": np.amax, "tessera.min": np.amin,
       "tessera.amax": np.amax, "tessera.amin": np.amin}


@pytest.mark.parametrize("op_name", list(_NP))
@pytest.mark.parametrize("dtype,tol", [
    (np.float32, 1e-5), (np.float16, 4e-3), ("bf16", 3e-2),
])
@pytest.mark.parametrize("shape,axis", [
    ((8, 64), -1),        # last-axis, cols multiple of BD chunk
    ((4, 130), -1),       # ragged cols
    ((3, 5, 16), -1),     # rank-3 last axis
    ((6, 48), None),      # reduce all
])
def test_reduce_matches_numpy(op_name, dtype, tol, shape, axis):
    rt = _reduce_or_skip()
    if dtype == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(7 + len(shape) + shape[-1])
    x = (rng.standard_normal(shape) * 1.5).astype(dtype)
    res = rt.launch(_artifact(rt, op_name, {"axis": axis}), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_reduce_compiled"
    out = np.asarray(res["output"]).astype(np.float32)
    ref = _NP[op_name](x.astype(np.float32), axis=axis)
    np.testing.assert_allclose(out, np.asarray(ref), atol=tol, rtol=tol)


@pytest.mark.parametrize("op_name", ["tessera.sum", "tessera.max"])
def test_reduce_keepdims(op_name):
    rt = _reduce_or_skip()
    rng = np.random.default_rng(99)
    x = (rng.standard_normal((4, 32)) * 1.5).astype(np.float32)
    res = rt.launch(_artifact(rt, op_name, {"axis": -1, "keepdims": True}),
                    (x,))
    assert res["ok"] is True, res.get("reason")
    out = np.asarray(res["output"]).astype(np.float32)
    ref = _NP[op_name](x, axis=-1, keepdims=True)
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=0)


def test_reduce_unknown_op_rejected():
    from tessera import runtime as rt
    x = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="rocm_reduce_compiled executor"):
        rt._execute_rocm_compiled_reduce(
            _artifact(rt, "tessera.softmax", {"axis": -1}), (x,))


# ── GPU-free codegen gate (needs only tessera-opt, not a GPU) ────────────────
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

_OPT = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"


def _opt(directive, *passes):
    if not _OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    return subprocess.run([str(_OPT), "-", *passes], input=directive,
                          capture_output=True, text=True)


@pytest.mark.parametrize("kind", ["sum", "mean", "max", "min"])
def test_reduce_codegen_and_lowers(kind):
    import re
    d = ('module {\n  "tessera_rocm.reduce"() {name = "rd", '
         f'kind = "{kind}", dtype = "f32"}} : () -> ()\n}}\n')
    ir = _opt(d, "--generate-rocm-reduce-kernel")
    assert ir.returncode == 0, ir.stderr
    m = re.search(r"gpu\.func @rd\(([^)]*)\)", ir.stdout)
    assert m and len([a for a in m.group(1).split(",") if a.strip()]) == 4
    assert "gpu.barrier" in ir.stdout and "scf.for" in ir.stdout
    low = _opt(d, "--pass-pipeline=builtin.module(generate-rocm-reduce-kernel,"
               "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
               "reconcile-unrealized-casts))")
    assert low.returncode == 0 and "llvm." in low.stdout


def test_reduce_codegen_bad_kind_rejected():
    d = ('module {\n  "tessera_rocm.reduce"() {name = "rd", kind = "zzz"} '
         ': () -> ()\n}\n')
    r = _opt(d, "--generate-rocm-reduce-kernel")
    assert r.returncode != 0 and \
        "kind must be sum, mean, max, min, or prod" in r.stderr
