"""Compiler-generated unary predicates on gfx1151 (P2b of
S_SERIES_GAP_CLOSURE_PLAN) — isnan / isinf / isfinite. The Tessera compiler
GENERATES the kernel (generate-rocm-predicate-kernel, kind StrAttr → ROCDL →
hsaco), then HIP launches it. Reachable via
`compiler_path="rocm_predicate_compiled"`. Validated vs numpy on gfx1151.
Skip-clean: tessera-opt not built / no GPU.
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


def _art(rt, op):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_predicate_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["a"], "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": ["a"], "kwargs": {}}],
    })


_X = np.array([1.0, np.nan, np.inf, -np.inf, 0.0, -3.5, 1e30, -1e-30],
              np.float32)


@pytest.mark.parametrize("op,ref", [
    ("tessera.isnan", np.isnan),
    ("tessera.isinf", np.isinf),
    ("tessera.isfinite", np.isfinite),
])
def test_predicate(op, ref):
    rt = _rocm_or_skip()
    res = rt.launch(_art(rt, op), (_X,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_predicate_compiled"
    out = np.asarray(res["output"])
    np.testing.assert_array_equal(out, ref(_X))


def test_predicate_shape_preserved():
    rt = _rocm_or_skip()
    x = np.array([[1.0, np.nan], [np.inf, 2.0]], np.float32)
    res = rt.launch(_art(rt, "tessera.isfinite"), (x,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_array_equal(np.asarray(res["output"]), np.isfinite(x))


@pytest.mark.parametrize("shape", [(0,), (0, 3), (2, 0)])
def test_predicate_empty_input(shape):
    # Empty input must short-circuit (a 0-sized grid is undefined for
    # hipModuleLaunchKernel) — return the empty bool result without launching.
    rt = _rocm_or_skip()
    x = np.zeros(shape, np.float32)
    res = rt.launch(_art(rt, "tessera.isnan"), (x,))
    assert res["ok"] is True, res.get("reason")
    out = np.asarray(res["output"])
    assert out.shape == shape and out.dtype == np.bool_


@pytest.mark.parametrize("kind", ["isnan", "isinf", "isfinite"])
def test_predicate_codegen_lowers(kind):
    import subprocess
    from pathlib import Path
    opt = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"
    if not opt.is_file():
        pytest.skip("build tessera-opt")
    d = (f'module {{\n  "tessera_rocm.predicate"() {{name = "p", kind = "{kind}"}} '
         ': () -> ()\n}\n')
    low = subprocess.run(
        [str(opt), "-",
         "--pass-pipeline=builtin.module(generate-rocm-predicate-kernel,"
         "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
         "reconcile-unrealized-casts))"],
        input=d, capture_output=True, text=True)
    assert low.returncode == 0 and "llvm." in low.stdout
