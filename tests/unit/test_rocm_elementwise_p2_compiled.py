"""Compiler-generated elementwise completion on gfx1151 (P2a of
S_SERIES_GAP_CLOSURE_PLAN) — binary arithmetic add / mul / mod / floor_div (new
kinds in generate-rocm-binary-kernel) + abs (numeric_helper alias on the
generate-rocm-unary-kernel). Reachable via the rocm_binary_compiled /
rocm_unary_compiled lanes. Validated vs numpy on gfx1151. Skip-clean: no opt/GPU.
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


def _art(rt, op, path, operands):
    names = [f"a{i}" for i in range(len(operands))]
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": path,
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names, "kwargs": {}}],
    })


@pytest.mark.parametrize("op,ref", [
    ("tessera.add", lambda a, b: a + b),
    ("tessera.mul", lambda a, b: a * b),
    ("tessera.mod", np.mod),
    ("tessera.floor_div", np.floor_divide),
])
def test_binary(op, ref):
    rt = _rocm_or_skip()
    rng = np.random.default_rng(1 + hash(op) % 100)
    a = rng.standard_normal(40).astype(np.float32)
    b = (rng.standard_normal(40) + 1.5).astype(np.float32)
    res = rt.launch(_art(rt, op, "rocm_binary_compiled", [a, b]), (a, b))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]), ref(a, b), atol=1e-4)


def test_abs():
    rt = _rocm_or_skip()
    a = np.random.default_rng(7).standard_normal(40).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.abs", "rocm_unary_compiled", [a]), (a,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]), np.abs(a), atol=1e-5)


@pytest.mark.parametrize("kind", ["add", "mul", "mod", "floor_div"])
def test_binary_codegen_lowers(kind):
    import subprocess
    from pathlib import Path
    opt = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"
    if not opt.is_file():
        pytest.skip("build tessera-opt")
    d = (f'module {{\n  "tessera_rocm.binary"() {{name = "b", kind = "{kind}", '
         'dtype = "f32"} : () -> ()\n}\n')
    low = subprocess.run(
        [str(opt), "-",
         "--pass-pipeline=builtin.module(generate-rocm-binary-kernel,"
         "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
         "reconcile-unrealized-casts))"],
        input=d, capture_output=True, text=True)
    assert low.returncode == 0 and "llvm." in low.stdout
