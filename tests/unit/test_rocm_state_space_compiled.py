"""Compiler-generated state-space scan on gfx1151 (state_space PR) —
selective_ssm (Mamba2). The Tessera compiler GENERATES the selective-scan kernel
(generate-rocm-selective-ssm-kernel → ROCDL → hsaco), then HIP launches it.
Reachable via `compiler_path="rocm_selective_ssm_compiled"`. Validated vs the
tessera.ops.selective_ssm reference on gfx1151. Skip-clean: no opt / no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _art(rt, operands, extras):
    names = [f"a{i}" for i in range(len(operands))]
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_selective_ssm_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": "tessera.selective_ssm", "result": "o",
                 "operands": names, "kwargs": {"extras": extras}}],
    })


def _inputs(rng, bsz, s, d, n):
    x = rng.standard_normal((bsz, s, d)).astype(np.float32)
    a = (-rng.random((d, n))).astype(np.float32)
    b = rng.standard_normal((bsz, s, n)).astype(np.float32)
    c = rng.standard_normal((bsz, s, n)).astype(np.float32)
    delta = (rng.random((bsz, s, d)) * 0.5).astype(np.float32)
    return x, a, b, c, delta


@pytest.mark.parametrize("n", [16, 10])
def test_selective_ssm(n):
    rt = _rocm_or_skip()
    rng = np.random.default_rng(1 + n)
    x, a, b, c, delta = _inputs(rng, 2, 7, 4, n)
    res = rt.launch(_art(rt, [x, a, b, c, delta], []), (x, a, b, c, delta))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_selective_ssm_compiled"
    ref = np.asarray(tessera.ops.selective_ssm(x, a, b, c, delta))
    np.testing.assert_allclose(np.asarray(res["output"]), ref, atol=1e-3)


@pytest.mark.parametrize("dtype_tag,atol", [("f16", 3e-2), ("bf16", 1e-1)])
@pytest.mark.parametrize("n", [16, 10])
def test_selective_ssm_lowp(dtype_tag, atol, n):
    # Half-I/O storage (f16/bf16 memrefs, loads extf->f32, y truncf) with f32
    # state + exp + accumulate: matches the f32 reference to storage rounding.
    rt = _rocm_or_skip()
    if dtype_tag == "bf16":
        store = rt._bfloat16_dtype()
        if store is None:
            pytest.skip("no bf16 dtype (ml_dtypes)")
    else:
        store = np.float16
    rng = np.random.default_rng(30 + n)
    x, a, b, c, delta = (v.astype(store) for v in _inputs(rng, 2, 7, 4, n))
    res = rt.launch(_art(rt, [x, a, b, c, delta], []), (x, a, b, c, delta))
    assert res["ok"] is True, res.get("reason")
    out = np.asarray(res["output"])
    assert out.dtype == store, f"expected {store} output, got {out.dtype}"
    ref = np.asarray(tessera.ops.selective_ssm(
        x.astype(np.float32), a.astype(np.float32), b.astype(np.float32),
        c.astype(np.float32), delta.astype(np.float32)))
    np.testing.assert_allclose(out.astype(np.float32), ref, atol=atol)


def test_selective_ssm_gate_state():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(9)
    x, a, b, c, delta = _inputs(rng, 2, 7, 4, 16)
    gate = rng.standard_normal((2, 7, 4)).astype(np.float32)
    state = rng.standard_normal((2, 4, 16)).astype(np.float32)
    res = rt.launch(_art(rt, [x, a, b, c, delta, gate, state],
                         ["gate", "state"]),
                    (x, a, b, c, delta, gate, state))
    assert res["ok"] is True, res.get("reason")
    ref = np.asarray(tessera.ops.selective_ssm(x, a, b, c, delta, gate=gate,
                                               state=state))
    np.testing.assert_allclose(np.asarray(res["output"]), ref, atol=1e-3)


def test_ssm_codegen_lowers():
    import subprocess
    from pathlib import Path
    opt = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"
    if not opt.is_file():
        pytest.skip("build tessera-opt")
    d = ('module {\n  "tessera_rocm.selective_ssm"() {name = "ss"} '
         ': () -> ()\n}\n')
    low = subprocess.run(
        [str(opt), "-",
         "--pass-pipeline=builtin.module(generate-rocm-selective-ssm-kernel,"
         "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
         "reconcile-unrealized-casts))"],
        input=d, capture_output=True, text=True)
    assert low.returncode == 0 and "llvm." in low.stdout
