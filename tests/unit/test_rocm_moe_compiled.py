"""Compiler-generated MoE compute on gfx1151 (moe PR) — the routed per-token
expert matmuls (top-1). The Tessera compiler GENERATES the kernel (generate-rocm-
moe-kernel → ROCDL → hsaco), then HIP launches it. Reachable via
`compiler_path="rocm_moe_compiled"`. Validated vs tessera.ops.moe on gfx1151.
Skip-clean: tessera-opt not built / no GPU.
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
        "target": "rocm", "compiler_path": "rocm_moe_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": "tessera.moe", "result": "o", "operands": names,
                 "kwargs": {"extras": extras}}],
    })


def test_moe_round_robin():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(1)
    x = rng.standard_normal((10, 5)).astype(np.float32)
    experts = rng.standard_normal((4, 5, 6)).astype(np.float32)
    res = rt.launch(_art(rt, [x, experts], []), (x, experts))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_moe_compiled"
    np.testing.assert_allclose(np.asarray(res["output"]),
                               np.asarray(tessera.ops.moe(x, experts)), atol=1e-3)


def test_moe_scores():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(2)
    x = rng.standard_normal((10, 5)).astype(np.float32)
    experts = rng.standard_normal((4, 5, 6)).astype(np.float32)
    scores = rng.standard_normal((10, 4)).astype(np.float32)
    res = rt.launch(_art(rt, [x, experts, scores], ["scores"]), (x, experts, scores))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(
        np.asarray(res["output"]),
        np.asarray(tessera.ops.moe(x, experts, scores=scores)), atol=1e-3)


def test_moe_route():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(3)
    x = rng.standard_normal((10, 5)).astype(np.float32)
    experts = rng.standard_normal((4, 5, 6)).astype(np.float32)
    route = rng.integers(0, 4, size=10).astype(np.int64)
    res = rt.launch(_art(rt, [x, experts, route], ["route"]), (x, experts, route))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(
        np.asarray(res["output"]),
        np.asarray(tessera.ops.moe(x, experts, route=route)), atol=1e-3)


def test_moe_single_expert_nd():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 3, 5)).astype(np.float32)
    w = rng.standard_normal((5, 6)).astype(np.float32)
    res = rt.launch(_art(rt, [x, w], []), (x, w))
    assert res["ok"] is True, res.get("reason")
    out = np.asarray(res["output"])
    ref = np.asarray(tessera.ops.moe(x, w))
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, atol=1e-3)


def test_moe_codegen_lowers():
    import subprocess
    from pathlib import Path
    opt = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"
    if not opt.is_file():
        pytest.skip("build tessera-opt")
    d = 'module {\n  "tessera_rocm.moe"() {name = "mo"} : () -> ()\n}\n'
    low = subprocess.run(
        [str(opt), "-",
         "--pass-pipeline=builtin.module(generate-rocm-moe-kernel,"
         "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
         "reconcile-unrealized-casts))"],
        input=d, capture_output=True, text=True)
    assert low.returncode == 0 and "llvm." in low.stdout
