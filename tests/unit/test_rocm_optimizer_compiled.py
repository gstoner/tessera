"""Compiler-generated fused optimizer steps on gfx1151 (P3 of
S_SERIES_GAP_CLOSURE_PLAN) — sgd / momentum / adam / adamw / lion. The Tessera
compiler GENERATES the kernel (generate-rocm-optimizer-kernel, kind StrAttr →
ROCDL → hsaco), then HIP launches it. Reachable via
`compiler_path="rocm_optimizer_compiled"`. Validated vs tessera.optim on
gfx1151. Skip-clean: tessera-opt not built / no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import optim


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _art(rt, op, operands, extras, kw):
    names = [f"a{i}" for i in range(len(operands))]
    kw = dict(kw)
    kw["extras"] = extras
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_optimizer_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names, "kwargs": kw}],
    })


SHAPE = (3, 7)


def test_adamw_multistep():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(1)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    m = np.zeros(SHAPE, np.float32)
    v = np.zeros(SHAPE, np.float32)
    state = None
    for step in range(1, 4):
        g = rng.standard_normal(SHAPE).astype(np.float32)
        res = rt.launch(_art(rt, "tessera.adamw", [p, g, m, v], ["m", "v"],
                             {"lr": 1e-3, "beta1": 0.9, "beta2": 0.999,
                              "eps": 1e-8, "weight_decay": 0.01, "step": step}),
                        (p, g, m, v))
        assert res["ok"] is True, res.get("reason")
        assert res["compiler_path"] == "rocm_optimizer_compiled"
        pn, m, v = (np.asarray(x) for x in res["output"])
        ref_p, state = optim.adamw(p, g, state, lr=1e-3, beta1=0.9, beta2=0.999,
                                   eps=1e-8, weight_decay=0.01)
        np.testing.assert_allclose(pn, np.asarray(ref_p), atol=1e-4)
        np.testing.assert_allclose(v, np.asarray(state["v"]), atol=1e-5)
        p = pn


def test_sgd():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(2)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    g = rng.standard_normal(SHAPE).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.sgd", [p, g], [], {"lr": 0.1}), (p, g))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]),
                               np.asarray(optim.sgd(p, g, lr=0.1)), atol=1e-5)


def test_momentum():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(3)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    g = rng.standard_normal(SHAPE).astype(np.float32)
    v0 = np.zeros(SHAPE, np.float32)
    res = rt.launch(_art(rt, "tessera.momentum", [p, g, v0], ["v"],
                         {"lr": 0.1, "momentum": 0.9}), (p, g, v0))
    assert res["ok"] is True, res.get("reason")
    rp, rst = optim.momentum(p, g, None, lr=0.1, momentum=0.9)
    pn, vn = (np.asarray(x) for x in res["output"])
    np.testing.assert_allclose(pn, np.asarray(rp), atol=1e-5)
    np.testing.assert_allclose(vn, np.asarray(rst["velocity"]), atol=1e-5)


def test_nesterov_multistep():
    """Look-ahead momentum: v=β·v+g ; p -= lr·(g+β·v). Multi-step vs
    optim.nesterov on gfx1151 so the carried velocity is exercised."""
    rt = _rocm_or_skip()
    rng = np.random.default_rng(7)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    v = np.zeros(SHAPE, np.float32)
    state = None
    for _ in range(5):
        g = rng.standard_normal(SHAPE).astype(np.float32)
        res = rt.launch(_art(rt, "tessera.nesterov", [p, g, v], ["v"],
                             {"lr": 1e-2, "momentum": 0.9}), (p, g, v))
        assert res["ok"] is True, res.get("reason")
        assert res["compiler_path"] == "rocm_optimizer_compiled"
        pn, v = (np.asarray(x) for x in res["output"])
        rp, state = optim.nesterov(p, g, state, lr=1e-2, momentum=0.9)
        np.testing.assert_allclose(pn, np.asarray(rp), atol=1e-5)
        np.testing.assert_allclose(v, np.asarray(state["velocity"]), atol=1e-5)
        p = pn


def test_adam_and_lion():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(4)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    g = rng.standard_normal(SHAPE).astype(np.float32)
    z = np.zeros(SHAPE, np.float32)
    res = rt.launch(_art(rt, "tessera.adam", [p, g, z, z], ["m", "v"],
                         {"lr": 1e-3, "step": 1}), (p, g, z, z))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"][0]),
                               np.asarray(optim.adam(p, g, None, lr=1e-3)[0]),
                               atol=1e-4)
    res = rt.launch(_art(rt, "tessera.lion", [p, g, z], ["m"],
                         {"lr": 1e-4, "beta1": 0.9, "beta2": 0.99,
                          "weight_decay": 0.01}), (p, g, z))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(
        np.asarray(res["output"][0]),
        np.asarray(optim.lion(p, g, None, lr=1e-4, beta1=0.9, beta2=0.99,
                              weight_decay=0.01)[0]), atol=1e-5)


@pytest.mark.parametrize("kind", ["sgd", "momentum", "adam", "adamw", "lion"])
def test_optimizer_codegen_lowers(kind):
    import subprocess
    from pathlib import Path
    opt = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"
    if not opt.is_file():
        pytest.skip("build tessera-opt")
    d = (f'module {{\n  "tessera_rocm.optimizer"() {{name = "o", kind = "{kind}"}} '
         ': () -> ()\n}\n')
    low = subprocess.run(
        [str(opt), "-",
         "--pass-pipeline=builtin.module(generate-rocm-optimizer-kernel,"
         "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
         "reconcile-unrealized-casts))"],
        input=d, capture_output=True, text=True)
    assert low.returncode == 0 and "llvm." in low.stdout
