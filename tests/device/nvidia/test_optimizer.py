from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia import nvidia_mma_runtime_available
from tessera import optim


def _art(rt, op, vals, extras, kw):
    names = [f"a{i}" for i in range(len(vals))]
    return rt.RuntimeArtifact(
        metadata={
            "target": "nvidia_sm120",
            "compiler_path": "nvidia_optimizer_compiled",
            "executable": True,
            "execution_kind": "native_gpu",
            "arg_names": names,
            "output_name": "o",
            "ops": [{"op_name": op, "result": "o", "operands": names, "kwargs": dict(kw, extras=extras)}],
        }
    )


@pytest.mark.skipif(not nvidia_mma_runtime_available(), reason="requires nvcc and NVIDIA GPU")
@pytest.mark.hardware_nvidia
def test_sgd_momentum_nesterov():
    from tessera import runtime as rt

    rng = np.random.default_rng(5)
    p = rng.standard_normal((3, 7)).astype(np.float32)
    g = rng.standard_normal(p.shape).astype(np.float32)
    z = np.zeros_like(p)
    s = rt.launch(_art(rt, "tessera.sgd", [p, g], [], {"lr": 0.1}), (p, g))["output"]
    np.testing.assert_allclose(s, optim.sgd(p, g, lr=0.1), atol=1e-6)
    for name, fn in (("momentum", optim.momentum), ("nesterov", optim.nesterov)):
        out = rt.launch(_art(rt, f"tessera.{name}", [p, g, z], ["v"], {"lr": 0.01, "momentum": 0.9}), (p, g, z))["output"]
        rp, state = fn(p, g, None, lr=0.01, momentum=0.9)
        np.testing.assert_allclose(out[0], rp, atol=1e-6)
        np.testing.assert_allclose(out[1], state["velocity"], atol=1e-6)


@pytest.mark.skipif(not nvidia_mma_runtime_available(), reason="requires nvcc and NVIDIA GPU")
@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("name", ["adam", "adamw", "lion"])
def test_adaptive_optimizer_contracts(name):
    from tessera import runtime as rt

    rng = np.random.default_rng(13 + len(name))
    p = rng.standard_normal((4, 6)).astype(np.float32)
    g = rng.standard_normal(p.shape).astype(np.float32)
    z = np.zeros_like(p)
    if name == "lion":
        vals, extras, kw = [p, g, z], ["m"], {"lr": 1e-4, "beta1": 0.9, "beta2": 0.99, "weight_decay": 0.01}
        ref = optim.lion(p, g, None, **kw)
    else:
        vals, extras, kw = [p, g, z, z], ["m", "v"], {"lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8}
        if name == "adamw":
            kw["weight_decay"] = 0.01
        ref = getattr(optim, name)(p, g, None, **kw)
    out = rt.launch(_art(rt, f"tessera.{name}", vals, extras, kw), tuple(vals))["output"]
    np.testing.assert_allclose(out[0], ref[0], atol=1e-5)
