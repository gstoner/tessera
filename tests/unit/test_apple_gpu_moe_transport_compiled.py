"""Apple GPU local MoE transport — native dispatch gather and combine scatter."""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.stdlib import moe


def _plan(seed: int = 41):
    rng = np.random.default_rng(seed)
    routes = rng.integers(0, 4, size=(12, 2), dtype=np.int64)
    weights = rng.random((12, 2), dtype=np.float32)
    weights /= weights.sum(axis=1, keepdims=True)
    return moe.plan_dispatch(routes, weights, 4, capacity=5)


def _artifact(op_name: str, names: list[str]):
    return rt.RuntimeArtifact(metadata={
        "target": "apple_gpu", "compiler_path": "apple_gpu_moe_transport_compiled",
        "executable": True, "execution_kind": "native_gpu", "arg_names": names,
        "ops": [{"op_name": op_name}],
    })


def test_moe_dispatch_matches_plan_oracle():
    x = np.random.default_rng(42).standard_normal((12, 8)).astype(np.float32)
    plan = _plan()
    res = rt.launch(_artifact("tessera.moe_dispatch", ["x", "plan"]), (x, plan))
    assert res["ok"], res.get("reason")
    np.testing.assert_allclose(res["output"], moe.dispatch(x, plan), rtol=0, atol=0)


def test_moe_combine_matches_weighted_scatter_oracle():
    x = np.random.default_rng(43).standard_normal((12, 8)).astype(np.float32)
    plan = _plan(43)
    partials = moe.dispatch(x, plan) * np.float32(1.25)
    res = rt.launch(
        _artifact("tessera.moe_combine", ["partials", "plan"]), (partials, plan))
    assert res["ok"], res.get("reason")
    np.testing.assert_allclose(res["output"], moe.combine(partials, plan),
                               rtol=1e-5, atol=1e-6)


def test_moe_transport_reports_native_gpu_on_metal():
    if __import__("sys").platform != "darwin" or not rt.DeviceTensor.is_metal():
        pytest.skip("requires an available Apple Metal runtime")
    x = np.random.default_rng(44).standard_normal((12, 8)).astype(np.float32)
    plan = _plan(44)
    dispatch = rt.launch(_artifact("tessera.moe_dispatch", ["x", "plan"]), (x, plan))
    combine = rt.launch(_artifact("tessera.moe_combine", ["partials", "plan"]),
                        (dispatch["output"], plan))
    assert dispatch["execution_kind"] == "native_gpu"
    assert combine["execution_kind"] == "native_gpu"
