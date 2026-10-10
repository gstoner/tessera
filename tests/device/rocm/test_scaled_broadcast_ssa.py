"""Exact gfx1201 numerical proof for native shared/mapped result sums."""
import os
import subprocess

import numpy as np
import pytest
from tessera import runtime
from tests.unit.test_scaled_broadcast_ssa import case
from tests.device.rocm.test_composed_scaled_maps import expected

pytestmark = pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
                               reason="owning gfx1201 required")


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("mode", [None, "forward", "reverse"])
def test_shared_result_broadcast_and_seed_reduction(side, depth, mode, monkeypatch):
    assert runtime._rocm_live_arch() == "gfx1201"
    _, owner, values, directions, axes, prefix = case(side, depth, mode)
    seed = np.random.default_rng(19103).uniform(-.2, .2, (*prefix, 3, 5)).astype(np.float32)

    def invoke(frame, tangent, cotangent):
        if mode == "forward":
            return owner.native_jvp(*frame, tangents=tangent)
        if mode == "reverse":
            return owner.native_backward(*frame, out_cotangents=cotangent)
        return (owner(*frame),)

    def oracle(frame, tangent, cotangent):
        result = expected(owner, frame, tangent if mode is not None else (), axes, prefix, False,
                          cotangent if mode == "reverse" else None)
        return result[:1] if mode is None else result

    actual = invoke(values, directions, seed)
    for got, want in zip(actual, oracle(values, directions, seed), strict=True):
        np.testing.assert_allclose(got, want, rtol=3e-4, atol=3e-5)
    receipt = owner.last_jvp_execution if mode == "forward" else owner.last_backward_execution if mode == "reverse" else owner._native_descriptor_last_receipt
    assert receipt["execution_kind"] == "native_gpu"
    retained = tuple(value.copy() for value in actual)
    changed = tuple(value if index < 2 else np.ascontiguousarray(value*np.float32(-.875))
                    for index, value in enumerate(values))
    changed_directions = tuple(-value for value in directions)
    wanted = oracle(changed, changed_directions, -seed)

    def forbidden(*args, **kwargs):
        raise AssertionError("warm broadcast escaped native compiler-owned package")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(owner, "_fn", forbidden)
    repeated = invoke(changed, changed_directions, -seed)
    for got, want in zip(repeated, wanted, strict=True):
        np.testing.assert_allclose(got, want, rtol=3e-4, atol=3e-5)
    for got, saved in zip(actual, retained, strict=True):
        np.testing.assert_array_equal(got, saved)
