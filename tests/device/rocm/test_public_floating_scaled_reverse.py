"""Owning public Python f32 matrix and scale reverse through native MLIR."""
import json
import os
import subprocess

import numpy as np
import pytest

from tessera import runtime
from tests.device.rocm.test_floating_scaled_adjoint import inputs, oracle
from tests.unit.test_public_floating_scaled_reverse import floating_owner

pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="owning gfx1201 required")


@pytest.mark.parametrize("ta,tb", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("roles", [("a",), ("b",), ("sa",), ("sb",),
                                  ("a", "b", "sa", "sb"), ("sb", "b", "a", "sa")])
def test_public_floating_reverse_executes_native_and_reuses_package(ta, tb, roles, monkeypatch):
    assert runtime._rocm_live_arch() == "gfx1201"
    values = inputs(ta, tb)
    owner = floating_owner(ta, tb, roles)
    expected = oracle(values, ta, tb)
    actual = owner.native_backward(*values[:4], out_cotangents=values[4])
    positions = {"a": 0, "b": 1, "sa": 2, "sb": 3}
    for result, name in zip(actual, roles, strict=True):
        np.testing.assert_allclose(result, expected[positions[name]], rtol=4e-5, atol=3e-6)
    receipt = owner.last_backward_execution
    assert receipt["execution_kind"] == "native_gpu"
    assert receipt["evidence_target"] == "rocm_gfx1201"
    assert receipt["frontend_authority"] == "tracer"
    assert receipt["compiler_path"] == "rocm_scaled_vjp_program_compiled"
    program = json.loads(owner._native_backward_artifact.program_json)
    assert program["gradient_roles"] == [positions[name] for name in roles]
    retained = tuple(result.copy() for result in actual)

    def forbidden(*args, **kwargs):
        raise AssertionError("warm reverse invoked a compiler")

    monkeypatch.setattr(subprocess, "run", forbidden)
    repeated = owner.native_backward(*values[:4], out_cotangents=values[4] * np.float32(-.5))
    for result, name in zip(repeated, roles, strict=True):
        np.testing.assert_allclose(result, expected[positions[name]] * -.5, rtol=4e-5, atol=3e-6)
    for result, old in zip(actual, retained, strict=True):
        np.testing.assert_array_equal(result, old)


@pytest.mark.parametrize("policy", ["shared_lhs", "shared_rhs_rows", "independent_rhs", "broadcast"])
@pytest.mark.parametrize("ta,tb", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("roles", [("a", "b", "sa", "sb"), ("sb", "b", "a", "sa")])
def test_public_floating_batch_reverse_matches_shared_reductions(policy, ta, tb, roles, monkeypatch):
    from tests.device.rocm.test_floating_scaled_adjoint import batch_inputs, batch_oracle
    from tests.unit.test_public_floating_scaled_reverse import floating_batch_owner
    assert runtime._rocm_live_arch() == "gfx1201"
    values = batch_inputs(policy, ta, tb)
    owner = floating_batch_owner(policy, ta, tb, roles)
    expected = batch_oracle(values, ta, tb)
    actual = owner.native_backward(*values[:4], out_cotangents=values[4])
    positions = {"a": 0, "b": 1, "sa": 2, "sb": 3}
    for result, name in zip(actual, roles, strict=True):
        np.testing.assert_allclose(result, expected[positions[name]], rtol=4e-5, atol=3e-6)
    receipt = owner.last_backward_execution
    assert receipt["execution_kind"] == "native_gpu"
    assert receipt["frontend_authority"] == "tracer"
    assert receipt["evidence_target"] == "rocm_gfx1201"
    retained = tuple(result.copy() for result in actual)

    def forbidden(*args, **kwargs):
        raise AssertionError("warm batched reverse invoked a compiler")

    monkeypatch.setattr(subprocess, "run", forbidden)
    repeated = owner.native_backward(*values[:4], out_cotangents=values[4] * np.float32(-.5))
    for result, name in zip(repeated, roles, strict=True):
        np.testing.assert_allclose(result, expected[positions[name]] * -.5, rtol=4e-5, atol=3e-6)
    for result, old in zip(actual, retained, strict=True):
        np.testing.assert_array_equal(result, old)
