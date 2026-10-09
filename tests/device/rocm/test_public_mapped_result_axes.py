"""Owning gfx1201 public vmap with compiler-produced nonleading outputs."""
import json
import os
import subprocess

import numpy as np
import pytest
from tessera import runtime
from tessera.autodiff import vmap
from tests.unit.test_native_typed_scaled_vmap import case

pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="owning gfx1201 required")


@pytest.mark.parametrize("fmt", ["fp32", "e8m0"])
@pytest.mark.parametrize("policy", ["independent_rhs", "shared_lhs", "shared_rhs_rows"])
@pytest.mark.parametrize("out_axis", [1, -1])
def test_public_nonleading_map_executes_native_permutation(fmt, policy, out_axis, monkeypatch):
    assert runtime._rocm_live_arch() == "gfx1201"
    scalar, leading, values, oracle = case(policy, fmt, False, (2, 7, 19, 256))
    owner = vmap(scalar, in_axes=leading._frontend_batch_axes, out_axes=out_axis)
    expected = np.moveaxis(oracle, 0, out_axis)
    result = owner(*values)
    np.testing.assert_allclose(result, expected, rtol=4e-5, atol=2e-5)
    receipt = owner._native_descriptor_last_receipt
    assert receipt["execution_kind"] == "native_gpu"
    assert json.loads(owner._native_composed_scaled_last_program.program_json)["steps"][-1]["operation"] == "tessera.transpose"
    retained = result.copy()

    def forbidden(*args, **kwargs):
        raise AssertionError("warm mapped execution invoked a compiler")
    monkeypatch.setattr(subprocess, "run", forbidden)
    changed = list(values)
    changed[2] = changed[2] * np.float32(.5) if fmt == "fp32" else changed[2] - np.uint8(1)
    np.testing.assert_allclose(owner(*changed), expected * .5, rtol=4e-5, atol=2e-5)
    np.testing.assert_array_equal(result, retained)

@pytest.mark.parametrize("policy", ["independent_rhs", "shared_lhs", "shared_rhs_rows"])
@pytest.mark.parametrize("out_axis", [1, -1])
def test_public_mapped_scale_jvp_moves_primal_and_tangent(policy, out_axis, monkeypatch):
    import tessera as ts
    from tests.device.rocm.test_public_scaled_jvp import oracle as scalar_oracle
    assert runtime._rocm_live_arch() == "gfx1201"
    scalar, leading, values, primal = case(policy, "fp32", False, (2, 7, 19, 256))
    forward = ts.jit(target="rocm_gfx1201", autodiff="forward",
                     wrt=("sa", "sb"))(scalar._fn)
    owner = vmap(forward, in_axes=leading._frontend_batch_axes, out_axes=out_axis)
    rng = np.random.default_rng(905)
    seeds = tuple(rng.uniform(-.2, .2, value.shape).astype(np.float32)
                  for value in values[2:])
    tangent = []
    for plane in range(2):
        arguments = [value[plane] if axis is not None else value
                     for value, axis in zip(values, leading._frontend_batch_axes, strict=True)]
        selected = [seed[plane] if axis is not None else seed
                    for seed, axis in zip(seeds, leading._frontend_batch_axes[2:], strict=True)]
        tangent.append(scalar_oracle(*arguments, *selected)[1])
    expected = (np.moveaxis(primal, 0, out_axis),
                np.moveaxis(np.stack(tangent), 0, out_axis))
    actual = owner.native_jvp(*values, tangents=seeds)
    for output, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(output, reference, rtol=4e-5, atol=3e-5)
    assert owner.last_jvp_execution["execution_kind"] == "native_gpu"
    def forbidden(*args, **kwargs):
        raise AssertionError("warm JVP invoked a compiler")
    monkeypatch.setattr(subprocess, "run", forbidden)
    repeated = owner.native_jvp(*values, tangents=tuple(seed * -.5 for seed in seeds))
    np.testing.assert_allclose(repeated[0], expected[0], rtol=4e-5, atol=3e-5)
    np.testing.assert_allclose(repeated[1], expected[1] * -.5, rtol=4e-5, atol=3e-5)


@pytest.mark.parametrize("mode", ["primal", "jvp"])
def test_nested_nonleading_map_executes_native_result_order(mode):
    import tessera as ts
    assert runtime._rocm_live_arch() == "gfx1201"
    fmt = "e8m0" if mode == "primal" else "fp32"
    scalar, _, flat, oracle = case("independent_rhs", fmt, False, (6, 7, 19, 256))
    if mode == "jvp":
        scalar = ts.jit(target="rocm_gfx1201", autodiff="forward",
                        wrt=("sa", "sb"))(scalar._fn)
    owner = vmap(vmap(scalar, in_axes=0, out_axes=1), in_axes=0, out_axes=2)
    values = tuple(value.reshape(2, 3, *value.shape[1:]) for value in flat)
    expected = oracle.reshape(2, 3, 7, 19).transpose(2, 1, 0, 3)
    if mode == "primal":
        actual = (owner(*values),)
        references = (expected,)
        assert owner._native_descriptor_last_receipt["execution_kind"] == "native_gpu"
    else:
        seeds = (values[2] * np.float32(.1), values[3] * np.float32(.2))
        actual = owner.native_jvp(*values, tangents=seeds)
        references = (expected, expected * .3)
        assert owner.last_jvp_execution["execution_kind"] == "native_gpu"
    for result, reference in zip(actual, references, strict=True):
        np.testing.assert_allclose(result, reference, rtol=4e-5, atol=3e-5)
