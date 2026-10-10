"""Owning gfx1201 public continuous primal/JVP pipeline and warm reuse."""
import itertools
import json
import os
import subprocess

import numpy as np
import pytest

from tessera import runtime
from tests.unit.test_public_floating_scaled_primal_jvp import case
from tests.device.rocm.test_floating_scaled_product import reference

pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="owning gfx1201 required")


def placed_reference(values, directions, ta, tb, owner, jvp):
    expected = reference([*values, *directions] if jvp else values, ta, tb, jvp)
    permutation = owner._frontend_output_permutation
    return tuple(np.transpose(value, permutation) if permutation is not None else value
                 for value in expected)


@pytest.mark.parametrize("ta,tb", tuple(itertools.product((False, True), repeat=2)))
@pytest.mark.parametrize("mode", [None, "forward"])
@pytest.mark.parametrize("mask", range(1,16))
@pytest.mark.parametrize("prefix", [(2,), (2,3)])
@pytest.mark.parametrize("out_axes", [0, -1])
def test_public_continuous_maps_execute_native_and_reuse_packages(
        ta,tb,mode,mask,prefix,out_axes,monkeypatch):
    assert runtime._rocm_live_arch() == "gfx1201"
    _, owner, values, directions = case(ta,tb,mode,mask,prefix,out_axes)
    expected = placed_reference(values,directions,ta,tb,owner,mode=="forward")
    ordinary = owner(*values)
    np.testing.assert_allclose(ordinary,expected[0],rtol=4e-5,atol=3e-6)
    receipt = owner._native_descriptor_last_receipt
    assert receipt["execution_kind"] == "native_gpu"
    assert receipt["compiler_path"] == "rocm_scaled_primal_program_compiled"
    program = json.loads(owner._native_composed_scaled_last_program.program_json)
    assert any(step.get("lowering") == "structured_f32_scaled_product" for step in program["steps"])
    actual = owner.native_jvp(*values,tangents=directions) if mode=="forward" else (ordinary,)
    if mode=="forward":
        assert owner.last_jvp_execution["execution_kind"] == "native_gpu"
    for got,want in zip(actual,expected,strict=True):
        np.testing.assert_allclose(got,want,rtol=4e-5,atol=3e-6)
    retained = tuple(value.copy() for value in actual)
    changed = tuple(np.ascontiguousarray(value*np.float32(1.125)) for value in values)
    changed_directions = tuple(np.ascontiguousarray(value*np.float32(-.5)) for value in directions)
    wanted = placed_reference(changed,changed_directions,ta,tb,owner,mode=="forward")
    def forbidden(*args,**kwargs):
        raise AssertionError("warm continuous public execution invoked a compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    np.testing.assert_allclose(owner(*changed),wanted[0],rtol=4e-5,atol=3e-6)
    repeated = (owner.native_jvp(*changed,tangents=changed_directions)
                if mode=="forward" else (owner(*changed),))
    for got,want in zip(repeated,wanted,strict=True):
        np.testing.assert_allclose(got,want,rtol=4e-5,atol=3e-6)
    for got,old in zip(actual,retained,strict=True):
        np.testing.assert_array_equal(got,old)


@pytest.mark.parametrize("ta,tb", tuple(itertools.product((False, True), repeat=2)))
@pytest.mark.parametrize("mode", [None, "forward"])
def test_direct_public_continuous_pipeline(ta,tb,mode):
    assert runtime._rocm_live_arch() == "gfx1201"
    _,owner,values,directions = case(ta,tb,mode,prefix=())
    ordinary = owner(*values)
    expected = placed_reference(values,directions,ta,tb,owner,mode=="forward")
    np.testing.assert_allclose(ordinary,expected[0],rtol=4e-5,atol=3e-6)
    assert owner._native_descriptor_last_receipt["execution_kind"] == "native_gpu"
    if mode=="forward":
        actual = owner.native_jvp(*values,tangents=directions)
        assert owner.last_jvp_execution["execution_kind"] == "native_gpu"
        for got,want in zip(actual,expected,strict=True):
            np.testing.assert_allclose(got,want,rtol=4e-5,atol=3e-6)


@pytest.mark.parametrize("ta,tb", tuple(itertools.product((False,True),repeat=2)))
@pytest.mark.parametrize("mode",[None,"forward"])
@pytest.mark.parametrize("nested",[False,True])
def test_public_continuous_mixed_axes_follow_independent_coordinates(ta,tb,mode,nested,monkeypatch):
    from tests.unit.test_public_floating_scaled_primal_jvp import mixed_case
    assert runtime._rocm_live_arch() == "gfx1201"
    _,owner,values,directions,canonical,seeds,permutation = mixed_case(ta,tb,mode,nested)
    # The oracle consumes original canonical fixtures, never implementation
    # normalization; the literal map placement was derived independently.
    # Outer-only B/sb batches occupy the first logical map coordinate;
    # NumPy trailing broadcasting alone would incorrectly align them with
    # the inner coordinate. This frame follows the explicit fixture axes.
    def oracle_frame(frame):
        if not nested:
            return frame
        return (frame[0], frame[1][:, None, ...], frame[2], frame[3][:, None, ...])
    canonical_frame, seed_frame = oracle_frame(canonical), oracle_frame(seeds)
    expected = tuple(np.transpose(value,permutation)
                     for value in reference([*canonical_frame,*seed_frame] if mode=="forward"
                                            else canonical_frame, ta,tb,mode=="forward"))
    ordinary = owner(*values)
    np.testing.assert_allclose(ordinary,expected[0],rtol=4e-5,atol=3e-6)
    assert owner._native_descriptor_last_receipt["execution_kind"] == "native_gpu"
    actual = owner.native_jvp(*values,tangents=directions) if mode=="forward" else (ordinary,)
    for got,want in zip(actual,expected,strict=True):
        np.testing.assert_allclose(got,want,rtol=4e-5,atol=3e-6)
    def forbidden(*args,**kwargs):
        raise AssertionError("warm mixed-axis execution invoked a compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    repeated = (owner.native_jvp(*values,tangents=tuple(seed*np.float32(-.5) for seed in directions))
                if mode=="forward" else (owner(*values),))
    for index,(got,want) in enumerate(zip(repeated,expected,strict=True)):
        np.testing.assert_allclose(got,want*(-.5 if index==1 else 1),rtol=4e-5,atol=3e-6)
