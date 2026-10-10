"""Exact gfx1201 public JVP projection for native continuous scaled products."""
import copy
import itertools
import os
import subprocess

import numpy as np
import pytest

from tessera import runtime
from tessera.autodiff import jvp
from tests.unit.test_public_floating_scaled_primal_jvp import case
from tests.device.rocm.test_floating_scaled_product import reference

pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="requires explicit owning gfx1201 execution")


@pytest.mark.parametrize("ta,tb", tuple(itertools.product((False, True), repeat=2)))
@pytest.mark.parametrize("prefix,out_axes", [((), 0), ((2,), -1), ((2, 3), 0)])
@pytest.mark.parametrize("active", [(0, 1, 2, 3), (0,), (2, 3)])
def test_public_jvp_native_scaled_preserves_owner_and_warm_outputs(
        ta, tb, prefix, out_axes, active, monkeypatch):
    assert runtime._rocm_live_arch() == "gfx1201"
    _, owner, values, directions = case(ta, tb, prefix=prefix, out_axes=out_axes)
    graph = copy.deepcopy(owner.graph_ir)
    request = owner.differentiation_request
    seeds = tuple(seed if index in active else None
                  for index, seed in enumerate(directions))
    oracle_seeds = tuple(seed if index in active else np.zeros_like(seed)
                         for index, seed in enumerate(directions))
    expected = reference([*values, *oracle_seeds], ta, tb, True)
    permutation = owner._frontend_output_permutation
    if permutation is not None:
        expected = tuple(np.transpose(value, permutation) for value in expected)
    try:
        actual = jvp(owner, values, seeds)
        for got, want in zip(actual, expected, strict=True):
            np.testing.assert_allclose(got, want, rtol=4e-5, atol=3e-6)
        receipt = owner.last_jvp_execution
        assert receipt["execution_kind"] == "native_gpu"
        assert receipt["compiler_path"] == "rocm_jvp_compiled"
        assert receipt["evidence_target"] == "rocm_gfx1201"
        assert receipt["family"] == "scaled_product_program"
        assert receipt["public_transform"] == "jvp"
        assert receipt["wrt_indices"] == active
        assert owner.graph_ir == graph
        assert owner.differentiation_request is request
        retained = tuple(value.copy() for value in actual)
        def forbidden(*args, **kwargs):
            raise AssertionError("warm public JVP invoked a compiler")
        monkeypatch.setattr(subprocess, "run", forbidden)
        changed = tuple(None if seed is None else seed * np.float32(-.5) for seed in seeds)
        repeated = jvp(owner, values, changed)
        np.testing.assert_allclose(repeated[0], expected[0], rtol=4e-5, atol=3e-6)
        np.testing.assert_allclose(repeated[1], expected[1] * -.5, rtol=4e-5, atol=3e-6)
        for output, previous in zip(actual, retained, strict=True):
            np.testing.assert_array_equal(output, previous)
        assert owner.graph_ir == graph
        assert len(owner._native_public_jvp_owners) == 1
    finally:
        owner.close_native_storage()
    assert not owner._native_public_jvp_owners

