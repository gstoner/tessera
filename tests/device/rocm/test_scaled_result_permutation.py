"""Owning gfx1201 textual Graph -> Schedule -> Tile -> native permutation."""
import os
import subprocess

import numpy as np
import pytest

from tests.unit.test_native_scaled_result_permutation import source

pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="owning gfx1201 required")


@pytest.mark.parametrize("shape", [(17, 19, 64), (1, 33, 32), (33, 1, 65),
                                  (31, 47, 95), (64, 65, 128)])
def test_compiled_scaled_product_permutation_preserves_native_output_lifetime(monkeypatch, shape):
    from tessera import runtime
    from tessera.compiler.native_scaled_program import (
        NativeScaledProgram, PreparedScaledProgram, package_native_scaled_primal)
    from tests.unit.test_rocm_independent_scaled_batch import batch_inputs
    assert runtime._rocm_live_arch() == "gfx1201"
    arrays, oracle = batch_inputs(shape=(1, *shape), fmt="e8m0",
                                 nk=False, policy="independent_rhs")
    arrays = tuple(value[0] for value in arrays)
    expected = np.ascontiguousarray(oracle[0].T)
    package = package_native_scaled_primal(source(shape))
    package = NativeScaledProgram.from_manifest(package.to_manifest())
    lib = os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]
    with PreparedScaledProgram(package, arrays, runtime_library=lib) as owner:
        generation, _ = owner.invoke()
        output = owner.read(generation)[0]
        np.testing.assert_allclose(output, expected, rtol=4e-5, atol=2e-5)
        retained = output.copy()
        def forbidden(*args, **kwargs):
            raise AssertionError("warm native program called a compiler")
        monkeypatch.setattr(subprocess, "run", forbidden)
        changed = list(arrays)
        changed[2] = changed[2] - np.uint8(1)
        owner.update(changed)
        previous_generation = generation
        generation, _ = owner.invoke()
        with pytest.raises(RuntimeError, match="status 10"):
            owner.read(previous_generation)
        updated = owner.read(generation)[0]
        np.testing.assert_allclose(updated, expected * .5, rtol=4e-5, atol=2e-5)
        np.testing.assert_array_equal(output, retained)
        generation, member_ms = owner.profile_members(repeats=64)
        assert len(member_ms) == 2 and all(value > 0 for value in member_ms)
        np.testing.assert_allclose(owner.read(generation)[0], expected * .5,
                                   rtol=4e-5, atol=2e-5)
        with pytest.raises(ValueError, match="repetition"):
            owner.profile_members(repeats=0)
        _, kernel_ms = owner.invoke(repeats=25, timed=True)
        assert kernel_ms is not None and kernel_ms > 0
    np.testing.assert_array_equal(output, retained)
