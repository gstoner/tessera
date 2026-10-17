"""Exact gfx1201 resident expansion/product/BF16 activation ownership."""
import ctypes as C
import os

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.rocm_nvfp4_expansion_program import package_native_nvfp4_expansion_program
from tests.device.rocm.test_prepared_nvfp4_expansion_program import inputs_and_oracle
from tests.device.rocm.test_native_parent_view_bf16_jit import oracle, assert_result
from tests.unit.test_native_nvfp4_parent_view_program import resident_source

pytestmark = pytest.mark.skipif(
    os.getenv("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="requires owning gfx1201",
)


@pytest.mark.parametrize("rows,width,k", [(3, 17, 48), (16, 64, 80)])
def test_resident_bf16_parent_remains_private_across_activation_updates(rows, width, k, monkeypatch):
    assert rt._rocm_live_arch() == "gfx1201"
    package = package_native_nvfp4_expansion_program(resident_source(rows, width, k))
    roots, parent = inputs_and_oracle(rows, 2 * width, k, (0, 1, 2, 3, 4), "bf16")
    original = [value.copy() for value in roots]
    expected = oracle(parent, width, False)
    expected_changed = oracle(-parent, width, False)
    import subprocess
    def forbidden(*args, **kwargs):
        pytest.fail("resident BF16 invocation called compiler or eager SiLU arithmetic")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(np, "exp", forbidden)
    monkeypatch.setattr(rt, "_execute_rocm_compiled_silu_mul", forbidden)
    with package.prepare(roots, runtime_library=os.environ["TESSERA_ROCM_MULTI_OUTPUT_LIB"]) as owner:
        with pytest.raises(RuntimeError, match="status 10"):
            owner.invoke_consumer()
        imported, _ = owner.import_weights()
        with pytest.raises(RuntimeError):
            owner.read(imported)
        generation, _ = owner.invoke_consumer()
        result, = owner.read(generation)
        assert_result(result, expected)
        retained = result.copy()
        # The matmul parent is private storage, never an exported host result.
        probe = np.full(parent.shape, 123, dtype=parent.dtype)
        status = owner.lib.tessera_rocm_program_read(
            owner.handle, 7, generation, C.c_void_p(probe.ctypes.data), probe.nbytes)
        assert status != 0
        np.testing.assert_array_equal(probe, np.full_like(probe, 123))
        changed = -roots[3]
        owner.update_activations(changed, roots[4])
        with pytest.raises(RuntimeError):
            owner.read(generation)
        updated, _ = owner.invoke_consumer()
        again, = owner.read(updated)
        assert_result(again, expected_changed)
        np.testing.assert_array_equal(result, retained)
        assert not np.shares_memory(result, again)
        for actual, before in zip(roots, original, strict=True):
            np.testing.assert_array_equal(actual, before)
    assert owner.handle.value == 0
