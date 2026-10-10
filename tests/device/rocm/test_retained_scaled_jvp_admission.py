"""Admission and snapshot guards for caller-owned native scaled JVP plans."""
import copy
import os

import numpy as np
import pytest

from tessera import runtime
from tessera.autodiff import jvp
from tessera.compiler.native_jvp import NativeJVPArtifact, child_digest, _digest
from tessera.compiler.native_scaled_jvp_call import NativeScaledJVPCall
from tests.unit.test_public_floating_scaled_primal_jvp import case

pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="requires owning gfx1201 compiled artifact")


@pytest.mark.parametrize("field", ["inputs", "outputs", "child_digest", "execution_kind"])
def test_retained_scaled_jvp_rejects_resealed_wrong_binding_before_prepare(field, monkeypatch):
    _, owner, values, directions = case(prefix=())
    try:
        jvp(owner, values, directions)
        child_owner = owner._native_public_jvp_owners[(0, 1, 2, 3)][1]
        package = next(iter(child_owner._native_jvp_packages.values()))
        contract = copy.deepcopy(package.contract)
        step = contract["steps"][0]
        if field == "inputs":
            step["inputs"] = list(reversed(step["inputs"]))
        elif field == "outputs":
            step["outputs"] = ["tangent", "primal"]
        elif field == "child_digest":
            step["child_digest"] = "f"*64
        else:
            step["child_metadata"]["execution_kind"] = "native_cpu"
            step["child_digest"] = child_digest(step["child_metadata"])
        body = dict(contract)
        body.pop("artifact_hash")
        contract["artifact_hash"] = _digest(body)
        def forbidden(*args, **kwargs):
            raise AssertionError("invalid binding reached native preparation")
        monkeypatch.setattr(runtime, "_load_rocm_native_movement_runtime", forbidden)
        with pytest.raises(ValueError, match="child binding"):
            NativeScaledJVPCall(NativeJVPArtifact(contract), [*values, *directions])
    finally:
        owner.close_native_storage()


def test_retained_scaled_jvp_snapshot_cannot_be_rebound_by_export_mutation():
    _, owner, values, directions = case(prefix=())
    try:
        expected = jvp(owner, values, directions)
        child_owner = owner._native_public_jvp_owners[(0, 1, 2, 3)][1]
        package = next(iter(child_owner._native_jvp_packages.values()))
        original_hash = owner.last_jvp_execution["artifact_hash"]
        original = copy.deepcopy(package.contract)
        try:
            package.contract["steps"][0]["child_metadata"]["arg_names"].reverse()
            with pytest.raises(ValueError, match="stale content"):
                package.runtime_metadata()
            # An already owned native plan executes its validated snapshot;
            # altered serialized metadata cannot rebind that native program.
            actual = jvp(owner, values, directions)
            for got, want in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(got, want)
            assert owner.last_jvp_execution["artifact_hash"] == original_hash
        finally:
            package.contract.clear()
            package.contract.update(original)
    finally:
        owner.close_native_storage()



def test_retained_scaled_jvp_collection_closes_the_native_handle():
    import ctypes as c
    import gc
    import weakref
    _, owner, values, directions = case(prefix=())
    del _  # The direct fixture returns scalar and owner as the same object.
    jvp(owner, values, directions)
    child_owner = owner._native_public_jvp_owners[(0, 1, 2, 3)][1]
    call = next(iter(child_owner._native_prepared_jvp_calls.values()))
    handle = call._owner.handle.value
    library = call._owner.lib
    reference = weakref.ref(call)
    del call, child_owner, owner
    gc.collect()
    assert reference() is None
    generation = c.c_uint64()
    # The C++ registry no longer admits the collected caller's handle.
    assert library.tessera_rocm_program_invoke(
        handle, 1, c.byref(generation), None) != 0

