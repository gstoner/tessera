"""Opt-in test-only observation of bounded source allocation and frame invariants."""
import ctypes as ct
import os
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.device.nvidia.test_bounded_source_reverse_owner import RawBoundedSourceOwner, guarded_outputs

pytestmark = pytest.mark.hardware_nvidia


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("rows", [False, True])
@pytest.mark.parametrize("capacity", [(7, 11, 4, 3), (17, 19, 35, 7), (17, 19, 257, 13)])
def test_source_owner_keeps_allocations_and_rejects_private_roots(storage, rows, capacity):
    if os.environ.get("TESSERA_TEST_SOURCE_OWNER_OBSERVER") != "1":
        pytest.skip("requires the test-only observer provider on owning SM120")
    dtype = np.dtype(np.float16 if storage == "f16" else ml_dtypes.bfloat16)
    sq_cap, sk_cap, d, dv = capacity
    trace = (np.zeros((1, 2, 1, d), dtype), np.zeros((1, 1, 1, d), dtype),
             np.zeros((1, 1, 1, dv), dtype))
    owner = RawBoundedSourceOwner(trace, storage, rows, "normal", (0, 1, 2), capacity[:2])
    try:
        query = owner.lib.tessera_test_source_owner_state
        query.argtypes, query.restype = [ct.c_uint64, ct.POINTER(ct.c_uint64), ct.c_size_t], ct.c_int

        def state():
            values = (ct.c_uint64 * 71)()
            owner.check(query(owner.handle, values, 71))
            return tuple(values)

        initial = state()
        for step, (sq, sk) in enumerate(((1, 1), (sq_cap, sk_cap), (3, 2), (1, 1))):
            rng = np.random.default_rng(20261019 + step)
            shapes = ((1, 2, sq, d), (1, 1, sk, d), (1, 1, sk, dv), (1, 2, sq, dv), (1, 2, sq))
            values = tuple(rng.normal(0, .2, shape).astype(np.float32 if i == 4 else dtype)
                           for i, shape in enumerate(shapes))
            _, outputs = guarded_outputs(values, (0, 1, 2))
            dims = (1, 2, 1, sq, sk, d, dv)
            owner.check(owner.call(dims, values, outputs))
            current = state()
            assert current[:44] == initial[:44], "allocation addresses/capacities/offsets changed"
            assert current[44:51] == dims
            assert current[64:66] == (2 * sq, d)
            assert all(actual <= maximum for actual, maximum in zip(current[51:64], current[31:44], strict=True))
            invalid = (1, 2, 1, sq_cap + 1, sk, d, dv)
            assert owner.call(invalid, values, outputs) != 0
            assert state() == current, "rejected frame changed active geometry"
            other_sq = sq_cap if sq != sq_cap else 1
            other_valid_geometry = (1, 2, 1, other_sq, sk, d, dv)
            assert owner.call(other_valid_geometry, values, outputs) != 0
            assert state() == current, "valid changed geometry committed before root/output span validation"
            sizes = [x.nbytes for x in values]
            sizes[1] -= 2
            assert owner.call(dims, values, outputs, sizes=sizes) != 0
            assert state() == current, "rejected root span changed active geometry"

            with NvidiaDeviceSession() as session:
                resident = tuple(session.upload(x) for x in values)
                assert session.synchronize() == 0
                for private_pointer in current[:3]:
                    borrowed = list(resident)
                    borrowed[0] = SimpleNamespace(__cuda_array_interface__={
                        "data": (private_pointer, True), "stream": session.stream})
                    _, guarded = guarded_outputs(values, (0, 1, 2))
                    assert owner.call(dims, values, guarded, resident=borrowed) != 0
                    assert b"private" in owner.lib.tessera_nvidia_attention_vjp_last_error()
                    assert state() == current, "private-root rejection changed active geometry"
                    for x in guarded:
                        np.testing.assert_array_equal(x, x.dtype.type(23))
            owner.check(owner.call(dims, values, outputs))
            assert state() == current
    finally:
        owner.close()
