"""Pre-reserved native results must order pending external producer streams."""
from contextlib import ExitStack
import ctypes as ct

import numpy as np
import pytest
import tessera as ts
from benchmarks.nvidia.active_source_jvp_reference import inputs, oracle, product
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.native_tensor_result import _Batch, NativeTensorResult, bind
from tessera.compiler.resident_nvidia_tensor import ordered_resident_views
from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed, queue_delayed_upload
from tests.device.nvidia.test_resident_active_source_jvp import check

pytestmark = pytest.mark.hardware_nvidia


def pending_frame(storage):
    a, b, da = inputs({"shape_mnk": (17, 19, 35), "storage": storage})
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=("a",))(product)
    fn.native_jvp(a, b, tangents=(da,))
    owner = next(iter(fn._native_prepared_jvp_calls.values()))
    bind(owner.lib)
    changed = tuple((value.astype(np.float32) * .75 + .0625).astype(value.dtype)
                    for value in (a, b, da))
    expected, budgets = oracle(*changed)
    pair = None
    try:
        with ExitStack() as stack:
            sessions = [stack.enter_context(NvidiaDeviceSession()) for _ in range(3)]
            buffers = [session.upload(np.zeros_like(value)) for session, value in
                       zip(sessions, changed, strict=True)]
            for session in sessions:
                assert session.synchronize() == 0
            roots = tuple(Borrowed(value) for value in buffers)
            owner(roots)
            handle = ct.c_uint64()
            owner._check(owner.lib.tessera_nvidia_tensor_program_reserve_result(
                owner.handle, ct.byref(handle)))
            batch = _Batch(owner.lib, handle.value)
            pair = tuple(NativeTensorResult(batch, i) for i in range(2))
            # No GPU allocation occurs between pending writes and submit.
            with pytest.raises(RuntimeError, match="closed or malformed"):
                _ = pair[0].__cuda_array_interface__
            completed = []
            callbacks = [queue_delayed_upload(session, buffer, value, session.stream, completed)
                         for session, buffer, value in zip(sessions, buffers, changed, strict=True)]
            assert not completed
            views, streams = ordered_resident_views(roots, None, writable_from=3)
            producers = (ct.c_uint64 * 3)(*streams)
            owner._check(owner.lib.tessera_nvidia_tensor_program_submit_result(
                owner.handle, handle.value, views, 3, producers, 3))
            actual = tuple(value.to_host() for value in pair)
            assert len(callbacks) == 3
            return actual, expected, budgets
    finally:
        if pair is not None:
            for value in pair:
                value.close()
        fn.close_native_storage()


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_preallocated_result_orders_pending_producer_writes(storage):
    actual, expected, budgets = pending_frame(storage)
    check(actual, expected, budgets)
