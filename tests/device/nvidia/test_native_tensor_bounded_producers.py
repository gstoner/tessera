"""Exact SM120 bounded original add/matmul programs and retained native state."""
import ctypes as ct
import subprocess

import numpy as np
import pytest
import tessera as ts
from tessera import runtime
from tests.device.nvidia.test_native_tensor_binary_producers import check, roots
from tests.unit.test_native_tensor_binary_producers import lhs_add

pytestmark = pytest.mark.hardware_nvidia


def forbidden(*args, **kwargs):
    pytest.fail("bounded warm native call traced, compiled or executed eager arithmetic")


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("kn", [(32, 11), (35, 19)])
def test_public_bounded_binary_reuses_original_program_and_capacity(storage, kn, monkeypatch):
    assert runtime._nvidia_device_name() == "sm_120"
    k, n = kn
    fn = ts.jit(target="nvidia_sm120", shape_bounds={"M": 65})(lhs_add)
    try:
        initial = roots("lhs", storage, (17, k, n), 14101)
        first = fn(*initial)
        check(first, "lhs", initial)
        saved = first.copy()
        program, owner = fn._native_tensor_last_program, fn._native_tensor_last_owner
        assert program.frame["schema"] == "tessera.native.tensor_program_frame.v2"
        assert program.frame["row_bound"] == 65
        stats = owner.stats()
        assert stats["allocations"] == 1
        for m in (1, 5, 65, 2, 64, 1, 17):
            values = roots("lhs", storage, (m, k, n), 14102 + m)
            snapshots = tuple(value.copy() for value in values)
            with monkeypatch.context() as guard:
                guard.setattr(fn, "_fn", forbidden)
                guard.setattr(fn, "_trace_frontend_capture", forbidden)
                guard.setattr(subprocess, "run", forbidden)
                guard.setattr(subprocess, "Popen", forbidden)
                actual = fn(*values)
            check(actual, "lhs", values)
            assert fn._native_tensor_last_program is program
            assert fn._native_tensor_last_owner is owner
            assert owner.stats()["allocations"] == stats["allocations"]
            assert owner.stats()["capacity"] == stats["capacity"]
            np.testing.assert_array_equal(first, saved)
            assert not np.shares_memory(first, actual)
            for value, snapshot in zip(values, snapshots, strict=True):
                np.testing.assert_array_equal(value, snapshot)
        restored = runtime.RuntimeArtifact.from_json(fn.runtime_artifact().to_json())
        replay_values = roots("lhs", storage, (5, k, n), 14190)
        receipt = runtime.launch(restored, dict(zip(restored.metadata["arg_names"], replay_values, strict=True)))
        assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
        check(receipt["output"], "lhs", replay_values)
    finally:
        fn.close_native_storage()


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_native_bounded_rejections_preserve_last_completed_frame(storage):
    from tessera.compiler.prepared_native_tensor_frame import _descriptor

    assert runtime._nvidia_device_name() == "sm_120"
    fn = ts.jit(target="nvidia_sm120", shape_bounds={"M": 65})(lhs_add)
    try:
        values = roots("lhs", storage, (5, 32, 11), 14103)
        expected = fn(*values)
        owner = fn._native_tensor_last_owner
        before = owner.stats()
        # Bypass Python's shape gate to exercise native transactional checks.
        active = owner._active_specs(values)
        inputs, _ = owner._root_views(values, False, active)
        output = np.full(expected.shape, 999, dtype=np.float32)
        view = _descriptor(active[owner.frame["output"]], output.ctypes.data, output.strides)
        for bad_rows in (0, -1, 66):
            original = inputs[0].shape[0]
            inputs[0].shape[0] = bad_rows
            status = owner.lib.tessera_nvidia_native_frame_invoke_v1(
                owner.handle, inputs, len(inputs), ct.byref(view), 1)
            inputs[0].shape[0] = original
            assert status != 0
            assert owner.stats() == before
            np.testing.assert_array_equal(output, np.full_like(output, 999))
        # A residual with incompatible active rows, even within capacity,
        # cannot change the prior frame's geometry or generation.
        inputs[1].shape[0] = 4
        assert owner.lib.tessera_nvidia_native_frame_invoke_v1(
            owner.handle, inputs, len(inputs), ct.byref(view), 1) != 0
        assert owner.stats() == before
        inputs[1].shape[0] = 5
        view.bytes -= 4
        assert owner.lib.tessera_nvidia_native_frame_invoke_v1(
            owner.handle, inputs, len(inputs), ct.byref(view), 1) != 0
        assert owner.stats() == before
        np.testing.assert_array_equal(output, np.full_like(output, 999))
        # Profiling remains legal for the last completed binding after refusal.
        timing = owner.profile(2)
        assert len(timing["stage_ms"]) == 2 and timing["program_ms"] > 0
        check(owner(values), "lhs", values)
    finally:
        fn.close_native_storage()


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_bounded_resident_roots_order_streams_after_shrink_and_grow(storage, monkeypatch):
    from contextlib import ExitStack
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed, queue_delayed_upload

    assert runtime._nvidia_device_name() == "sm_120"
    fn = ts.jit(target="nvidia_sm120", shape_bounds={"M": 65})(lhs_add)
    with ExitStack() as stack:
        sessions = [stack.enter_context(NvidiaDeviceSession()) for _ in range(3)]
        try:
            stats = None
            for m in (17, 1, 65, 5):
                values = roots("lhs", storage, (m, 35, 19), 14110 + m)
                buffers = [session.upload(np.zeros_like(value))
                           for session, value in zip(sessions, values, strict=True)]
                assert all(session.synchronize() == 0 for session in sessions)
                borrowed = [Borrowed(buffer, session.stream)
                            for session, buffer in zip(sessions, buffers, strict=True)]
                completed = [[] for _ in values]
                callbacks = [queue_delayed_upload(session, buffer, value, session.stream, done)
                             for session, buffer, value, done in zip(
                                 sessions, buffers, values, completed, strict=True)]
                assert not any(completed)
                if stats is None:
                    actual = fn(*borrowed)
                else:
                    with monkeypatch.context() as guard:
                        guard.setattr(fn, "_fn", forbidden)
                        guard.setattr(fn, "_trace_frontend_capture", forbidden)
                        guard.setattr(subprocess, "run", forbidden)
                        guard.setattr(subprocess, "Popen", forbidden)
                        actual = fn(*borrowed)
                check(actual, "lhs", values)
                assert all(completed) and len(callbacks) == len(values)
                owner = fn._native_tensor_last_owner
                if stats is None:
                    stats = owner.stats()
                assert owner.stats()["allocations"] == stats["allocations"] == 1
                assert owner.stats()["capacity"] == stats["capacity"]
                for session, buffer, expected in zip(sessions, buffers, values, strict=True):
                    np.testing.assert_array_equal(session.download(buffer), expected)
        finally:
            fn.close_native_storage()
