"""Compiled original-source reverse program with native intermediate ownership."""
from contextlib import ExitStack
import ctypes as ct
import subprocess

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.prepared_nvidia_source_reverse import PreparedSourceReverse
from tests.unit.test_native_sm120_source_reverse import source
from tests.device.nvidia.test_ordered_resident_tensor_dag import queue_delayed_upload

pytestmark = pytest.mark.hardware_nvidia


def inputs(storage, permuted, seed=87, witness=False):
    import ml_dtypes
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .75, (17, 35)).astype(dtype)
    b = rng.normal(0, .2, (35, 19)).astype(dtype)
    dy = rng.normal(0, .2, (17, 19)).astype(np.float32)
    if witness:
        b.fill(0)
        b[:, :2] = 1
        dy.fill(0)
        dy[:, 0], dy[:, 1] = 1 + 2.0**(-12 if storage == "f16" else -9), -1
    return (b, x, dy) if permuted else (x, b, dy)


def oracle(roots, permuted):
    x, b, dy = (roots[1], roots[0], roots[2]) if permuted else roots
    xx, bb, seed = (v.astype(np.float64) for v in (x, b, dy))
    wide = seed @ bb.T
    # The native AD Graph stores the matmul adjoint to the primal half dtype.
    edge = wide.astype(x.dtype).astype(np.float64)
    g = np.where(xx > 0, edge, 0)
    r = 1 / np.sqrt(np.mean(xx * xx, axis=1, keepdims=True) + 1e-5)
    reference = g * r - xx * r**3 * np.mean(g * xx, axis=1, keepdims=True)
    rounded = reference.astype(x.dtype)
    ulp = np.abs(np.nextafter(rounded, x.dtype.type(np.inf)).astype(np.float64) -
                 rounded.astype(np.float64))
    gamma = 19 * np.finfo(np.float32).eps / (1 - 19 * np.finfo(np.float32).eps)
    edge_ulp = np.abs(np.nextafter(wide.astype(x.dtype), x.dtype.type(np.inf)).astype(np.float64) -
                      wide.astype(x.dtype).astype(np.float64))
    edge_budget = gamma * (np.abs(seed) @ np.abs(bb.T)) + edge_ulp
    propagated = r * edge_budget + np.abs(xx) * r**3 * np.mean(
        np.abs(xx) * edge_budget, axis=1, keepdims=True)
    budget = propagated + 35 * 32 * np.finfo(np.float32).eps * (np.abs(reference) + .01) + ulp
    return reference, budget


def check(actual, roots, permuted):
    reference, budget = oracle(roots, permuted)
    assert len(actual) == 1 and actual[0].dtype == roots[int(permuted)].dtype
    assert np.isfinite(actual[0]).all()
    assert np.all(np.abs(actual[0].astype(np.float64) - reference) <= budget)


@pytest.fixture(scope="module", params=[("f16", False), ("f16", True), ("bf16", False), ("bf16", True)])
def prepared(request):
    assert rt._nvidia_device_name() == "sm_120"
    storage, permuted = request.param
    owner = PreparedSourceReverse(source(storage, permuted))
    yield owner, storage, permuted
    owner.close()


@pytest.mark.parametrize("resident", [False, True])
def test_composed_reverse_replays_without_python_member_launches(prepared, resident, monkeypatch):
    owner, storage, permuted = prepared
    with NvidiaDeviceSession() as session:
        for seed in (87, 193):
            roots = inputs(storage, permuted, seed)
            values = tuple(session.upload(v) for v in roots) if resident else roots
            with monkeypatch.context() as warm:
                warm.setattr(subprocess, "run", lambda *a, **k: pytest.fail("warm reverse compiled"))
                actual = owner(values)
                # Replays and caller mutation cannot reuse an old returned gradient.
                retained = actual[0].copy()
                replay = owner(values)
            check(actual, roots, permuted)
            check(replay, roots, permuted)
            assert not np.shares_memory(actual[0], replay[0])
            np.testing.assert_array_equal(actual[0], retained)
        for captured in (False, True):
            actual, timing = owner.profile(values, repeats=16, captured=captured)
            check(actual, roots, permuted)
            assert len(timing["stage_ms"]) == 2
            assert all(np.isfinite(v) and v > 0 for v in timing["stage_ms"])
            assert np.isfinite(timing["program_ms"]) and timing["program_ms"] > 0


def test_composed_reverse_preserves_fp32_cotangent_witness(prepared):
    owner, storage, permuted = prepared
    roots = inputs(storage, permuted, witness=True)
    actual = owner(roots)
    check(actual, roots, permuted)
    assert np.any(actual[0] != 0)
    narrowed = (*roots[:2], roots[2].astype(roots[int(permuted)].dtype).astype(np.float32))
    broken_reference, _ = oracle(narrowed, permuted)
    assert not np.any(broken_reference)


def test_composed_reverse_orders_all_pending_resident_roots(prepared):
    owner, storage, permuted = prepared
    roots = inputs(storage, permuted, 427)
    with ExitStack() as stack:
        sessions = [stack.enter_context(NvidiaDeviceSession()) for _ in range(3)]
        values = tuple(session.upload(np.zeros_like(root))
                       for session, root in zip(sessions, roots, strict=True))
        for session in sessions:
            assert session.synchronize() == 0
        owner(values)  # Allocate scratch before delayed producer writes.
        completed = []
        callbacks = [queue_delayed_upload(session, value, root, session.stream, completed)
                     for session, value, root in zip(sessions, values, roots, strict=True)]
        assert not completed
        actual = owner(values)
        assert len(completed) == len(callbacks) == 3
        check(actual, roots, permuted)


def test_native_reverse_rejects_forged_capacity_without_output_commit(prepared):
    owner, storage, permuted = prepared
    roots = inputs(storage, permuted)
    with NvidiaDeviceSession() as session:
        values = tuple(session.upload(v) for v in roots)
        small = session.empty((1, 1), np.float32)
        output, views, streams = owner._frame(values)
        output.fill(123)
        views[0].data = small.ptr
        status = owner.lib.tessera_nvidia_source_reverse_invoke_v1(owner.handle, views, 4, streams, 3)
        assert status != 0
        assert b"allocation context/capacity mismatch" in owner.lib.tessera_nvidia_matmul_last_error()
        np.testing.assert_array_equal(output, 123)
        check(owner(values), roots, permuted)
