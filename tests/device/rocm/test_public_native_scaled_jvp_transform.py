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



def retained_call(owner, active=(0, 1, 2, 3)):
    child = owner._native_public_jvp_owners[active][1]
    return next(iter(child._native_prepared_jvp_calls.values()))


def test_retained_public_jvp_warm_call_does_not_decode_or_prepare(monkeypatch):
    from tessera.compiler.native_jvp import NativeJVPArtifact
    from tessera.compiler.native_scaled_program import NativeScaledProgram, PreparedScaledProgram
    _, owner, values, directions = case(prefix=())
    try:
        expected = jvp(owner, values, directions)
        call = retained_call(owner)
        handle = call._owner.handle.value
        def forbidden(*args, **kwargs):
            raise AssertionError("warm retained execution decoded or prepared an artifact")
        monkeypatch.setattr(NativeJVPArtifact, "runtime_metadata", forbidden)
        monkeypatch.setattr(NativeJVPArtifact, "validate", forbidden)
        monkeypatch.setattr(NativeScaledProgram, "from_manifest", forbidden)
        monkeypatch.setattr(PreparedScaledProgram, "__init__", forbidden)
        actual = jvp(owner, values, tuple(seed*-.5 for seed in directions))
        np.testing.assert_allclose(actual[0], expected[0], rtol=4e-5, atol=3e-6)
        np.testing.assert_allclose(actual[1], expected[1]*-.5, rtol=4e-5, atol=3e-6)
        assert retained_call(owner) is call and call._owner.handle.value == handle
        assert owner.last_jvp_execution["retained_native_owner"] is True
    finally:
        owner.close_native_storage()


def test_retained_public_jvp_owners_are_isolated_and_close():
    owners = [case(prefix=())[1] for _ in range(2)]
    _, _, values, directions = case(prefix=())
    try:
        outputs = [jvp(owner, values, directions) for owner in owners]
        calls = [retained_call(owner) for owner in owners]
        assert calls[0] is not calls[1]
        assert calls[0]._owner.handle.value != calls[1]._owner.handle.value
        changed = tuple(value*np.float32(.75) for value in values)
        expected = reference([*changed, *directions], False, False, True)
        for got, want in zip(jvp(owners[1], changed, directions), expected, strict=True):
            np.testing.assert_allclose(got, want, rtol=4e-5, atol=3e-6)
        for got, want in zip(jvp(owners[0], values, directions), outputs[0], strict=True):
            np.testing.assert_array_equal(got, want)
        owners[0].close_native_storage()
        assert calls[0]._owner.handle.value == 0
        with pytest.raises(RuntimeError, match="closed"):
            calls[0].invoke([*values, *directions])
        for got, want in zip(jvp(owners[1], values, directions), outputs[1], strict=True):
            np.testing.assert_array_equal(got, want)
    finally:
        for owner in owners:
            owner.close_native_storage()


def test_retained_public_jvp_bad_frame_preserves_last_generation():
    _, owner, values, directions = case(prefix=())
    try:
        expected = jvp(owner, values, directions)
        call = retained_call(owner)
        bad = [values[0].reshape(-1), *values[1:], *directions]
        with pytest.raises(ValueError, match="storage/shape/layout"):
            call.invoke(bad)
        for got, want in zip(jvp(owner, values, directions), expected, strict=True):
            np.testing.assert_array_equal(got, want)
    finally:
        owner.close_native_storage()


def test_retained_public_jvp_concurrent_frames_are_transactions():
    import ctypes as c
    from concurrent.futures import ThreadPoolExecutor
    _, owner, values, directions = case(prefix=())
    hip = c.CDLL("/opt/rocm/lib/libamdhip64.so")
    jvp(owner, values, directions)
    def invoke(index):
        assert hip.hipSetDevice(0) == 0
        roots = tuple(value*np.float32(1+index*.125) for value in values)
        seeds = tuple(seed*np.float32(1-index*.125) for seed in directions)
        expected = reference([*roots, *seeds], False, False, True)
        for _ in range(3):
            actual = jvp(owner, roots, seeds)
            for got, want in zip(actual, expected, strict=True):
                np.testing.assert_allclose(got, want, rtol=4e-5, atol=3e-6)
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(invoke, range(4)))
    finally:
        owner.close_native_storage()


def test_retained_public_jvp_fork_refuses_before_driver_and_locks(monkeypatch):
    import select
    import signal
    _, owner, values, directions = case(prefix=())
    expected = jvp(owner, values, directions)
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read)
        try:
            def forbidden(*args, **kwargs):
                raise AssertionError("inherited public JVP touched the HIP driver")
            monkeypatch.setattr(runtime, "_rocm_chip", forbidden)
            jvp(owner, values, directions)
            message = b"unexpected success"
        except Exception as error:
            message = str(error).encode()
        os.write(write, message)
        os._exit(0)
    os.close(write)
    try:
        ready, _, _ = select.select([read], [], [], 8)
        if not ready:
            os.kill(pid, signal.SIGKILL)
            pytest.fail("inherited retained JVP blocked on a parent lock")
        assert b"cannot cross fork" in os.read(read, 4096)
        for got, want in zip(jvp(owner, values, directions), expected, strict=True):
            np.testing.assert_array_equal(got, want)
    finally:
        os.close(read)
        os.waitpid(pid, 0)
        owner.close_native_storage()

