"""Native bounded source reverse owner; public/portable integration is separate."""
from contextlib import ExitStack
import ctypes as ct
import hashlib
import json
import re
import time

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.native_source_attention_reverse import export_native_source_attention_reverse
from tessera.compiler.nvidia_native import _compile_tile_ir
from tests.device.nvidia.test_source_attention_reverse import oracle, source

pytestmark = pytest.mark.hardware_nvidia


class RawBoundedSourceOwner:
    def __init__(self, roots, storage, rows, variant, requested, bounds):
        self.lib = rt._load_nvidia_ptx_launch()
        self.api = ("tessera_nvidia_source_attention_reverse_rows_bounded_" if rows else
                    "tessera_nvidia_source_attention_reverse_bounded_")
        assert self.lib is not None and hasattr(self.lib, self.api + "prepare"), "matching bounded source owner required"
        self.requested, self.rows, self.handle = requested, rows, ct.c_uint64()
        compile_start = time.perf_counter()
        graph = source(roots, variant, requested, rows=rows).replace(
            "module attributes {", "module attributes {tessera.attention_sequence_bounds = array<i64: "
            + ", ".join(map(str, bounds)) + ">, ", 1)
        program = export_native_source_attention_reverse(graph)
        self.plan, self.images = program.manifest, []
        image_args = []
        for i, stages in enumerate(program.lower_members()):
            entry = re.search(r"\bllvm.func @(\w+)\(", stages[2])[1]
            _, ptx, *_ = _compile_tile_ir(stages[2], entry)
            image = ct.create_string_buffer(ptx.encode())
            self.images.append(image)
            image_args.extend((image, len(ptx.encode()), entry.encode()))
            if i == 0:
                image_args.append(int("_cooperative_128_" in entry))  # Preserve compiler-selected norm geometry.
        self.compile_host_ms = (time.perf_counter() - compile_start) * 1000
        P, S, I, L, U = ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_int64, ct.c_uint64
        self.prepare = getattr(self.lib, self.api + "prepare")
        self.prepare.argtypes = ([P, S, ct.c_char_p, I] + [P, S, ct.c_char_p] * (len(self.images) - 1)
                                 + [ct.POINTER(L), ct.POINTER(I), ct.POINTER(I), ct.POINTER(I), S, I, ct.POINTER(U)])
        self.prepare.restype = I
        self.invoke = getattr(self.lib, self.api + "invoke")
        self.invoke.argtypes = [U, ct.POINTER(L), ct.POINTER(P), ct.POINTER(S), S,
                                ct.POINTER(U), S, ct.POINTER(P), ct.POINTER(S), S]
        self.invoke.restype = I
        self.times = getattr(self.lib, "tessera_nvidia_source_attention_reverse_"
                             + ("rows_" if rows else "") + "last_times")
        self.times.argtypes, self.times.restype = [U, ct.POINTER(ct.c_float), S], I
        self.lib.tessera_nvidia_source_attention_reverse_close.argtypes = [U]
        self.lib.tessera_nvidia_source_attention_reverse_close.restype = I
        self.lib.tessera_nvidia_attention_vjp_last_error.restype = ct.c_char_p
        roles = self.plan["role_indices"]
        selected = [roles.index(i) for i in requested]
        prepare_start = time.perf_counter()
        self.check(self.prepare(
            *image_args, (L * 7)(*self.plan["shape_bounds"]), (I * 3)(*roles),
            (I * 2)(*self.plan["cotangent_indices"]), (I * len(selected))(*selected),
            len(selected), {"f16": 2, "bf16": 3}[storage], ct.byref(self.handle)))
        self.native_prepare_host_ms = (time.perf_counter() - prepare_start) * 1000

    def check(self, status):
        assert status == 0, self.lib.tessera_nvidia_attention_vjp_last_error().decode()

    def call(self, dims, values, outputs, *, resident=None, sizes=None):
        addresses = [x.__cuda_array_interface__["data"][0] for x in resident] if resident is not None else [x.ctypes.data for x in values]
        streams = None
        if resident is not None:
            streams = [x.__cuda_array_interface__["stream"] for x in resident]
        return self.invoke(
            self.handle, (ct.c_int64 * 7)(*dims) if dims is not None else None,
            (ct.c_void_p * 5)(*addresses),
            (ct.c_size_t * 5)(*(sizes if sizes is not None else [x.nbytes for x in values])), 5,
            (ct.c_uint64 * 5)(*streams) if streams is not None else None,
            5 if streams is not None else 0,
            (ct.c_void_p * len(outputs))(*(x.ctypes.data for x in outputs)),
            (ct.c_size_t * len(outputs))(*(x.nbytes for x in outputs)), len(outputs))

    def close(self):
        if self.handle.value:
            self.check(self.lib.tessera_nvidia_source_attention_reverse_close(self.handle))
            self.handle.value = 0


def guarded_outputs(values, requested):
    roots, outputs = [], []
    for i in requested:
        arena = np.full(values[i].size + 16, 23, values[i].dtype)
        roots.append(arena)
        outputs.append(arena[8:-8].reshape(values[i].shape))
    return roots, tuple(outputs)


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("rows", [False, True])
@pytest.mark.parametrize("variant,requested", [
    ("normal", (0, 1, 2)), ("permuted", (2, 0)), ("swapped", (2, 0, 1))])
@pytest.mark.parametrize("capacity", [(7, 11, 4, 3), (17, 19, 35, 7), (17, 19, 257, 13)])
def test_native_bounded_source_owner_reuses_one_capacity_frame(
        storage, rows, variant, requested, capacity, monkeypatch, record_property):
    assert rt._nvidia_device_name() == "sm_120"
    dtype = np.dtype(np.float16 if storage == "f16" else ml_dtypes.bfloat16)
    sq_cap, sk_cap, d, dv = capacity
    trace = (np.zeros((1, 2, 1, d), dtype), np.zeros((1, 1, 1, d), dtype),
             np.zeros((1, 1, 1, dv), dtype))
    roots = (trace[2], trace[0], trace[1]) if variant == "permuted" else trace
    owner = RawBoundedSourceOwner(roots, storage, rows, variant, requested, capacity[:2])
    handle = owner.handle.value
    retained, stage_times = [], []
    try:
        def forbidden(*args, **kwargs):
            pytest.fail("retained bounded source invocation compiled or prepared again")
        with monkeypatch.context() as guard:
            import subprocess
            guard.setattr(subprocess, "run", forbidden)
            guard.setattr(subprocess, "Popen", forbidden)
            guard.setattr(owner, "prepare", forbidden)
            for step, (sq, sk) in enumerate(((1, 1), (sq_cap, sk_cap), (3, 2),
                                             (1, sk_cap), (sq_cap, 1))):
                rng = np.random.default_rng(20261019 + step)
                q = rng.normal(0, .4, (1, 2, sq, d)).astype(dtype)
                if step == 2:
                    q.fill(0)
                k = rng.normal(0, .3, (1, 1, sk, d)).astype(dtype)
                v = rng.normal(0, .3, (1, 1, sk, dv)).astype(dtype)
                do = rng.normal(0, .2, (1, 2, sq, dv)).astype(dtype)
                dlse = rng.normal(0, .2, (1, 2, sq)).astype(np.float32)
                roots = (v, q, k) if variant == "permuted" else (q, k, v)
                values = (*roots, *((dlse, do) if variant == "swapped" else (do, dlse)))
                frozen = tuple(x.tobytes() for x in values)
                for x in values:
                    x.flags.writeable = False
                dims = (1, 2, 1, sq, sk, d, dv)
                arenas, outputs = guarded_outputs(values, requested)
                owner.check(owner.call(dims, values, outputs))
                expected = oracle(q, k, v, do, dlse)
                expected = (expected[2], expected[0], expected[1]) if variant == "permuted" else expected
                for value, i in zip(outputs, requested, strict=True):
                    np.testing.assert_allclose(value.astype(np.float32), expected[i].astype(np.float32),
                                               rtol=.03 if storage == "bf16" else .006, atol=.002)
                for arena in arenas:
                    np.testing.assert_array_equal(arena[:8], arena.dtype.type(23))
                    np.testing.assert_array_equal(arena[-8:], arena.dtype.type(23))
                for x, original in zip(values, frozen, strict=True):
                    assert x.tobytes() == original
                for old, saved in retained:
                    np.testing.assert_array_equal(old, saved)
                    assert all(not np.shares_memory(old, new) for new in outputs)
                retained.extend((x, x.copy()) for x in outputs)
                times = (ct.c_float * (6 if rows else 5))()
                owner.check(owner.times(owner.handle, times, len(times)))
                assert all(np.isfinite(x) and x >= 0 for x in times)
                stage_times.append(list(times))
                assert owner.handle.value == handle

                bad_dims = [(0, 2, 1, sq, sk, d, dv), (1, 2, 1, -1, sk, d, dv),
                            (1, 2, 1, sq_cap + 1, sk, d, dv), (1, 2, 1, sq, sk_cap + 1, d, dv),
                            (1, 3, 1, sq, sk, d, dv), (1, 2, 1, sq, sk, d + 1, dv),
                            (1, 2, 1, 2**63 - 1, sk, d, dv), None]
                for invalid in bad_dims:
                    _, guarded = guarded_outputs(values, requested)
                    assert owner.call(invalid, values, guarded) != 0
                    for x in guarded:
                        np.testing.assert_array_equal(x, x.dtype.type(23))
                    assert owner.times(owner.handle, times, len(times)) != 0
                if step == 0:
                    legacy = getattr(owner.lib, owner.api.replace("bounded_", "") + "invoke")
                    P, S, U = ct.c_void_p, ct.c_size_t, ct.c_uint64
                    legacy.argtypes = [U, ct.POINTER(P), ct.POINTER(S), S,
                                       ct.POINTER(U), S, ct.POINTER(P), ct.POINTER(S), S]
                    legacy.restype = ct.c_int
                    _, guarded = guarded_outputs(values, requested)
                    assert legacy(owner.handle, (P * 5)(*(x.ctypes.data for x in values)),
                                  (S * 5)(*(x.nbytes for x in values)), 5, None, 0,
                                  (P * len(guarded))(*(x.ctypes.data for x in guarded)),
                                  (S * len(guarded))(*(x.nbytes for x in guarded)), len(guarded)) != 0
                    assert owner.times(owner.handle, times, len(times)) != 0
                    for x in guarded:
                        np.testing.assert_array_equal(x, x.dtype.type(23))
                _, guarded = guarded_outputs(values, requested)
                sizes = [x.nbytes for x in values]
                sizes[0] -= 2
                assert owner.call(dims, values, guarded, sizes=sizes) != 0
                assert owner.times(owner.handle, times, len(times)) != 0
                owner.check(owner.call(dims, values, guarded))
                for a, b in zip(guarded, outputs, strict=True):
                    np.testing.assert_array_equal(a, b)
        record_property("stage_and_program_device_ms", json.dumps(stage_times))
        record_property("program_sha256", owner.plan["program_graph_sha256"])
        record_property("member_image_sha256", json.dumps([
            hashlib.sha256(bytes(image)[:-1]).hexdigest() for image in owner.images]))
    finally:
        owner.close()
    owner.close()


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("rows", [False, True])
@pytest.mark.parametrize("mode", ["different", "legacy", "per_thread"])
def test_bounded_source_owner_orders_producers_and_recovers_context(storage, rows, mode):
    from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed, queue_delayed_upload
    assert rt._nvidia_device_name() == "sm_120"
    dtype = np.dtype(np.float16 if storage == "f16" else ml_dtypes.bfloat16)
    rng = np.random.default_rng(933)
    shapes = ((1, 2, 3, 4), (1, 1, 5, 4), (1, 1, 5, 3), (1, 2, 3, 3), (1, 2, 3))
    values = tuple(rng.normal(0, .2, shape).astype(np.float32 if i == 4 else dtype)
                   for i, shape in enumerate(shapes))
    trace = tuple(np.zeros((*x.shape[:2], 1, x.shape[-1]), dtype) for x in values[:3])
    owner = RawBoundedSourceOwner(trace, storage, rows, "normal", (0, 1, 2), (7, 11))
    try:
        with ExitStack() as stack:
            sessions = tuple(stack.enter_context(NvidiaDeviceSession()) for _ in values)
            resident = tuple(s.upload(np.zeros_like(v)) for s, v in zip(sessions, values, strict=True))
            for session in sessions:
                assert session.synchronize() == 0
            completed, callbacks, borrowed = [], [], []
            for session, buffer, value in zip(sessions, resident, values, strict=True):
                stream = 1 if mode == "legacy" else 2 if mode == "per_thread" else session.stream
                callbacks.append(queue_delayed_upload(session, buffer, value, stream, completed))
                borrowed.append(Borrowed(buffer, stream))
            assert len(completed) < 5, "producer proof requires pending writes"
            dims = (1, 2, 1, 3, 5, 4, 3)
            _, outputs = guarded_outputs(values, (0, 1, 2))
            owner.check(owner.call(dims, values, outputs, resident=borrowed))
            assert len(completed) == 5 and len(callbacks) == 5
            expected = oracle(*values)
            for a, b in zip(outputs, expected, strict=True):
                np.testing.assert_allclose(a.astype(np.float32), b.astype(np.float32),
                                           rtol=.03 if storage == "bf16" else .006, atol=.002)
            _, guarded = guarded_outputs(values, (0, 1, 2))
            invalid = [Borrowed(buffer, 0) for buffer in resident]
            assert owner.call(dims, values, guarded, resident=invalid) != 0
            for x in guarded:
                np.testing.assert_array_equal(x, x.dtype.type(23))
            times = (ct.c_float * (6 if rows else 5))()
            assert owner.times(owner.handle, times, len(times)) != 0
            driver = ct.CDLL("libcuda.so.1")
            driver.cuCtxGetCurrent.argtypes = [ct.POINTER(ct.c_void_p)]
            driver.cuCtxSetCurrent.argtypes = [ct.c_void_p]
            context = ct.c_void_p()
            assert driver.cuCtxGetCurrent(ct.byref(context)) == 0
            assert driver.cuCtxSetCurrent(None) == 0
            try:
                assert owner.call(dims, values, guarded) != 0
                assert owner.times(owner.handle, times, len(times)) != 0
            finally:
                assert driver.cuCtxSetCurrent(context) == 0
            owner.check(owner.call(dims, values, guarded))
            for a, b in zip(guarded, outputs, strict=True):
                np.testing.assert_array_equal(a, b)
    finally:
        owner.close()
