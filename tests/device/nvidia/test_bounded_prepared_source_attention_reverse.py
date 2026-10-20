"""Owning SM120 prepared bounded source VJP; no ordinary JIT claim yet."""
from contextlib import ExitStack
import subprocess

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.native_source_attention_reverse import export_native_source_attention_reverse
from tessera.compiler.prepared_source_attention_reverse import PreparedSourceAttentionReverse
from tests.device.nvidia.test_source_attention_reverse import source, oracle

pytestmark = pytest.mark.hardware_nvidia


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("rows", [False, True])
@pytest.mark.parametrize("variant,requested", [
    ("normal", (0, 1, 2)), ("permuted", (2, 0)), ("swapped", (2, 0, 1))])
@pytest.mark.parametrize("capacity", [(7, 11, 4, 3), (17, 19, 257, 13)])
def test_prepared_bounded_source_reuses_images_and_owner(
        storage, rows, variant, requested, capacity, monkeypatch):
    assert rt._nvidia_device_name() == "sm_120"
    dtype = np.dtype(np.float16 if storage == "f16" else ml_dtypes.bfloat16)
    sq_cap, sk_cap, d, dv = capacity
    trace = (np.zeros((1, 2, 1, d), dtype), np.zeros((1, 1, 1, d), dtype),
             np.zeros((1, 1, 1, dv), dtype))
    roots = (trace[2], trace[0], trace[1]) if variant == "permuted" else trace
    graph = source(roots, variant, requested, rows=rows).replace(
        "module attributes {", f"module attributes {{tessera.attention_sequence_bounds = array<i64: {sq_cap}, {sk_cap}>, ", 1)
    program = export_native_source_attention_reverse(graph)
    with PreparedSourceAttentionReverse(program) as owner:
        handle, images, retained = None, None, []
        for step, (sq, sk) in enumerate(((1, 1), (sq_cap, sk_cap), (3, 2), (1, sk_cap), (sq_cap, 1))):
            rng = np.random.default_rng(20261020 + step)
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
            expected = oracle(q, k, v, do, dlse)
            expected = (expected[2], expected[0], expected[1]) if variant == "permuted" else expected
            def forbidden(*args, **kwargs):
                pytest.fail("warm bounded prepared call compiled or prepared again")
            with monkeypatch.context() as guard:
                guard.setattr(subprocess, "run", forbidden)
                guard.setattr(subprocess, "Popen", forbidden)
                if handle is not None:
                    guard.setattr(owner, "_prepare", forbidden)
                result = owner(*values)
            assert owner.receipt["execution_kind"] == "native_gpu"
            assert owner.receipt["runtime_dims"] == (1, 2, 1, sq, sk, d, dv)
            assert owner.receipt["shape_bounds"] == (1, 2, 1, sq_cap, sk_cap, d, dv)
            assert owner.receipt["native_call_binding"].endswith("_bounded")
            assert len(owner.receipt["stage_device_ms"]) == (5 if rows else 4)
            if handle is None:
                handle, images = owner._handle, owner.receipt["image_digests"]
            assert owner._handle == handle and owner.receipt["image_digests"] == images
            for actual, i in zip(result, requested, strict=True):
                np.testing.assert_allclose(actual.astype(np.float32), expected[i].astype(np.float32),
                                           rtol=.03 if storage == "bf16" else .006, atol=.002)
            for old, saved in retained:
                np.testing.assert_array_equal(old, saved)
                assert all(not np.shares_memory(old, x) for x in result)
            retained.extend((x, x.copy()) for x in result)
            if step == 0:
                with ExitStack() as stack:
                    sessions = tuple(stack.enter_context(NvidiaDeviceSession()) for _ in values)
                    resident = tuple(s.upload(x) for s, x in zip(sessions, values, strict=True))
                    with monkeypatch.context() as guard:
                        guard.setattr(subprocess, "run", forbidden)
                        guard.setattr(subprocess, "Popen", forbidden)
                        guard.setattr(owner, "_prepare", forbidden)
                        third = owner(*resident)
                    for a, b in zip(result, third, strict=True):
                        np.testing.assert_array_equal(a, b)
                    assert owner._handle == handle
            for value, original in zip(values, frozen, strict=True):
                assert value.tobytes() == original
            bad = list(values)
            bad[0] = bad[0].astype(np.float64)
            with pytest.raises(ValueError, match="storage"):
                owner(*bad)
            assert owner.receipt is None
            with pytest.raises(ValueError):
                owner(*values[:-1])
            assert owner.receipt is None
    owner.close()
    with pytest.raises(ValueError, match="closed"):
        owner(*values)
