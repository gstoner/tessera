"""Owning SM120 proof of the original-source RMSNorm/attention pullback."""
from contextlib import ExitStack
import ctypes as ct
import subprocess

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.native_source_attention_reverse import export_native_source_attention_reverse
from tessera.compiler.prepared_source_attention_reverse import PreparedSourceAttentionReverse
from tests.device.nvidia.test_saved_lse_half_backward_kernel import reference
from tests.device.nvidia.test_source_attention_prefix import trace
from tests.unit.test_native_attention_prefix_program import (
    prefix_attention, prefix_attention_permuted, prefix_attention_swapped_outputs)

pytestmark = pytest.mark.hardware_nvidia


def source(roots, variant, requested, *, rows=False):
    function = {"normal": prefix_attention, "permuted": prefix_attention_permuted,
                "swapped": prefix_attention_swapped_outputs}[variant]
    text = trace(function, roots)
    text = text.replace("module attributes {",
                        "module attributes {tessera.native.source_attention_reverse_request = unit, ", 1)
    if rows:
        text = text.replace("module attributes {",
                            "module attributes {tessera.native.source_attention_reverse_rows_request = unit, ", 1)
    index = text.index("attributes {", text.index("func.func @")) + len("attributes {")
    return text[:index] + ('tessera.autodiff = "reverse", tessera.autodiff.wrt_indices = '
                          + str(list(requested)) + ", ") + text[index:]


def oracle(q, k, v, do, dlse):
    # Match the explicit storage edges, while deriving attention and RMSNorm
    # derivatives independently in float64 (FP32 normalization statistics).
    qf = q.astype(np.float32)
    inv = 1 / np.sqrt(np.mean(qf * qf, axis=-1, keepdims=True) + np.float32(1e-5))
    normalized = (qf * inv).astype(q.dtype)
    _, gradients = reference(normalized, k, v, do, None, dlse, causal=False)
    dn = gradients[0].astype(q.dtype).astype(np.float64)
    raw = q.astype(np.float64)
    inverse = inv.astype(np.float64)
    dq = dn * inverse - raw * inverse**3 * np.mean(dn * raw, axis=-1, keepdims=True)
    return (dq.astype(q.dtype), gradients[1].astype(k.dtype), gradients[2].astype(v.dtype))


@pytest.mark.parametrize("rows", [False, True])
@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("variant,requested,profile", [
    ("normal", (0, 1, 2), "random"), ("permuted", (2, 0), "random"),
    ("swapped", (2, 0, 1), "random"), ("normal", (0, 1, 2), "zero"),
    ("normal", (0,), "tiny"), ("normal", (0, 1, 2), "wide"),
    ("normal", (0, 1, 2), "rows136")])
def test_source_reverse_original_roots_and_both_cotangents(storage, variant, requested, profile, rows, monkeypatch):
    assert rt._nvidia_device_name() == "sm_120", "owning SM120 device required"
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    sq, sk, d, dv = (17, 19, 257, 13) if profile == "wide" else (
        (68, 5, 35, 7) if profile == "rows136" else (3, 5, 4, 3))
    rng = np.random.default_rng(20261018)
    q = rng.normal(0, .4, (1, 2, sq, d)).astype(dtype)
    if profile in ("zero", "tiny"):
        q = (np.zeros(q.shape) if profile == "zero" else q.astype(np.float32) * 1e-4).astype(dtype)
    k = rng.normal(0, .3, (1, 1, sk, d)).astype(dtype)
    v = rng.normal(0, .3, (1, 1, sk, dv)).astype(dtype)
    do = rng.normal(0, .2, (1, 2, sq, dv)).astype(dtype)
    dlse = rng.normal(0, .2, (1, 2, sq)).astype(np.float32)
    roots = (v, q, k) if variant == "permuted" else (q, k, v)
    seeds = (dlse, do) if variant == "swapped" else (do, dlse)
    values = (*roots, *seeds)
    frozen = tuple(x.copy() for x in values)
    for x in values:
        x.flags.writeable = False
    expected = oracle(q, k, v, do, dlse)
    expected = (expected[2], expected[0], expected[1]) if variant == "permuted" else expected
    expected = tuple(expected[i] for i in requested)
    with PreparedSourceAttentionReverse(export_native_source_attention_reverse(source(roots, variant, requested, rows=rows))) as owner:
        first = owner(*values)
        assert owner.receipt["execution_kind"] == "native_gpu"
        assert len(owner.receipt["stage_device_ms"]) == (5 if rows else 4)
        for actual, ref in zip(first, expected, strict=True):
            np.testing.assert_allclose(actual.astype(np.float32), ref.astype(np.float32),
                                       rtol=.03 if storage == "bf16" else .006,
                                       atol=.02 if profile in ("zero", "tiny") else .002)
        retained = tuple(x.copy() for x in first)
        def forbidden(*args, **kwargs):
            pytest.fail("warm source reverse invoked compiler")
        with monkeypatch.context() as guard:
            guard.setattr(subprocess, "run", forbidden)
            guard.setattr(subprocess, "Popen", forbidden)
            second = owner(*values)
        for old, saved, new in zip(first, retained, second, strict=True):
            np.testing.assert_array_equal(old, saved)
            np.testing.assert_array_equal(new, saved)
            assert not np.shares_memory(old, new)
        if profile == "random":
            with ExitStack() as stack:
                sessions = tuple(stack.enter_context(NvidiaDeviceSession()) for _ in values)
                resident = tuple(s.upload(x) for s, x in zip(sessions, values, strict=True))
                # Each root has a separate producer stream. Native event waits
                # must establish dependencies; no explicit synchronize here.
                third = owner(*resident)
                for actual, saved in zip(third, retained, strict=True):
                    np.testing.assert_array_equal(actual, saved)
        for value, saved in zip(values, frozen, strict=True):
            np.testing.assert_array_equal(value, saved)
        with pytest.raises(ValueError, match="storage"):
            owner(*values[:3], *(x.astype(np.float64) for x in values[3:]))
        assert owner.receipt is None
    owner.close()
    with pytest.raises(ValueError, match="closed"):
        owner(*values)


@pytest.mark.parametrize("rows", [False, True])
@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("mode", ["different", "legacy", "per_thread"])
def test_source_reverse_waits_pending_writes_and_recovers_after_rejection(storage, mode, rows):
    from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed, queue_delayed_upload
    assert rt._nvidia_device_name() == "sm_120"
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    rng = np.random.default_rng(732)
    shapes = ((1, 2, 3, 4), (1, 1, 5, 4), (1, 1, 5, 3), (1, 2, 3, 3), (1, 2, 3))
    values = tuple(rng.normal(0, .2, shape).astype(np.float32 if i == 4 else dtype)
                   for i, shape in enumerate(shapes))
    with PreparedSourceAttentionReverse(export_native_source_attention_reverse(
            source(values[:3], "normal", (0, 1, 2), rows=rows))) as owner, ExitStack() as stack:
        expected = owner(*values)
        sessions = tuple(stack.enter_context(NvidiaDeviceSession()) for _ in values)
        resident = tuple(s.upload(np.zeros_like(v)) for s, v in zip(sessions, values, strict=True))
        for session in sessions:
            assert session.synchronize() == 0
        completed, callbacks, borrowed = [], [], []
        for session, buffer, value in zip(sessions, resident, values, strict=True):
            stream = 1 if mode == "legacy" else 2 if mode == "per_thread" else session.stream
            callbacks.append(queue_delayed_upload(session, buffer, value, stream, completed))
            borrowed.append(Borrowed(buffer, stream))
        assert len(completed) < 5, "stream proof requires pending producer writes at invocation"
        actual = owner(*borrowed)
        assert len(completed) == 5 and len(callbacks) == 5
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a, b)
        wrong = (ct.c_size_t * 5)(*(v.nbytes for v in values))
        wrong[0] -= 2
        guarded = tuple(np.full_like(v, 23) for v in expected)
        status = getattr(owner._lib, owner._api + "invoke")(
            owner._handle, (ct.c_void_p * 5)(*(v.ctypes.data for v in values)), wrong, 5, None, 0,
            (ct.c_void_p * 3)(*(v.ctypes.data for v in guarded)),
            (ct.c_size_t * 3)(*(v.nbytes for v in guarded)), 3)
        assert status != 0
        assert b"extent" in owner._lib.tessera_nvidia_attention_vjp_last_error()
        for value in guarded:
            np.testing.assert_array_equal(value, np.full_like(value, 23))
        changed = tuple((v.astype(np.float32) * .7).astype(v.dtype) for v in values)
        result = owner(*changed)
        for a, b in zip(result, oracle(*changed), strict=True):
            np.testing.assert_allclose(a.astype(np.float32), b.astype(np.float32), rtol=.03, atol=.002)
        driver = ct.CDLL("libcuda.so.1")
        driver.cuCtxGetCurrent.argtypes = [ct.POINTER(ct.c_void_p)]
        driver.cuCtxSetCurrent.argtypes = [ct.c_void_p]
        context = ct.c_void_p()
        assert driver.cuCtxGetCurrent(ct.byref(context)) == 0
        assert driver.cuCtxSetCurrent(None) == 0
        try:
            with pytest.raises(RuntimeError, match="context generation"):
                owner(*values)
            assert owner.receipt is None
        finally:
            assert driver.cuCtxSetCurrent(context) == 0
        for a, b in zip(owner(*values), expected, strict=True):
            np.testing.assert_array_equal(a, b)
