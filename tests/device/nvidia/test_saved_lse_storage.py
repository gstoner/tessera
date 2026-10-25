"""Exact-SM120 mixed-storage saved-LSE forward packages."""
from contextlib import ExitStack
import ctypes as ct
import json
import subprocess

import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed, queue_delayed_upload

pytestmark = pytest.mark.hardware_nvidia


def saved(q, k, v):
    return ts.ops.flash_attn(q, k, v, causal=True, lse_checkpoint="saved")


def saved_open(q, k, v):
    return ts.ops.flash_attn(q, k, v, causal=False, lse_checkpoint="saved")


def saved_bias(v, q, k, bias):
    return ts.ops.flash_attn(q, k, v, bias=bias, causal=True, lse_checkpoint="saved")


def saved_bias_open(v, q, k, bias):
    return ts.ops.flash_attn(q, k, v, bias=bias, causal=False, lse_checkpoint="saved")


def reference(values, causal):
    q, k, v = (values[name].astype(np.float64) for name in ("q", "k", "v"))
    groups = q.shape[1] // k.shape[1]
    scores = q @ np.repeat(k, groups, axis=1).swapaxes(-1, -2) / np.sqrt(q.shape[-1])
    if "bias" in values:
        scores += values["bias"].astype(np.float64)
    if causal:
        sq, sk = q.shape[2], k.shape[2]
        allowed = np.arange(sk)[None, :] <= np.arange(sq)[:, None] + max(sk-sq, 0)
        scores = np.where(allowed, scores, -np.inf)
    maximum = scores.max(axis=-1, keepdims=True)
    exponent = np.exp(scores-maximum)
    denominator = exponent.sum(axis=-1, keepdims=True)
    return exponent/denominator @ np.repeat(v, groups, axis=1), maximum[..., 0]+np.log(denominator[..., 0])


def compare(pair, expected, storage):
    assert isinstance(pair, tuple) and len(pair) == 2
    assert pair[0].dtype == storage and pair[1].dtype == np.dtype("float32")
    assert all(np.isfinite(value).all() for value in pair)
    rounded = expected[0].astype(storage)
    upper = np.nextafter(rounded, storage.type(np.inf)).astype(np.float64)
    lower = np.nextafter(rounded, storage.type(-np.inf)).astype(np.float64)
    exact = rounded.astype(np.float64)
    budget = np.maximum(upper-exact, exact-lower) + 2e-6
    assert np.all(np.abs(pair[0].astype(np.float64)-exact) <= budget)
    np.testing.assert_allclose(pair[1], expected[1], rtol=3e-6, atol=3e-6)


def _dtype(name):
    return np.dtype(ml_dtypes.bfloat16 if name == "bfloat16" else name)


@pytest.mark.parametrize("storage_name", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("sk", [5, 129])
@pytest.mark.parametrize("bias_kind", ["none", "broadcast", "dense"])
@pytest.mark.parametrize("causal", [False, True])
def test_saved_storage_public_resident_and_serialized_byte_boundaries(
    storage_name, sk, bias_kind, causal, monkeypatch,
):
    _run_saved_storage_case(storage_name, storage_name, sk, bias_kind, causal, monkeypatch)


# SM120-SAVED-LSE-MIXED-STORAGE-20261025: K/V share a storage that differs
# from Q; O takes Q's storage (the frontend result type), LSE stays f32.
@pytest.mark.parametrize("storage_name,kv_name", [
    ("float16", "bfloat16"), ("bfloat16", "float16"), ("float32", "float16"),
    ("float32", "bfloat16"), ("float16", "float32"), ("bfloat16", "float32")])
@pytest.mark.parametrize("sk", [5, 129])
@pytest.mark.parametrize("bias_kind", ["none", "broadcast", "dense"])
@pytest.mark.parametrize("causal", [False, True])
def test_mixed_kv_storage_public_resident_and_serialized_byte_boundaries(
    storage_name, kv_name, sk, bias_kind, causal, monkeypatch,
):
    _run_saved_storage_case(storage_name, kv_name, sk, bias_kind, causal, monkeypatch)


def _run_saved_storage_case(storage_name, kv_name, sk, bias_kind, causal, monkeypatch):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    storage, kv_storage = _dtype(storage_name), _dtype(kv_name)
    rng = np.random.default_rng(20261016)
    values = {name: rng.normal(0, .2, shape).astype(storage if name == "q" else kv_storage)
              for name, shape in {"q": (1, 2, 3, 4), "k": (1, 1, sk, 4), "v": (1, 1, sk, 3)}.items()}
    if bias_kind != "none":
        shape = (1, 2, 1, 1) if bias_kind == "broadcast" else (1, 2, 3, sk)
        values["bias"] = rng.normal(0, .1, shape).astype(np.float32)
    function = (saved_bias if causal else saved_bias_open) if "bias" in values else (
        saved if causal else saved_open)
    expected = reference(values, causal)
    fn = ts.jit(target="nvidia_sm120")(function)
    try:
        host = fn(**values)
        assert fn._native_prepared_attention_calls, "saved-LSE call did not create a native owner"
        assert fn._native_descriptor_last_receipt["execution_kind"] == "native_gpu"
        assert ("_kv_" in fn._native_descriptor_last_receipt["entry"]) == (kv_storage != storage)
        compare(host, expected, storage)
        preserved = tuple(value.copy() for value in host)
        owner = next(iter(fn._native_prepared_attention_calls.values()))
        artifact = rt.RuntimeArtifact.from_dict(json.loads(json.dumps(owner.artifact.to_dict())))
        handle = owner.handle
        with ExitStack() as stack:
            sessions = {name: stack.enter_context(NvidiaDeviceSession()) for name in values}
            buffers = {name: sessions[name].upload(np.zeros_like(value)) for name, value in values.items()}
            roots = {name: Borrowed(value) for name, value in buffers.items()}
            for session in sessions.values():
                assert session.synchronize() == 0
            fn(**roots)
            completed = []
            callback = queue_delayed_upload(sessions["k"], buffers["k"], values["k"],
                                            sessions["k"].stream, completed)
            for name, value in values.items():
                if name != "k":
                    assert sessions[name].lib.tessera_nvidia_device_upload(
                        ct.c_void_p(buffers[name].ptr), ct.c_void_p(value.ctypes.data),
                        value.nbytes, ct.c_void_p(sessions[name].stream)) == 0
            assert not completed
            def forbidden(*args, **kwargs):
                pytest.fail("warm saved-LSE forward compiled or evaluated concrete frontend")
            with monkeypatch.context() as guard:
                guard.setattr(subprocess, "run", forbidden)
                guard.setattr(fn, "_fn", forbidden)
                guard.setattr(fn, "_trace_frontend_capture", forbidden)
                resident = fn(**roots)
            assert completed and callback is not None and owner.handle == handle
            compare(resident, expected, storage)
            assert fn._native_descriptor_last_receipt["execution_kind"] == "native_gpu"
            for actual, original in zip(resident, host, strict=True):
                np.testing.assert_array_equal(actual, original)
            for actual, original in zip(host, preserved, strict=True):
                np.testing.assert_array_equal(actual, original)

        ins = sorted((binding for binding in artifact.launch_descriptor.buffers
                      if binding.direction == "input"), key=lambda binding: binding.ordinal)
        outs = sorted((binding for binding in artifact.launch_descriptor.buffers
                       if binding.direction == "output"), key=lambda binding: binding.ordinal)
        arguments = {binding.name: values[fn.arg_names[position]]
                     for binding, position in zip(ins, owner.positions, strict=True)}
        arenas = [np.full(value.size+16, 37, value.dtype) for value in host]
        returned = tuple(arena[8:-8].reshape(value.shape)
                         for arena, value in zip(arenas, host, strict=True))
        arguments.update((binding.name, value) for binding, value in zip(outs, returned, strict=True))
        arguments.update(zip(("B", "Hq", "Hkv", "Sq", "Sk", "D", "Dv"), owner.dims, strict=True))
        if owner.bias_scalars:
            arguments.update(zip(("BiasB", "BiasH", "BiasQ", "BiasK"), owner.bias_shape, strict=True))
        receipt = rt.launch(artifact, arguments)
        assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
        compare(returned, expected, storage)
        for actual, original in zip(returned, host, strict=True):
            np.testing.assert_array_equal(actual, original)
        for arena in arenas:
            np.testing.assert_array_equal(arena[:8], np.full(8, 37, arena.dtype))
            np.testing.assert_array_equal(arena[-8:], np.full(8, 37, arena.dtype))
    finally:
        fn.close_native_storage()

