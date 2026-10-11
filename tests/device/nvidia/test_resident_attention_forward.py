"""Exact-device ordinary resident attention forward and saved LSE."""
import ctypes as ct
from contextlib import ExitStack
import subprocess

import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed, queue_delayed_upload

pytestmark = pytest.mark.hardware_nvidia


def plain(q, k, v):
    return ts.ops.flash_attn(q, k, v, causal=True)


def biased(v, q, k, bias):
    return ts.ops.flash_attn(q, k, v, bias=bias, causal=True)


def saved(v, q, k, bias):
    return ts.ops.flash_attn(q, k, v, bias=bias, causal=True, lse_checkpoint="saved")


def oracle(values, bias=None):
    q, k, v = (value.astype(np.float64) for value in values)
    sk, sq = k.shape[2], q.shape[2]
    groups = q.shape[1] // k.shape[1]
    scores = (q @ np.swapaxes(np.repeat(k, groups, axis=1), -1, -2)) / np.sqrt(q.shape[-1])
    if bias is not None:
        scores += bias.astype(np.float64)
    mask = np.arange(sk)[None, :] <= np.arange(sq)[:, None] + max(sk - sq, 0)
    scores = np.where(mask, scores, -np.inf)
    maximum = scores.max(axis=-1, keepdims=True)
    exponent = np.exp(scores - maximum)
    denominator = exponent.sum(axis=-1, keepdims=True)
    return exponent / denominator @ np.repeat(v, groups, axis=1), maximum[..., 0] + np.log(denominator[..., 0])


@pytest.mark.parametrize("with_lse", [False, True])
@pytest.mark.parametrize("sk", [5, 129])
def test_resident_ordinary_forward_orders_inputs_and_retains_tuple(with_lse, sk, monkeypatch):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070 required")
    fn = ts.jit(target="nvidia_sm120")(saved if with_lse else plain)
    rng = np.random.default_rng(20261009)
    shapes = {"q": (1, 2, 3, 4), "k": (1, 1, sk, 4), "v": (1, 1, sk, 3)}
    if with_lse:
        shapes["bias"] = (1, 2, 1, 1)
    values = {name: rng.normal(0, .2, shape).astype(np.float32) for name, shape in shapes.items()}
    order = ("v", "q", "k", "bias") if with_lse else ("q", "k", "v")
    with ExitStack() as stack:
        sessions = {name: stack.enter_context(NvidiaDeviceSession()) for name in shapes}
        buffers = {name: sessions[name].upload(np.zeros(shape, np.float32)) for name, shape in shapes.items()}
        roots = {name: Borrowed(buffer) for name, buffer in buffers.items()}
        for session in sessions.values():
            assert session.synchronize() == 0
        first = fn(*(roots[name] for name in order))
        owner = next(iter(fn._native_prepared_attention_calls.values()))
        stack.callback(fn.close_native_storage)
        held = tuple(value.copy() for value in first) if with_lse else (first.copy(),)
        handle = owner.handle
        def forbidden(*args, **kwargs):
            pytest.fail("warm forward retraced/compiled/evaluated frontend")
        for generation in (1, 2):
            current = {name: value * generation for name, value in values.items()}
            completed = []
            callback = queue_delayed_upload(sessions["k"], buffers["k"], current["k"],
                                             sessions["k"].stream, completed)
            for name in shapes:
                if name == "k":
                    continue
                value = current[name]
                assert sessions[name].lib.tessera_nvidia_device_upload(
                    ct.c_void_p(buffers[name].ptr), ct.c_void_p(value.ctypes.data),
                    value.nbytes, ct.c_void_p(sessions[name].stream)) == 0
            assert not completed
            with monkeypatch.context() as guard:
                guard.setattr(subprocess, "run", forbidden)
                guard.setattr(fn, "_fn", forbidden)
                guard.setattr(fn, "_trace_frontend_capture", forbidden)
                result = fn(**roots)
            assert completed and callback is not None
            expected = oracle(tuple(current[name] for name in ("q", "k", "v")), current.get("bias"))
            results = result if with_lse else (result,)
            for actual, reference in zip(results, expected[:len(results)], strict=True):
                np.testing.assert_allclose(actual, reference, rtol=3e-5, atol=3e-5)
            prior = first if with_lse else (first,)
            for actual, reference in zip(prior, held, strict=True):
                np.testing.assert_array_equal(actual, reference)
            assert owner.handle == handle and owner.last_device_ms > 0
            assert fn._native_descriptor_last_receipt["execution_kind"] == "native_gpu"

@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("sk", [5, 129])
@pytest.mark.parametrize("with_bias", [False, True])
def test_plain_forward_storage_matches_compiler_and_oracle(dtype, sk, with_bias, monkeypatch):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070 required")
    import ml_dtypes
    storage = ml_dtypes.bfloat16 if dtype == "bfloat16" else np.dtype(dtype)
    rng = np.random.default_rng(721)
    shapes = {"q": (1, 2, 3, 4), "k": (1, 1, sk, 4), "v": (1, 1, sk, 3)}
    values = {name: rng.normal(0, .2, shape).astype(storage) for name, shape in shapes.items()}
    if with_bias:
        values["bias"] = rng.normal(0, .1, (1, 2, 3, sk)).astype(np.float32)
    fn = ts.jit(target="nvidia_sm120")(biased if with_bias else plain)
    with ExitStack() as stack:
        sessions = {name: stack.enter_context(NvidiaDeviceSession()) for name in values}
        buffers = {name: sessions[name].upload(np.zeros_like(value)) for name, value in values.items()}
        roots = {name: Borrowed(buffer) for name, buffer in buffers.items()}
        for session in sessions.values():
            assert session.synchronize() == 0
        held = fn(**roots)
        preserved = held.copy()
        stack.callback(fn.close_native_storage)
        completed = []
        callback = queue_delayed_upload(sessions["k"], buffers["k"], values["k"],
                                        sessions["k"].stream, completed)
        for name, value in values.items():
            if name == "k":
                continue
            assert sessions[name].lib.tessera_nvidia_device_upload(
                ct.c_void_p(buffers[name].ptr), ct.c_void_p(value.ctypes.data),
                value.nbytes, ct.c_void_p(sessions[name].stream)) == 0
        assert not completed
        def forbidden(*args, **kwargs):
            pytest.fail("warm half-result forward retraced/compiled/evaluated")
        with monkeypatch.context() as guard:
            guard.setattr(subprocess, "run", forbidden)
            guard.setattr(fn, "_fn", forbidden)
            guard.setattr(fn, "_trace_frontend_capture", forbidden)
            actual = fn(**roots)
        assert completed and callback is not None
        expected = oracle(tuple(values[name] for name in ("q", "k", "v")), values.get("bias"))[0].astype(storage)
        assert actual.dtype == np.dtype(storage)
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
        np.testing.assert_array_equal(held, preserved)
        host = fn(**values)
        np.testing.assert_array_equal(host, actual)
        assert fn._native_descriptor_last_receipt["execution_kind"] == "native_gpu"


@pytest.mark.parametrize("violation", ["capacity", "alignment", "stream_count", "outputs_overlap"])
def test_forward_native_rejections_preserve_outputs_and_owner(violation):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070 required")
    fn = ts.jit(target="nvidia_sm120")(saved)
    shapes = ((1, 1, 5, 3), (1, 2, 3, 4), (1, 1, 5, 4), (1, 2, 1, 1))
    values = tuple(np.full(shape, .1, np.float32) for shape in shapes)
    with NvidiaDeviceSession() as session:
        roots = tuple(Borrowed(session.upload(value)) for value in values)
        expected = fn(*roots)
        owner = next(iter(fn._native_prepared_attention_calls.values()))
        try:
            ordered = tuple(roots[i] for i in owner.positions)
            pointers = (ct.c_void_p * len(ordered))(*(value.__cuda_array_interface__["data"][0] for value in ordered))
            sizes = (ct.c_size_t * len(ordered))(*(np.prod(shape) * 4 for shape in owner.shapes))
            streams = (ct.c_uint64 * len(ordered))(*(session.stream for _ in ordered))
            outputs = tuple(np.full(shape, 123, np.float32) for shape in owner.output_shapes)
            destinations = (ct.c_void_p * 2)(*(value.ctypes.data for value in outputs))
            lengths = (ct.c_size_t * 2)(*(value.nbytes for value in outputs))
            count = len(ordered)
            if violation == "capacity":
                pointers[0] = session.empty((1,), np.float32).ptr
            if violation == "alignment":
                pointers[0] += 1
            if violation == "stream_count":
                count -= 1
            if violation == "outputs_overlap":
                destinations[1] = destinations[0]
            assert owner.lib.tessera_nvidia_attention_forward_invoke(
                owner.handle, pointers, sizes, len(ordered), streams, count,
                destinations, lengths, 2, None) != 0
            for value in outputs:
                np.testing.assert_array_equal(value, np.full(value.shape, 123, np.float32))
            for actual, reference in zip(fn(*roots), expected, strict=True):
                np.testing.assert_array_equal(actual, reference)
            assert owner.lib.tessera_nvidia_attention_jvp_close(owner.handle) != 0
            assert owner.lib.tessera_nvidia_attention_vjp_close(owner.handle) != 0
        finally:
            fn.close_native_storage()


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_serialized_half_result_raw_launcher_preserves_adjacent_bytes(dtype, with_bias):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070 required")
    import json
    import ml_dtypes
    storage = np.dtype(ml_dtypes.bfloat16 if dtype == "bfloat16" else dtype)
    rng = np.random.default_rng(8721)
    values = {name: rng.normal(0, .2, shape).astype(storage) for name, shape in
              {"q": (1, 2, 3, 4), "k": (1, 1, 129, 4), "v": (1, 1, 129, 3)}.items()}
    if with_bias:
        values["bias"] = rng.normal(0, .1, (1, 2, 3, 129)).astype(np.float32)
    fn = ts.jit(target="nvidia_sm120")(biased if with_bias else plain)
    try:
        expected = fn(**values)
        owner = next(iter(fn._native_prepared_attention_calls.values()))
        artifact = rt.RuntimeArtifact.from_dict(json.loads(json.dumps(owner.artifact.to_dict())))
        arena = np.full(expected.size + 16, 42, storage)
        output = arena[8:-8].reshape(expected.shape)
        outputs = [binding for binding in artifact.launch_descriptor.buffers if binding.direction == "output"]
        inputs = sorted((binding for binding in artifact.launch_descriptor.buffers
                         if binding.direction == "input"), key=lambda binding: binding.ordinal)
        args = {binding.name: values[fn.arg_names[position]]
                for binding, position in zip(inputs, owner.positions, strict=True)}
        args[outputs[0].name] = output
        args.update(zip(("B", "Hq", "Hkv", "Sq", "Sk", "D", "Dv"), owner.dims, strict=True))
        receipt = rt.launch(artifact, args)
        assert receipt["ok"], str(receipt)
        assert receipt["execution_kind"] == "native_gpu"
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(arena[:8], np.full(8, 42, storage))
        np.testing.assert_array_equal(arena[-8:], np.full(8, 42, storage))
    finally:
        fn.close_native_storage()
