"""Public typed reverse, retained native ownership and portable replay on SM120."""
from __future__ import annotations

from contextlib import ExitStack
import json
import subprocess

import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.device.nvidia.test_saved_lse_half_backward_kernel import reference, compare
from tests.device.nvidia.test_ordered_resident_tensor_dag import queue_delayed_upload

pytestmark = pytest.mark.hardware_nvidia


def function(saved=False, bias=False, causal=True):
    if bias:
        if saved:
            def attention(bias, v, q, k):
                return ts.ops.flash_attn(q, k, v, attn_bias=bias, causal=causal, lse_checkpoint="saved")
        else:
            def attention(bias, v, q, k):
                return ts.ops.flash_attn(q, k, v, attn_bias=bias, causal=causal)
    elif saved:
        def attention(q, k, v):
            return ts.ops.flash_attn(q, k, v, causal=causal, lse_checkpoint="saved")
    else:
        def attention(q, k, v):
            return ts.ops.flash_attn(q, k, v, causal=causal)
    return attention


@pytest.mark.parametrize("storage", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dims", [(1, 2, 1, 3, 5, 4, 3), (2, 2, 2, 1, 129, 8, 6), (1, 2, 1, 7, 3, 4, 5)])
@pytest.mark.parametrize("subset", [False, True])
def test_public_typed_reverse_replays_without_compilers_and_retains_results(storage, saved, bias, causal, dims, subset, monkeypatch):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    dtype = np.dtype(ml_dtypes.bfloat16 if storage == "bfloat16" else storage)
    rng = np.random.default_rng(20261016)
    b, hq, hkv, sq, sk, d, dv = dims
    values = {name: rng.normal(0, .3, shape).astype(dtype) for name, shape in {
        "q": (b, hq, sq, d), "k": (b, hkv, sk, d), "v": (b, hkv, sk, dv)}.items()}
    if bias:
        values["bias"] = rng.normal(0, .1, (1, hq, 1, sk)).astype(np.float32)
    do = rng.normal(0, .2, (b, hq, sq, dv)).astype(dtype)
    seed = rng.normal(0, .2, (b, hq, sq)).astype(np.float32) if saved else np.zeros((b, hq, sq), np.float32)
    _, expected = reference(values["q"], values["k"], values["v"], do, values.get("bias"), seed, causal=causal)
    wrt = ("bias", "v", "k", "q") if bias else ("v", "k", "q")
    role = {"q": 0, "k": 1, "v": 2, "bias": 3}
    if subset:
        wrt = ("bias", "q") if bias else ("v", "q")
    fn = ts.jit(target="nvidia_sm120", autodiff="reverse", wrt=wrt)(function(saved, bias, causal))
    try:
        seeds = (do, seed) if saved else do
        first = fn.native_backward(**values, out_cotangents=seeds)
        for name, result in zip(wrt, first, strict=True):
            assert result.dtype == (np.dtype("float32") if name == "bias" else dtype)
            compare(result, expected[role[name]])
        receipt = fn.last_backward_execution
        assert receipt["execution_kind"] == "native_gpu"
        assert receipt["compiler_path"] == "nvidia_sm120_attention_vjp_compiled"
        artifact = rt.RuntimeArtifact.from_json(fn.native_backward_runtime_artifact().to_json())
        assert json.loads(artifact.metadata["program_json"])["program"]["schema"] == (
            "tessera.native_attention_vjp_program.v1" if storage == "float32" else "tessera.native_attention_vjp_program.v2")
        held = tuple(x.copy() for x in first)
        def forbidden(*args, **kwargs):
            pytest.fail("warm typed backward invoked compiler or concrete frontend")
        ordered = tuple(values[name] for name in fn.arg_names)
        with monkeypatch.context() as guard:
            guard.setattr(subprocess, "run", forbidden)
            guard.setattr(subprocess, "check_output", forbidden)
            guard.setattr(subprocess, "Popen", forbidden)
            guard.setattr(fn, "_fn", forbidden)
            warm = fn.native_backward(*ordered, out_cotangents=seeds)
            replay = rt.launch(artifact, (*ordered, do, *((seed,) if saved else ())))
        assert replay["ok"] and replay["execution_kind"] == "native_gpu", replay
        for collection in (warm, replay["output"]):
            for name, result in zip(wrt, collection, strict=True):
                compare(result, expected[role[name]])
        with ExitStack() as stack:
            host_inputs = (*ordered, do, *((seed,) if saved else ()))
            sessions = tuple(stack.enter_context(NvidiaDeviceSession()) for _ in host_inputs)
            resident = tuple(session.upload(x) for session, x in zip(sessions, host_inputs, strict=True))
            for session in sessions:
                assert session.synchronize() == 0
            resident_replay = rt.launch(artifact, resident)
            assert resident_replay["ok"] and resident_replay["execution_kind"] == "native_gpu", resident_replay
            for name, result in zip(wrt, resident_replay["output"], strict=True):
                compare(result, expected[role[name]])
            k_index = fn.arg_names.index("k")
            changed_key = (values["k"].astype(np.float32)*.7).astype(dtype)
            completed = []
            callback = queue_delayed_upload(sessions[k_index], resident[k_index], changed_key,
                                            sessions[k_index].stream, completed)
            assert not completed
            changed_seed = (resident[len(ordered)], resident[-1]) if saved else resident[-1]
            resident_public = fn.native_backward(*resident[:len(ordered)], out_cotangents=changed_seed)
            assert completed and callback is not None
            assert fn.last_frontend_differential.contract["concrete_executions"] == 0
            _, changed_expected = reference(values["q"], changed_key, values["v"], do, values.get("bias"), seed, causal=causal)
            for name, result in zip(wrt, resident_public, strict=True):
                compare(result, changed_expected[role[name]])
            with monkeypatch.context() as guard:
                guard.setattr(subprocess, "run", forbidden)
                guard.setattr(subprocess, "check_output", forbidden)
                guard.setattr(subprocess, "Popen", forbidden)
                guard.setattr(fn, "_fn", forbidden)
                repeated = fn.native_backward(*resident[:len(ordered)], out_cotangents=changed_seed)
            for name, result in zip(wrt, repeated, strict=True):
                compare(result, changed_expected[role[name]])
        for original, snapshot in zip(first, held, strict=True):
            np.testing.assert_array_equal(original, snapshot)
    finally:
        fn.close_native_storage()
        from tessera.compiler.prepared_attention_vjp import clear_prepared
        clear_prepared()


@pytest.mark.parametrize("storage", ["float16", "bfloat16"])
@pytest.mark.parametrize("violation", ["capacity", "alignment", "stream", "storage_flag", "output_flag", "seed_flag"])
def test_typed_owner_rejects_bad_native_binding_without_retiring_valid_owner(storage, violation):
    import ctypes as ct
    from tessera.compiler.prepared_attention_vjp import prepared, clear_prepared
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    dtype = np.dtype(ml_dtypes.bfloat16 if storage == "bfloat16" else storage)
    rng = np.random.default_rng(507016)
    values = tuple(rng.normal(0, .2, shape).astype(dtype) for shape in (
        (1, 2, 3, 8), (1, 1, 129, 8), (1, 1, 129, 6), (1, 2, 3, 6)))
    fn = ts.jit(target="nvidia_sm120", autodiff="reverse", wrt=("q", "v"))(function())
    try:
        first = fn.native_backward(*values[:3], out_cotangents=values[3])
        artifact = fn.native_backward_runtime_artifact()
        owner = prepared(artifact.metadata)
        handle = owner.handle
        if violation.endswith("_flag"):
            pair = owner.program.pair
            lib = owner._library()
            fimage = ct.create_string_buffer(pair.forward.image.payload)
            bimage = ct.create_string_buffer(pair.backward.image.payload)
            dims = (ct.c_int64*7)(*owner.dims)
            mapping = (ct.c_int*3)(*owner.program.input_indices)
            roles = (ct.c_int*2)(*owner.program.active)
            code = 2 if storage == "float16" else 3
            input_code = (3 if code == 2 else 2) if violation == "storage_flag" else code
            output_code = 1 if violation == "output_flag" else code
            seed_code = int(violation == "seed_flag")
            invalid = ct.c_uint64()
            status = lib.tessera_nvidia_attention_vjp_prepare_typed(
                fimage, len(pair.forward.image.payload), pair.forward.descriptor.entry_symbol.encode(),
                bimage, len(pair.backward.image.payload), pair.backward.descriptor.entry_symbol.encode(),
                dims, None, mapping, roles, 2, input_code, output_code, seed_code, ct.byref(invalid))
            try:
                assert status != 0 and invalid.value == 0
            finally:
                if invalid.value:
                    lib.tessera_nvidia_attention_vjp_close(invalid.value)
        else:
            with NvidiaDeviceSession() as session:
                roots = [session.upload(value) for value in values]
                assert session.synchronize() == 0
                data = dict(roots[1].__cuda_array_interface__)
                if violation == "capacity":
                    small = session.empty((1,), dtype)
                    data["data"] = (small.ptr, False)
                elif violation == "alignment":
                    data["data"] = (roots[1].ptr+1, False)
                else:
                    data["stream"] = 0
                class Invalid:
                    shape = tuple(data["shape"])
                    dtype = roots[1].dtype
                    @property
                    def __cuda_array_interface__(self):
                        return data
                broken = [*roots]
                broken[1] = Invalid()
                receipt = rt.launch(artifact, tuple(broken))
                assert not receipt["ok"], receipt
                message = str(receipt.get("reason", ""))
                assert ("allocation" in message or "capacity" in message) if violation == "capacity" else violation in message
                assert owner.last_device_ms is None
                restored = rt.launch(artifact, tuple(roots))
                assert restored["ok"], restored
                for actual, expected in zip(restored["output"], first, strict=True):
                    np.testing.assert_array_equal(actual, expected)
        assert owner.handle == handle
        again = fn.native_backward(*values[:3], out_cotangents=values[3])
        for actual, expected in zip(again, first, strict=True):
            np.testing.assert_array_equal(actual, expected)
    finally:
        fn.close_native_storage()
        clear_prepared()


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("threads", [64, 128])
def test_textual_half_graph_with_fp32_primary_reaches_checked_public_owner(storage, saved, bias, threads, monkeypatch):
    import os
    from tessera.compiler.native_attention_program import compile_attention_vjp_program
    from tessera.compiler.prepared_attention_vjp import clear_prepared
    from tests.unit.test_saved_lse_half_backward import source
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    dtype = np.dtype(np.float16 if storage == "f16" else ml_dtypes.bfloat16)
    rng = np.random.default_rng(507132)
    q, k, v = tuple(rng.normal(0, .2, shape).astype(dtype) for shape in (
        (1, 2, 3, 4), (1, 1, 5, 4), (1, 1, 5, 3)))
    do = rng.normal(0, .2, (1, 2, 3, 3)).astype(np.float32)
    seed = rng.normal(0, .2, (1, 2, 3)).astype(np.float32) if saved else np.zeros((1, 2, 3), np.float32)
    bias_value = rng.normal(0, .1, (1, 2, 1, 5)).astype(np.float32) if bias else None
    rt.backend_capabilities("nvidia_sm120")
    original = source(storage, saved=saved, bias=bias, output="f32")
    count = 3+int(bias)
    program = compile_attention_vjp_program(original, tuple(range(count)),
        compiler=os.environ["TESSERA_OPT"], compact_gradients=True, compact_threads=threads)
    artifact = rt.RuntimeArtifact(graph_ir=original, target_ir=program.pair.backward.target_ir,
        metadata=dict(target="nvidia_sm120", compiler_path="nvidia_sm120_attention_vjp_compiled",
            execution_kind="native_gpu", execution_mode="cuda_runtime", executable=True,
            arg_names=[f"primal_{i}" for i in range(count)]+["cotangent"]+(["lse_cotangent"] if saved else []),
            program_json=program.to_json(), program_digest=program.program_digest))
    _, expected = reference(q, k, v, do, bias_value, seed)
    values = (q, k, v)+((bias_value,) if bias else ())+ (do,)+((seed,) if saved else ())
    def forbidden(*args, **kwargs):
        pytest.fail("textual typed public owner replay invoked a compiler")
    try:
        with monkeypatch.context() as guard:
            guard.setattr(subprocess, "run", forbidden)
            guard.setattr(subprocess, "check_output", forbidden)
            guard.setattr(subprocess, "Popen", forbidden)
            receipt = rt.launch(artifact, values)
        assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
        for index, (actual, oracle) in enumerate(zip(receipt["output"], expected, strict=True)):
            assert actual.dtype == (np.dtype("float32") if index == 3 else dtype)
            compare(actual, oracle)
    finally:
        clear_prepared()
