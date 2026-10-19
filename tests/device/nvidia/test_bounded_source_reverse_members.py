"""Exact-SM120 projected source-VJP images; composed retained owner is separate."""
import ctypes as ct
import hashlib
import json
import math
import re

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.native_source_attention_reverse import export_native_source_attention_reverse
from tessera.compiler.nvidia_native import _compile_tile_ir
from tests.device.nvidia.test_scalar_adjoint_contraction import driver
from tests.device.nvidia.test_saved_lse_half_backward_kernel import reference
from tests.device.nvidia.test_source_attention_reverse import oracle, source

pytestmark = pytest.mark.hardware_nvidia


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("rows", [False, True])
@pytest.mark.parametrize("variant", ["normal", "permuted", "swapped"])
@pytest.mark.parametrize("capacity", [(7, 11, 4, 3), (17, 19, 35, 7)])
def test_projected_source_reverse_images_reuse_capacities(
        storage, rows, variant, capacity, monkeypatch, record_property):
    assert rt._nvidia_device_name() == "sm_120", "owning SM120 required"
    dtype = np.dtype(np.float16 if storage == "f16" else ml_dtypes.bfloat16)
    sq_cap, sk_cap, d, dv = capacity
    trace = (np.zeros((1, 2, 1, d), dtype), np.zeros((1, 1, 1, d), dtype),
             np.zeros((1, 1, 1, dv), dtype))
    roots = (trace[2], trace[0], trace[1]) if variant == "permuted" else trace
    requested = (2, 0) if variant == "permuted" else (2, 0, 1) if variant == "swapped" else (0, 1, 2)
    graph = source(roots, variant, requested, rows=rows).replace(
        "module attributes {",
        f"module attributes {{tessera.attention_sequence_bounds = array<i64: {sq_cap}, {sk_cap}>, ", 1)
    program = export_native_source_attention_reverse(graph)
    plan, compiled = program.manifest, []
    for stages in program.lower_members():
        entry = re.search(r"\bllvm.func @(\w+)\(", stages[2])[1]
        _, ptx, *_ = _compile_tile_ir(stages[2], entry)
        compiled.append((entry, ptx))
    record_property("scope", "projected native member sequence; not public/portable/composed-owner proof")
    record_property("member_image_sha256", json.dumps([
        hashlib.sha256(ptx.encode()).hexdigest() for _, ptx in compiled]))
    record_property("program_graph_sha256", plan["program_graph_sha256"])
    lib, modules, functions = driver(), [], []
    with NvidiaDeviceSession() as session:
        try:
            for entry, ptx in compiled:
                module, function = ct.c_void_p(), ct.c_void_p()
                assert lib.cuModuleLoadData(ct.byref(module), ct.create_string_buffer(ptx.encode())) == 0
                modules.append(module)
                assert lib.cuModuleGetFunction(ct.byref(function), module, entry.encode()) == 0
                functions.append(function)
            allocations, host = [], []
            for buffer in plan["buffers"]:
                dt = np.float32 if buffer["storage"] == "f32" else dtype
                values = np.full(math.prod(buffer["capacity_shape"]) + 16, 23, dtype=dt)
                host.append(values)
                allocations.append(session.upload(values))
            addresses = tuple(x.ptr for x in allocations)
            roles = plan["role_indices"]
            key_ids = {roles[1], roles[2], 10 if rows else 9, 11 if rows else 10}

            def forbidden(*args, **kwargs):
                pytest.fail("projected image replay invoked a compiler or external tool")

            with monkeypatch.context() as guard:
                import subprocess
                guard.setattr(subprocess, "run", forbidden)
                guard.setattr(subprocess, "Popen", forbidden)
                for step, (sq, sk) in enumerate(((1, 1), (sq_cap, sk_cap), (3, 2),
                                                 (1, sk_cap), (sq_cap, 1))):
                    rng = np.random.default_rng(19102026 + step)
                    q = rng.normal(0, .4, (1, 2, sq, d)).astype(dtype)
                    if step == 2:
                        q.fill(0)
                    k = rng.normal(0, .3, (1, 1, sk, d)).astype(dtype)
                    v = rng.normal(0, .3, (1, 1, sk, dv)).astype(dtype)
                    do = rng.normal(0, .2, (1, 2, sq, dv)).astype(dtype)
                    dlse = rng.normal(0, .2, (1, 2, sq)).astype(np.float32)
                    roots = (v, q, k) if variant == "permuted" else (q, k, v)
                    seeds = (dlse, do) if variant == "swapped" else (do, dlse)
                    inputs = (*roots, *seeds)
                    frozen = tuple(x.tobytes() for x in inputs)
                    shapes = []
                    for buffer in plan["buffers"]:
                        shape = list(buffer["capacity_shape"])
                        shape[2] = sk if buffer["id"] in key_ids else sq
                        shapes.append(tuple(shape))
                    for values in host:
                        values.fill(23)
                    for i, value in enumerate(inputs):
                        host[i][:value.size] = value.reshape(-1)
                    # Refill existing capacity allocations; no device allocation
                    # or recompilation occurs during the shrink/grow sequence.
                    for arena, values in zip(allocations, host, strict=True):
                        assert session.lib.tessera_nvidia_device_upload(
                            ct.c_void_p(arena.ptr), ct.c_void_p(values.ctypes.data),
                            values.nbytes, ct.c_void_p(session.stream)) == 0
                    logical_dims = (1, 2, 1, sq, sk, d, dv)
                    for index, (call, function) in enumerate(zip(plan["steps"], functions, strict=True)):
                        ids = call["inputs"] + call["outputs"]
                        args = [ct.c_uint64(allocations[i].ptr) for i in ids]
                        scalar = (2 * sq, d) if index in (0, len(functions) - 1) else logical_dims
                        args += [ct.c_int64(x) for x in scalar]
                        params = (ct.c_void_p * len(args))(*(ct.addressof(x) for x in args))
                        threads = 32 if rows and index == 2 else 128
                        if index in (0, len(functions) - 1):
                            elements = 2 * sq
                        elif index == 1:
                            elements = math.prod(shapes[6])
                        else:
                            elements = sum(math.prod(shapes[i]) for i in call["outputs"])
                        assert lib.cuLaunchKernel(
                            function, (elements + threads - 1) // threads, 1, 1,
                            threads, 1, 1, 0, ct.c_void_p(session.stream), params, None) == 0
                    assert session.synchronize() == 0
                    actual = []
                    for buffer, allocation in zip(plan["buffers"], allocations, strict=True):
                        values = session.download(allocation)
                        active = math.prod(shapes[buffer["id"]])
                        np.testing.assert_array_equal(values[active:], values.dtype.type(23))
                        actual.append(values[:active].reshape(shapes[buffer["id"]]))
                    for i in range(5):
                        assert actual[i].tobytes() == inputs[i].tobytes() == frozen[i]
                    expected = oracle(q, k, v, do, dlse)
                    expected = (expected[2], expected[0], expected[1]) if variant == "permuted" else expected
                    for id_, requested_index in zip(plan["outputs"], requested, strict=True):
                        np.testing.assert_allclose(
                            actual[id_].astype(np.float32), expected[requested_index].astype(np.float32),
                            rtol=.03 if storage == "bf16" else .006, atol=.002)
                    qf = q.astype(np.float32)
                    normalized = (qf / np.sqrt(np.mean(qf * qf, axis=-1, keepdims=True)
                                               + np.float32(1e-5))).astype(dtype)
                    (out_ref, lse_ref), _ = reference(normalized, k, v, do, None, dlse, causal=False)
                    np.testing.assert_allclose(actual[6].astype(np.float32), out_ref, rtol=.03, atol=.002)
                    np.testing.assert_allclose(actual[7], lse_ref, rtol=3e-5, atol=3e-5)
                    assert tuple(x.ptr for x in allocations) == addresses
        finally:
            assert session.synchronize() == 0
            for module in reversed(modules):
                assert lib.cuModuleUnload(module) == 0
