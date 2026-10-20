"""Exact SM120 binary producer programs through ordinary JIT and checked owners."""
import subprocess

import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera import runtime
from tests.unit.test_native_tensor_binary_producers import CASES

pytestmark = pytest.mark.hardware_nvidia


def roots(case, storage, shape, seed):
    m, k, n = shape
    dtype = {"f16": np.float16, "bf16": ml_dtypes.bfloat16, "f32": np.float32}[storage]
    rng = np.random.default_rng(seed)
    def value(dimensions):
        return rng.normal(0, .2, dimensions).astype(np.float32).astype(dtype)
    if case == "rhs":
        return value((m, k)), value((k, n)), value((k, n))
    if case == "both":
        return value((m, k)), value((m, k)), value((k, n)), value((k, n))
    return value((m, k)), value((m, k)), value((k, n))


def operands(case, values):
    # Independent value-domain oracle: round each producer result to its
    # declared storage before it is used by another producer or the matmul.
    def add(a, b):
        return (a.astype(np.float64) + b.astype(np.float64)).astype(a.dtype)
    if case == "rhs":
        a, b, residual = values
        return a, add(b, residual)
    if case == "both":
        a, residual_a, b, residual_b = values
        return add(a, residual_a), add(b, residual_b)
    a, residual, b = values
    x = add(a, residual)
    if case == "chain":
        x = np.maximum(np.float64(0), x.astype(np.float64)).astype(a.dtype)
    elif case == "duplicate":
        x = add(x, x)
    return x, b


def check(output, case, values):
    if not 0 < values[0].shape[1] < 2**24:
        raise ValueError("the FP32 gamma-K oracle requires 0 < K < 2**24")
    a, b = operands(case, values)
    left, right = a.astype(np.float64), b.astype(np.float64)
    expected = left @ right
    u = 2.**-24
    gamma = a.shape[1] * u / (1 - a.shape[1] * u)
    bound = gamma * (np.abs(left) @ np.abs(right))
    assert output.dtype == np.float32 and output.shape == expected.shape
    assert np.all(np.abs(output.astype(np.float64) - expected) <= bound), (
        np.max(np.abs(output.astype(np.float64) - expected)), np.max(bound))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("storage", ["f16", "bf16", "f32"])
@pytest.mark.parametrize("shape", [(17, 35, 19), (64, 128, 64)])
def test_public_binary_native_program_reuses_checked_owner(case, storage, shape, monkeypatch):
    assert runtime._nvidia_device_name() == "sm_120"
    fn = ts.jit(target="nvidia_sm120")(CASES[case][0])
    values = roots(case, storage, shape, 13021)
    snapshots = tuple(value.copy() for value in values)
    try:
        first = fn(*values)
        check(first, case, values)
        assert fn.execution_kind == "native_gpu"
        program, owner = fn._native_tensor_last_program, fn._native_tensor_last_owner
        assert program is not None and len(program.members) == len(CASES[case][2])
        assert [stage["inputs"] for stage in program.frame["stages"]] == CASES[case][2]
        assert [buffer["storage"] for buffer in program.frame["buffers"][len(values):]] == (
            [storage] * (len(program.members) - 1) + ["f32"])
        assert "schedule.binary_kernel" in program.schedule_ir
        assert "tile.elementwise_kernel" in program.tile_ir
        for value, snapshot in zip(values, snapshots, strict=True):
            np.testing.assert_array_equal(value, snapshot)
        before = owner.stats()
        saved = first.copy()
        changed = roots(case, storage, shape, 13022)
        changed_snapshots = tuple(value.copy() for value in changed)
        def forbidden(*args, **kwargs):
            pytest.fail("warm binary program used eager execution, tracing or compiler tools")
        with monkeypatch.context() as guard:
            guard.setattr(fn, "_fn", forbidden)
            guard.setattr(fn, "_trace_frontend_capture", forbidden)
            guard.setattr(subprocess, "run", forbidden)
            guard.setattr(subprocess, "Popen", forbidden)
            second = fn(*changed)
            check(second, case, changed)
            assert fn._native_tensor_last_program is program
            assert fn._native_tensor_last_owner is owner
            assert owner.stats()["allocations"] == before["allocations"]
            np.testing.assert_array_equal(first, saved)
            assert not np.shares_memory(first, second)
            for value, snapshot in zip(changed, changed_snapshots, strict=True):
                np.testing.assert_array_equal(value, snapshot)
        restored = runtime.RuntimeArtifact.from_json(fn.runtime_artifact().to_json())
        receipt = runtime.launch(
            restored, dict(zip(restored.metadata["arg_names"], changed, strict=True)))
        assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
        check(receipt["output"], case, changed)
    finally:
        fn.close_native_storage()


@pytest.mark.parametrize("case", ["lhs", "rhs", "both"])
@pytest.mark.parametrize("storage", ["f16", "bf16", "f32"])
def test_binary_resident_roots_wait_for_every_pending_producer(case, storage, monkeypatch):
    from contextlib import ExitStack
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed, queue_delayed_upload

    assert runtime._nvidia_device_name() == "sm_120"
    values = roots(case, storage, (17, 35, 19), 13023)
    fn = ts.jit(target="nvidia_sm120")(CASES[case][0])
    with ExitStack() as stack:
        sessions = [stack.enter_context(NvidiaDeviceSession()) for _ in values]
        buffers = [session.upload(np.zeros_like(value))
                   for session, value in zip(sessions, values, strict=True)]
        assert all(session.synchronize() == 0 for session in sessions)
        borrowed = [Borrowed(buffer, session.stream)
                    for session, buffer in zip(sessions, buffers, strict=True)]
        try:
            fn(*borrowed)
            owner = fn._native_tensor_last_owner
            before = owner.stats()
            completed = [[] for _ in values]
            callbacks = [
                queue_delayed_upload(session, buffer, value, session.stream, done)
                for session, buffer, value, done in zip(
                    sessions, buffers, values, completed, strict=True)]
            assert not any(completed)
            def forbidden(*args, **kwargs):
                pytest.fail("resident binary roots used eager execution, tracing, tools or host coercion")
            with monkeypatch.context() as guard:
                guard.setattr(fn, "_fn", forbidden)
                guard.setattr(fn, "_trace_frontend_capture", forbidden)
                guard.setattr(subprocess, "run", forbidden)
                guard.setattr(subprocess, "Popen", forbidden)
                actual = fn(*borrowed)
            assert all(completed) and len(callbacks) == len(values)
            assert fn._native_tensor_last_owner is owner
            assert owner.stats()["allocations"] == before["allocations"] == 1
            assert owner.stats()["capacity"] == before["capacity"]
            check(actual, case, values)
            for session, buffer, expected in zip(sessions, buffers, values, strict=True):
                np.testing.assert_array_equal(session.download(buffer), expected)
            saved = actual.copy()
        finally:
            fn.close_native_storage()
    np.testing.assert_array_equal(actual, saved)


@pytest.mark.parametrize("storage", ["f16", "bf16", "f32"])
def test_compiler_projected_add_member_rounding_and_canaries(storage):
    """Diagnostic prefix exposes stored producer bits before matmul erases them.

    The member image and launch geometry are unchanged compiler products. The
    shortened frame is diagnostic transport, not a public standalone add route.
    """
    import copy
    from tessera.compiler.prepared_native_tensor_frame import PreparedNativeTensorFrame

    assert runtime._nvidia_device_name() == "sm_120"
    values = roots("lhs", storage, (17, 35, 19), 13071)
    fn = ts.jit(target="nvidia_sm120")(CASES["lhs"][0])
    try:
        check(fn(*values), "lhs", values)
        program = fn._native_tensor_last_program
        frame = program.frame
        stage = copy.deepcopy(frame["stages"][0])
        assert stage["inputs"] == [0, 1]
        specs = copy.deepcopy([frame["buffers"][0], frame["buffers"][1],
                               frame["buffers"][stage["output"]]])
        for index, spec in enumerate(specs):
            spec["id"] = index
            spec["role"] = "output" if index == 2 else "input"
        stage["output"] = 2
        frame.update(buffers=specs, roots=[0, 1], output=2, stages=[stage])
        dtype = values[0].dtype
        bits = np.uint32 if storage == "f32" else np.uint16
        # Halfway cases alternate even/odd retained mantissas. Include signed
        # zeros, cancellation, smallest subnormals, overflow and nonfinites.
        precision = {"f16": 11, "bf16": 8, "f32": 24}[storage]
        one = np.array(1, dtype=dtype)
        odd = np.array([one], dtype=dtype).view(bits).copy()
        odd[0] += 1
        odd = odd.view(dtype)[0]
        tiny = np.array([1], dtype=bits).view(dtype)[0]
        maximum = np.array([{ "f16": 0x7bff, "bf16": 0x7f7f,
                              "f32": 0x7f7fffff}[storage]], dtype=bits).view(dtype)[0]
        lhs = [0., -0., -0., 1., odd, -1., tiny, -tiny, maximum,
               np.inf, -np.inf, np.inf, np.nan]
        rhs = [0., -0., 0., 2.**-precision, 2.**-precision, 1., tiny, -tiny,
               maximum, 1., -1., -np.inf, 1.]
        a = np.resize(np.asarray(lhs, dtype=dtype), values[0].shape)
        b = np.resize(np.asarray(rhs, dtype=dtype), values[1].shape)
        saved = a.copy(), b.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            expected = (a.astype(np.float64) + b.astype(np.float64)).astype(dtype)
        canary = np.full(a.size + 32, -123, dtype=dtype)
        output = canary[16:-16].reshape(a.shape)
        with PreparedNativeTensorFrame(frame, (program.images[0],)) as owner:
            returned = owner((a, b), output=output)
            assert returned is output
            assert owner.stats()["allocations"] == 1
        finite = np.isfinite(expected.astype(np.float64))
        np.testing.assert_array_equal(output.view(bits)[finite], expected.view(bits)[finite])
        np.testing.assert_array_equal(np.isnan(output.astype(np.float64)),
                                      np.isnan(expected.astype(np.float64)))
        infinite = np.isinf(expected.astype(np.float64))
        np.testing.assert_array_equal(output.view(bits)[infinite], expected.view(bits)[infinite])
        assert np.all(canary[:16] == -123) and np.all(canary[-16:] == -123)
        for value, snapshot in zip((a, b), saved, strict=True):
            np.testing.assert_array_equal(value.view(bits), snapshot.view(bits))
    finally:
        fn.close_native_storage()


@pytest.mark.parametrize("storage", ["f16", "bf16", "f32"])
def test_wide_compiler_add_scalar_metadata_without_arena(storage):
    """Compiler/PTX/native metadata proof only; no huge allocation or execution."""
    import base64
    import ctypes as ct
    import os
    import copy
    import json
    import re
    from tessera.compiler.nvidia_native import _compile_tile_ir
    from tessera.compiler.prepared_native_tensor_frame import NativeFrameStep, _descriptor
    from tessera.compiler.prepared_nvidia_matmul import HostView
    from tests.unit.test_native_tensor_binary_producers import binary_source
    from tests.unit.test_native_tensor_program_pipeline import _run

    assert runtime._nvidia_device_name() == "sm_120"
    source = binary_source("lhs", storage)
    source = source.replace(f"tensor<17x32x{storage}>", f"tensor<32768x65536x{storage}>")
    source = source.replace(f"tensor<32x11x{storage}>", f"tensor<65536x1x{storage}>")
    source = source.replace("tensor<17x11xf32>", "tensor<32768x1xf32>")
    tiled = _run(source, "--tessera-graph-to-schedule", "--tessera-schedule-to-tile")
    assert tiled.returncode == 0, tiled.stderr
    frame = json.loads(base64.b64decode(re.search(
        r'tessera.native.tensor_program_frame_json = "([^"]+)"', tiled.stdout)[1]))
    leaf = _run(tiled.stdout, "--tessera-schedule-to-tile=tensor-member=0")
    assert leaf.returncode == 0, leaf.stderr
    stage = copy.deepcopy(frame["stages"][0])
    assert stage["scalars"] == [2**31]
    _, ptx, *_ = _compile_tile_ir(leaf.stdout, stage["entry"])
    specs = copy.deepcopy([frame["buffers"][0], frame["buffers"][1],
                           frame["buffers"][stage["output"]]])
    for index, spec in enumerate(specs):
        spec.update(id=index, role="output" if index == 2 else "input")
    stage["output"] = 2
    frame.update(buffers=specs, roots=[0, 1], output=2, stages=[stage])
    signature = re.search(re.escape(stage["entry"])+r"\((.*?)\)", ptx, re.S)
    assert signature and len(re.findall(r"\.param\s+\.u64\b", signature[1])) == 4
    library = os.environ.get("TESSERA_NATIVE_FRAME_OBSERVER_LIB")
    if not library:
        pytest.fail("wide native metadata proof requires the matching test-only observer")
    observer = ct.CDLL(library)
    probe = observer.tessera_nvidia_native_frame_test_metadata_v1
    probe.argtypes = [ct.POINTER(HostView), ct.POINTER(ct.c_uint8), ct.c_size_t,
                      ct.POINTER(NativeFrameStep), ct.c_size_t,
                      ct.POINTER(ct.c_int64), ct.POINTER(ct.c_uint64)]
    probe.restype = ct.c_int
    descriptors = (HostView * 3)(*(_descriptor(spec) for spec in specs))
    roles = (ct.c_uint8 * 3)(0, 0, 2)
    blob = ct.create_string_buffer(ptx.encode())
    step = NativeFrameStep()
    step.image, step.image_bytes = ct.addressof(blob), len(ptx.encode())
    step.entry, step.input_count, step.scalar_count = stage["entry"].encode(), 2, 1
    step.buffers[:3] = [0, 1, 2]
    step.scalars[0] = 2**31
    step.grid[:], step.block[:] = stage["geometry"][:3], stage["geometry"][3:]
    scalar, arena = ct.c_int64(), ct.c_uint64()
    assert probe(descriptors, roles, 3, ct.byref(step), 1,
                 ct.byref(scalar), ct.byref(arena)) == 0
    assert scalar.value == 2**31
    assert arena.value == sum(spec["bytes"] for spec in specs)
    for invalid in (0, -1, -(2**63)):
        step.scalars[0] = invalid
        assert probe(descriptors, roles, 3, ct.byref(step), 1,
                     ct.byref(scalar), ct.byref(arena)) != 0
        assert scalar.value == arena.value == 0
