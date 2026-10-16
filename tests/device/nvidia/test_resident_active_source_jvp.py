"""Exact-SM120 resident roots for compiler-owned source-active JVP."""
from contextlib import ExitStack
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from benchmarks.nvidia.active_source_jvp_reference import inputs, oracle, product, product_reversed
from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed

pytestmark = pytest.mark.hardware_nvidia


def check(outputs, expected, budgets):
    assert len(outputs) == 2 and not np.shares_memory(*outputs)
    for actual, reference, budget in zip(outputs, expected, budgets, strict=True):
        assert actual.dtype == np.float32 and np.isfinite(actual).all()
        assert np.all(np.abs(actual.astype(np.float64) - reference) <= budget)


@pytest.fixture(scope="module", params=[
    (storage, permuted, shape)
    for storage in ("f16", "bf16") for permuted in (False, True)
    for shape in ((1, 1, 1), (17, 19, 35), (129, 33, 257))
])
def compiled(request):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    storage, permuted, shape = request.param
    a, b, da = inputs({"shape_mnk": shape, "storage": storage})
    roots = (b, a, da) if permuted else (a, b, da)
    expected, budgets = oracle(a, b, da)
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=("a",))(
        product_reversed if permuted else product)
    check(fn.native_jvp(*roots[:2], tangents=(da,)), expected, budgets)
    try:
        yield fn, roots, expected, budgets
    finally:
        fn.close_native_storage()


def test_public_resident_source_jvp_matches_independent_oracle(compiled):
    fn, roots, expected, budgets = compiled
    with ExitStack() as stack:
        sessions = [stack.enter_context(NvidiaDeviceSession()) for _ in roots]
        buffers = [session.upload(value) for session, value in zip(sessions, roots, strict=True)]
        device_roots = tuple(Borrowed(buffer) for buffer in buffers)
        pair = fn.native_jvp(*device_roots[:2], tangents=(device_roots[2],))
        check(pair, expected, budgets)
        assert fn.last_jvp_execution["native_call_binding"] == "prepared_cpp_active_source_program"


@pytest.mark.parametrize("mode", ["different", "shared", "legacy", "per_thread"])
@pytest.mark.parametrize("route", ["public", "portable", "captured", "events"])
def test_pending_resident_writes_order_before_public_replay_and_profile(
    compiled, mode, route, monkeypatch,
):
    import subprocess
    from tests.device.nvidia.test_ordered_resident_tensor_dag import queue_delayed_upload
    fn, roots, _, _ = compiled
    owner = next(iter(fn._native_prepared_jvp_calls.values()))
    role = owner.plan["role_indices"][0]
    changed = tuple((value.astype(np.float32) * .75 + .0625).astype(value.dtype)
                    for value in roots)
    expected, budgets = oracle(changed[role], changed[1 - role], changed[2])
    artifact = rt.RuntimeArtifact.from_json(fn.native_jvp_runtime_artifact().to_json())
    with ExitStack() as stack:
        first = stack.enter_context(NvidiaDeviceSession())
        sessions = ([first] + [stack.enter_context(NvidiaDeviceSession()) for _ in range(2)]
                    if mode == "different" else [first] * 3)
        buffers = [session.upload(np.zeros_like(value))
                   for session, value in zip(sessions, roots, strict=True)]
        for session in set(sessions):
            assert session.synchronize() == 0
        streams = [1 if mode == "legacy" else 2 if mode == "per_thread" else session.stream
                   for session in sessions]
        borrowed = tuple(Borrowed(buffer, stream) for buffer, stream in
                         zip(buffers, streams, strict=True))
        # Prime private storage before pending writes, so allocation fences
        # cannot substitute for the explicit producer ordering contract.
        owner(borrowed)
        completed = []
        callbacks = [queue_delayed_upload(session, buffer, value, stream, completed)
                     for session, buffer, value, stream in
                     zip(sessions, buffers, changed, streams, strict=True)]
        assert not completed
        with monkeypatch.context() as warm:
            warm.setattr(subprocess, "run", lambda *a, **k: pytest.fail("warm resident call compiled"))
            if route == "public":
                pair = fn.native_jvp(*borrowed[:2], tangents=(borrowed[2],))
            elif route == "portable":
                result = rt.launch(artifact, borrowed)
                assert result["ok"], result
                assert all(r["input_binding"] == "cuda_resident_roots" and r["root_uploads"] == 0
                           for r in result["component_receipts"])
                pair = result["output"]
            else:
                pair, events = owner.profile(borrowed, repeats=8, captured=route == "captured")
                assert len(events["stage_ms"]) == 3
                assert all(value > 0 for value in events["stage_ms"])
                assert events["program_ms"] > 0
        assert len(completed) == len(callbacks) == 3
        check(pair, expected, budgets)
        assert owner.receipt["input_binding"] == "cuda_resident_roots"
        assert owner.receipt["root_uploads"] == 0
        saved = tuple(value.copy() for value in pair)
    for actual, snapshot in zip(pair, saved, strict=True):
        np.testing.assert_array_equal(actual, snapshot)


@pytest.mark.parametrize("field", [
    "capacity", "pointer", "dtype", "shape", "layout", "output-overlap", "input-output-overlap",
    "zero-stream", "invalid-stream",
])
def test_native_resident_admission_preserves_outputs_on_bad_frames(compiled, field):
    import ctypes as ct
    fn, roots, _, _ = compiled
    owner = next(iter(fn._native_prepared_jvp_calls.values()))
    with NvidiaDeviceSession() as session:
        buffers = [session.upload(value) for value in roots]
        _, outputs, views, streams = owner._frame(tuple(Borrowed(buffer) for buffer in buffers))
        for output in outputs:
            output.fill(91)
        if field == "capacity":
            small = session.upload(np.zeros((1,), np.uint8))
            views[0].data = small.ptr
        elif field == "pointer":
            views[0].data = 0
        elif field == "dtype":
            views[0].dtype = 1
        elif field == "shape":
            views[0].shape[0] += 1
        elif field == "layout":
            active_axes = [axis for axis in range(2) if views[0].shape[axis] > 1]
            if active_axes:
                views[0].strides[active_axes[0]] += 2
            else:
                # A 1x1 view has no strided access: changing either pitch is
                # valid and must not invent an extra read span.
                views[0].strides[0] += 4096
                assert owner.lib.tessera_nvidia_active_source_program_invoke_resident_inputs(
                    owner.handle, views, 5, streams, 3) == 0
                role = owner.plan["role_indices"][0]
                expected, budgets = oracle(roots[role], roots[1-role], roots[2])
                check(outputs, expected, budgets)
                for output in outputs:
                    output.fill(91)
                # Rank remains part of the physical frame even at unit extent.
                views[0].rank = 1
        elif field == "output-overlap":
            views[4].data = views[3].data
        elif field == "input-output-overlap":
            views[3].data = views[0].data
        elif field == "zero-stream":
            streams[0] = 0
        else:
            streams[0] = 3
        result = owner.lib.tessera_nvidia_active_source_program_invoke_resident_inputs(
            owner.handle, views, 5, streams, 3)
        assert result != 0
        message = owner.lib.tessera_nvidia_matmul_last_error().decode()
        if field == "capacity":
            assert "capacity" in message
        assert all(np.all(output == 91) for output in outputs)
        assert ct.sizeof(streams) == 24


def test_readonly_source_and_seed_may_share_device_storage(compiled):
    fn, roots, _, _ = compiled
    owner = next(iter(fn._native_prepared_jvp_calls.values()))
    role = owner.plan["role_indices"][0]
    expected, budgets = oracle(roots[role], roots[1 - role], roots[role])
    with NvidiaDeviceSession() as session:
        buffers = [session.upload(value) for value in roots[:2]]
        pair = fn.native_jvp(Borrowed(buffers[0]), Borrowed(buffers[1]),
                             tangents=(Borrowed(buffers[role]),))
        check(pair, expected, budgets)


def test_fresh_process_resident_replay_needs_no_compiler(compiled, tmp_path):
    import os
    import subprocess
    import sys
    fn, roots, expected, budgets = compiled
    artifact = tmp_path / "program.json"
    artifact.write_text(fn.native_jvp_runtime_artifact().to_json())
    source = tmp_path / "roots.npz"
    np.savez(source, *(value.view(np.uint16) for value in roots))
    destination = tmp_path / "outputs.npz"
    storage = "bf16" if roots[0].dtype.name == "bfloat16" else "f16"
    worker = """
from contextlib import ExitStack
from pathlib import Path
import sys
import subprocess
import ml_dtypes
import numpy as np
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
artifact = rt.RuntimeArtifact.from_json(Path(sys.argv[1]).read_text())
dtype = ml_dtypes.bfloat16 if sys.argv[4] == 'bf16' else np.float16
with np.load(sys.argv[2]) as packed:
    arrays = tuple(packed['arr_' + str(i)].view(dtype) for i in range(3))
with ExitStack() as stack:
    sessions = [stack.enter_context(NvidiaDeviceSession()) for _ in arrays]
    roots = tuple(session.upload(value) for session, value in zip(sessions, arrays, strict=True))
    def forbidden(*args, **kwargs):
        raise AssertionError('standalone resident replay invoked a compiler/subprocess')
    subprocess.run = forbidden
    result = rt.launch(artifact, roots)
    assert result['ok'], result
    assert all(r['root_uploads'] == 0 and r['input_binding'] == 'cuda_resident_roots'
               for r in result['component_receipts'])
    outputs = result['output']
np.savez(sys.argv[3], *outputs)
"""
    env = dict(os.environ)
    for name in ("TESSERA_OPT", "TESSERA_NVIDIA_OPT", "TESSERA_LLVM_BIN", "LLVM_BIN"):
        env[name] = "/compiler-intentionally-unavailable"
    completed = subprocess.run(
        [sys.executable, "-c", worker, str(artifact), str(source), str(destination), storage],
        env=env, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    with np.load(destination) as saved:
        check((saved["arr_0"], saved["arr_1"]), expected, budgets)
