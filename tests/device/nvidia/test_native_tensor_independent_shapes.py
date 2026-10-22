"""Exact RTX5070 execution of compiler-owned independent M/N/K frames."""
import subprocess

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime
from tessera.compiler.native_tensor_program import compile_native_tensor_program
from tests.unit.test_native_tensor_shape_ownership import shape_source
from tests.unit.test_native_tensor_shape_plan import oracle as shape_oracle

pytestmark = pytest.mark.hardware_nvidia


@pytest.fixture(scope="module", params=[(case, storage) for case in
    ("norm_lhs", "rhs_add", "chain", "two_matmuls") for storage in ("f16", "bf16")])
def independent_program(request):
    assert runtime._nvidia_device_name() == "sm_120"
    case, storage = request.param
    program = compile_native_tensor_program(shape_source(case, storage))
    assert program is not None and program.shape_companion is not None
    return case, storage, program


def roots(case, storage, m, n, k, seed=23001):
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    rng = np.random.default_rng(seed)
    def array(shape):
        return rng.normal(0, .2, shape).astype(dtype)
    if case == "two_matmuls":
        return array((m, 32)), array((32, k)), array((k, n))
    a, b = array((m, k)), array((k, n))
    return (a, b, array(b.shape if case == "rhs_add" else a.shape)) if case in {"rhs_add", "chain"} else (a, b)


def check(case, actual, args):
    a, b, *extra = args
    dtype = a.dtype
    input_uncertainty = 0
    if case == "two_matmuls":
        lhs, weights, b = (value.astype(np.float64) for value in args)
        wide = lhs @ weights
        hidden = np.maximum(wide.astype(np.float32), 0).astype(np.float64)
        gamma = 32 * 2**-24 / (1 - 32 * 2**-24)
        input_uncertainty = 2 * gamma * (np.abs(lhs) @ np.abs(weights)) + np.abs(np.spacing(wide.astype(np.float32)))
        a = hidden
    elif case == "norm_lhs":
        wide = a.astype(np.float64)
        ideal = wide / np.sqrt(np.mean(wide * wide, axis=-1, keepdims=True) + 1e-5)
        # The contract computes the norm in FP32 before its storage conversion.
        # FP64->f16 rounding alone is not an oracle at a half-way boundary:
        # a valid FP32 result can land exactly on that boundary and round to
        # the other adjacent f16 value. Carry a narrow FP32 arithmetic interval
        # through the storage conversion, then propagate only its ambiguous
        # stored elements through the independent final product.
        # For these finite, non-underflowing K<=64 inputs, use a conservative
        # gamma_(K+8) budget for same-sign square summation, mean/epsilon,
        # round-to-nearest sqrt and division, doubled for reciprocal sensitivity.
        # Summation reference: https://nhigham.com/wp-content/uploads/2023/10/high93s.pdf
        count = wide.shape[1] + 8
        relative = 2 * (count * 2**-24 / (1 - count * 2**-24))
        radius = relative * np.abs(ideal)
        lower = np.nextafter(ideal - radius, -np.inf).astype(dtype).astype(np.float64)
        upper = np.nextafter(ideal + radius, np.inf).astype(dtype).astype(np.float64)
        a = ideal.astype(dtype)
        rounded = a.astype(np.float64)
        input_uncertainty = np.maximum(np.abs(lower - rounded), np.abs(upper - rounded))
    else:
        if case == "chain":
            a = (a.astype(np.float64) + extra[0].astype(np.float64)).astype(dtype)
        if case == "rhs_add":
            b = (b.astype(np.float64) + extra[0].astype(np.float64)).astype(dtype)
        a = np.maximum(a, 0).astype(dtype)
    left, right = a.astype(np.float64), b.astype(np.float64)
    expected = left @ right
    gamma = left.shape[1] * 2**-24 / (1 - left.shape[1] * 2**-24)
    bound = (input_uncertainty @ np.abs(right) if np.ndim(input_uncertainty) else 0)
    bound += 3 * gamma * ((np.abs(left) + input_uncertainty) @ np.abs(right)) + 3e-7
    assert np.isfinite(actual).all()
    error = np.abs(actual.astype(np.float64) - expected)
    assert np.all(error <= bound), (float(error.max()), float(np.max(bound)))
    return float(error.max())


def test_changed_mnk_uses_native_plan_and_one_arena(independent_program, monkeypatch):
    case, storage, program = independent_program
    kept = []
    def forbidden(*args, **kwargs):
        pytest.fail("warm independent-shape execution invoked a compiler")
    with program.prepare() as owner:
        initial = owner.stats()
        library_path = owner._shape_library._name
        from pathlib import Path
        assert library_path in Path("/proc/self/maps").read_text()
        for index, (m, n, k) in enumerate(((1, 1, 1), (17, 11, 23), (65, 47, 64), (5, 31, 7), (17, 17, 17))):
            args = roots(case, storage, m, n, k, seed=23001 + index)
            snapshots = [value.copy() for value in args]
            before_query = owner.stats()
            active = owner._active_specs(args)
            root_shapes, expected_words = shape_oracle(case, m, n, k)
            expected_shapes = np.array(expected_words[:2 * len(active)]).reshape(-1, 2).tolist()
            assert [spec["shape"] for spec in active] == expected_shapes
            assert [active[i]["shape"] for i in program.frame["roots"]] == root_shapes
            assert owner.stats() == before_query
            tail = np.full(m * n + 16, 123, np.float32)
            output = tail[8:-8].reshape(m, n)
            with monkeypatch.context() as guard:
                guard.setattr(subprocess, "run", forbidden)
                guard.setattr(subprocess, "Popen", forbidden)
                assert owner(args, output=output) is output
            check(case, output, args)
            np.testing.assert_array_equal(tail[:8], 123)
            np.testing.assert_array_equal(tail[-8:], 123)
            for value, saved in zip(args, snapshots, strict=True):
                np.testing.assert_array_equal(value, saved)
            for value, saved in kept:
                np.testing.assert_array_equal(value, saved)
            kept.append((output, output.copy()))
        final = owner.stats()
        assert initial["allocations"] == final["allocations"] == 1
        assert initial["capacity"] == final["capacity"]
    assert owner._shape_library is None
    assert library_path not in Path("/proc/self/maps").read_text()


def test_failed_shape_does_not_commit_frame_or_output(independent_program):
    case, storage, program = independent_program
    with program.prepare() as owner:
        valid = roots(case, storage, 17, 11, 23)
        output = owner(valid)
        saved, before = output.copy(), owner.stats()
        for m, n, k in ((66, 11, 23), (17, 48, 23), (17, 11, 65)):
            with pytest.raises(RuntimeError, match="shape program rejected"):
                owner(roots(case, storage, m, n, k), output=output)
            assert owner.stats() == before
            np.testing.assert_array_equal(output, saved)
        broken = list(valid)
        index = 2 if case == "two_matmuls" else 1
        broken[index] = broken[index][:-1].copy()
        with pytest.raises(RuntimeError, match="shape program rejected"):
            owner(broken, output=output)
        assert owner.stats() == before
        np.testing.assert_array_equal(output, saved)
        check(case, owner(valid), valid)


@pytest.mark.parametrize("case", ["norm_lhs", "rhs_add", "chain", "two_matmuls"])
@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("trace", [(1, 1, 1), (17, 17, 17)])
def test_public_jit_reuses_one_native_program_across_independent_axes(case, storage, trace, monkeypatch):
    import tessera as ts
    from tests.unit import test_native_tensor_shape_frontend as frontend

    fn = ts.jit(target="nvidia_sm120", shape_bounds={"M": 65, "N": 47, "K": 64})(getattr(frontend, case))
    def forbidden(*args, **kwargs):
        pytest.fail("warm independent-shape public JIT traced, compiled or ran eager")
    try:
        args = roots(case, storage, *trace)
        first = fn(*args)
        check(case, first, args)
        saved = first.copy()
        program, owner = fn._native_tensor_last_program, fn._native_tensor_last_owner
        assert program.frame["schema"] == "tessera.native.tensor_program_frame.v3"
        assert fn.execution_kind == "native_gpu"
        initial = owner.stats()
        for m, n, k in ((17, 11, 23), (65, 47, 64), (5, 31, 7)):
            args = roots(case, storage, m, n, k, seed=23101 + m)
            with monkeypatch.context() as guard:
                guard.setattr(fn, "_fn", forbidden)
                guard.setattr(fn, "_trace_frontend_capture", forbidden)
                guard.setattr(subprocess, "run", forbidden)
                guard.setattr(subprocess, "Popen", forbidden)
                actual = fn(*args)
            check(case, actual, args)
            assert fn._native_tensor_last_program is program
            assert fn._native_tensor_last_owner is owner
            assert owner.stats()["allocations"] == initial["allocations"] == 1
            np.testing.assert_array_equal(first, saved)
        artifact = runtime.RuntimeArtifact.from_json(fn.runtime_artifact().to_json())
        assert artifact.metadata["native_program"]["schema"] == "tessera.native.tensor_program.v2"
        args = roots(case, storage, 5, 31, 7, seed=23200)
        result = runtime.launch(artifact, dict(zip(artifact.metadata["arg_names"], args, strict=True)))
        assert result["ok"] and result["execution_kind"] == "native_gpu", result
        check(case, result["output"], args)
    finally:
        fn.close_native_storage()


@pytest.mark.parametrize("mode", ["explicit", "legacy", "per_thread"])
def test_resident_independent_shapes_order_pending_producers(independent_program, mode, monkeypatch):
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed, queue_delayed_upload

    case, storage, program = independent_program
    args = roots(case, storage, 17, 11, 23, seed=23301)
    with NvidiaDeviceSession() as session, program.prepare() as owner:
        device = [session.upload(np.zeros_like(value)) for value in args]
        assert session.synchronize() == 0
        stream = 1 if mode == "legacy" else 2 if mode == "per_thread" else session.stream
        borrowed = [Borrowed(value, stream) for value in device]
        owner(borrowed)
        before = owner.stats()
        completed, callbacks = [], []
        for buffer, value in zip(device, args, strict=True):
            callbacks.append(queue_delayed_upload(session, buffer, value, stream, completed))
        assert len(completed) < len(args)
        def forbidden(*args, **kwargs):
            pytest.fail("resident native shape frame compiled or read host values")
        with monkeypatch.context() as guard:
            guard.setattr(subprocess, "run", forbidden)
            guard.setattr(subprocess, "Popen", forbidden)
            output = owner(borrowed)
        assert len(completed) == len(callbacks) == len(args)
        check(case, output, args)
        for buffer, value in zip(device, args, strict=True):
            np.testing.assert_array_equal(session.download(buffer), value)
        assert owner.stats()["allocations"] == before["allocations"] == 1
        kept = output.copy()
    np.testing.assert_array_equal(output, kept)



@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_norm_rounding_boundary_seed_matches_fp32_storage_contract(storage):
    import tessera as ts
    from tests.unit.test_native_tensor_shape_frontend import norm_lhs

    args = roots("norm_lhs", storage, 65, 47, 64, seed=23403)
    fn = ts.jit(target="nvidia_sm120", shape_bounds={"M": 65, "N": 47, "K": 64})(norm_lhs)
    try:
        check("norm_lhs", fn(*args), args)
    finally:
        fn.close_native_storage()
