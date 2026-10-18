"""Exact SM120 bounded host preparation through the native capacity owner."""
from pathlib import Path
import os

import ml_dtypes
import numpy as np
import pytest
import tessera as ts

from tessera import runtime as rt
from tessera.compiler import native_attention_jvp_runtime as runtime
from tests.device.nvidia.test_public_typed_attention_vjp import function
from tests.device.nvidia.test_public_typed_attention_jvp import reference
from tests.device.nvidia.test_saved_lse_half_backward_kernel import compare

pytestmark = [pytest.mark.hardware_nvidia, pytest.mark.compiler_route]


@pytest.mark.parametrize("storage", ["float16", "bfloat16"])
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("subset", [False, True])
def test_bounded_prepared_host_reuses_capacity_owner(storage, saved, bias, subset, monkeypatch):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    dtype = np.dtype(ml_dtypes.bfloat16 if storage == "bfloat16" else storage)
    rng = np.random.default_rng(20261017)
    def inputs(sq, sk):
        values = {name: rng.normal(0, .4, shape).astype(dtype) for name, shape in {
            "q": (1, 2, sq, 16), "k": (1, 1, sk, 16), "v": (1, 1, sk, 16),
        }.items()}
        if bias:
            values["bias"] = rng.normal(0, .1, (1, 2, 1, 1)).astype(np.float32)
        return values
    traced = inputs(1, 1)
    active = (("bias", "q") if bias else ("v",)) if subset else tuple(traced)
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=active)(
        function(saved, bias, True))
    program = fn.compile_native_attention_jvp(
        **traced, compiler=Path(os.environ["TESSERA_OPT"]),
        llvm_bin=Path(os.environ["TESSERA_LLVM_BIN"]), sequence_bounds=(7, 9))
    names = [f"primal_{i}" for i in range(len(program.input_indices))]
    names += [f"tangent_{program.input_indices[i]}" for i in program.active]
    metadata = dict(program_json=program.to_json(), program_digest=program.program_digest,
                    arg_names=names)
    frontend = ("bias", "v", "q", "k") if bias else ("q", "k", "v")
    identity = (program.program_digest, program.tangent.binding_digest,
                program.pair.forward.image.image_digest)
    def forbidden(*args, **kwargs):
        pytest.fail("bounded prepared replay attempted compilation")
    retained = []
    runtime.clear_prepared()
    try:
        handle = None
        with monkeypatch.context() as guard:
            guard.setattr("subprocess.run", forbidden)
            for sq, sk in ((1, 1), (3, 4), (7, 9), (7, 1), (1, 9), (3, 4)):
                values = inputs(sq, sk)
                seeds = {n: rng.normal(0, .1, x.shape).astype(x.dtype) if n in active
                         else np.zeros_like(x) for n, x in values.items()}
                result = runtime.execute(metadata, tuple(values[n] for n in frontend) +
                                         tuple(seeds[n] for n in active))
                expected = reference(*(values[n] for n in ("q", "k", "v")),
                    *(seeds[n] for n in ("q", "k", "v")), causal=True,
                    bias=values.get("bias"), dbias=seeds.get("bias"))
                for actuals, targets in zip(result, expected, strict=True):
                    actuals = actuals if saved else (actuals,)
                    targets = targets if saved else targets[:1]
                    for index, (actual, target) in enumerate(zip(actuals, targets, strict=True)):
                        assert actual.dtype == (np.dtype("float32") if index else dtype)
                        compare(actual, target)
                owner = runtime.prepared(metadata)
                if handle is None:
                    handle = owner.handle
                assert owner.handle == handle
                assert identity == (owner.program.program_digest,
                    owner.program.tangent.binding_digest,
                    owner.program.pair.forward.image.image_digest)
                snapshots = tuple(tuple(x.copy() for x in group) if saved else group.copy()
                                  for group in result)
                retained.append((result, snapshots))
                for prior, snapshots in retained:
                    for actual, snapshot in zip(prior, snapshots, strict=True):
                        for a, b in zip(actual if saved else (actual,),
                                        snapshot if saved else (snapshot,), strict=True):
                            np.testing.assert_array_equal(a, b)
            for sq, sk in ((8, 4), (3, 10), (0, 4), (3, 0)):
                values = inputs(sq, sk)
                with pytest.raises(ValueError, match="sequence|capacity|envelope"):
                    runtime.execute(metadata, tuple(values[n] for n in frontend) +
                                    tuple(np.zeros_like(values[n]) for n in active))
            # Capacity refusal must leave the same owner usable.
            values = inputs(3, 4)
            runtime.execute(metadata, tuple(values[n] for n in frontend) +
                            tuple(np.zeros_like(values[n]) for n in active))
            assert runtime.prepared(metadata).handle == handle
    finally:
        runtime.clear_prepared()


@pytest.mark.parametrize("storage", ["float16", "bfloat16"])
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_public_bounded_jvp_keeps_one_compiled_owner(storage, saved, bias, monkeypatch):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    dtype = np.dtype(ml_dtypes.bfloat16 if storage == "bfloat16" else storage)
    rng = np.random.default_rng(20261018)
    def values(sq, sk):
        roots = {n: rng.normal(0, .4, shape).astype(dtype) for n, shape in {
            "q": (1, 2, sq, 16), "k": (1, 1, sk, 16), "v": (1, 1, sk, 16),
        }.items()}
        if bias:
            roots["bias"] = rng.normal(0, .1, (1, 2, 1, 1)).astype(np.float32)
        return roots
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=tuple(values(1, 1)))(
        function(saved, bias, True))
    held = []
    first = True
    handle = None
    def forbidden(*args, **kwargs):
        pytest.fail("warm bounded public JVP attempted compilation or concrete frontend evaluation")
    try:
        for sq, sk in ((1, 1), (3, 4), (7, 9), (1, 9), (7, 1), (3, 4)):
            roots = values(sq, sk)
            directions = {n: rng.normal(0, .1, x.shape).astype(x.dtype) for n, x in roots.items()}
            expected = reference(*(roots[n] for n in ("q", "k", "v")),
                *(directions[n] for n in ("q", "k", "v")), causal=True,
                bias=roots.get("bias"), dbias=directions.get("bias"))
            if first:
                result = fn.native_jvp(**roots, tangents=tuple(directions.values()),
                                       sequence_bounds=(7, 9))
                first = False
            else:
                with monkeypatch.context() as guard:
                    guard.setattr("subprocess.run", forbidden)
                    guard.setattr(fn, "_fn", forbidden)
                    guard.setattr(fn, "_trace_frontend_capture", forbidden)
                    guard.setattr(fn, "compile_native_attention_jvp", forbidden)
                    result = fn.native_jvp(**roots, tangents=tuple(directions.values()),
                                           sequence_bounds=(7, 9))
            assert len(fn._native_prepared_jvp_calls) == 1
            call = next(iter(fn._native_prepared_jvp_calls.values()))
            if handle is None:
                handle = call.owner.handle
            assert call.owner.handle == handle
            for actuals, targets in zip(result, expected, strict=True):
                actuals = actuals if saved else (actuals,)
                targets = targets if saved else targets[:1]
                for actual, target in zip(actuals, targets, strict=True):
                    compare(actual, target)
            held.append((result, tuple(tuple(x.copy() for x in group) if saved else group.copy()
                                       for group in result)))
            for prior, snapshots in held:
                for group, snapshot in zip(prior, snapshots, strict=True):
                    for a, b in zip(group if saved else (group,),
                                    snapshot if saved else (snapshot,), strict=True):
                        np.testing.assert_array_equal(a, b)
            assert fn.last_jvp_execution["execution_kind"] == "native_gpu"
            assert fn.last_jvp_execution["host_preparation"] == "native_bounded_capacity_owner"
            assert len(fn.last_jvp_execution["forward_schedule_ir_digest"]) == 64
    finally:
        fn.close_native_storage()
    assert not fn._native_prepared_jvp_calls


@pytest.mark.parametrize("storage", ["f16", "bf16", "f32"])
@pytest.mark.parametrize("layout", ["query_broadcast", "key_broadcast", "logical"])
def test_bounded_dynamic_bias_and_fp32_output(storage, layout, monkeypatch):
    from tessera.compiler.native_attention_program import compile_attention_program
    from tests.unit.test_saved_lse_half_backward import source
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    dtype = np.dtype({"f16": np.float16, "bf16": ml_dtypes.bfloat16,
                     "f32": np.float32}[storage])
    original = source(storage, saved=True, bias=True, output="f32",
                      dims=(1, 2, 1, 3, 4, 4, 3)).replace('"reverse"', '"forward"')
    bias_type = {"query_broadcast": "1x2x1x4", "key_broadcast": "1x2x3x1",
                 "logical": "1x2x3x4"}[layout]
    original = original.replace("tensor<1x2x1x4xf32>", f"tensor<{bias_type}xf32>")
    original = original.replace('tessera.arch = "sm_120"',
        'tessera.arch = "sm_120", tessera.attention_sequence_bounds = array<i64: 7, 9>')
    program = compile_attention_program(original, (0, 1, 2, 3),
        compiler=Path(os.environ["TESSERA_OPT"]), llvm_bin=Path(os.environ["TESSERA_LLVM_BIN"]))
    metadata = dict(program_json=program.to_json(), program_digest=program.program_digest,
        arg_names=[f"primal_{i}" for i in range(4)] + [f"tangent_{i}" for i in range(4)])
    rng = np.random.default_rng(20261019)
    def forbidden(*args, **kwargs):
        pytest.fail("dynamic-bias bounded owner attempted compilation")
    runtime.clear_prepared()
    try:
        handle = None
        with monkeypatch.context() as guard:
            guard.setattr("subprocess.run", forbidden)
            for sq, sk in ((3, 4), (1, 1), (7, 9), (1, 9), (7, 1), (3, 4)):
                roots = tuple(rng.normal(0, .3, shape).astype(dtype) for shape in
                              ((1, 2, sq, 4), (1, 1, sk, 4), (1, 1, sk, 3)))
                shape = (1, 2, 1 if layout == "query_broadcast" else sq,
                         1 if layout == "key_broadcast" else sk)
                bias = rng.normal(0, .1, shape).astype(np.float32)
                values = (*roots, bias)
                seeds = tuple(rng.normal(0, .1, x.shape).astype(x.dtype) for x in values)
                result = runtime.execute(metadata, values + seeds)
                expected = reference(*roots, *seeds[:3], bias=bias, dbias=seeds[3], causal=True)
                for actuals, targets in zip(result, expected, strict=True):
                    for actual, target in zip(actuals, targets, strict=True):
                        assert actual.dtype == np.dtype("float32")
                        compare(actual, target)
                owner = runtime.prepared(metadata)
                if handle is None:
                    handle = owner.handle
                assert owner.handle == handle
    finally:
        runtime.clear_prepared()


def test_native_bounded_abi_refuses_bad_spans_and_recovers():
    import ctypes as ct
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    rng = np.random.default_rng(20261020)
    roots = {n: rng.normal(0, .3, shape).astype(np.float16) for n, shape in {
        "q": (1, 2, 3, 16), "k": (1, 1, 4, 16), "v": (1, 1, 4, 16),
    }.items()}
    directions = tuple(rng.normal(0, .1, x.shape).astype(x.dtype) for x in roots.values())
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=("q", "k", "v"))(
        function(True, False, True))
    try:
        first = fn.native_jvp(**roots, tangents=directions, sequence_bounds=(7, 9))
        call = next(iter(fn._native_prepared_jvp_calls.values()))
        owner = call.owner
        lib = owner._library()
        values = (*roots.values(), *directions)
        P, S, I64 = ct.c_void_p, ct.c_size_t, ct.c_int64
        pointers = (P*len(values))(*(x.ctypes.data for x in values))
        lengths = (S*len(values))(*(x.nbytes for x in values))
        results = tuple(np.full_like(x, 37) for group in first for x in group)
        destinations = (P*4)(*(x.ctypes.data for x in results))
        sizes = (S*4)(*(x.nbytes for x in results))
        dims = (I64*7)(1, 2, 1, 3, 4, 16, 16)
        too_large = (I64*7)(1, 2, 1, 8, 4, 16, 16)
        wrong_fixed = (I64*7)(1, 2, 1, 3, 4, 15, 16)
        unexpected_bias = (I64*4)(1, 2, 1, 1)
        truncated = (S*len(values))(*(x.nbytes - (2 if i == 0 else 0)
                                    for i, x in enumerate(values)))
        times = (ct.c_float*2)(123, 456)
        invoke = lib.tessera_nvidia_attention_jvp_invoke_bounded
        for shape, bias_shape, spans, count in (
            (None, None, lengths, 4), (too_large, None, lengths, 4),
            (wrong_fixed, None, lengths, 4), (dims, unexpected_bias, lengths, 4),
            (dims, None, lengths, 3), (dims, None, truncated, 4),
        ):
            status = invoke(owner.handle, pointers, spans, len(values), shape, bias_shape,
                            None, 0, destinations, sizes, count, times)
            assert status != 0
            assert lib.tessera_nvidia_attention_jvp_last_error()
            for result in results:
                np.testing.assert_array_equal(result, np.full_like(result, 37))
            assert tuple(times) == (123, 456)
        # A static ABI cannot silently launch a bounded handle.
        legacy = lib.tessera_nvidia_attention_jvp_invoke_lse
        legacy.argtypes = [ct.c_uint64, ct.POINTER(P), ct.POINTER(S), S,
                           ct.POINTER(P), ct.POINTER(S), S, ct.POINTER(ct.c_float)]
        legacy.restype = ct.c_int
        assert legacy(owner.handle, pointers, lengths, len(values),
                      destinations, sizes, 4, times) != 0
        for result in results:
            np.testing.assert_array_equal(result, np.full_like(result, 37))
        handle = owner.handle
        recovered = fn.native_jvp(**roots, tangents=directions, sequence_bounds=(7, 9))
        assert call.owner.handle == handle
        for actuals, targets in zip(recovered, first, strict=True):
            for actual, target in zip(actuals, targets, strict=True):
                np.testing.assert_array_equal(actual, target)
    finally:
        fn.close_native_storage()


@pytest.mark.parametrize("storage", ["float16", "bfloat16"])
@pytest.mark.parametrize("saved", [False, True])
def test_public_bounded_jvp_orders_pending_producer_stream(storage, saved, monkeypatch):
    from contextlib import ExitStack
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed, queue_delayed_upload
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    dtype = np.dtype(ml_dtypes.bfloat16 if storage == "bfloat16" else storage)
    rng = np.random.default_rng(20261017)
    shapes = {"q": (1, 2, 3, 16), "k": (1, 1, 5, 16), "v": (1, 1, 5, 16)}
    values = {n: rng.normal(0, .4, shape).astype(dtype) for n, shape in shapes.items()}
    seeds = {n: rng.normal(0, .1, shape).astype(dtype) for n, shape in shapes.items()}
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=("q", "k", "v"))(
        function(saved, False, True))
    try:
        with ExitStack() as stack:
            sessions = {n: stack.enter_context(NvidiaDeviceSession()) for n in shapes}
            roots = {n: session.upload(np.zeros(shapes[n], dtype)) for n, session in sessions.items()}
            directions = tuple(Borrowed(sessions[n].upload(seeds[n])) for n in shapes)
            for session in sessions.values():
                assert session.synchronize() == 0
            borrowed = {n: Borrowed(value) for n, value in roots.items()}
            fn.native_jvp(**borrowed, tangents=directions, sequence_bounds=(7, 9))
            held = []
            callback = queue_delayed_upload(sessions["k"], roots["k"], values["k"], sessions["k"].stream, held)
            for n in ("q", "v"):
                import ctypes as ct
                assert sessions[n].lib.tessera_nvidia_device_upload(ct.c_void_p(roots[n].ptr),
                    ct.c_void_p(values[n].ctypes.data), values[n].nbytes, ct.c_void_p(sessions[n].stream)) == 0
            assert not held
            def forbidden(*args, **kwargs):
                pytest.fail("resident typed replay attempted compilation or host coercion")
            with monkeypatch.context() as guard:
                guard.setattr("subprocess.run", forbidden)
                guard.setattr(fn, "_fn", forbidden)
                guard.setattr(fn, "_trace_frontend_capture", forbidden)
                result = fn.native_jvp(**borrowed, tangents=directions, sequence_bounds=(7, 9))
            assert held and callback is not None
            expected = reference(*values.values(), *seeds.values(), causal=True)
            for actuals, targets in zip(result, expected, strict=True):
                for actual, target in zip(actuals if saved else (actuals,), targets if saved else targets[:1], strict=True):
                    compare(actual, target)
            assert fn.last_jvp_execution["host_preparation"] == "native_ordered_resident_snapshot"
    finally:
        fn.close_native_storage()

@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("variation", ["storage", "fixed_dimensions"])
def test_public_bounded_jvp_specializes_fixed_inputs(resident, bias, variation, monkeypatch):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    layouts = ([("float16", 1, 2, 1, 16, 16),
                ("bfloat16", 1, 2, 1, 16, 16),
                ("float32", 1, 2, 1, 16, 16)] if variation == "storage" else
               [("float16", 1, 2, 1, 16, 16),
                ("float16", 2, 4, 2, 8, 12),
                ("float16", 1, 2, 1, 16, 8)])
    active = ("q", "k", "v", "bias") if bias else ("q", "k", "v")
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=active)(
        function(True, bias, True))
    compile_original = fn.compile_native_attention_jvp
    compilations = []
    def compile_once(*args, **kwargs):
        compilations.append(1)
        return compile_original(*args, **kwargs)
    monkeypatch.setattr(fn, "compile_native_attention_jvp", compile_once)
    rng = np.random.default_rng(20261019)
    handles = {}
    session = NvidiaDeviceSession() if resident else None
    def forbidden(*args, **kwargs):
        pytest.fail("compatible bounded specialization reached frontend/compiler")
    try:
        for warm in (False, True):
            for index, (storage, b, hq, hkv, d, dv) in enumerate(layouts):
                sq, sk = ((1, 1) if not warm else (3 + index * 2, 9 - index))
                dtype = np.dtype(ml_dtypes.bfloat16 if storage == "bfloat16" else storage)
                roots = {n: rng.normal(0, .25, shape).astype(dtype) for n, shape in {
                    "q": (b, hq, sq, d), "k": (b, hkv, sk, d),
                    "v": (b, hkv, sk, dv)}.items()}
                if bias:
                    roots["bias"] = rng.normal(0, .1, (1, hq, 1, 1)).astype(np.float32)
                seeds = {n: rng.normal(0, .1, x.shape).astype(x.dtype) for n, x in roots.items()}
                expected = reference(*(roots[n] for n in ("q", "k", "v")),
                    *(seeds[n] for n in ("q", "k", "v")), bias=roots.get("bias"),
                    dbias=seeds.get("bias"), causal=True)
                bound = {n: session.upload(x) for n, x in roots.items()} if session else roots
                tangents = tuple(session.upload(seeds[n]) if session else seeds[n] for n in active)
                if warm:
                    with monkeypatch.context() as guard:
                        guard.setattr(fn, "compile_native_attention_jvp", forbidden)
                        guard.setattr(fn, "_trace_frontend_capture", forbidden)
                        guard.setattr(fn, "_fn", forbidden)
                        guard.setattr("subprocess.run", forbidden)
                        result = fn.native_jvp(**bound, tangents=tangents, sequence_bounds=(7, 9))
                else:
                    result = fn.native_jvp(**bound, tangents=tangents, sequence_bounds=(7, 9))
                for actuals, targets in zip(result, expected, strict=True):
                    for actual, target in zip(actuals, targets, strict=True):
                        compare(actual, target)
                digest = fn.last_jvp_execution["program_digest"]
                owner = next(call.owner for call in fn._native_prepared_jvp_calls.values()
                             if call.program.program_digest == digest)
                if warm:
                    assert owner.handle == handles[index]
                else:
                    handles[index] = owner.handle
        assert len(compilations) == len(layouts)
        assert len(set(handles.values())) == len(layouts)
        assert len(fn._native_prepared_jvp_calls) == len(layouts)
    finally:
        fn.close_native_storage()
        if session:
            session.close()

def test_public_bounded_jvp_preserves_bias_specializations_and_capacity_guards(monkeypatch):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    active = ("q", "k", "v", "bias")
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=active)(
        function(True, True, True))
    rng = np.random.default_rng(20261020)
    def inputs(sq, sk, logical):
        roots = {n: rng.normal(0, .25, shape).astype(np.float16) for n, shape in {
            "q": (1, 2, sq, 16), "k": (1, 1, sk, 16),
            "v": (1, 1, sk, 16)}.items()}
        roots["bias"] = rng.normal(0, .1, (1, 2, sq if logical else 1,
                                                   sk if logical else 1)).astype(np.float32)
        seeds = {n: rng.normal(0, .1, x.shape).astype(x.dtype) for n, x in roots.items()}
        return roots, seeds
    def forbidden(*args, **kwargs):
        pytest.fail("warm physical bias/capacity selection attempted compilation")
    handles = {}
    try:
        for logical, sq, sk, cold in ((False, 1, 1, True), (False, 3, 4, False),
                                      (True, 3, 4, True), (True, 1, 9, False),
                                      (True, 7, 1, False), (True, 1, 1, False),
                                      (False, 3, 4, False)):
            roots, seeds = inputs(sq, sk, logical)
            expected = reference(*(roots[n] for n in ("q", "k", "v")),
                *(seeds[n] for n in ("q", "k", "v")), bias=roots["bias"],
                dbias=seeds["bias"], causal=True)
            with monkeypatch.context() as guard:
                if not cold:
                    guard.setattr(fn, "compile_native_attention_jvp", forbidden)
                    guard.setattr(fn, "_trace_frontend_capture", forbidden)
                    guard.setattr("subprocess.run", forbidden)
                result = fn.native_jvp(**roots, tangents=tuple(seeds[n] for n in active),
                                       sequence_bounds=(7, 9))
            digest = fn.last_jvp_execution["program_digest"]
            owner = next(call.owner for call in fn._native_prepared_jvp_calls.values()
                         if call.program.program_digest == digest)
            if cold:
                handles[logical] = owner.handle
            else:
                assert owner.handle == handles[logical]
            for actuals, targets in zip(result, expected, strict=True):
                for actual, target in zip(actuals, targets, strict=True):
                    compare(actual, target)
        assert len(fn._native_prepared_jvp_calls) == 2
        assert handles[False] != handles[True]
        roots, seeds = inputs(8, 4, True)
        with monkeypatch.context() as guard:
            guard.setattr(fn, "compile_native_attention_jvp", forbidden)
            guard.setattr("subprocess.run", forbidden)
            with pytest.raises(ts.TesseraJitError, match="compiled envelope"):
                fn.native_jvp(**roots, tangents=tuple(seeds[n] for n in active),
                              sequence_bounds=(7, 9))
        roots, seeds = inputs(3, 4, True)
        result = fn.native_jvp(**roots, tangents=tuple(seeds[n] for n in active),
                               sequence_bounds=(7, 9))
        expected = reference(*(roots[n] for n in ("q", "k", "v")),
            *(seeds[n] for n in ("q", "k", "v")), bias=roots["bias"],
            dbias=seeds["bias"], causal=True)
        for actuals, targets in zip(result, expected, strict=True):
            for actual, target in zip(actuals, targets, strict=True):
                compare(actual, target)
    finally:
        fn.close_native_storage()
