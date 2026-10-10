"""Exact SM120 public JVP routes through native attention AD packages."""
import copy
import subprocess

import numpy as np
import pytest
import tessera as ts

from tessera import runtime as rt
from tessera.autodiff import jvp
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed
from tests.device.nvidia.test_resident_attention_forward import oracle, plain, biased, saved

pytestmark = pytest.mark.hardware_nvidia


@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("profile", ["plain", "biased", "saved"])
@pytest.mark.parametrize("sk", [5, 129])
def test_public_jvp_native_attention_preserves_original_and_reuses_product(
        resident, profile, sk, monkeypatch):
    with_lse = profile == 'saved'
    with_bias = profile != 'plain'
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070 required")
    rng = np.random.default_rng(933)
    shapes = {"q": (1, 2, 3, 4), "k": (1, 1, sk, 4), "v": (1, 1, sk, 3)}
    values = {name: rng.normal(0, .2, shape).astype(np.float32) for name, shape in shapes.items()}
    seeds = {name: rng.normal(0, .1, shape).astype(np.float32) for name, shape in shapes.items()}
    order = ("v", "q", "k", "bias") if with_bias else ("q", "k", "v")
    if with_bias:
        values["bias"] = rng.normal(0, .15, (1, 2, 1, sk)).astype(np.float32)
        seeds["bias"] = None
    fn = ts.jit(target="nvidia_sm120")({"plain": plain, "biased": biased, "saved": saved}[profile])
    before = copy.deepcopy(fn.graph_ir)
    with NvidiaDeviceSession() as session:
        inputs = tuple(Borrowed(session.upload(values[name])) if resident else values[name] for name in order)
        directions = tuple(Borrowed(session.upload(seeds[name])) if resident and seeds[name] is not None
                           else seeds[name] for name in order)
        result = jvp(fn, inputs, directions)
        assert fn.graph_ir == before and fn.differentiation_request is None
        active = tuple(index for index, value in enumerate(directions) if value is not None)
        owner = fn._native_public_jvp_owners[active][1]
        assert owner.last_jvp_execution["execution_kind"] == "native_gpu"
        h = 1e-4
        plus = tuple(values[name].astype(np.float64) + h * seeds[name] for name in ("q", "k", "v"))
        minus = tuple(values[name].astype(np.float64) - h * seeds[name] for name in ("q", "k", "v"))
        expected = oracle(tuple(values[name] for name in ("q", "k", "v")), values.get("bias"))
        positive, negative = oracle(plus, values.get("bias")), oracle(minus, values.get("bias"))
        primals = result[0] if with_lse else (result[0],)
        tangents = result[1] if with_lse else (result[1],)
        for actual, reference in zip(primals, expected[:len(primals)], strict=True):
            np.testing.assert_allclose(actual, reference, rtol=4e-5, atol=3e-6)
        for actual, positive_value, negative_value in zip(tangents, positive, negative, strict=False):
            np.testing.assert_allclose(actual, (positive_value-negative_value)/(2*h), rtol=4e-5, atol=3e-6)
        held = tuple(value.copy() for value in (*primals, *tangents))
        def forbidden(*args, **kwargs):
            pytest.fail("warm public native JVP recompiled or evaluated the frontend")
        with monkeypatch.context() as guard:
            guard.setattr(subprocess, "run", forbidden)
            guard.setattr(owner, "_fn", forbidden)
            repeated = jvp(fn, inputs, directions)
        assert fn._native_public_jvp_owners[active][1] is owner
        assert fn.last_jvp_execution["public_transform"] == "jvp"
        new_primals = repeated[0] if with_lse else (repeated[0],)
        new_tangents = repeated[1] if with_lse else (repeated[1],)
        for actual, reference in zip((*new_primals, *new_tangents), held, strict=True):
            np.testing.assert_array_equal(actual, reference)
        fn.close_native_storage()

@pytest.mark.parametrize("active_name", ["v", "bias"])
@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("sk", [5, 129])
def test_saved_lse_public_jvp_value_only_and_bias_only(active_name, resident, sk):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070 required")
    rng = np.random.default_rng(935)
    shapes = {"q": (1, 2, 3, 4), "k": (1, 1, sk, 4),
              "v": (1, 1, sk, 3), "bias": (1, 2, 1, sk)}
    order = ("v", "q", "k", "bias")
    values = {name: rng.normal(0, .2, shape).astype(np.float32) for name, shape in shapes.items()}
    seed = rng.normal(0, .1, shapes[active_name]).astype(np.float32)
    fn = ts.jit(target="nvidia_sm120")(saved)
    with NvidiaDeviceSession() as session:
        inputs = tuple(Borrowed(session.upload(values[name])) if resident else values[name] for name in order)
        seeds = tuple((Borrowed(session.upload(seed)) if resident else seed)
                      if name == active_name else None for name in order)
        primal, tangent = jvp(fn, inputs, seeds)
        h = 1e-4
        plus = {name: value.astype(np.float64) + (h*seed if name == active_name else 0)
                for name, value in values.items()}
        minus = {name: value.astype(np.float64) - (h*seed if name == active_name else 0)
                 for name, value in values.items()}
        expected = oracle(tuple(values[name] for name in ("q", "k", "v")), values["bias"])
        positive = oracle(tuple(plus[name] for name in ("q", "k", "v")), plus["bias"])
        negative = oracle(tuple(minus[name] for name in ("q", "k", "v")), minus["bias"])
        for actual, reference in zip(primal, expected, strict=True):
            np.testing.assert_allclose(actual, reference, rtol=4e-5, atol=3e-6)
        for actual, upper, lower in zip(tangent, positive, negative, strict=True):
            np.testing.assert_allclose(actual, (upper-lower)/(2*h), rtol=4e-5, atol=3e-6)
        if active_name == "v":
            np.testing.assert_array_equal(tangent[1], np.zeros((1, 2, 3), np.float32))
        fn.close_native_storage()

@pytest.mark.parametrize("defect", ["count", "extent", "overlap", "input_overlap"])
def test_saved_lse_native_output_span_guards_precede_launch(defect):
    import ctypes as ct
    from tessera.compiler.native_attention_jvp_runtime import prepared
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070 required")
    rng = np.random.default_rng(936)
    values = tuple(rng.normal(0, .2, shape).astype(np.float32) for shape in
                   ((1, 1, 5, 3), (1, 2, 3, 4), (1, 1, 5, 4), (1, 2, 1, 5)))
    directions = tuple(np.ones_like(value)*.01 for value in values[:3]) + (None,)
    fn = ts.jit(target="nvidia_sm120")(saved)
    jvp(fn, values, directions)
    child = fn._native_public_jvp_owners[(0, 1, 2)][1]
    package = next(iter(child._native_jvp_packages.values()))
    owner = prepared(package.contract["steps"][0]["child_metadata"])
    inputs = (*values, *directions[:3])
    arrays = tuple(np.full(shape, 934., np.float32) for shape in
                   ((1, 2, 3, 3), (1, 2, 3), (1, 2, 3, 3), (1, 2, 3)))
    pointers = (ct.c_void_p*len(inputs))(*(value.ctypes.data for value in inputs))
    lengths = (ct.c_size_t*len(inputs))(*(value.nbytes for value in inputs))
    outputs = (ct.c_void_p*4)(*(value.ctypes.data for value in arrays))
    sizes = (ct.c_size_t*4)(*(value.nbytes for value in arrays))
    count = 4
    if defect == "count":
        count = 3
    elif defect == "extent":
        sizes[3] -= 4
    elif defect == "overlap":
        outputs[2] = outputs[0]
    else:
        outputs[0] = values[1].ctypes.data
    before = tuple(value.copy() for value in inputs)
    invoke = owner._library().tessera_nvidia_attention_jvp_invoke_lse
    status = invoke(owner.handle, pointers, lengths, len(inputs), outputs, sizes,
                    count, (ct.c_float*2)())
    assert status != 0
    for value in arrays:
        np.testing.assert_array_equal(value, np.full_like(value, 934.))
    for value, reference in zip(inputs, before, strict=True):
        np.testing.assert_array_equal(value, reference)
    fn.close_native_storage()


def test_saved_lse_portable_result_selection_cannot_be_dropped():
    from dataclasses import replace
    from tessera.compiler.native_attention_jvp_runtime import prepared
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070 required")
    values = tuple(np.ones(shape, np.float32) for shape in
                   ((1, 1, 5, 3), (1, 2, 3, 4), (1, 1, 5, 4), (1, 2, 1, 5)))
    fn = ts.jit(target="nvidia_sm120")(saved)
    jvp(fn, values, (*values[:3], None))
    child = fn._native_public_jvp_owners[(0, 1, 2)][1]
    package = next(iter(child._native_jvp_packages.values()))
    program = prepared(package.contract["steps"][0]["child_metadata"]).program
    assert program.saved_lse
    with pytest.raises(ValueError, match="saved-LSE selection differs"):
        replace(program, saved_lse=False).validate()
    fn.close_native_storage()

def test_saved_lse_public_jvp_orders_all_pending_primal_and_tangent_streams(monkeypatch):
    from contextlib import ExitStack
    from tests.device.nvidia.test_ordered_resident_tensor_dag import queue_delayed_upload
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070 required")
    rng = np.random.default_rng(937)
    shapes = ((1, 1, 129, 3), (1, 2, 3, 4), (1, 1, 129, 4), (1, 2, 1, 129))
    values = tuple(rng.normal(0, .2, shape).astype(np.float32) for shape in shapes)
    directions = tuple(rng.normal(0, .1, shape).astype(np.float32) for shape in shapes)
    fn = ts.jit(target="nvidia_sm120")(saved)
    with ExitStack() as stack:
        sessions = tuple(stack.enter_context(NvidiaDeviceSession()) for _ in range(8))
        buffers = tuple(session.upload(np.zeros(shape, np.float32))
                        for session, shape in zip(sessions, (*shapes, *shapes), strict=True))
        for session in sessions:
            assert session.synchronize() == 0
        roots = tuple(Borrowed(buffer) for buffer in buffers)
        jvp(fn, roots[:4], roots[4:])
        completed = []
        callbacks = tuple(queue_delayed_upload(session, buffer, value, session.stream, completed)
                          for session, buffer, value in zip(sessions, buffers, (*values, *directions), strict=True))
        def forbidden(*args, **kwargs):
            pytest.fail("warm resident saved-LSE JVP evaluated or recompiled the frontend")
        child = fn._native_public_jvp_owners[(0, 1, 2, 3)][1]
        with monkeypatch.context() as guard:
            guard.setattr(subprocess, "run", forbidden)
            guard.setattr(child, "_fn", forbidden)
            primal, tangent = jvp(fn, roots[:4], roots[4:])
        assert len(completed) == len(callbacks) == 8
        v, q, k, bias = values
        dv, dq, dk, dbias = directions
        expected = oracle((q, k, v), bias)
        h = 1e-4
        positive = oracle((q.astype(np.float64)+h*dq, k.astype(np.float64)+h*dk,
                           v.astype(np.float64)+h*dv), bias.astype(np.float64)+h*dbias)
        negative = oracle((q.astype(np.float64)-h*dq, k.astype(np.float64)-h*dk,
                           v.astype(np.float64)-h*dv), bias.astype(np.float64)-h*dbias)
        for actual, reference in zip(primal, expected, strict=True):
            np.testing.assert_allclose(actual, reference, rtol=4e-5, atol=3e-6)
        for actual, upper, lower in zip(tangent, positive, negative, strict=True):
            np.testing.assert_allclose(actual, (upper-lower)/(2*h), rtol=4e-5, atol=3e-6)
        fn.close_native_storage()
