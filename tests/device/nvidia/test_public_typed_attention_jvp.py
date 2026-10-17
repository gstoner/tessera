"""Public typed attention JVP compilation, portable replay and resident ownership."""
from pathlib import Path
import os

import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.native_attention_program import NativeAttentionJVPProgram
from tests.device.nvidia.test_public_typed_attention_vjp import function
from tests.device.nvidia.test_saved_lse_half_backward_kernel import compare
from benchmarks.nvidia.benchmark_jit_attention_vjp import download

pytestmark = [pytest.mark.hardware_nvidia, pytest.mark.compiler_route]

def reference(q,k,v,dq,dk,dv,causal=True,bias=None,dbias=None):
 q,k,v,dq,dk,dv=[x.astype(np.float64) for x in (q,k,v,dq,dk,dv)]
 groups=q.shape[1]//k.shape[1]
 k,v,dk,dv=[np.repeat(x,groups,axis=1) for x in (k,v,dk,dv)]
 scale=q.shape[-1]**-.5
 scores=scale*(q@k.swapaxes(-1,-2))
 ds=scale*(dq@k.swapaxes(-1,-2)+q@dk.swapaxes(-1,-2))
 if bias is not None:scores=scores+bias.astype(np.float64)
 if dbias is not None:ds=ds+dbias.astype(np.float64)
 if causal:
  sq,sk=q.shape[2],k.shape[2]
  mask=np.arange(sk)[None,:]<=np.arange(sq)[:,None]+max(sk-sq,0)
  scores=np.where(mask,scores,-np.inf)
 maximum=scores.max(axis=-1,keepdims=True);ex=np.exp(scores-maximum);den=ex.sum(axis=-1,keepdims=True);p=ex/den
 lse=(maximum+np.log(den))[...,0];dlse=np.sum(p*ds,axis=-1)
 return (p@v,lse),((p*(ds-dlse[...,None]))@v+p@dv,dlse)


@pytest.mark.parametrize("storage", ["float16", "bfloat16"])
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("active", [("q", "k", "v"), ("v",), ("q",)])
def test_public_typed_jvp_portable_resident_replay(storage, saved, bias, causal, active, monkeypatch):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    dtype = np.dtype(ml_dtypes.bfloat16 if storage == "bfloat16" else storage)
    rng = np.random.default_rng(20261017)
    values = {n: rng.normal(0, .7, shape).astype(dtype) for n, shape in {
        "q": (1, 2, 3, 16), "k": (1, 1, 5, 16), "v": (1, 1, 5, 16)}.items()}
    if bias:
        values["bias"] = rng.normal(0, .1, (1, 2, 1, 5)).astype(np.float32)
    directions = {n: rng.normal(0, .2, x.shape).astype(x.dtype) for n, x in values.items()}
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=active)(function(saved, bias, causal))
    program = fn.compile_native_attention_jvp(**values, compiler=Path(os.environ["TESSERA_OPT"]),
                                            llvm_bin=Path(os.environ["TESSERA_LLVM_BIN"]))
    program = NativeAttentionJVPProgram.from_json(program.to_json(), expected_digest=program.program_digest)
    def forbidden(*args, **kwargs):
        pytest.fail("portable resident replay attempted compilation")
    monkeypatch.setattr("subprocess.run", forbidden)
    with NvidiaDeviceSession() as session:
        roots = {n: session.upload(x) for n, x in values.items()}
        tangents = tuple(session.upload(directions[n]) for n in active)
        expected = reference(*(values[n] for n in ("q", "k", "v")),
            *(directions[n] if n in active else np.zeros_like(values[n]) for n in ("q", "k", "v")),
            causal=causal, bias=values.get("bias"))
        with program.capture(**roots) as frame:
            result = frame.jvp(*tangents)
            result = result if saved else (result,)
            wanted = expected[1] if saved else expected[1][:1]
            held = tuple(download(session, item) for item in result)
            for actual, target in zip(held, wanted, strict=True):
                compare(actual, target)
            # A second derivative call must not overwrite the retained first result.
            frame.jvp(*tangents)
            for prior, item in zip(held, result, strict=True):
                np.testing.assert_array_equal(download(session, item), prior)


@pytest.mark.parametrize("storage", ["float16", "bfloat16"])
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("subset", [False, True])
def test_public_typed_jvp_ordinary_prepared_host_route(storage, saved, bias, subset, monkeypatch):
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    dtype = np.dtype(ml_dtypes.bfloat16 if storage == "bfloat16" else storage)
    rng = np.random.default_rng(20261017)
    values = {n: rng.normal(0, .7, shape).astype(dtype) for n, shape in {
        "q": (1, 2, 3, 16), "k": (1, 1, 5, 16), "v": (1, 1, 5, 16)}.items()}
    if bias:
        values["bias"] = rng.normal(0, .1, (1, 2, 1, 5)).astype(np.float32)
    active = (("bias", "q") if bias else ("v",)) if subset else tuple(values)
    directions = {n: rng.normal(0, .2, x.shape).astype(x.dtype) if n in active
                  else np.zeros_like(x) for n, x in values.items()}
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=active)(function(saved, bias, True))
    def forbidden(*args, **kwargs):
        pytest.fail("warm prepared JVP attempted compilation or concrete frontend evaluation")
    try:
        first = fn.native_jvp(**values, tangents=tuple(directions[n] for n in active))
        held = tuple(tuple(x.copy() for x in value) if saved else value.copy() for value in first)
        for generation in (1, 2):
            current = {n: x*generation for n, x in values.items()}
            expected = reference(*(current[n] for n in ("q", "k", "v")),
                *(directions[n] for n in ("q", "k", "v")),
                bias=current.get("bias"), dbias=directions.get("bias"))
            with monkeypatch.context() as guard:
                guard.setattr("subprocess.run", forbidden)
                guard.setattr(fn, "_fn", forbidden)
                guard.setattr(fn, "_trace_frontend_capture", forbidden)
                primal, tangent = fn.native_jvp(**current, tangents=tuple(directions[n] for n in active))
            for actuals, targets in zip((primal, tangent), expected, strict=True):
                actuals = actuals if saved else (actuals,)
                targets = targets if saved else targets[:1]
                for i, (actual, target) in enumerate(zip(actuals, targets, strict=True)):
                    assert actual.dtype == (np.dtype("float32") if i else dtype)
                    compare(actual, target)
            for actual, prior in zip(first, held, strict=True):
                for x, y in zip(actual if saved else (actual,), prior if saved else (prior,), strict=True):
                    np.testing.assert_array_equal(x, y)
        assert fn.last_jvp_execution["execution_kind"] == "native_gpu"
    finally:
        fn.close_native_storage()


@pytest.mark.parametrize("storage", ["float16", "bfloat16"])
@pytest.mark.parametrize("saved", [False, True])
def test_public_typed_jvp_orders_pending_producer_stream(storage, saved, monkeypatch):
    from contextlib import ExitStack
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
            fn.native_jvp(**borrowed, tangents=directions)
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
                result = fn.native_jvp(**borrowed, tangents=directions)
            assert held and callback is not None
            expected = reference(*values.values(), *seeds.values())
            for actuals, targets in zip(result, expected, strict=True):
                for actual, target in zip(actuals if saved else (actuals,), targets if saved else targets[:1], strict=True):
                    compare(actual, target)
            assert fn.last_jvp_execution["host_preparation"] == "native_ordered_resident_snapshot"
    finally:
        fn.close_native_storage()
