"""Typed bounded saved-LSE generations, native reverse and retained CUDA storage."""
import os
import subprocess

import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.native_attention_program import NativeAttentionVJPProgram
from benchmarks.nvidia.benchmark_jit_attention_vjp import download
from tests.device.nvidia.test_public_typed_attention_vjp import function
from tests.device.nvidia.test_saved_lse_half_backward_kernel import reference,compare

pytestmark=pytest.mark.hardware_nvidia


@pytest.mark.parametrize('storage',[np.float16,ml_dtypes.bfloat16])
@pytest.mark.parametrize('saved',[False,True])
@pytest.mark.parametrize('compact',[False,True])
@pytest.mark.parametrize('bias_kind',['none','query','key'])
@pytest.mark.parametrize('asynchronous',[False,True])
def test_typed_capture_replays_bounded_shapes_and_retains_gradients(storage,saved,compact,bias_kind,asynchronous,monkeypatch):
    if rt._nvidia_device_name()!='sm_120':
        pytest.skip('owning RTX5070 required')
    biased=bias_kind!='none'
    def inputs(sq,sk):
        rng=np.random.default_rng(20261017+sq*13+sk)
        roots={name:rng.normal(0,.2,shape).astype(storage) for name,shape in {
            'q':(1,2,sq,8),'k':(1,1,sk,8),'v':(1,1,sk,6)}.items()}
        if biased:
            shape=(1,2,1,sk) if bias_kind=='query' else (1,2,sq,1)
            roots['bias']=rng.normal(0,.1,shape).astype(np.float32)
        do=rng.normal(0,.1,(1,2,sq,6)).astype(storage)
        seed=rng.normal(0,.1,(1,2,sq)).astype(np.float32) if saved else np.zeros((1,2,sq),np.float32)
        return roots,do,seed
    trace=(3,1) if bias_kind=='key' else (1,4)
    initial,_,_=inputs(*trace)
    wrt=('bias','v','q','k') if biased else ('v','q','k')
    fn=ts.jit(target='nvidia_sm120',autodiff='reverse',wrt=wrt)(function(saved,biased,True))
    program=fn.compile_native_attention_vjp(**initial,compiler=os.environ['TESSERA_OPT'],
        sequence_bounds=(9,11),compact_gradients=compact)
    program=NativeAttentionVJPProgram.from_json(program.to_json(),expected_digest=program.program_digest)
    def forbidden(*args,**kwargs):
        pytest.fail('compiler reentry during typed resident capture/reverse')
    monkeypatch.setattr(subprocess,'run',forbidden)
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    with NvidiaDeviceSession() as session:
        for sq,sk in (trace,(3,5),(9,11),(9,1)):
            roots,do,seed=inputs(sq,sk)
            expected,gradients=reference(roots['q'],roots['k'],roots['v'],do,roots.get('bias'),seed,causal=True)
            ordered=[session.upload(roots[name]) for name in fn.arg_names]
            with program.capture(*ordered,asynchronous=asynchronous) as frame:
                frame.wait_on(session.stream)
                primal=frame.primal if saved else (frame.primal,)
                for actual,wanted in zip(primal,expected if saved else expected[:1],strict=True):
                    compare(download(session,actual),wanted)
                seeds=(session.upload(do),session.upload(seed)) if saved else session.upload(do)
                first=frame.backward(seeds)
                frame.wait_on(session.stream)
                held=[download(session,value) for value in first]
                for actual,role in zip(held,program.active,strict=True):compare(actual,gradients[role])
                second=frame.backward(seeds)
                frame.wait_on(session.stream)
                for prior,snapshot,new in zip(first,held,second,strict=True):
                    np.testing.assert_array_equal(download(session,prior),snapshot)
                    np.testing.assert_array_equal(download(session,new),snapshot)
            with pytest.raises(ValueError,match='closed'):
                _=first[0].__cuda_array_interface__


@pytest.mark.parametrize('storage',[np.float16,ml_dtypes.bfloat16])
@pytest.mark.parametrize('asynchronous',[False,True])
def test_typed_capture_orders_producer_writes_and_owns_saved_inputs(storage,asynchronous):
    from tests.device.nvidia.test_ordered_resident_tensor_dag import queue_delayed_upload
    if rt._nvidia_device_name()!='sm_120':
        pytest.skip('owning RTX5070 required')
    rng=np.random.default_rng(202610172)
    roots={name:rng.normal(0,.2,shape).astype(storage) for name,shape in {
        'q':(1,2,3,8),'k':(1,1,5,8),'v':(1,1,5,6)}.items()}
    do=rng.normal(0,.1,(1,2,3,6)).astype(storage)
    seed=rng.normal(0,.1,(1,2,3)).astype(np.float32)
    changed=(roots['q'].astype(np.float32)*.7).astype(storage)
    fn=ts.jit(target='nvidia_sm120',autodiff='reverse',wrt=('q','k','v'))(function(True,False,True))
    program=fn.compile_native_attention_vjp(**roots,compiler=os.environ['TESSERA_OPT'],sequence_bounds=(9,11))
    expected,gradients=reference(changed,roots['k'],roots['v'],do,None,seed,causal=True)
    with NvidiaDeviceSession() as session:
        resident=[session.upload(roots[name]) for name in fn.arg_names]
        completed=[]
        before=queue_delayed_upload(session,resident[0],changed,session.stream,completed)
        with program.capture(*resident,asynchronous=asynchronous) as frame:
            after=queue_delayed_upload(session,resident[0],np.zeros_like(changed),session.stream,completed)
            frame.wait_on(session.stream)
            assert before is not None and after is not None
            for actual,wanted in zip(frame.primal,expected,strict=True):compare(download(session,actual),wanted)
            got=frame.backward((session.upload(do),session.upload(seed)))
            frame.wait_on(session.stream)
            for actual,wanted in zip(got,gradients,strict=True):compare(download(session,actual),wanted)


@pytest.mark.parametrize('storage',[np.float16,ml_dtypes.bfloat16])
def test_typed_forward_only_uses_verified_storage_without_reverse_image(storage):
    from tessera.compiler.nvidia_native import AttentionForwardCheckpoint
    if rt._nvidia_device_name()!='sm_120':
        pytest.skip('owning RTX5070 required')
    rng=np.random.default_rng(202610173)
    roots=[rng.normal(0,.2,shape).astype(storage) for shape in ((1,2,3,8),(1,1,5,8),(1,1,5,6))]
    fn=ts.jit(target='nvidia_sm120',autodiff='reverse',wrt=('q','k','v'))(function(False,False,True))
    program=fn.compile_native_attention_vjp(*roots,compiler=os.environ['TESSERA_OPT'],sequence_bounds=(9,11))
    forward=AttentionForwardCheckpoint(program.pair.forward,program.pair.contract_digest)
    expected,_=reference(*roots,np.zeros((1,2,3,6),storage),None,np.zeros((1,2,3),np.float32))
    with NvidiaDeviceSession() as session:
        with forward.capture(*(session.upload(value) for value in roots)) as frame:
            compare(download(session,frame.primal),expected[0])
            with pytest.raises(ValueError,match='no reverse executable'):
                frame.backward(None)


@pytest.mark.parametrize('storage_name,storage',[('f16',np.float16),('bf16',ml_dtypes.bfloat16)])
def test_authored_fp32_output_keeps_half_primal_and_gradient_storage(storage_name,storage):
    from tests.unit.test_saved_lse_half_backward import source
    from tessera.compiler.native_attention_program import compile_attention_vjp_program
    if rt._nvidia_device_name()!='sm_120':
        pytest.skip('owning RTX5070 required')
    text=source(storage_name,saved=True,output='f32').replace(
        'tessera.arch = "sm_120"','tessera.arch = "sm_120", tessera.attention_sequence_bounds = array<i64: 9, 11>')
    program=compile_attention_vjp_program(text,(0,1,2),compiler=os.environ['TESSERA_OPT'])
    rng=np.random.default_rng(202610174)
    roots=[rng.normal(0,.2,shape).astype(storage) for shape in ((1,2,7,4),(1,1,9,4),(1,1,9,3))]
    do=rng.normal(0,.1,(1,2,7,3)).astype(np.float32)
    seed=rng.normal(0,.1,(1,2,7)).astype(np.float32)
    expected,gradients=reference(*roots,do,None,seed)
    with NvidiaDeviceSession() as session:
        with program.capture(*(session.upload(value) for value in roots)) as frame:
            assert all(value.dtype==np.dtype(np.float32) for value in frame.primal)
            for actual,wanted in zip(frame.primal,expected,strict=True):compare(download(session,actual),wanted)
            got=frame.backward((session.upload(do),session.upload(seed)))
            for actual,wanted in zip(got,gradients,strict=True):
                assert actual.dtype==np.dtype(storage)
                compare(download(session,actual),wanted)
