"""A source reverse product must not bypass its exception-bearing forward."""
import json
from types import SimpleNamespace
import pytest
from tessera.compiler.native_public_result import NativeSourceVJP


def program(role,pair='same'):
    values={'tessera.source_state':json.dumps({'result_count':1,'groups':[]}),
            'tessera.autodiff.product_abi':json.dumps({'role':role}),
            'tessera.autodiff.product_pair':pair}
    return SimpleNamespace(source='module attributes {'+', '.join(k+' = '+json.dumps(v) for k,v in values.items())+'} {}')


def test_exception_failure_prevents_backward_construction(monkeypatch):
    import tessera.compiler.native_public_result as module
    failure=ValueError('forward failed')
    def failed(*args,**kwargs):
        assert kwargs['snapshot'] is True
        raise failure
    monkeypatch.setattr(module,'PublicResultFrame',failed)
    with pytest.raises(ValueError) as caught:
        NativeSourceVJP(program('forward'),program('backward')).run(object(),cotangents=(object(),))
    assert caught.value is failure


def test_mismatched_pair_refuses_before_any_device_work(monkeypatch):
    import tessera.compiler.native_public_result as module
    def unexpected(*args,**kwargs):raise AssertionError('device work started')
    monkeypatch.setattr(module,'PublicResultFrame',unexpected)
    with pytest.raises(ValueError,match='generated pair'):
        NativeSourceVJP(program('forward'),program('backward','other')).run(object(),cotangents=(object(),))


def test_public_scalar_residual_keeps_integer_storage():
    import os
    from pathlib import Path
    import numpy as np
    from tessera.compiler.trace import trace
    from tessera.compiler.source_control_flow import to_native_source_ir
    from tessera.compiler.native_gpu_storage import _run
    from tessera.compiler.native_public_result import _prepare
    compiler=Path(os.environ.get('TESSERA_OPT','/missing'))
    if not compiler.is_file():pytest.skip('native compiler required')
    def branch(x):
        if x<x+x:return x*x
        return x+x
    source=to_native_source_ir(trace(branch,np.ones(1,np.float32),source_control_flow=True),autodiff='reverse')
    exported=_run(compiler,'--tessera-autodiff-paired=box-product-scalars=true export-product=forward',source=source)
    native=_run(compiler,'--tessera-to-linalg',source=exported)
    buffered=_run(Path('/usr/lib/llvm-23/bin/mlir-opt'),'--allow-unregistered-dialect','--convert-elementwise-to-linalg',
        '--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map',
        '--convert-linalg-to-loops','--canonicalize',source=native)
    _,metadata,_=_prepare(buffered,compiler,'nvidia',4)
    scalar=metadata['results'][-1]
    assert scalar['rank']==0 and scalar['capacity']==1
    assert metadata['arguments'][scalar['data']]['storage']=='i8'


def test_async_vjp_waits_for_forward_before_enqueuing_backward(monkeypatch):
    import tessera.compiler.native_public_result as module
    calls=[]
    ready=iter((False,True))
    done=iter((False,True))
    forward=SimpleNamespace(poll=lambda:next(ready),results=('primal','residual'),close=lambda:calls.append('close-forward'))
    backward=SimpleNamespace(poll=lambda:next(done),results=('derivative',),close=lambda:calls.append('close-backward'))
    def submit(*args):
        calls.append('backward')
        return backward
    def factory(*args,**kwargs):
        assert kwargs=={'stream':1,'snapshot':True,'scoped':False}
        return forward
    monkeypatch.setattr(module,'PublicResultFrame',factory)
    program=SimpleNamespace(_validate_pair=lambda seeds:({'primal_results':1},1),_forward=object(),
        _backward=SimpleNamespace(submit=submit),_backward_inputs=lambda *args:('saved-input','seed','residual'))
    frame=module.AsyncSourceVJPFrame(program,1,(object(),),(object(),))
    assert not frame.poll() and calls==[]
    assert not frame.poll() and calls==['backward']
    assert not hasattr(frame,'derivatives')
    assert frame.poll() and frame.derivatives==('derivative',)
    frame.close()
    assert calls==['backward','close-backward','close-forward']


def test_async_vjp_failure_retains_identity_without_backward(monkeypatch):
    import tessera.compiler.native_public_result as module
    error=ValueError('failed forward')
    def fail():raise error
    monkeypatch.setattr(module,'PublicResultFrame',lambda *args,**kwargs:SimpleNamespace(poll=fail,close=lambda:None))
    program=SimpleNamespace(_validate_pair=lambda seeds:({'primal_results':1},1),_forward=object())
    frame=module.AsyncSourceVJPFrame(program,1,(object(),),(object(),))
    for _ in range(3):
        with pytest.raises(ValueError) as caught:frame.poll()
        assert caught.value is error and frame._backward is None
        assert not hasattr(frame,'derivatives')
    frame.close()


def test_scoped_vjp_retirement_never_uses_synchronous_close(monkeypatch):
    import threading
    import tessera.compiler.native_public_result as module
    calls=[]
    ready=iter((False,True))
    def child(name,poll):
        result = SimpleNamespace(_owner=SimpleNamespace(_active=0), _retiring=False,
            poll_retired=poll, close=lambda:pytest.fail('synchronous close used'))
        def retire(stream):
            calls.append((name,stream))
            result._retiring = True
        result.retire = retire
        return result
    frame=object.__new__(module.AsyncSourceVJPFrame)
    frame._lock=threading.RLock()
    frame._scoped=True;frame._retiring=False;frame.closed=False
    frame._forward=child('forward',lambda:True)
    frame._backward=child('backward',lambda:next(ready))
    frame._cotangents=(object(),)
    with pytest.raises(ValueError,match='stream'):frame.retire(0)
    assert not frame._retiring
    frame._forward._owner._active=1
    with pytest.raises(ValueError,match='reader'):frame.retire(7)
    assert not calls and not frame._retiring
    frame._forward._owner._active=0
    frame.retire(7)
    assert calls==[('backward',7),('forward',7)]
    assert not frame.poll_retired() and frame._cotangents
    assert frame.poll_retired() and frame.closed and not frame._cotangents


def test_gpu_custom_class_requires_binding_before_compilation(monkeypatch):
    import tessera.compiler.native_public_result as module
    class DomainError(RuntimeError):pass
    contract={'schema':1,'error_specs':[[[1],'f32']],'arguments':[{}],
              'groups':[],'error_table':[['DomainError',['bad']]]}
    source='module attributes {tessera.source_state = '+json.dumps(json.dumps(contract))+'} {}'
    calls=[]
    def compile_product(*args,**kwargs):
        calls.append(1)
        raise RuntimeError('compiler reached')
    monkeypatch.setattr(module,'_materialize_ad_product',compile_product)
    with pytest.raises(ValueError,match='explicit host binding'):
        module.materialize_source_vjp(source,compiler='/missing',llvm_bin='/missing',backend='nvidia',chip='sm_120',capacity=4)
    assert not calls
    with pytest.raises(RuntimeError,match='compiler reached'):
        module.materialize_source_vjp(source,compiler='/missing',llvm_bin='/missing',backend='nvidia',chip='sm_120',capacity=4,exception_types={'DomainError':DomainError})
    assert calls==[1]


def test_repoll_preserves_real_host_failure_frames_without_growth(monkeypatch):
    import tessera.compiler.native_public_result as module
    failure=RuntimeError('host completion constructor failed')
    def constructor_frame():raise failure
    forward=SimpleNamespace(poll=constructor_frame)
    monkeypatch.setattr(module,'PublicResultFrame',lambda *args,**kwargs:forward)
    program=SimpleNamespace(_validate_pair=lambda seeds:({'primal_results':1},1),_forward=object())
    frame=module.AsyncSourceVJPFrame(program,1,(object(),),(object(),))
    lengths=[]
    for _ in range(8):
        with pytest.raises(RuntimeError) as caught:frame.poll()
        assert caught.value is failure
        tb=caught.value.__traceback__;codes=[]
        while tb is not None:
            codes.append(tb.tb_frame.f_code);tb=tb.tb_next
        assert constructor_frame.__code__ in codes
        lengths.append(len(codes))
    assert len(set(lengths))==1
