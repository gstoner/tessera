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
        assert kwargs=={'stream':1,'snapshot':True}
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
