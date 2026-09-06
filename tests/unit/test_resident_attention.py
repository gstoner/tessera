"""Physical policy mutations must fail before CUDA is touched."""
from dataclasses import replace
import pytest
from tessera.compiler import nvidia_native as native
from tessera.compiler.resident_attention import checkpoint_shapes
from tessera.compiler.scheduled_matmul import find_tessera_opt
from test_nvidia_checkpoint_pair import checkpoint_modules


@pytest.fixture
def pair(monkeypatch):
    if find_tessera_opt() is None:
        pytest.skip('requires native scheduling compiler')
    monkeypatch.setattr(native,'_compile_tile_ir',lambda text,entry:(text,'// PTX',{},'compiler','toolchain',(),'cold'))
    return native.package_attention_checkpoint_pair(*checkpoint_modules(),pipeline_name='tessera-nvidia-pipeline-sm120')


def test_resident_checkpoint_shapes(pair):
    dims,shapes=checkpoint_shapes(pair)
    assert dims==(1,2,1,3,4,4,3)
    assert shapes[-1]==(1,2,3)


@pytest.mark.parametrize('mutation',['identity','shape','abi','direction','geometry','scale','guard','scalar','huge_scale','tiny_scale','huge_dimension'])
def test_resident_checkpoint_rejects_changed_contract_before_driver(pair,monkeypatch,mutation):
    import ctypes
    monkeypatch.setattr(ctypes,'CDLL',lambda *a,**k:pytest.fail('loaded CUDA before validation'))
    package=pair.backward
    desc=package.descriptor
    if mutation=='identity':
        pair=replace(pair,contract_digest='0'*64)
    elif mutation in ('shape','scale','huge_scale','tiny_scale','huge_dimension'):
        provenance=dict(desc.provenance)
        key='shape' if mutation in ('shape','huge_dimension') else 'scale'
        provenance[key]={'shape':[1,2,1,4,4,4,3], 'scale':True,
                         'huge_scale':1e100, 'tiny_scale':1e-100,
                         'huge_dimension':[1,2,1,1 << 64,4,4,3]}[mutation]
        desc=replace(desc,provenance=provenance)
    elif mutation=='abi':
        desc=replace(desc,abi_id='wrong')
    elif mutation=='direction':
        desc=replace(desc,buffers=(replace(desc.buffers[0],direction='output'),*desc.buffers[1:]))
    elif mutation=='geometry':
        desc=replace(desc,geometry=replace(desc.geometry,policy='runtime_default'))
    elif mutation=='guard':
        desc=replace(desc,shape_guards=desc.shape_guards[1:])
    elif mutation=='scalar':
        desc=replace(desc,scalars=(replace(desc.scalars[0],name='unexpected'),*desc.scalars[1:]))
    pair=replace(pair,backward=replace(package,descriptor=desc))
    with pytest.raises((ValueError,RuntimeError)):
        pair.capture(None,None,None)


def test_failed_backward_releases_only_new_generation_allocations():
    import ctypes as ct
    import threading
    from types import SimpleNamespace
    from tessera.compiler.resident_attention import ResidentAttentionTape
    frame=ResidentAttentionTape.__new__(ResidentAttentionTape)
    frame._lock=threading.RLock()
    frame._ready=lambda:None
    frame._resident=lambda value,shape:512
    frame.shapes=((1,1,2,2),)*4+((1,1,2),)
    prior=SimpleNamespace(pointer=ct.c_void_p(128))
    frame.buffers=[prior]
    frame._saved=[prior]*5
    freed=[]
    def alloc(destination,nbytes):
        destination._obj.value=1024+len(frame.buffers)*256
        return 0
    frame.alloc=alloc
    frame.free=lambda pointer:freed.append(pointer.value) or 0
    frame.sync=lambda:0
    frame.copy=lambda *args:0
    def failed(*args):
        raise RuntimeError('injected launch failure')
    frame._execute=failed
    with pytest.raises(RuntimeError,match='injected'):
        frame.backward(None)
    assert frame.buffers==[prior]
    assert len(freed)==4 and len(set(freed))==4
    assert prior.pointer.value==128


def test_resident_frame_rejects_context_change_before_allocating():
    import ctypes as ct
    import threading
    from tessera.compiler.resident_attention import ResidentAttentionTape
    frame=ResidentAttentionTape.__new__(ResidentAttentionTape)
    frame._lock=threading.RLock()
    frame.closed=False
    frame.context=ct.c_void_p(128)
    def current(destination):
        destination._obj.value=256
        return 0
    frame._current=current
    frame.alloc=lambda *a:pytest.fail('allocated in another CUDA context')
    with pytest.raises(ValueError,match='owning CUDA context'):
        frame.backward(None)
    with pytest.raises(ValueError,match='owning CUDA context'):
        frame.close()


def test_backward_launch_covers_concatenated_gradient_ranges():
    import ctypes as ct
    from types import SimpleNamespace
    from tessera.compiler.resident_attention import ResidentAttentionTape
    frame=ResidentAttentionTape.__new__(ResidentAttentionTape)
    frame.dims=(1,2,1,3,129,4,3)
    frame._functions=(ct.c_void_p(1),ct.c_void_p(2))
    launches=[]
    frame._launch=lambda *args:launches.append(args) or 0
    frame.sync=lambda:0
    frame._execute(True,[SimpleNamespace(pointer=ct.c_void_p(i+1)) for i in range(8)])
    assert launches[0][1]==8  # ceil((24 + 516 + 387) / 128), not max/128.
