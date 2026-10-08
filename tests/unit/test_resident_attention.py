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


@pytest.mark.parametrize("bias_gradient", [False, True])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("lse_cotangent", [False, True])
def test_failed_backward_releases_only_new_generation_allocations(bias_gradient, compact, lse_cotangent):
    import ctypes as ct
    import threading
    from types import SimpleNamespace
    from tessera.compiler.resident_attention import ResidentAttentionTape
    frame=ResidentAttentionTape.__new__(ResidentAttentionTape)
    frame.asynchronous=False
    frame._pending_sources=[]
    frame._consumer_streams=set()
    frame._lock=threading.RLock()
    frame._ready=lambda:None
    frame._resident=lambda value,shape,**kwargs:512
    frame.shapes=((1,1,2,2),)*4+((1,1,2),)
    prior=SimpleNamespace(pointer=ct.c_void_p(128))
    frame.buffers=[prior]
    frame._saved=[prior]*5
    frame._has_backward=True
    frame._has_lse_cotangent=lse_cotangent
    frame._has_bias=bias_gradient
    frame._has_bias_gradient=bias_gradient
    frame._gradient_roles=(2,) if compact else tuple(range(3+int(bias_gradient)))
    frame._gradient_launch="logical_v1"
    frame._bias=prior
    frame.dims=(1,1,1,2,2,2,2)
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
        frame.backward((None,None) if lse_cotangent else None)
    assert frame.buffers==[prior]
    expected=(2 if compact else 4+int(bias_gradient))+int(lse_cotangent)
    assert len(freed)==expected and len(set(freed))==expected
    assert prior.pointer.value==128


def test_resident_frame_rejects_context_change_before_allocating():
    import ctypes as ct
    import threading
    from tessera.compiler.resident_attention import ResidentAttentionTape
    frame=ResidentAttentionTape.__new__(ResidentAttentionTape)
    frame.asynchronous=False
    frame._pending_sources=[]
    frame._consumer_streams=set()
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
    frame.asynchronous=False
    frame._pending_sources=[]
    frame._consumer_streams=set()
    frame.dims=(1,2,1,3,129,4,3)
    frame._backward_threads=128
    frame._backward_scalars=frame.dims
    frame._has_bias_gradient=False
    frame.shapes=((1,2,3,4),(1,1,129,4),(1,1,129,3))
    frame._gradient_roles=(0,1,2)
    frame._gradient_launch="logical_v1"
    frame._functions=(ct.c_void_p(1),ct.c_void_p(2))
    frame._stream=ct.c_void_p(777)
    launches=[]
    frame._launch=lambda *args:launches.append(args) or 0
    frame.sync=lambda:0
    frame._execute(True,[SimpleNamespace(pointer=ct.c_void_p(i+1)) for i in range(8)])
    assert launches[0][1]==8  # ceil((24 + 516 + 387) / 128), not max/128.

def test_bias_gradient_launch_covers_the_fourth_output_range():
    import ctypes as ct
    from types import SimpleNamespace
    from tessera.compiler.resident_attention import ResidentAttentionTape
    frame=ResidentAttentionTape.__new__(ResidentAttentionTape)
    frame.asynchronous=False
    frame._pending_sources=[]
    frame._consumer_streams=set()
    frame.dims=(1,2,1,3,129,4,3)
    frame._backward_threads=128
    frame._backward_scalars=frame.dims
    frame._has_bias_gradient=True
    frame.shapes=((1,2,3,4),(1,1,129,4),(1,1,129,3))
    frame._bias_shape=(1,2,3,129)
    frame._gradient_roles=(0,1,2,3)
    frame._gradient_launch="logical_v1"
    frame._functions=(ct.c_void_p(1),ct.c_void_p(2))
    frame._stream=ct.c_void_p(777)
    launches=[]
    frame._launch=lambda *args:launches.append(args) or 0
    frame.sync=lambda:0
    frame._execute(True,[SimpleNamespace(pointer=ct.c_void_p(i+1)) for i in range(11)])
    assert launches[0][1]==14  # ceil((24 + 516 + 387 + 774)/128).

@pytest.mark.parametrize("launch,blocks", [("packed_v1",4),("logical_v1",8)])
@pytest.mark.parametrize("threads", [64,128])
def test_compact_value_gradient_launch_layouts_keep_one_output(launch,blocks,threads):
    import ctypes as ct
    from types import SimpleNamespace
    from tessera.compiler.resident_attention import ResidentAttentionTape
    frame=ResidentAttentionTape.__new__(ResidentAttentionTape)
    frame.asynchronous=False
    frame._pending_sources=[]
    frame._consumer_streams=set()
    frame.dims=(1,2,1,3,129,4,3)
    frame._backward_threads=threads
    frame._backward_scalars=frame.dims
    frame.shapes=((1,2,3,4),(1,1,129,4),(1,1,129,3))
    frame._has_bias_gradient=False
    frame._gradient_roles=(2,)
    frame._gradient_launch=launch
    frame._functions=(ct.c_void_p(1),ct.c_void_p(2))
    frame._stream=ct.c_void_p(777)
    launches=[]
    frame._launch=lambda *args:launches.append(args) or 0
    frame.sync=lambda:0
    frame._execute(True,[SimpleNamespace(pointer=ct.c_void_p(i+1)) for i in range(7)])
    assert launches[0][1]==(blocks if threads == 128 else (7 if launch == "packed_v1" else 15))
    assert launches[0][4]==threads

def test_forward_checkpoint_refuses_reverse_before_buffer_access():
    import threading
    from tessera.compiler.resident_attention import ResidentAttentionTape
    frame=ResidentAttentionTape.__new__(ResidentAttentionTape)
    frame.asynchronous=False
    frame._pending_sources=[]
    frame._consumer_streams=set()
    frame._lock=threading.RLock()
    frame._ready=lambda:None
    frame._has_backward=False
    frame._resident=lambda *a:pytest.fail("touched buffer before reverse capability check")
    with pytest.raises(ValueError,match="no reverse executable"):
        frame.backward(None)

def test_forward_checkpoint_preserves_shape_validation(pair):
    from tessera.compiler.nvidia_native import AttentionForwardCheckpoint
    forward=AttentionForwardCheckpoint(pair.forward,pair.contract_digest)
    assert checkpoint_shapes(forward)==checkpoint_shapes(pair)
    with pytest.raises(ValueError,match="identity"):
        checkpoint_shapes(replace(forward,contract_digest="0"*64))
