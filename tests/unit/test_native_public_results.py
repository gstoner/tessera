"""Native shape-sidecar and nested-gate projection boundaries."""
import pytest
from benchmarks.record_deep_native_contracts import public_source
from tessera.compiler.native_public_result import _prepare
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt


def compiler():
    tool=find_tessera_opt()
    if tool is None:pytest.skip('native compiler required')
    return tool


@pytest.mark.parametrize('backend',['nvidia','rocm'])
def test_native_result_shapes_and_nested_scalar_exits(backend):
    text,contract,specs=_prepare(public_source(),compiler(),backend)
    assert contract['results']==[{'data':1,'shape':2,'capacity':8}]
    assert contract['arguments'][0]['writable'] is False
    assert 'cf.assert' not in text
    assert 'scf.if' in text and len(specs)==5


@pytest.mark.parametrize('replacement',['tessera.result_shape = 1 : i64','tessera.result_shape = 9 : i64'])
def test_invalid_shape_slot_refuses(replacement):
    with pytest.raises(RuntimeError):
        run_tessera_opt(compiler(),public_source().replace('tessera.result_shape = 2 : i64',replacement),'--tessera-native-tape-to-gpu=status-buffer=true')


def test_public_output_requires_checked_status():
    with pytest.raises(RuntimeError):run_tessera_opt(compiler(),public_source(),'--tessera-native-tape-to-gpu')


def test_public_input_cannot_be_written_behind_readonly_projection():
    source=public_source().replace('memref.store %v, %out[%count]', 'memref.store %v, %x[%count]')
    assert source != public_source()
    with pytest.raises(RuntimeError):
        run_tessera_opt(compiler(),source,'--tessera-native-tape-to-gpu=status-buffer=true')


def test_asynchronous_result_does_not_expose_uncompleted_or_failed_output():
    import ctypes as ct
    from types import SimpleNamespace
    from tessera.compiler.native_public_result import PublicResultFrame
    frame=object.__new__(PublicResultFrame)
    import threading
    frame._lock=threading.RLock()
    frame.closed=False
    frame.context_type=ct.c_int
    frame.context=ct.c_int(0)
    frame.check=lambda status: None
    frame.current=lambda context: 0
    ticket=SimpleNamespace(poll=lambda: False)
    frame._submission=SimpleNamespace(ticket=ticket)
    frame._status=object()
    frame._integer=lambda status: 1
    assert not frame.poll()
    assert not hasattr(frame,'results')
    ticket.poll=lambda: True
    with pytest.raises(RuntimeError,match='guard failed'):
        frame.poll()
    assert not hasattr(frame,'results')


@pytest.mark.parametrize("streams", [(21,), (21, 23)])
def test_scoped_dynamic_public_results_require_checked_reader_scopes(streams):
    import ctypes as ct
    from types import SimpleNamespace
    from tests.unit.test_native_reader_retirement import setup
    from tessera.compiler.native_public_result import PublicResultFrame
    owner, native = setup()
    frame = object.__new__(PublicResultFrame)
    frame.__dict__.update(vars(owner.frame))
    del frame._ready  # Exercise the public frame's real lifecycle guard.
    owner.frame = frame
    frame.closed = False
    frame._scoped, frame._retiring, frame._owner = True, False, owner
    frame.context_type, frame.context = ct.c_int, ct.c_int(0)
    frame.current = lambda _: 0
    frame._submission = owner._submission
    frame._status = object()
    frame._integer = lambda _: 0
    frame._integers = lambda *args: (2,)
    frame._arguments = list(owner._buffers)
    frame._metadata = {'results': [{'data': 0, 'shape': 1, 'capacity': 4}]}
    frame.binding = SimpleNamespace(close_if_complete=lambda **kwargs: True)
    frame.program = SimpleNamespace(source='module {}')
    with pytest.raises(ValueError, match='successful completion'):
        frame.read(21)
    assert frame.poll() and not hasattr(frame, 'results')
    with frame.read_many(*streams) as readers:
        assert owner._active == len(streams)
        borrowed = readers[streams[-1]][0]
        assert borrowed.__cuda_array_interface__['shape'] == (2,)
        with pytest.raises(ValueError, match='scopes'):
            frame.retire(22)
    with pytest.raises(ValueError, match='lease is closed'):
        _ = borrowed.__cuda_array_interface__
    frame.retire(22)
    assert frame.poll_retired() and frame.closed and not frame.buffers


@pytest.mark.parametrize('root,admitted',[('%x',False),('%out',True)])
def test_public_view_write_preserves_root_ownership(root,admitted):
    source=public_source().replace('memref.store %v, %out[%count] : memref<8xf32>',
        f'%view = memref.subview {root}[0] [8] [1] : memref<8xf32> to memref<8xf32, strided<[1]>>\n'
        '            memref.store %v, %view[%count] : memref<8xf32, strided<[1]>>')
    assert source!=public_source()
    if admitted:
        assert run_tessera_opt(compiler(),source,'--tessera-native-tape-to-gpu=status-buffer=true')
    else:
        with pytest.raises(RuntimeError):run_tessera_opt(compiler(),source,'--tessera-native-tape-to-gpu=status-buffer=true')


def test_repolling_exception_does_not_accumulate_host_tracebacks():
    import ctypes as ct
    import json
    from types import SimpleNamespace
    from tessera.compiler.native_public_result import PublicResultFrame
    contract={'error_specs':[[[1],'f32']],'error_dynamic':False}
    frame=object.__new__(PublicResultFrame)
    frame.program=SimpleNamespace(source='module attributes {tessera.source_state = '+json.dumps(json.dumps(contract))+'} {}')
    frame._status=None
    frame._integer=lambda _:0
    frame._integers=lambda *args:(1,)
    frame._metadata={'results':[{'data':0,'shape':1,'capacity':1}]}
    frame._arguments=[SimpleNamespace(pointer=ct.c_void_p(1),__cuda_array_interface__={'shape':(1,),'typestr':'<f4','data':(1,True),'version':3}),None]
    frame.copy_out=lambda *args:0
    frame.check=lambda status:None
    frame._source_exception=ValueError('same native failure')
    lengths=[]
    for _ in range(10):
        try:frame._expose()
        except ValueError as error:
            assert error is frame._source_exception
            tb=error.__traceback__;count=0
            while tb is not None:count+=1;tb=tb.tb_next
            lengths.append(count)
        else:raise AssertionError('failed frame exposed results')
    assert min(lengths)==max(lengths)
    assert not hasattr(frame,'results')


def _closing_frame():
    import ctypes as ct
    import threading
    from types import SimpleNamespace
    from tessera.compiler.native_public_result import PublicResultFrame
    frame=object.__new__(PublicResultFrame)
    frame._lock=threading.RLock()
    frame.closed=False
    frame._scoped=False
    frame.context_type=ct.c_int
    frame.context=ct.c_int(0)
    frame.current=lambda _:0
    frame.check=lambda code: None if code==0 else (_ for _ in ()).throw(RuntimeError('unknown free'))
    frame.sync=lambda:0
    frame.buffers=[SimpleNamespace(pointer=ct.c_void_p(7))]
    frame.binding=SimpleNamespace(close=lambda:None)
    return frame


def test_synchronous_free_failure_retains_frame_and_never_retries():
    from tessera.compiler.native_public_result import _QUARANTINED_PUBLIC_FRAMES
    frame=_closing_frame();calls=[]
    frame.free=lambda pointer:calls.append(pointer.value) or 1
    try:
        with pytest.raises(RuntimeError,match='unknown free'):frame.close()
        with pytest.raises(RuntimeError,match='owner retained'):frame.close()
        assert calls==[7] and frame.buffers[0].pointer.value==7
        assert frame in _QUARANTINED_PUBLIC_FRAMES
    finally:_QUARANTINED_PUBLIC_FRAMES.discard(frame)


def test_close_drops_owned_exception_roots_without_mutating_external_error():
    import weakref
    import gc
    import numpy as np
    frame=_closing_frame();frame.free=lambda _:0
    payload=np.ones(128)
    reference=weakref.ref(payload)
    error=ValueError(payload)
    error.__context__=error
    frame._source_exception=error
    frame._source_exception_traceback=None
    frame.close()
    assert error.__context__ is error and error.args[0] is payload
    assert not hasattr(frame,'_source_exception')
    del payload,error
    gc.collect()
    assert reference() is None


def test_paired_dynamic_readers_close_both_products_on_external_failure():
    import threading
    from types import SimpleNamespace
    from tests.unit.test_native_reader_retirement import setup
    from tessera.compiler.native_public_result import AsyncSourceVJPFrame
    primal, primal_native = setup()
    derivative, derivative_native = setup()
    derivative._reader_buffers[0].__cuda_array_interface__['shape'] = (2, 2)
    frame = object.__new__(AsyncSourceVJPFrame)
    frame._lock = threading.RLock()
    frame._scoped, frame._complete, frame.closed, frame._retiring = True, False, False, False
    frame._forward = SimpleNamespace(read=primal.read)
    frame._backward = SimpleNamespace(read=derivative.read)
    frame._public_count = 1
    with pytest.raises(ValueError, match='completed'):
        with frame.read_many(21, 22):
            pass
    assert primal._active == derivative._active == 0
    frame._complete = True
    with pytest.raises(LookupError):
        with frame.read_many(21, 22) as products:
            assert primal._active == derivative._active == 2
            p, d = products[21]
            assert p[0].__cuda_array_interface__['shape'] == (4,)
            assert d[0].__cuda_array_interface__['shape'] == (2, 2)
            raise LookupError('external enqueue failed')
    assert primal._active == derivative._active == 0
    for owner, native in ((primal, primal_native), (derivative, derivative_native)):
        assert sum(call[0] == 'record' for call in native.calls) == 2
        owner.retire(23)
        assert owner.poll()


@pytest.mark.parametrize('failed_child', ['forward', 'backward'])
def test_paired_retirement_resumes_only_unsubmitted_children(failed_child):
    import threading
    from types import SimpleNamespace
    from tessera.compiler.native_public_result import AsyncSourceVJPFrame
    class Child:
        def __init__(self, fail):
            self._retiring = False
            self._owner = SimpleNamespace(_active=0)
            self.fail = fail
            self.frees = 0
        def retire(self, stream):
            if self.fail:
                raise RuntimeError('dependency completion unproven')
            assert not self._retiring
            self._retiring = True
            self.frees += 1
    frame = AsyncSourceVJPFrame.__new__(AsyncSourceVJPFrame)
    frame._lock = threading.RLock()
    frame._scoped = True
    frame.closed = frame._retiring = False
    frame._forward = Child(failed_child == 'forward')
    frame._backward = Child(failed_child == 'backward')
    with pytest.raises(RuntimeError, match='unproven'):
        frame.retire(21)
    assert frame._retiring == (failed_child == 'forward')
    if frame._retiring:
        with pytest.raises(ValueError, match='original stream'):
            frame.retire(22)
    frame._forward.fail = frame._backward.fail = False
    frame.retire(21)
    frame.retire(21)
    assert frame._forward.frees == frame._backward.frees == 1
