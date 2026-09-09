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


def test_scoped_dynamic_public_results_require_checked_reader_scopes():
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
    with pytest.raises(ValueError, match='successful completion'):
        frame.read(21)
    assert frame.poll() and not hasattr(frame, 'results')
    with frame.read(21) as results:
        borrowed = results[0]
        assert borrowed.__cuda_array_interface__['shape'] == (2,)
        with pytest.raises(ValueError, match='scopes'):
            frame.retire(22)
    with pytest.raises(ValueError, match='lease is closed'):
        _ = borrowed.__cuda_array_interface__
    frame.retire(22)
    assert frame.poll_retired() and frame.closed and not frame.buffers
