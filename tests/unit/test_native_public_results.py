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
