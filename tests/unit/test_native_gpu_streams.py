"""Native submissions retain ownership and complete through events, not device waits."""
import ctypes as ct
import threading
from types import SimpleNamespace
import pytest
from tessera.compiler.native_gpu_storage import BoundNativeGPUStorage


def binding():
    native = object.__new__(BoundNativeGPUStorage)
    native._lock = threading.RLock()
    native._pending = []
    native._module, native._function = ct.c_void_p(1), ct.c_void_p(2)
    native._types = (ct.c_int64,)
    native.package = SimpleNamespace(abi=('index',))
    native._size = lambda *args: 128
    calls = []
    def record(name):
        def call(*args):
            calls.append(name)
            return 0
        return call
    def create(ptr, flags):
        ct.cast(ptr, ct.POINTER(ct.c_void_p))[0] = len(calls) + 10
        calls.append('create')
        return 0
    native._event_create = create
    for name in ('event_record', 'event_sync', 'event_destroy', 'stream_wait', 'stream_sync', 'launch'):
        setattr(native, '_' + name, record(name))
    native._sync = lambda: pytest.fail('device-wide synchronization')
    return native, calls


def test_event_dependency_and_completion_keep_resources_alive():
    native, calls = binding()
    owner = object()
    ticket = native.submit((32,), grid=(1, 1, 1), block=(32, 1, 1), stream=7,
                           producer_streams=(8,), keepalive=(owner,))
    assert calls == ['create', 'event_record', 'stream_wait', 'create', 'launch', 'event_record']
    assert ticket._keepalive == (owner,)
    ticket.wait_on(9)
    assert calls[-1] == 'stream_wait'
    assert ticket.wait() == 128
    assert not native._pending and ticket._keepalive == ()
    assert calls[-3:] == ['event_sync', 'event_destroy', 'event_destroy']
    count = len(calls)
    ticket.wait()
    assert len(calls) == count


@pytest.mark.parametrize('stream', [0, True, -1, 1 << 64])
def test_invalid_stream_rejected_before_driver(stream):
    native, calls = binding()
    with pytest.raises(ValueError, match='stream'):
        native.submit((32,), grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
    assert not calls


def test_failed_completion_record_waits_for_queued_work_before_cleanup():
    native, calls = binding()
    native._event_record = lambda *args: 1
    with pytest.raises(RuntimeError, match='driver status'):
        native.submit((32,), grid=(1, 1, 1), block=(32, 1, 1), stream=7)
    assert calls == ['create', 'launch', 'stream_sync', 'event_destroy']
    assert not native._pending


def test_failed_cleanup_retains_owners_until_a_later_successful_wait():
    native, calls = binding()
    owner = object()
    native._event_record = lambda *args: 1
    native._stream_sync = lambda *args: 2
    with pytest.raises(RuntimeError, match='driver status 2'):
        native.submit((32,), grid=(1, 1, 1), block=(32, 1, 1), stream=7, keepalive=(owner,))
    assert len(native._pending) == 1
    ticket = native._pending[0]
    assert ticket._keepalive == (owner,)
    native._stream_sync = lambda *args: 0
    ticket.wait()
    assert not native._pending and not ticket._keepalive


def test_shared_readonly_inputs_do_not_serialize_independent_outputs():
    import inspect
    from tessera.compiler.native_gpu_tensor import NativeTensorCall, TensorSpec
    call = object.__new__(NativeTensorCall)
    call._lock, call._inflight = threading.RLock(), []
    call.signature = inspect.signature(lambda source, output: None)
    call.specs = (TensorSpec('source', 'float32', (32,)), TensorSpec('output', 'float32', (32,), True))
    tickets = []
    def submit(*args, **kwargs):
        waits = []
        ticket = SimpleNamespace(done=False, wait_on=lambda stream: waits.append(stream), waits=waits)
        tickets.append(ticket)
        return ticket
    call._bound = SimpleNamespace(submit=submit)
    call._resident = lambda src, dst: ((), [(src.pointer, 128), (dst.pointer, 128)], (1, 1, 1), (32, 1, 1), (dst,))
    def array(pointer):
        return SimpleNamespace(pointer=pointer, __cuda_array_interface__={})
    source, first, second = array(4096), array(8192), array(12288)
    call.submit(7, source, first)
    call.submit(8, source, second)
    assert tickets[0].waits == []
    call.submit(9, source, first)
    assert tickets[0].waits == [9]
    assert tickets[1].waits == []



def test_completed_cleanup_retries_without_requerying_destroyed_event():
    from tessera.compiler.native_gpu_storage import NativeSubmission
    calls=[]; failures=[True]
    def destroy(event):
        calls.append(('destroy',event))
        if event==11 and failures and failures.pop():
            return 1
        return 0
    def check(code):
        if code:raise RuntimeError('destroy failed')
    owner=SimpleNamespace(_lock=threading.RLock(),_pending=[],_check=check,
        _event_query=lambda event:calls.append(('query',event)) or 0,
        _event_destroy=destroy,_event_sync=lambda event:pytest.fail('completed event waited'))
    ticket=NativeSubmission(owner,12,[11,12],(object(),),0,7)
    owner._pending.append(ticket)
    with pytest.raises(RuntimeError):ticket.poll()
    assert ticket._keepalive and not ticket.done
    ticket.wait_on(9)
    assert ticket.poll() and not ticket._keepalive
    assert [c for c in calls if c[0]=='query']==[('query',12)]


def test_idle_module_close_queries_completion_without_context_wait():
    native,calls=binding()
    native._directory=SimpleNamespace(cleanup=lambda: calls.append('cleanup'))
    native._unload=lambda module: calls.append('unload') or 0
    native._event_query=lambda event: 600
    native.submit((32,),grid=(1,1,1),block=(32,1,1),stream=7)
    assert not native.close_if_complete()
    assert 'unload' not in calls
    native._event_query=lambda event: 0
    assert native.close_if_complete()
    assert calls[-2:]==['unload','cleanup']
    assert 'event_sync' not in calls
    with pytest.raises(ValueError,match='closed'):
        native.submit((32,),grid=(1,1,1),block=(32,1,1),stream=7)


def test_module_query_close_refuses_failed_synchronous_launch_completion():
    native,calls=binding()
    native._directory=SimpleNamespace(cleanup=lambda: calls.append('cleanup'))
    native._unload=lambda module: calls.append('unload') or 0
    native._sync=lambda: 1
    with pytest.raises(RuntimeError,match='driver status'):
        native.launch((32,),grid=(1,1,1),block=(32,1,1))
    assert not native.close_if_complete()
    assert 'unload' not in calls
    native._sync=lambda: 0
    native.close()
    assert calls[-2:]==['unload','cleanup']
