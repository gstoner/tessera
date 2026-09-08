"""Reader completeness and failure retention without a device dependency."""

import ctypes as ct
import threading
from types import SimpleNamespace
import pytest
from tessera.compiler.native_reader_retirement import TrackedDerivativeGeneration


class Native:
    def __init__(self):
        self._lock = threading.RLock()
        self._pending = []
        self.calls = []
        self.ready = True
        self.record_failure = False
        self.counter = 10

    def _check(self, status):
        if status:
            raise RuntimeError("injected driver failure")

    def _event_create(self, pointer, flags):
        self.counter += 1
        ct.cast(pointer, ct.POINTER(ct.c_void_p))[0] = self.counter
        return 0

    def _event_record(self, event, stream):
        self.calls.append(("record", stream.value))
        return int(self.record_failure)

    def _event_query(self, event):
        return 0 if self.ready else 600

    def _event_sync(self, event):
        self.calls.append(("event_wait", event.value))
        return 0

    def _event_destroy(self, event):
        return 0

    def _stream_wait(self, stream, event, flags):
        self.calls.append(("stream_wait", stream.value))
        return 0

    def _stream_sync(self, stream):
        self.calls.append(("stream_sync", stream.value))
        return 0


def setup():
    native = Native()
    producer = SimpleNamespace(
        wait_on=lambda s: native.calls.append(("producer_wait", s)), wait=lambda: 0, poll=lambda: True
    )
    frame = SimpleNamespace(
        _lock=threading.RLock(), _ready=lambda **kwargs: None, check=native._check, buffers=[], _submissions=[]
    )

    def free(pointer, stream):
        native.calls.append(("free", pointer.value, stream.value))
        return 0

    frame.free_async = free
    buffers = tuple(
        SimpleNamespace(
            pointer=ct.c_void_p(i), __cuda_array_interface__=dict(shape=(4,), typestr="<f4", data=(i, False), version=3)
        )
        for i in (1, 2)
    )
    frame.buffers.extend(buffers)
    owner = TrackedDerivativeGeneration(frame, SimpleNamespace(ticket=producer), buffers, native)
    frame._submissions.append(owner)
    return owner, native


def test_retirement_orders_every_reader_and_invalidates_borrowed_views():
    owner, native = setup()
    for stream in (21, 22):
        with owner.read(stream) as outputs:
            view = outputs[0]
            assert view.__cuda_array_interface__["stream"] == stream
            with pytest.raises(ValueError, match="scopes"):
                owner.retire(23)
        with pytest.raises(ValueError, match="lease is closed"):
            view.__cuda_array_interface__
    native.ready = False
    owner.retire(23)
    assert native.calls.index(("stream_wait", 23)) < next(i for i, c in enumerate(native.calls) if c[0] == "free")
    assert sum(c == ("stream_wait", 23) for c in native.calls) == 2
    assert not owner.poll()
    with pytest.raises(ValueError, match="acquisition"):
        with owner.read(21):
            pass
    native.ready = True
    assert owner.poll()
    assert not owner.frame.buffers and not owner.frame._submissions
    assert not any(c[0] == "stream_sync" for c in native.calls)


def test_failed_reader_event_retains_owner_until_explicit_wait():
    owner, native = setup()
    native.record_failure = True
    with pytest.raises(RuntimeError):
        with owner.read(21):
            pass
    assert len(owner.frame.buffers) == 2
    assert not owner._active
    assert not owner._readers[0].poll()
    native.record_failure = False
    owner.retire(23)
    assert ("stream_sync", 21) in native.calls
    owner.wait()
    assert owner._released


def test_partial_free_is_never_submitted_twice():
    owner, native = setup()

    def free(pointer, stream):
        native.calls.append(("free", pointer.value, stream.value))
        return int(pointer.value == 2)

    owner.frame.free_async = free
    with pytest.raises(RuntimeError):
        owner.retire(23)
    assert [b.pointer.value for b in owner.frame.buffers] == [2]
    with pytest.raises(RuntimeError, match="quarantined"):
        owner.wait()
    assert owner.frame._retirement_poisoned
    owner.frame.closed = False
    from tessera.compiler.native_persistent_tape import PersistentTapeFrame

    with pytest.raises(RuntimeError, match="quarantined"):
        PersistentTapeFrame._ready(owner.frame)
    from tessera.compiler.native_reader_retirement import _QUARANTINED_GENERATIONS

    assert owner in _QUARANTINED_GENERATIONS
    _QUARANTINED_GENERATIONS.remove(owner)
    with pytest.raises(ValueError, match="already requested"):
        owner.retire(23)
    assert [c[1] for c in native.calls if c[0] == "free"] == [1, 2]


def test_failed_retirement_marker_retains_until_wait():
    owner, native = setup()
    native.record_failure = True
    with pytest.raises(RuntimeError):
        owner.retire(23)
    assert not owner.poll()
    assert owner in owner.frame._submissions
    owner.wait()
    assert owner._released
    assert ("stream_sync", 23) in native.calls


@pytest.mark.parametrize("stream", [0, -1, True, 1 << 64, 1.5])
def test_invalid_reader_streams_fail_before_ownership_changes(stream):
    owner, native = setup()
    with pytest.raises(ValueError):
        owner.read(stream)
    with pytest.raises(ValueError):
        owner.retire(stream)
    assert not native.calls


def test_tensor_binding_rejects_undeclared_reader_stream():
    import inspect
    from tessera.compiler.native_gpu_tensor import NativeTensorCall, TensorSpec

    value = SimpleNamespace(_tessera_reader_stream=21)
    call = SimpleNamespace(
        _lock=threading.RLock(),
        _resident=lambda *args, **kwargs: ([], [], (1, 1, 1), (1, 1, 1), []),
        signature=inspect.Signature([inspect.Parameter("x", inspect.Parameter.POSITIONAL_ONLY)]),
        specs=(TensorSpec("x", "fp32", (4,), False),),
    )
    with pytest.raises(ValueError, match="declared reader stream"):
        NativeTensorCall.submit(call, 22, value)


def test_frame_close_cannot_complete_an_active_reader_scope():
    owner, native = setup()
    with owner.read(21):
        with pytest.raises(ValueError, match="active reader"):
            owner.wait()
    owner.retire(23).wait()
    assert owner._released


@pytest.mark.parametrize('failure',[False,True])
def test_scoped_native_consumer_closes_reader_on_enqueue_failure(failure):
    from tessera.compiler.native_gpu_tensor import NativeTensorCall
    owner,native=setup()
    binding=object.__new__(NativeTensorCall)
    def submit(stream,*values):
        assert stream==21 and owner._active==1
        assert values[-1]=='output'
        assert values[0].__cuda_array_interface__['data']==(1,True)
        if failure:
            raise RuntimeError('submission already enqueued')
        return 'ticket'
    binding.submit=submit
    if failure:
        with pytest.raises(RuntimeError,match='already enqueued'):
            owner.submit_to(binding,21,'output')
    else:
        assert owner.submit_to(binding,21,'output')=='ticket'
    assert owner._active==0 and len(owner._readers)==1
    owner.retire(23).wait()


def test_scoped_reverse_composition_owns_cotangent_reader():
    from tessera.compiler.native_persistent_tape import PersistentTapeFrame
    owner,native=setup()
    frame=object.__new__(PersistentTapeFrame)
    def backward(stream,*cotangents,tracked):
        assert stream==22 and tracked
        assert owner._active==1 and len(cotangents)==2
        return 'child'
    frame.backward_async=backward
    assert owner.backward_into(frame,22)=='child'
    assert owner._active==0 and len(owner._readers)==1
    owner.retire(23).wait()
