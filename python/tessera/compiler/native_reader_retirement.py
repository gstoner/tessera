"""Scoped readers and stream-ordered retirement for persistent derivatives.

Borrowed views may only be submitted on the lease's declared stream before
leaving its scope. They are not unrestricted external pointer exports.
"""

from contextlib import ExitStack, contextmanager
import ctypes as ct
from .native_gpu_storage import NativeSubmission

# A failed allocator API cannot prove whether a free was submitted. Keep the
# frame reachable and never retry its pointers; teardown owns recovery.
_QUARANTINED_GENERATIONS: set["TrackedDerivativeGeneration"] = set()


def _stream(value):
    if type(value) is not int or not 0 < value < (1 << 64):
        raise ValueError("reader ownership requires a non-null native stream")
    return value



@contextmanager
def _read_many(read, *streams):
    """Compose checked reader scopes, unwinding every entered stream on failure."""
    streams = tuple(_stream(stream) for stream in streams)
    if not streams or len(set(streams)) != len(streams):
        raise ValueError('reader streams must be nonempty and distinct')
    with ExitStack() as stack:
        yield {stream: stack.enter_context(read(stream)) for stream in streams}


def _record(native, stream, keepalive, tickets):
    event = ct.c_void_p()
    events = []
    try:
        native._check(native._event_create(ct.byref(event), 2))
        events.append(event)
        native._check(native._event_record(event, ct.c_void_p(stream)))
    except BaseException:
        # A failed record cannot prove absence of previously queued work.
        ticket = NativeSubmission(native, None, events, keepalive, 0, stream)
        native._pending.append(ticket)
        tickets.append(ticket)
        raise
    ticket = NativeSubmission(native, event, events, keepalive, 0, stream)
    native._pending.append(ticket)
    tickets.append(ticket)
    return ticket


class _BorrowedView:
    def __init__(self, lease, buffer):
        self._lease, self._buffer = lease, buffer

    @property
    def _tessera_reader_stream(self):
        return self._lease.stream

    @property
    def __cuda_array_interface__(self):
        if not self._lease.active:
            raise ValueError("persistent reader lease is closed")
        return {
            **self._buffer.__cuda_array_interface__,
            "data": (self._buffer.pointer.value, True),
            "stream": self._lease.stream,
        }


class _ReaderLease:
    def __init__(self, owner, stream):
        self.owner, self._stream = owner, _stream(stream)
        self._active = False
        self.used = False

    @property
    def stream(self):
        return self._stream

    @property
    def active(self):
        return self._active

    def __enter__(self):
        owner = self.owner
        with owner.frame._lock:
            owner.frame._ready()
            if self.used or owner.retiring:
                raise ValueError("persistent reader acquisition is closed")
            owner._submission.ticket.wait_on(self.stream)
            self.used = self._active = True
            owner._active += 1
            return tuple(_BorrowedView(self, buffer) for buffer in owner._reader_buffers)

    def __exit__(self, *exc):
        owner = self.owner
        with owner.frame._lock:
            if not self.active:
                raise ValueError("persistent reader lease is not active")
            try:
                _record(owner.native, self.stream, (owner,), owner._readers)
            finally:
                self._active = False
                owner._active -= 1


class TrackedDerivativeGeneration:
    """Pool-allocated generation whose readers are closed before retirement.

    Healthy retire/poll calls enqueue or query work without a context wait.
    Failed event recording requires explicit wait/close and retains ownership.
    """

    def __init__(self, frame, submission, outputs, native):
        self.frame, self._submission, self._buffers = frame, submission, outputs
        self.native = native
        self._reader_buffers = outputs
        self._readers = []
        self._retirements = []
        self._active = 0
        self.retiring = False
        self._free_failed = False
        self._all_queued = False
        self._released = False

    def read(self, stream):
        return _ReaderLease(self, stream)

    @contextmanager
    def read_many(self, *streams):
        """Borrow read-only views on every declared external consumer stream.

        Consumers must enqueue all reads before leaving this scope and must
        not retain raw pointers. Every stream records its own completion even
        when another consumer raises. Views do not authorize cross-device use.
        """
        with _read_many(self.read, *streams) as views:
            yield views

    def submit_to(self, binding, stream, *args, **kwargs):
        """Submit derivatives as the leading arguments of a native consumer.

        The lease covers validation and enqueue, including a failed submission
        that may already have queued device work. Remaining arguments normally
        provide the consumer's output storage and scalar dimensions.
        """
        from .native_gpu_tensor import NativeTensorCall
        if not isinstance(binding, NativeTensorCall):
            raise TypeError("scoped derivative submission requires a native tensor binding")
        with self.read(stream) as outputs:
            return binding.submit(stream, *outputs, *args, **kwargs)

    def backward_into(self, frame, stream, *, tracked=True, indices=None):
        """Consume this generation as another persistent frame's cotangents."""
        from .native_persistent_tape import PersistentTapeFrame
        from .resident_ssd import ResidentSSDFrame
        if not isinstance(frame, (PersistentTapeFrame, ResidentSSDFrame)):
            raise TypeError("scoped derivative composition requires a persistent tape frame")
        if indices is not None and (not isinstance(indices,tuple) or not indices or
                any(type(i) is not int or not 0 <= i < len(self._reader_buffers) for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError("gradient projection requires distinct valid result indices")
        with self.read(stream) as outputs:
            selected = outputs if indices is None else tuple(outputs[i] for i in indices)
            return frame.backward_async(stream, *selected, tracked=tracked)

    def retire(self, stream):
        stream = _stream(stream)
        with self.frame._lock:
            self.frame._ready()
            if self._active:
                raise ValueError("persistent retirement requires all reader scopes to close")
            if self.retiring:
                raise ValueError("persistent generation retirement already requested")
            # Dependencies may lack completion events after a record failure.
            # No free has been attempted yet: keep retirement retryable after
            # the caller establishes completion with an explicit wait.
            self._submission.ticket.wait_on(stream)
            for reader in self._readers:
                reader.wait_on(stream)
            self.retiring = True
            try:
                for buffer in self._buffers:
                    try:
                        self.frame.check(self.frame.free_async(buffer.pointer, ct.c_void_p(stream)))
                    except BaseException:
                        self._free_failed = True
                        self.frame._retirement_poisoned = True
                        _QUARANTINED_GENERATIONS.add(self)
                        raise
                    # Never submit the same pointer to a second free on failure.
                    buffer.pointer = ct.c_void_p()
                    self.frame.buffers.remove(buffer)
                self._all_queued = True
            finally:
                # Retain partially queued frees even if an API call failed.
                _record(self.native, stream, (self,), self._retirements)
        return self

    def _finish(self):
        if self._all_queued:
            self._released = True
            if self in self.frame._submissions:
                self.frame._submissions.remove(self)
        return self._released

    def poll(self):
        with self.frame._lock:
            self.frame._ready(allow_retirement_failure=True)
            if self._released:
                return True
            if not self.retiring:
                return False
            tickets = [self._submission.ticket, *self._readers, *self._retirements]
            if not all(ticket.poll() for ticket in tickets):
                return False
            if self._free_failed:
                raise RuntimeError("failed asynchronous free quarantined the frame; device teardown is required")
            return self._finish()

    def wait(self):
        with self.frame._lock:
            self.frame._ready(allow_retirement_failure=True)
            if self._active:
                raise ValueError("cannot close a generation with active reader scopes")
            for ticket in [self._submission.ticket, *self._readers, *self._retirements]:
                ticket.wait()
            if self._free_failed:
                raise RuntimeError("failed asynchronous free quarantined the frame; device teardown is required")
            return self._finish()


class CheckedTrackedDerivativeGeneration(TrackedDerivativeGeneration):
    """Checked pool generation: gated or status-checked readers, async retirement.

    There is no unrestricted output export. Completion status is propagated on
    the device; retiring a failed generation is safe after its readers finish.
    """
    def __init__(self, frame, submission, outputs, status, native):
        super().__init__(frame, submission, (*outputs, status), native)
        self._reader_buffers = outputs
        self._status_buffer = status
        self._success_checked = False

    @property
    def submission(self):
        return self._submission

    def wait_success(self):
        """Explicit host status boundary for optional generic scoped readers."""
        with self.frame._lock:
            self.frame._ready()
            if self.retiring:
                raise ValueError('checked generation is retiring')
            self._submission.ticket.wait()
            self.frame._check_status(self._status_buffer)
            self._success_checked = True
        return self

    def read(self, stream):
        if not self._success_checked:
            raise ValueError('checked tracked derivatives require a compiler-gated reader or successful status check')
        return _ReaderLease(self, stream)

    def backward_into(self, frame, stream, *, tracked=True, indices=None, dependencies=()):
        """Consume this cotangent with additional checked status prerequisites.

        Every prerequisite owns a reader lease through consumer submission;
        status-only prerequisites do not add cotangent operands to the ABI.
        """
        from contextlib import ExitStack
        from .native_persistent_tape import PersistentTapeFrame, _input_status_count
        if not isinstance(frame, PersistentTapeFrame):
            raise TypeError('checked reader requires a persistent frame')
        if indices is not None and (not isinstance(indices, tuple) or not indices or
                any(type(i) is not int or not 0 <= i < len(self._reader_buffers) for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('gradient projection requires distinct valid result indices')
        if not isinstance(dependencies, tuple) or any(not isinstance(parent, CheckedTrackedDerivativeGeneration) for parent in dependencies):
            raise TypeError('dependencies must be a tuple of checked tracked generations')
        parents=(self,*dependencies)
        if len({id(parent) for parent in parents}) != len(parents):
            raise ValueError('status dependencies must be distinct')
        count=_input_status_count(frame.pair.backward)
        if not count or len(parents)>max(1,count-1):
            raise ValueError('incoming status count cannot cover every dependency')
        with ExitStack() as stack:
            owners={frame,*[parent.frame for parent in parents]}
            for owner in sorted(owners, key=id):
                stack.enter_context(owner._lock)
                owner._ready()
            target=(frame.pair.forward.backend,frame.pair.forward.chip)
            for parent in parents:
                if (parent.frame.pair.forward.backend,parent.frame.pair.forward.chip)!=target:
                    raise ValueError('checked reader requires the same owning target')
            outputs=stack.enter_context(_ReaderLease(self,stream))
            for parent in dependencies:
                stack.enter_context(_ReaderLease(parent,stream))
            selected = outputs if indices is None else tuple(outputs[i] for i in indices)
            return frame.backward_async(stream,*selected,tracked=tracked,_dependency=parents)
