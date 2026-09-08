"""Scoped readers and stream-ordered retirement for persistent derivatives.

Borrowed views may only be submitted on the lease's declared stream before
leaving its scope. They are not unrestricted external pointer exports.
"""

import ctypes as ct
from .native_gpu_storage import NativeSubmission

# A failed allocator API cannot prove whether a free was submitted. Keep the
# frame reachable and never retry its pointers; teardown owns recovery.
_QUARANTINED_GENERATIONS: set["TrackedDerivativeGeneration"] = set()


def _stream(value):
    if type(value) is not int or not 0 < value < (1 << 64):
        raise ValueError("reader ownership requires a non-null native stream")
    return value


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
            return tuple(_BorrowedView(self, buffer) for buffer in owner._buffers)

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
        self._readers = []
        self._retirements = []
        self._active = 0
        self.retiring = False
        self._free_failed = False
        self._all_queued = False
        self._released = False

    def read(self, stream):
        return _ReaderLease(self, stream)

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

    def backward_into(self, frame, stream, *, tracked=True):
        """Consume this generation as another persistent frame's cotangents."""
        from .native_persistent_tape import PersistentTapeFrame
        if not isinstance(frame, PersistentTapeFrame):
            raise TypeError("scoped derivative composition requires a persistent tape frame")
        with self.read(stream) as outputs:
            return frame.backward_async(stream, *outputs, tracked=tracked)

    def retire(self, stream):
        stream = _stream(stream)
        with self.frame._lock:
            self.frame._ready()
            if self._active:
                raise ValueError("persistent retirement requires all reader scopes to close")
            if self.retiring:
                raise ValueError("persistent generation retirement already requested")
            self.retiring = True
            try:
                self._submission.ticket.wait_on(stream)
                for reader in self._readers:
                    reader.wait_on(stream)
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
