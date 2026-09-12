"""Concurrent scoped readers and exclusive stream-ordered native mutations.

Writers serialize through events; they never race a collector against readers.
An event-record failure retains an eventless dependency until explicit recovery.
"""

from contextlib import contextmanager
from .native_reader_retirement import _ReaderLease, _read_many, _record, _stream
from .native_gpu_tensor import TensorSubmission
from .native_gpu_storage import NativeSubmission


class NativeStreamEpoch:
    def __init__(self, frame, native, buffers, submission):
        self.frame, self.native = frame, native
        self._reader_buffers = tuple(buffers)
        self._submission = submission
        self._active = 0
        self._readers = []
        self.retiring = False

    def read(self, stream):
        return _ReaderLease(self, stream)

    @contextmanager
    def read_many(self, *streams):
        with _read_many(self.read, *streams) as views:
            yield views

    @contextmanager
    def write(self, stream):
        stream = _stream(stream)
        with self.frame._lock:
            self.frame._ready()
            if self.retiring or self._active:
                raise ValueError("native mutation requires closed reader scopes")
            self._submission.ticket.wait_on(stream)
            for reader in self._readers:
                reader.wait_on(stream)
            self.retiring = True
            pending: list[NativeSubmission] = []
            try:
                yield
            finally:
                # Record even after a failed enqueue: earlier writes may be in
                # flight. _record retains an eventless ticket on uncertainty.
                try:
                    _record(self.native, stream, (self.frame,), pending)
                finally:
                    if pending:
                        self._submission = TensorSubmission(pending[-1], ())
                        self._readers.clear()
                    self.retiring = False

    def wait(self):
        with self.frame._lock:
            self.frame._ready()
            if self._active:
                raise ValueError("native epoch has active readers")
            self._submission.ticket.wait()
            for reader in self._readers:
                reader.wait()

    def poll(self):
        """Query every recorded dependency without an implicit synchronization."""
        with self.frame._lock:
            self.frame._ready()
            if self._active:
                raise ValueError("native epoch has active readers")
            complete = self._submission.ticket.poll()
            for reader in self._readers:
                complete = reader.poll() and complete
            return complete
