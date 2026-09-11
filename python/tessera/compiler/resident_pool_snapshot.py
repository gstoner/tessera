"""Owned immutable GPU pool epochs; snapshot readers can overlap live sweeping."""
import ctypes as ct
from .native_device_tape import _Buffer
from .native_reader_retirement import _record, _stream
from .native_gpu_tensor import TensorSubmission
from .native_stream_epoch import NativeStreamEpoch


class ResidentPoolSnapshot:
    def __init__(self, pool, stream):
        stream = _stream(stream)
        self.parent, self._lock = pool, pool._lock
        self.native = pool.native
        self.check, self.alloc, self.free = pool.check, pool.alloc, pool.free
        self.buffers, self.closed, self.epoch = [], False, None
        self._closing = False
        with self._lock:
            pool._ready()
            epoch = pool._access()
            if epoch._active or epoch.retiring:
                raise ValueError("pool snapshot requires closed live reader scopes")
            pool._snapshots.append(self)
            try:
                sources = pool._access()._reader_buffers
                copies = tuple(_Buffer(self, b.shape, b.dtype) for b in sources)
                with pool._access().write(stream):
                    for source, destination in zip(sources, copies, strict=True):
                        self.check(pool._copy_async(destination.pointer, source.pointer, source.nbytes, ct.c_void_p(stream)))
                ticket = _record(self.native, stream, (self,), [])
                self.epoch = NativeStreamEpoch(self, self.native, copies, TensorSubmission(ticket, ()))
            except BaseException:
                # Keep every allocation owned until close proves completion.
                # The parent retains this snapshot even if construction fails.
                raise

    def _ready(self):
        if self.closed:
            raise ValueError('pool snapshot is closed')
        if self._closing:
            self.parent._ready(recovery=True)
        else:
            self.parent._ready()

    def read(self, stream):
        self._ready()
        if self.epoch is None:
            raise ValueError('pool snapshot copy did not publish completion')
        return self.epoch.read(stream)

    def close(self):
        with self._lock:
            if self.closed:
                return
            self._closing = True
            try:
                self._ready()
                if self.epoch is not None:
                    self.epoch.wait()
                else:
                    self.check(self.native._sync())
                while self.buffers:
                    buffer = self.buffers[-1]
                    self.check(self.free(buffer.pointer))
                    buffer.pointer = ct.c_void_p()
                    self.buffers.pop()
                self.closed = True
                self.parent._snapshots.remove(self)
            finally:
                self._closing = False

    def __enter__(self):
        self._ready()
        return self

    def __exit__(self, *exc):
        self.close()
