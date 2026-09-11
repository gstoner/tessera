"""Automatically paired resident SSD forward/VJP with scoped GPU ownership.

Inputs, saved checkpoints, cotangents and all five gradients remain on the owning
CUDA/HIP device. Capture and first-order backward can compose through
stream-scoped generations. This does not turn the NumPy host tape into a GPU tape.
"""

from __future__ import annotations

import ctypes as ct
import threading
from .native_ssd import materialize_ssd
from .native_device_tape import _Buffer

_UNCERTAIN_FRAMES: list[ResidentSSDFrame] = []


class ResidentSSDProgram:
    def __init__(self, logical, **options):
        self.forward = materialize_ssd(logical, cooperative=True, **options)
        self.reverse = materialize_ssd(logical, adjoint=True, **options)
        self._forward = self.forward.bind()
        self._reverse = self.reverse.bind()
        self._lock = threading.RLock()
        self.frames = []
        self.closed = False
        self._closing = False

    def capture(self, *inputs):
        with self._lock:
            if self.closed or self._closing:
                raise ValueError("resident SSD program is closed or retiring")
            if any(hasattr(value, "_tessera_reader_stream") for value in inputs):
                raise ValueError("borrowed persistent view requires explicit stream ownership")
            frame = ResidentSSDFrame(self, inputs)
            self.frames.append(frame)
            return frame

    def capture_async(self, stream, *inputs):
        from .native_reader_retirement import _stream

        stream = _stream(stream)
        with self._lock:
            if self.closed or self._closing:
                raise ValueError("resident SSD program is closed or retiring")
            if any(getattr(value, "_tessera_reader_stream", stream) != stream for value in inputs):
                raise ValueError("capture requires the borrowed reader stream")
            frame = ResidentSSDFrame(self, inputs, stream=stream)
            self.frames.append(frame)
            return frame

    def __tessera_vjp__(self, *inputs, stream=None):
        frame = self.capture(*inputs) if stream is None else self.capture_async(stream, *inputs)
        return ResidentSSDValue(frame), ResidentSSDPullback(frame, stream)

    def value_and_grad(self, *inputs, cotangent):
        """Generate/capture the forward and execute its native resident VJP."""
        frame = self.capture(*inputs)
        try:
            frame.backward(cotangent)
            return frame
        except BaseException:
            try:
                frame.close()
            except BaseException:
                _UNCERTAIN_FRAMES.append(frame)
            raise

    def value_and_grad_async(self, stream, *inputs, cotangent):
        """Capture and differentiate without a context synchronization."""
        frame = self.capture_async(stream, *inputs)
        try:
            return frame, frame.backward_async(stream, cotangent)
        except BaseException:
            # The frame remains in this program until explicit close proves
            # completion, including a possibly failed enqueue.
            raise

    def retire_async(self, stream):
        """Retire every frame, then poll off-thread module unload completion.

        A partially submitted retirement stays closed to new captures. Calling
        this method again resumes frames whose retirement could not be queued.
        """
        from .native_reader_retirement import _stream
        stream = _stream(stream)
        with self._lock:
            if self.closed:
                return self
            # Refuse live leases across all frames before freeing any frame.
            for frame in self.frames:
                owners = [*frame._submissions]
                if frame._forward_epoch is not None:
                    owners.append(frame._forward_epoch)
                if any(owner._active for owner in owners):
                    raise ValueError("program retirement requires closed reader scopes")
            self._closing = True
            for frame in list(self.frames):
                if not frame.closed and frame._retirement is None:
                    frame.retire_async(stream)
        return self

    def poll_close(self):
        """Query frame completion and bounded-admission unload workers only."""
        with self._lock:
            if self.closed:
                return True
            if not self._closing:
                raise ValueError("program retirement has not started")
            ready = True
            for frame in list(self.frames):
                if frame._retirement is None or not frame.poll_close():
                    ready = False
            if not ready:
                return False
            # Poll both bindings even when one unload worker is still pending.
            complete = [binding.close_if_complete(defer_unload=True)
                        for binding in (self._reverse, self._forward)]
            self.closed = all(complete)
            return self.closed

    def close(self):
        with self._lock:
            if self._closing and not self.closed:
                if not self.poll_close():
                    raise RuntimeError("resident SSD retirement pending; poll completion")
                return
            if not self.closed:
                for frame in list(self.frames):
                    frame.close()
                self._reverse.close()
                self._forward.close()
                self.closed = True

    def __enter__(self):
        if self.closed or self._closing:
            raise ValueError("resident SSD program is closed or retiring")
        return self

    def __exit__(self, *exc):
        self.close()


class ResidentSSDFrame:
    def __init__(self, owner, inputs, *, stream=None):
        self.owner = owner
        self._lock = owner._lock
        self._submissions = []
        self._retirement = None
        self._capture = None
        self._capture_inputs = tuple(inputs)
        self._forward_epoch = None
        self._retirement_poisoned = False
        self.closed = False
        self.buffers = []
        self.gradients: tuple[_Buffer, ...] = ()
        self._backward_requested = False
        binding = owner._forward
        if binding._bound is None:
            binding._bound = owner.forward.package.bind()
        native = binding._bound
        self.check, self.sync = native._check, native._sync
        driver, cuda = native._driver, owner.forward.package.backend == "nvidia"
        P, S = ct.c_void_p, ct.c_size_t

        def bind(cu, hip, args):
            fn = getattr(driver, cu if cuda else hip)
            fn.argtypes, fn.restype = args, ct.c_int
            return fn

        self.alloc = bind("cuMemAlloc_v2", "hipMalloc", [ct.POINTER(P), S])
        self.free = bind("cuMemFree_v2", "hipFree", [P])
        self.copy_async = bind("cuMemcpyDtoDAsync_v2", "hipMemcpyDtoDAsync", [P, P, S, P])
        self.copy = bind("cuMemcpyDtoD_v2", "hipMemcpyDtoD", [P, P, S])
        self.zero = bind("cuMemsetD8_v2", "hipMemset", [P, ct.c_ubyte if cuda else ct.c_int, S])
        self.alloc_async = bind("cuMemAllocAsync", "hipMallocAsync", [ct.POINTER(P), S, P])
        self.free_async = bind("cuMemFreeAsync", "hipFreeAsync", [P, P])
        self.zero_async = bind("cuMemsetD8Async", "hipMemsetAsync", [P, ct.c_ubyte if cuda else ct.c_int, S, P])
        self.context_type = P if cuda else ct.c_int
        self.current = bind("cuCtxGetCurrent", "hipGetDevice", [ct.POINTER(self.context_type)])
        self.context = self.context_type()
        self.check(self.current(ct.byref(self.context)))
        try:
            specs = binding.specs
            self._inputs = [_Buffer(self, s.shape, stream=stream) for s in specs[:5]]
            outputs = [_Buffer(self, s.shape, stream=stream) for s in specs[5:8]]
            raw, _, _, _, _ = binding._resident(*inputs, *outputs, 1)
            self._outputs = tuple(outputs)
            self._checkpoints = outputs[2]
            if stream is None:
                self.check(self.sync())
                for i, buffer in enumerate(self._inputs):
                    self.check(self.copy(buffer.pointer, P(raw[i]), buffer.nbytes))
                binding(*self._inputs, *outputs, 1)
                self._capture_inputs = ()
            else:
                from .native_reader_retirement import _record, _stream
                from .native_stream_epoch import NativeStreamEpoch

                for value in inputs:
                    producer = value.__cuda_array_interface__.get("stream")
                    if producer is not None and producer != stream:
                        _record(native, _stream(producer), (value,), []).wait_on(stream)
                for i, buffer in enumerate(self._inputs):
                    self.check(self.copy_async(buffer.pointer, P(raw[i]), buffer.nbytes, P(stream)))
                self._capture = binding.submit(stream, *self._inputs, *outputs, 1)
                self._forward_epoch = NativeStreamEpoch(self, native, outputs, self._capture)
        except BaseException:
            try:
                self.close()
            except BaseException:
                _UNCERTAIN_FRAMES.append(self)
            raise

    def _ready(self, *, allow_retirement_failure=False):
        if self._retirement is not None and not allow_retirement_failure:
            raise ValueError("resident SSD frame is retiring")
        if self._retirement_poisoned and not allow_retirement_failure:
            raise RuntimeError("uncertain asynchronous retirement requires device recovery")
        if self.closed or self.owner.closed:
            raise ValueError("resident SSD frame is closed")
        context = self.context_type()
        self.check(self.current(ct.byref(context)))
        if context.value != self.context.value:
            raise ValueError("resident SSD requires its owning device context")
        if self._capture is not None and self._capture.ticket.poll():
            self._capture_inputs = ()

    def wait_forward(self):
        with self._lock:
            self._ready()
            if self._capture is not None:
                self._capture.ticket.wait()
                self._capture_inputs = ()
        return self

    @property
    def value(self):
        self._ready()
        if self._capture is not None and not self._capture.ticket.poll():
            raise ValueError("forward output is pending; use read_forward or wait_forward")
        return self._outputs[0]

    @property
    def carry(self):
        _ = self.value
        return self._outputs[1]

    def read_forward(self, stream):
        from .native_reader_retirement import _record, _stream

        stream = _stream(stream)
        from .native_gpu_tensor import TensorSubmission
        from .native_stream_epoch import NativeStreamEpoch

        with self._lock:
            self._ready()
            if self._forward_epoch is None:
                native = self.owner._forward._bound
                ticket = _record(native, stream, (self,), [])
                self._forward_epoch = NativeStreamEpoch(self, native, self._outputs, TensorSubmission(ticket, ()))
            return self._forward_epoch.read(stream)

    def retire_async(self, stream):
        from .native_reader_retirement import TrackedDerivativeGeneration, _record, _stream
        from .native_gpu_tensor import TensorSubmission

        stream = _stream(stream)
        with self._lock:
            self._ready()
            owners = [*self._submissions]
            if self._forward_epoch is not None:
                owners.append(self._forward_epoch)
            if any(owner._active for owner in owners):
                raise ValueError("whole-frame retirement requires closed reader scopes")
            native = self.owner._forward._bound
            submission = self._capture
            if submission is None:
                submission = TensorSubmission(_record(native, stream, (self,), []), ())
            retirement = TrackedDerivativeGeneration(self, submission, tuple(self.buffers), native)
            for owner in owners:
                retirement._readers.extend(
                    [owner._submission.ticket, *owner._readers, *getattr(owner, "_retirements", ())]
                )
            try:
                retirement.retire(stream)
            finally:
                if retirement.retiring:
                    self._retirement = retirement
                    for owner in owners:
                        owner.retiring = True
            return self

    def poll_close(self):
        with self._lock:
            if self.closed:
                return True
            if self._retirement is None:
                raise ValueError("whole-frame retirement has not started")
            if not self._retirement.poll():
                return False
            self.closed = True
            self._capture_inputs = ()
            self._submissions.clear()
            if self in self.owner.frames:
                self.owner.frames.remove(self)
            return True

    def backward(self, cotangent, *, carry_cotangent=None, checkpoint_cotangent=None):
        with self.owner._lock:
            self._ready()
            if self._backward_requested:
                raise ValueError("resident SSD frame allows one backward invocation")
            self.wait_forward()
            start = len(self.buffers)
            try:
                seeds = [cotangent]
                for supplied, shape in [
                    (carry_cotangent, self.carry.shape),
                    (checkpoint_cotangent, self._checkpoints.shape),
                ]:
                    if supplied is None:
                        supplied = _Buffer(self, shape)
                        self.check(self.zero(supplied.pointer, 0, supplied.nbytes))
                    seeds.append(supplied)
                grads = tuple(_Buffer(self, b.shape) for b in self._inputs)
                self.owner._reverse(*self._inputs, self._checkpoints, *seeds, *grads, 1)
                self.gradients = grads
                self._backward_requested = True
                return grads
            except BaseException:
                # Uncertain completion retains all buffers until a later proven
                # synchronization or isolation recovery. No premature free.
                self.check(self.sync())
                while len(self.buffers) > start:
                    buffer = self.buffers[-1]
                    self.check(self.free(buffer.pointer))
                    buffer.pointer = ct.c_void_p()
                    self.buffers.pop()
                raise

    def backward_async(self, stream, cotangent, *, carry_cotangent=None, checkpoint_cotangent=None, tracked=True):
        from .native_reader_retirement import TrackedDerivativeGeneration, _stream

        stream = _stream(stream)
        if tracked is not True:
            raise ValueError("resident SSD asynchronous gradients require scoped ownership")
        with self._lock:
            self._ready()
            if self._backward_requested:
                raise ValueError("resident SSD frame allows one backward invocation")
            if self._capture is not None:
                self._capture.ticket.wait_on(stream)
            start = len(self.buffers)
            try:
                seeds = [cotangent]
                for supplied, shape in [
                    (carry_cotangent, self._outputs[1].shape),
                    (checkpoint_cotangent, self._checkpoints.shape),
                ]:
                    if supplied is None:
                        supplied = _Buffer(self, shape, stream=stream)
                        self.check(self.zero_async(supplied.pointer, 0, supplied.nbytes, ct.c_void_p(stream)))
                    seeds.append(supplied)
                grads = tuple(_Buffer(self, b.shape, stream=stream) for b in self._inputs)
                submission = self.owner._reverse.submit(stream, *self._inputs, self._checkpoints, *seeds, *grads, 1)
                result = TrackedDerivativeGeneration(self, submission, grads, self.owner._reverse._bound)
                self._submissions.append(result)
                self._backward_requested = True
                return result
            except BaseException:
                self.check(self.sync())
                while len(self.buffers) > start:
                    buffer = self.buffers[-1]
                    self.check(self.free(buffer.pointer))
                    buffer.pointer = ct.c_void_p()
                    self.buffers.pop()
                raise

    def close(self):
        with self.owner._lock:
            if self.closed:
                return
            if self._retirement is not None:
                self._retirement.wait()
                self.poll_close()
                return
            self._ready()
            if self._forward_epoch is not None:
                self._forward_epoch.wait()
            for generation in list(self._submissions):
                generation.wait()
            self.check(self.sync())
            while self.buffers:
                buffer = self.buffers[-1]
                self.check(self.free(buffer.pointer))
                buffer.pointer = ct.c_void_p()
                self.buffers.pop()
            self.closed = True
            self._capture_inputs = ()
            if self in self.owner.frames:
                self.owner.frames.remove(self)

    def __enter__(self):
        self._ready()
        return self

    def __exit__(self, *exc):
        self.close()


class ResidentSSDValue:
    """Owned public value; asynchronous access requires a scoped reader."""

    def __init__(self, frame):
        self.frame = frame

    def read(self, stream):
        from contextlib import contextmanager

        @contextmanager
        def lease():
            with self.frame.read_forward(stream) as outputs:
                yield outputs[0]

        return lease()

    def wait(self):
        self.frame.wait_forward()
        return self


class ResidentSSDPullback:
    """Explicit owner returned by public vjp for a resident native program."""

    def __init__(self, frame, stream):
        self.frame, self.stream = frame, stream

    def __call__(self, cotangent):
        if self.stream is None:
            return self.frame.backward(cotangent)
        return self.frame.backward_async(self.stream, cotangent)

    def close(self):
        self.frame.close()

    def retire_async(self, stream):
        return self.frame.retire_async(stream)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
