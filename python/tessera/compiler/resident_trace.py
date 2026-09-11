"""Explicit first-order tracing of acyclic resident SSD calls with shared-input accumulation.

Python builds a call graph before any GPU work. Native packages still own each
operation and checkpoint. Repeated inputs and fan-out accumulate through replay-validated native additions.
Unused inputs and Python data-dependent control flow remain unsupported.
"""
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

_ACTIVE_TRACE: ContextVar = ContextVar('tessera_resident_trace', default=None)
_UNCERTAIN_TRACES: list = []


@dataclass(frozen=True, eq=False)
class _Value:
    kind: str
    index: int

    def __eq__(self, other):
        raise ValueError("resident trace does not support Python value comparisons")

    def __ne__(self, other):
        return self.__eq__(other)

    def __bool__(self):
        raise ValueError('resident trace does not support Python data-dependent control flow')


class _Recorder:
    def __init__(self):
        self.calls = []

    def call(self, program, inputs):
        if len(inputs) != 5 or any(type(v) is not _Value for v in inputs):
            raise ValueError('resident trace requires five traced inputs per SSD call')
        result = _Value('call', len(self.calls))
        self.calls.append((program, tuple(inputs)))
        return result


class ResidentSSDTrace:
    """Wrap a Python composition for public vjp(..., stream=...)."""
    def __init__(self, fn):
        if not callable(fn):
            raise TypeError('resident trace requires a callable')
        self.fn = fn

    def __tessera_vjp__(self, *inputs, stream=None):
        from .native_reader_retirement import _stream
        from .resident_ssd import ResidentSSDValue
        stream = _stream(stream)
        if _ACTIVE_TRACE.get() is not None:
            raise ValueError('nested resident tracing is not supported')
        recorder = _Recorder()
        token = _ACTIVE_TRACE.set(recorder)
        try:
            output = self.fn(*(_Value('input', i) for i in range(len(inputs))))
        finally:
            _ACTIVE_TRACE.reset(token)
        calls = recorder.calls
        if not calls or type(output) is not _Value or (output.kind, output.index) != ('call', len(calls) - 1):
            raise ValueError('resident trace must return its final SSD call')
        used: set[int] = set()
        reachable = {len(calls) - 1}
        for i in reversed(range(len(calls))):
            if i not in reachable:
                raise ValueError('resident trace contains a disconnected call')
            for value in calls[i][1]:
                if value.kind == 'call' and 0 <= value.index < i:
                    reachable.add(value.index)
                elif value.kind == 'input' and 0 <= value.index < len(inputs):
                    used.add(value.index)
                else:
                    raise ValueError('resident trace contains an invalid SSA reference')
        if used != set(range(len(inputs))):
            raise ValueError('resident trace requires every public input to be used')
        owner = ResidentTracePullback(stream, calls, len(inputs))
        try:
            for program, args in calls:
                with ExitStack() as leases:
                    values = [inputs[v.index] if v.kind == 'input' else
                              leases.enter_context(owner.frames[v.index].read_forward(stream))[0]
                              for v in args]
                    owner.frames.append(program.capture_async(stream, *values))
        except BaseException:
            try:
                owner.close()
            except BaseException:
                _UNCERTAIN_TRACES.append(owner)
            raise
        return ResidentSSDValue(owner.frames[-1]), owner


class ResidentTracePullback:
    def __init__(self, stream, calls, count):
        self.stream, self.calls = stream, calls
        self.count = count
        self.frames = []
        self.requested = False
        self.retiring = False

    def __call__(self, cotangent):
        if self.requested or self.retiring:
            raise ValueError('resident trace backward is already requested or retiring')
        self.requested = True
        contributions: dict[tuple[str, int], list[Any]] = {}
        for i in reversed(range(len(self.frames))):
            frame = self.frames[i]
            with ExitStack() as leases:
                if i == len(self.frames) - 1:
                    seed = cotangent
                else:
                    merged = frame.sum_terms(self.stream, contributions[('call', i)])
                    seed = leases.enter_context(merged[0].read(self.stream))[merged[1]]
                generation = frame.backward_async(self.stream, seed)
            for slot, value in enumerate(self.calls[i][1]):
                contributions.setdefault((value.kind, value.index), []).append((generation, slot))
        results = []
        for i in range(self.count):
            terms = contributions[('input', i)]
            # The first contributing frame owns the sum and its retirement.
            results.append(terms[0][0].frame.sum_terms(self.stream, terms))
        return ResidentTraceGradients(self, tuple(results))

    def retire_async(self, stream):
        # Preflight every frame before committing any retirement.
        for frame in self.frames:
            owners = [*frame._submissions]
            if frame._forward_epoch is not None:
                owners.append(frame._forward_epoch)
            if any(owner._active for owner in owners):
                raise ValueError('resident trace retirement requires closed reader scopes')
        self.retiring = True
        for frame in self.frames:
            if not frame.closed and frame._retirement is None:
                frame.retire_async(stream)
        return self

    def poll_close(self):
        return all([frame.closed or frame.poll_close() for frame in self.frames])

    def close(self):
        self.retiring = True
        for frame in reversed(self.frames):
            frame.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ResidentTraceGradients:
    def __init__(self, owner, results):
        self.owner, self.results = owner, results

    @contextmanager
    def read(self, stream):
        with ExitStack() as leases:
            views = {}
            for generation, _ in self.results:
                if id(generation) not in views:
                    views[id(generation)] = leases.enter_context(generation.read(stream))
            yield tuple(views[id(generation)][slot] for generation, slot in self.results)
