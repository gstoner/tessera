from contextlib import contextmanager
from types import SimpleNamespace
import pytest
from tessera.compiler.resident_trace import ResidentSSDTrace, _ACTIVE_TRACE
from tessera.control import vjp


class Program:
    def __init__(self):
        self.frames = []
    def __call__(self, *inputs):
        return _ACTIVE_TRACE.get().call(self, inputs)
    def capture_async(self, stream, *inputs):
        frame = Frame(inputs)
        self.frames.append(frame)
        return frame


class Frame:
    def __init__(self, inputs):
        self.inputs, self.closed = inputs, False
        self._submissions = []
        self._forward_epoch = None
        self._retirement = None
    @contextmanager
    def read_forward(self, stream):
        yield (sum(self.inputs),)
    def backward_async(self, stream, seed):
        frame = self
        class Generation:
            def __init__(self):
                self.frame = frame
            @contextmanager
            def read(self, stream):
                yield (seed,) * 5
        return Generation()
    def sum_terms(self, stream, terms):
        frame = self
        total = 0
        for owner, slot in terms:
            with owner.read(stream) as views:
                total += views[slot]
        class Sum:
            def __init__(self):
                self.frame = frame
            @contextmanager
            def read(self, stream):
                yield (total,)
        return Sum(), 0
    def close(self):
        self.closed = True
    def retire_async(self, stream):
        self._retirement = True
    def poll_close(self):
        self.closed = True
        return True


def test_public_vjp_traces_two_calls_and_projects_all_input_gradients():
    p = Program()
    trace = ResidentSSDTrace(lambda *x: p(p(*x[:5]), *x[5:]))
    _, pullback = vjp(trace, *range(9), stream=21)
    assert p.frames[1].inputs[0] == sum(range(5))
    gradients = pullback(3)
    with gradients.read(22) as values:
        assert values == (3,) * 9
    pullback.retire_async(23)
    assert pullback.poll_close()
    with pytest.raises(ValueError, match='already requested'):
        pullback(3)


@pytest.mark.parametrize('fn', [
    lambda p, x: p(x[0], x[0], x[2], x[3], x[4]),
    lambda p, x: p(*x[:5]) if x[0] else p(*x[:5]),
    lambda p, x: p(*x[:5]) if x[0] == x[1] else p(*x[:5]),
    lambda p, x: x[0],
])
def test_invalid_trace_fails_before_capture(fn):
    p = Program()
    with pytest.raises(ValueError):
        vjp(ResidentSSDTrace(lambda *x: fn(p, x)), *range(5), stream=21)
    assert p.frames == []
    assert _ACTIVE_TRACE.get() is None


def test_capture_failure_closes_prior_frames():
    p = Program()
    q = Program()
    def fail(*args):
        raise RuntimeError('capture failed')
    q.capture_async = fail
    with pytest.raises(RuntimeError, match='capture failed'):
        vjp(ResidentSSDTrace(lambda *x: q(p(*x[:5]), *x[5:])), *range(9), stream=21)
    assert p.frames[0].closed


def test_trace_retirement_preflights_all_readers():
    p = Program()
    _, pb = vjp(ResidentSSDTrace(lambda *x: p(p(*x[:5]), *x[5:])), *range(9), stream=21)
    p.frames[-1]._submissions = [SimpleNamespace(_active=1)]
    with pytest.raises(ValueError, match='closed reader scopes'):
        pb.retire_async(22)
    assert not pb.retiring
    assert all(f._retirement is None for f in p.frames)


def test_fanout_and_shared_public_inputs_accumulate_all_paths():
    p = Program()
    def fn(*x):
        first = p(*x)
        second = p(first, *x[1:])
        return p(second, x[1], first, x[3], x[4])
    _, pb = vjp(ResidentSSDTrace(fn), *range(5), stream=21)
    with pb(1).read(22) as gradients:
        assert gradients == (2, 4, 3, 4, 4)
    pb.close()
