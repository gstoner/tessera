"""Host isolation protocol tests for the CUDA twin, plus explicit sm_120 execution."""
from types import SimpleNamespace
import os
import time

import numpy as np
import pytest

from tessera.compiler.isolated_cuda_attention import IsolatedCUDAAttentionTape, reference_checkpoint_backward
from tessera.compiler.isolated_attention import _UNCERTAIN


def fake_pair(stall=False):
    return SimpleNamespace(contract_digest='identity', stall=stall,
                           forward=SimpleNamespace(descriptor=SimpleNamespace(provenance=dict(scale=.5, causal=True))))


def worker(connection, pair, q, k, v, device):
    assert device == 2
    connection.send(('ready', 'identity'))
    while True:
        message = connection.recv()
        if message[0] == 'close':
            connection.send(('closed',)); connection.close(); return
        if pair.stall:
            time.sleep(60)
        connection.send(('result', (message[1].astype(np.float32),)))


def _arrays():
    rng = np.random.default_rng(3)
    q = rng.normal(0, .2, (1, 2, 3, 4)).astype(np.float32)
    k = rng.normal(0, .2, (1, 1, 4, 4)).astype(np.float32)
    v = rng.normal(0, .2, (1, 1, 4, 3)).astype(np.float32)
    return q, k, v


def test_recovery_requires_death_and_replacement_uses_fresh_owner(monkeypatch):
    monkeypatch.setattr('tessera.compiler.isolated_cuda_attention._worker', worker)
    pair = fake_pair(stall=True)
    q, k, v = _arrays()
    tape = IsolatedCUDAAttentionTape(pair, q, k, v, device=2)
    assert tape._shape == (1, 2, 3, 3)
    tape.timeout = .1
    with pytest.raises(TimeoutError):
        tape.backward(np.ones((1, 2, 3, 3), np.float32))
    with pytest.raises(ValueError, match='confirmed'):
        tape.replacement()
    tape.recover()
    assert tape.closed and tape.lease.reusable and tape not in _UNCERTAIN
    pair.stall = False
    tape.timeout = 30
    with tape.replacement() as replacement:
        assert replacement._process.pid != tape._process.pid
        with pytest.raises(ValueError, match='shape'):
            replacement.backward(np.ones((1, 2, 3, 4), np.float32))
        with pytest.raises(ValueError, match='information'):
            replacement.backward(np.full((1, 2, 3, 3), .1, np.float64), casting='exact')
        assert not replacement.failed
        np.testing.assert_array_equal(replacement.backward(np.ones((1, 2, 3, 3), np.float64), casting='exact')[0], 1)


def test_owner_refuses_non_finite_or_misshaped_captures():
    q, k, v = _arrays()
    with pytest.raises(ValueError, match='finite rank-4'):
        IsolatedCUDAAttentionTape(fake_pair(), q[0], k, v)
    bad = q.copy(); bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite rank-4'):
        IsolatedCUDAAttentionTape(fake_pair(), bad, k, v)


def test_zero_only_worker_cannot_pass_nonzero_health_gate():
    from tessera.compiler.isolated_cuda_attention import _check_health
    q, k, v = _arrays()
    shape = (1, 2, 3, 3)
    driver = SimpleNamespace()
    calls = []

    def fake_run(driver_, tape, cotangent, shape_):
        calls.append(cotangent.copy())
        return tape.backward(cotangent)
    import tessera.compiler.isolated_cuda_attention as module
    original = module._run_backward
    module._run_backward = fake_run
    try:
        broken = SimpleNamespace(backward=lambda do: tuple(np.zeros(s, np.float32) for s in (q.shape, k.shape, v.shape)))
        with pytest.raises(RuntimeError, match='nonzero-VJP'):
            _check_health(driver, broken, fake_pair(), q, k, v, shape)
        healthy = SimpleNamespace(backward=lambda do: reference_checkpoint_backward(do, q, k, v, scale=.5, causal=True))
        _check_health(driver, healthy, fake_pair(), q, k, v, shape)
    finally:
        module._run_backward = original
    assert np.all(calls[0] == 0) and np.all(calls[1] == 1)


def test_reference_matches_the_pair_device_oracle_for_mha():
    """The GQA-aware reference reduces to the sm_120 pair test's numpy oracle."""
    q, k, v = _arrays()
    do = np.random.default_rng(5).normal(0, .2, (1, 2, 3, 3)).astype(np.float32)
    dq, dk, dv = reference_checkpoint_backward(do, q, k, v, scale=.5, causal=True)
    sq, sk = 3, 4
    scores = np.matmul(q, k.swapaxes(-1, -2)) * .5
    legal = np.arange(sk)[None, :] <= np.arange(sq)[:, None] + max(sk - sq, 0)
    scores = np.where(legal, scores, -np.inf)
    p = np.exp(scores - np.logaddexp.reduce(scores, axis=-1)[..., None])
    dp = do @ v.swapaxes(-1, -2)
    ds = p * (dp - (p * dp).sum(axis=-1, keepdims=True))
    np.testing.assert_allclose(dq, ds @ k * .5, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(dk, (ds.swapaxes(-1, -2) @ q * .5).sum(axis=1, keepdims=True), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(dv, (p.swapaxes(-1, -2) @ do).sum(axis=1, keepdims=True), rtol=1e-5, atol=1e-6)


@pytest.mark.hardware_nvidia
@pytest.mark.skipif(os.environ.get('TESSERA_SM120_DEVICE_PROOF') != '1', reason='owning sm_120 proof (TESSERA_SM120_DEVICE_PROOF=1)')
@pytest.mark.parametrize('sq,sk,causal', [(3, 4, True), (5, 3, True), (3, 4, False)])
def test_isolated_resident_attention_on_sm120(sq, sk, causal):
    from tests._support.nvidia import nvidia_cuda_host_ready
    from tessera.compiler.nvidia_native import package_attention_checkpoint_pair
    from tests.unit.test_nvidia_checkpoint_pair import checkpoint_modules
    if not nvidia_cuda_host_ready():
        pytest.skip('host WSL CUDA device/toolchain unavailable')
    forward, backward = checkpoint_modules(sq=sq, sk=sk, causal=causal)
    pair = package_attention_checkpoint_pair(forward, backward, pipeline_name='tessera-nvidia-pipeline-sm120')
    rng = np.random.default_rng(7121)
    q = rng.normal(0, .2, (1, 2, sq, 4)).astype(np.float32)
    k = rng.normal(0, .2, (1, 1, sk, 4)).astype(np.float32)
    v = rng.normal(0, .2, (1, 1, sk, 3)).astype(np.float32)
    do = rng.normal(0, .2, (1, 2, sq, 3)).astype(np.float32)
    with IsolatedCUDAAttentionTape(pair, q, k, v) as tape:
        for scale in (1, 2):
            actual = tape.backward(do * scale)
            expected = reference_checkpoint_backward(do * scale, q, k, v, scale=.5, causal=causal)
            for got, want in zip(actual, expected, strict=True):
                np.testing.assert_allclose(got, want, rtol=3e-5, atol=3e-6)
        # Force an uncertain worker outcome after real resident allocation;
        # process death, not a second driver operation, authorizes replacement.
        tape._process.terminate()
        with pytest.raises((EOFError, BrokenPipeError, ConnectionResetError, RuntimeError, TimeoutError)):
            tape.backward(do)
        tape.recover()
        assert tape.lease.reusable
        with tape.replacement() as replacement:
            actual = replacement.backward(do)
            expected = reference_checkpoint_backward(do, q, k, v, scale=.5, causal=causal)
            for got, want in zip(actual, expected, strict=True):
                np.testing.assert_allclose(got, want, rtol=3e-5, atol=3e-6)
