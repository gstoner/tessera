"""Host isolation protocol tests, plus explicit owning-device execution."""
from types import SimpleNamespace
import os
import time

import numpy as np
import pytest

from tessera.compiler.isolated_rocm_attention import IsolatedROCmAttentionTape, _UNCERTAIN


def fake_program(stall=False):
    arg = lambda name: SimpleNamespace(name=name,direction='input',layout='row_major')
    return SimpleNamespace(image=SimpleNamespace(image_digest='identity'),stall=stall,
        descriptors=[SimpleNamespace(buffers=[arg('q')]),SimpleNamespace(buffers=[arg('q'),arg('do')])])


def worker(connection,program,buffers,device,name):
    assert device == 2
    connection.send(('ready','identity'))
    while True:
        message = connection.recv()
        if message[0] == 'close':
            connection.send(('closed',));connection.close();return
        if program.stall:
            time.sleep(60)
        connection.send(('result',(message[1].astype(np.float32),)))


def test_recovery_requires_death_and_replacement_uses_fresh_owner(monkeypatch):
    monkeypatch.setattr('tessera.compiler.isolated_rocm_attention._worker',worker)
    program = fake_program(stall=True)
    buffers = dict(q=np.ones((2,2),np.float16),do=np.ones((2,2),np.float16))
    tape = IsolatedROCmAttentionTape(program,buffers,device=2)
    tape.timeout = .1
    with pytest.raises(TimeoutError):
        tape.backward(buffers['do'])
    with pytest.raises(ValueError,match='confirmed'):
        tape.replacement()
    tape.recover()
    assert tape.closed and tape.lease.reusable and tape not in _UNCERTAIN
    program.stall = False
    tape.timeout = 30
    with tape.replacement() as replacement:
        assert replacement._process.pid != tape._process.pid
        with pytest.raises(ValueError,match='information'):
            replacement.backward(np.full((2,2),.1,np.float32),casting='exact')
        assert not replacement.failed
        np.testing.assert_array_equal(replacement.backward(np.ones((2,2),np.float32),casting='exact')[0],1)


@pytest.mark.skipif(os.environ.get('TESSERA_GFX1201_DEVICE_PROOF') != '1',reason='owning gfx1201 proof')
@pytest.mark.parametrize('checkpoint',['saved','recompute'])
def test_isolated_resident_attention_on_gfx1201(checkpoint):
    from tessera import runtime as rt
    from tessera.compiler.scheduled_attention_backward import lower_scheduled_attention_backward
    from tessera.compiler.rocm_native import package_scheduled_attention_backward
    from benchmarks.rocm.benchmark_rocm_attention_backward_program import _module, _reference
    assert rt._rocm_live_arch() == 'gfx1201'
    schedule = lower_scheduled_attention_backward(_module(1,4,2,17,19,64,dtype='fp16',
        dropout_p=0,lse_checkpoint=checkpoint),target='rocm_gfx1201')
    program = package_scheduled_attention_backward(schedule,pipeline_name='tessera-lower-to-rocm')
    assert ('save_lse = true' in program.target_ir) == (checkpoint == 'saved')
    rng = np.random.default_rng(254)
    q,k,v = [(rng.normal(size=s)*.2).astype(np.float16) for s in [(1,4,17,64),(1,2,19,64),(1,2,19,64)]]
    do = np.ones(q.shape,np.float16)
    bias = np.zeros((1,4,17,19),np.float32)
    buffers = dict(q=q,key=k,v=v,do=do,bias=bias,dq=np.empty(q.shape,np.float32),
                   dk=np.empty(k.shape,np.float32),dv=np.empty(v.shape,np.float32))
    with IsolatedROCmAttentionTape(program,buffers) as tape:
        for scale in (1,2):
            actual = tape.backward(do.astype(np.float32)*scale,casting='exact')
            expected = _reference(do*scale,q,k,v,bias,dropout_p=0)
            for got,want in zip(actual,expected,strict=True):
                np.testing.assert_allclose(got,want,rtol=.04,atol=.003)
        # Force an uncertain worker outcome after real resident allocation;
        # process death, not a second driver operation, authorizes replacement.
        tape._process.terminate()
        with pytest.raises((EOFError,BrokenPipeError,ConnectionResetError,RuntimeError)):
            tape.backward(do)
        tape.recover()
        assert tape.lease.reusable
        with tape.replacement() as replacement:
            actual = replacement.backward(do)
            expected = _reference(do,q,k,v,bias,dropout_p=0)
            for got,want in zip(actual,expected,strict=True):
                np.testing.assert_allclose(got,want,rtol=.04,atol=.003)


def test_zero_only_worker_cannot_pass_nonzero_health_gate():
    from tessera.compiler.isolated_rocm_attention import _check_health
    from tessera.compiler.attention_contract import reference_attention_backward_split_reduced
    values = {name:np.ones((1,1,2,2),np.float16) for name in ('q','k','v','do')}
    provenance = dict(split_count=2,scale=1.,causal=False,window_left=-1,window_right=-1,softcap=0.,dropout_p=0.,dropout_seed=0)
    program = SimpleNamespace(descriptors=[SimpleNamespace(provenance=provenance,buffers=[
        SimpleNamespace(name=name,direction='input',layout='row_major') for name in ('q','k','v')])])
    broken = SimpleNamespace(backward=lambda do: tuple(np.zeros(do.shape,np.float32) for _ in range(3)))
    with pytest.raises(RuntimeError,match='nonzero-VJP'):
        _check_health(broken,program,values,'do')
    healthy = SimpleNamespace(backward=lambda do: reference_attention_backward_split_reduced(do,values['q'],values['k'],values['v'],**provenance))
    _check_health(healthy,program,values,'do')
