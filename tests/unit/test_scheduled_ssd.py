"""Shared SSD semantics and carry/checkpoint lineage, before device admission."""
from dataclasses import replace
import numpy as np
import pytest
from tessera.compiler.scheduled_ssd import lower_scheduled_ssd
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt


def compiler():
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip('native Schedule compiler required')
    return tool


@pytest.mark.parametrize('chunk', [1, 2, 5])
def test_ssd_native_recurrence_and_tail_checkpoints(chunk):
    from tessera import _jit_boundary as jit
    artifact = lower_scheduled_ssd(5, 2, 3, 2, chunk, compiler=compiler())
    artifact.validate(compiler())
    assert 'schedule.ssd' not in artifact.lowered_ir
    assert 'scf.for' in artifact.lowered_ir and 'fastmath' not in artifact.lowered_ir
    if jit._find_dylib() is None:
        pytest.skip('native CPU JIT required')
    rng = np.random.default_rng(74)
    x, decay, b, c, initial = [rng.uniform(-.5,.5,shape).astype(np.float32)
        for shape in [(5,2,2),(5,2),(5,2,3),(5,2,3),(2,3,2)]]
    original = initial.copy()
    state = initial.copy()
    expected = np.empty_like(x)
    checkpoints = []
    for t in range(5):
        state = decay[t,:,None,None] * state + b[t,:,:,None] * x[t,:,None,:]
        expected[t] = (c[t,:,:,None] * state).sum(axis=1)
        if (t+1)%chunk == 0 or t == 4:
            checkpoints.append(state.copy())
    outputs = [np.empty_like(x), np.empty_like(initial), np.empty((len(checkpoints),2,3,2),np.float32)]
    handle = jit.compile_module(artifact.lowered_ir)
    try:
        jit.invoke(handle, 'ssd', [x,decay,b,c,initial], outputs)
    finally:
        jit.destroy(handle)
    for got, want in zip(outputs, [expected,state,np.array(checkpoints)], strict=True):
        np.testing.assert_allclose(got,want,rtol=1e-5,atol=1e-6)
    np.testing.assert_array_equal(initial,original)


@pytest.mark.parametrize('before,after', [
    ('chunk_size = 2', 'chunk_size = -1'),
    ('chunk_size = 2', 'chunk_size = 6'),
    ('tensor<3x2x3x2xf32>', 'tensor<2x2x3x2xf32>'),
    ('tensor<5x2xf32>', 'tensor<5x3xf32>'),
    ('xf32>', 'xf64>'),
    ('tensor<5x2x2xf32>', 'tensor<0x2x2xf32>'),
    ('tensor<5x2x2xf32>', 'tensor<16777217x2x2xf32>'),
])
def test_ssd_rejects_invalid_schedule(before,after):
    tool = compiler()
    artifact = lower_scheduled_ssd(5,2,3,2,2,compiler=tool)
    assert before in artifact.schedule_ir
    with pytest.raises(RuntimeError):
        run_tessera_opt(tool,artifact.schedule_ir.replace(before,after),'--tessera-schedule-to-tile')


def test_ssd_replay_rejects_changed_recurrence():
    artifact = lower_scheduled_ssd(5,2,3,2,2,compiler=compiler())
    with pytest.raises(ValueError,match='replay'):
        replace(artifact,lowered_ir=artifact.lowered_ir.replace('arith.mulf','arith.addf')).validate(compiler())
