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


@pytest.mark.parametrize('chunk',[1,2,3])
def test_native_ssd_checkpoint_vjp_all_input_gradients(chunk):
    from tessera import _jit_boundary as jit
    from tessera.compiler.ssd_checkpoint_ad import lower_checkpoint_vjp, SSDCheckpointProgram
    tool = compiler()
    if jit._find_dylib() is None:
        pytest.skip('native CPU JIT required')
    parent = lower_scheduled_ssd(3,2,2,2,chunk,compiler=tool)
    adjoint = lower_checkpoint_vjp(parent,compiler=tool)
    adjoint.validate(tool)
    with pytest.raises(ValueError,match='replay'):
        replace(adjoint,lowered_ir=adjoint.lowered_ir.replace('arith.mulf','arith.addf')).validate(tool)
    rng = np.random.default_rng(53)
    inputs = [rng.uniform(-.4,.4,shape).astype(np.float32) for shape in [(3,2,2),(3,2),(3,2,2),(3,2,2),(2,2,2)]]
    def forward(values):
        x,d,b,c,state = values
        state = state.copy()
        ys,saved = [],[]
        for t in range(3):
            state = d[t,:,None,None]*state+b[t,:,:,None]*x[t,:,None,:]
            ys.append((c[t,:,:,None]*state).sum(axis=1))
            if (t+1)%chunk == 0 or t == 2:
                saved.append(state.copy())
        return np.array(ys),state,np.array(saved)
    results = forward(inputs)
    seeds = [rng.uniform(-.4,.4,r.shape).astype(np.float32) for r in results]
    grads = [np.empty_like(v) for v in inputs]
    original = [v.copy() for v in inputs]
    handle = jit.compile_module(adjoint.lowered_ir)
    try:
        jit.invoke(handle,'ssd_vjp',inputs+[results[2]]+seeds,grads)
    finally:
        jit.destroy(handle)
    def loss(values):
        return sum(np.sum(a*b) for a,b in zip(forward(values),seeds,strict=True))
    with SSDCheckpointProgram(parent,compiler=tool) as program:
        observed,owned_grads = program.vjp(inputs,seeds)
        for got,want in zip(observed,results,strict=True):
            np.testing.assert_allclose(got,want,rtol=1e-5,atol=1e-6)
        for got,want in zip(owned_grads,grads,strict=True):
            np.testing.assert_allclose(got,want,rtol=2e-5,atol=2e-6)
        with pytest.raises(ValueError,match='ABI'):
            program.vjp(inputs,seeds[:-1])
    with pytest.raises(ValueError,match='closed'):
        program.vjp(inputs,seeds)
    for i,value in enumerate(inputs):
        numeric = np.empty_like(value)
        for idx in np.ndindex(value.shape):
            plus,minus = [v.astype(np.float64) for v in inputs],[v.astype(np.float64) for v in inputs]
            plus[i][idx] += 1e-5
            minus[i][idx] -= 1e-5
            numeric[idx] = (loss(plus)-loss(minus))/2e-5
        np.testing.assert_allclose(grads[i],numeric,rtol=2e-5,atol=2e-6)
        np.testing.assert_array_equal(value,original[i])


def test_ssd_automatic_grad_uses_private_forward_checkpoints():
    from tessera import _jit_boundary as jit, ops
    from tessera.autodiff import grad
    from tessera.compiler.ssd_checkpoint_ad import SSDCheckpointProgram
    tool = compiler()
    if jit._find_dylib() is None:
        pytest.skip('native CPU JIT required')
    parent = lower_scheduled_ssd(3,1,2,2,2,compiler=tool)
    values = [np.full(shape,.2,np.float32) for shape in [(3,1,2),(3,1),(3,1,2),(3,1,2),(1,2,2)]]
    with SSDCheckpointProgram(parent,compiler=tool) as program:
        seeds = [np.ones((3,1,2),np.float32),np.zeros((1,2,2),np.float32),np.zeros((2,1,2,2),np.float32)]
        _,expected = program.vjp(values,seeds)
        def loss(*arrays):
            return ops.sum(program(*arrays))
        actual = grad(loss,argnums=(0,1,2,3,4))(*values)
        for got,want in zip(actual,expected,strict=True):
            np.testing.assert_allclose(got,want,rtol=1e-5,atol=1e-6)
