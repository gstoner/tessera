"""Source edges feed the existing tracer; no AST Graph emitter is introduced."""
import numpy as np
import pytest
from tessera import ops
from tessera.compiler.trace import trace, to_graph_ir_module
from tessera.compiler.source_control_flow import SourceControlFlowError


def normalized(value):
    import copy
    from tessera.compiler.graph_ir import IROp
    value = copy.deepcopy(value)
    def clear(node):
        if isinstance(node, IROp):
            node.source_span = None
            clear(node.kwargs)
        elif isinstance(node, dict):
            for item in node.values(): clear(item)
        elif isinstance(node, (list, tuple)):
            for item in node: clear(item)
    clear(value.body)
    return to_graph_ir_module(value).to_mlir(canonical=True)


def branches(x, y):
    if x < y:
        if ops.lt(x, ops.sub(ops.sub(y, y), y)):
            return ops.mul(x, x)
        x = ops.add(x, y)
    else:
        x = ops.sub(x, y)
    return ops.mul(x, y)


def loop(x, step, limit):
    while x < limit:
        x = x + step
    return x * x


@pytest.mark.parametrize('value', [-3., -1., 3.])
def test_nested_source_branches_and_early_returns(value):
    values = [np.array([value], np.float32), np.array([2.], np.float32)]
    captured = trace(branches, *values, source_control_flow=True)
    from tessera.control import cond
    def explicit(x, y):
        return cond(ops.lt(x, y), lambda: cond(ops.lt(x, ops.sub(ops.sub(y,y),y)),
                    lambda: ops.mul(x,x), lambda: ops.mul(ops.add(x,y),y)),
                    lambda: ops.mul(ops.sub(x,y),y))
    assert normalized(captured) == normalized(trace(explicit, *values))
    module = to_graph_ir_module(captured)
    assert module.verify().ok
    assert any(op.op_name == 'tessera.control_if' for op in captured.body)


def test_source_while_reaches_typed_region_consumer():
    values = [np.array([v], np.float32) for v in (0., 1., 3.)]
    captured = trace(loop, *values, source_control_flow=True, max_steps=5)
    from tessera.control import while_loop
    def explicit(x, step, limit):
        value = while_loop(lambda x: ops.lt(x,limit), lambda x: ops.add(x,step), x, max_steps=5)
        return ops.mul(value,value)
    assert normalized(captured) == normalized(trace(explicit,*values))
    assert to_graph_ir_module(captured).verify().ok


def test_source_unsupported_effects_fail_closed():
    calls = []
    def effect(x):
        calls.append(x)
        return x
    with pytest.raises(SourceControlFlowError):
        trace(effect, np.array(1., np.float32), source_control_flow=True)
    assert not calls


@pytest.mark.parametrize('fn,values', [(branches, [-3., 2.]), (branches, [-1., 2.]),
                                     (branches, [3., 2.]), (loop, [0., 1., 3.])])
def test_source_cfg_native_compiler_consumer(fn, values):
    from tessera.compiler.source_control_flow import to_native_source_ir
    from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
    tool = find_tessera_opt()
    if tool is None: pytest.skip('native compiler required')
    arrays = [np.array([v], np.float32) for v in values]
    text = to_native_source_ir(trace(fn, *arrays, source_control_flow=True, max_steps=5))
    lowered = run_tessera_opt(tool, text, '--tessera-to-linalg')
    assert 'scf.' in lowered and 'tessera.control_' not in lowered
    import os
    from pathlib import Path
    if not Path(os.environ.get('TESSERA_JIT_LIB','/nonexistent')).is_file():
        return  # Parser/lowering proof only; owning-host execution is separate.
    from tessera import _jit_boundary as jit
    handle = jit.compile_module(text)
    try:
        result = np.empty((1,), np.float32)
        jit.invoke(handle, 'source_program', arrays, result)
        np.testing.assert_allclose(result, fn(*arrays))
    finally:
        jit.destroy(handle)


def test_numeric_tensor_truth_is_not_reinterpreted_as_positive_mask():
    def ambiguous(x):
        if x:
            return ops.mul(x,x)
        return ops.add(x,x)
    with pytest.raises(SourceControlFlowError, match='explicit comparison'):
        trace(ambiguous, np.array([-1.], np.float32), source_control_flow=True)


def test_source_while_requires_an_explicit_bound():
    with pytest.raises(SourceControlFlowError, match='max_steps'):
        trace(loop, *[np.array([v], np.float32) for v in (0.,1.,3.)], source_control_flow=True)


def test_native_source_loop_has_exhaustion_guard_before_body():
    from tessera.compiler.source_control_flow import to_native_source_ir
    values = [np.array([v], np.float32) for v in (0.,1.,3.)]
    text = to_native_source_ir(trace(loop, *values, source_control_flow=True, max_steps=1))
    assert text.index('cf.assert') < text.index('} do {')
    assert 'source while exceeded max_steps' in text


def mixed_edges(x, step, limit):
    total = x - x
    while x < limit:
        x = x + step
        if x < step:
            continue
        total = total + x
        if total > limit:
            break
    assert total >= x - x, "negative accumulated value"
    return total * total


@pytest.mark.parametrize('values', [(-2.,1.,4.), (0.,1.,3.), (5.,1.,3.)])
def test_expanded_cfg_and_assertion_native_execution(values):
    from tessera.compiler.source_control_flow import to_native_source_ir
    from tessera.compiler.scheduled_matmul import find_tessera_opt
    import os
    import subprocess
    compiler = find_tessera_opt()
    if compiler is None: pytest.skip('native compiler required')
    arrays = [np.array([v], np.float32) for v in values]
    native = to_native_source_ir(trace(mixed_edges, *arrays, source_control_flow=True, max_steps=8))
    assert '"cf.assert"' in native and 'scf.if' in native
    result = subprocess.run([str(compiler), '--tessera-to-linalg'], input=native, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    if os.environ.get('TESSERA_JIT_LIB'):
        from tessera import _jit_boundary as jit
        handle = jit.compile_module(native)
        try:
            output = np.empty((1,),np.float32)
            jit.invoke(handle,'source_program',arrays,output)
            np.testing.assert_array_equal(output,mixed_edges(*arrays))
        finally: jit.destroy(handle)


def test_source_assertion_does_not_run_host_effects():
    calls=[]
    def bad(x,y):
        assert x < y, calls.append('bad')
        return x
    with pytest.raises(SourceControlFlowError, match='literal strings'):
        trace(bad,np.ones(1,np.float32),np.zeros(1,np.float32),source_control_flow=True)
    assert calls == []


def test_unmodelled_call_on_dead_path_is_rejected_before_capture():
    calls=[]
    def bad(x):
        return x*x
        unused = calls.append(x)
    with pytest.raises(SourceControlFlowError, match='unmodelled call effect'):
        trace(bad,np.ones(1,np.float32),source_control_flow=True)
    assert calls == []


def test_expansion_merges_iterations_before_the_next_step():
    from tessera.compiler.source_control_flow import to_native_source_ir
    values=[np.array([v],np.float32) for v in (0.,1.,2.)]
    sizes=[len(to_native_source_ir(trace(mixed_edges,*values,source_control_flow=True,max_steps=n))) for n in (4,8)]
    assert sizes[1] < 2.5*sizes[0]


@pytest.mark.parametrize('failure', ['budget','assertion'])
def test_native_source_failure_effect_is_not_erased(failure):
    import os
    import subprocess
    import sys
    from pathlib import Path
    if not Path(os.environ.get('TESSERA_JIT_LIB','/nonexistent')).is_file():
        pytest.skip('native JIT owning host required')
    code='''import numpy as np
from tessera.compiler.trace import trace
from tessera.compiler.source_control_flow import to_native_source_ir
from tessera import _jit_boundary as jit
from tests.unit.test_source_control_flow import mixed_edges, assertion_effect
arrays=[np.array([v],np.float32) for v in VALUES]
ir=to_native_source_ir(trace(FUNCTION,*arrays,source_control_flow=True,max_steps=BOUND))
h=jit.compile_module(ir)
print('invoking compiled source',flush=True)
jit.invoke(h,'source_program',arrays,np.empty(1,np.float32))
'''.replace('VALUES','(0.,1.,100.)' if failure=='budget' else '(-2.,1.,-3.)').replace('BOUND','2').replace('FUNCTION','mixed_edges' if failure=='budget' else 'assertion_effect')
    result=subprocess.run([sys.executable,'-c',code],text=True,capture_output=True, env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[2]/'python')})
    import signal
    assert result.returncode == -signal.SIGABRT, result.stderr
    assert 'invoking compiled source' in result.stdout


def assertion_effect(x,step,limit):
    assert x > step, "negative accumulated value"
    return x*x
