"""External full writes are SSA effects until explicit native commit."""
import os
from pathlib import Path
import numpy as np
import pytest
from tessera.compiler.native_source_state import compile_source_state, _aliases
from tessera.compiler.source_control_flow import SourceControlFlowError


def mutate(x,alias,step):
    local=x
    local[:]=local+step
    try:
        if alias < step+step:
            raise ValueError('small')
        x[:]=alias*alias
    except ValueError:
        x[:]=alias-step
    finally:
        local[:]=local+step
    return alias*alias


def mutate_loop(x,alias,step):
    while alias < step+step:
        x[:]=x+step
        if alias > step:
            return alias*alias
    return x*x


@pytest.mark.parametrize('fn,value',[(mutate,0.),(mutate,3.),(mutate_loop,0.),(mutate_loop,3.)])
def test_native_state_preserves_full_input_aliases(fn,value):
    if not Path(os.environ.get('TESSERA_JIT_LIB','/nonexistent')).is_file():pytest.skip('native CPU JIT required')
    x=np.array([value],np.float32);step=np.ones(1,np.float32)
    oracle=x.copy();expected=fn(oracle,oracle,step.copy())
    with compile_source_state(fn,x,x,step,mutable=(0,),max_steps=4) as program:
        np.testing.assert_array_equal(x,[value])
        assert 'tessera.source_state' in program.native_ir
        actual=program.run(x,x,step)
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(x,oracle)
        with pytest.raises(ValueError,match='alias topology'):
            program.run(x,x.copy(),step)
        readonly=x.view();readonly.flags.writeable=False
        with pytest.raises(ValueError,match='read-only'):
            program.run(readonly,readonly,step)
    with pytest.raises(ValueError,match='closed'):program.run(x,x,step)


def test_partial_aliases_refuse_before_execution():
    x=np.arange(8,dtype=np.float32)
    with pytest.raises(ValueError,match='partial'):_aliases((x[:4],x[1:5]))


def test_mutable_return_alias_is_not_silently_snapshotted():
    def bad(x):
        x[:]=x*x
        return x
    with pytest.raises(SourceControlFlowError,match='return aliases'):
        compile_source_state(bad,np.ones(1,np.float32),mutable=(0,))


def test_readonly_alias_is_not_made_writable_by_another_argument():
    x=np.ones(1,np.float32);alias=x.view();alias.flags.writeable=False
    with pytest.raises(ValueError,match='read-only'):
        compile_source_state(mutate,x,alias,np.ones(1,np.float32),mutable=(0,))


@pytest.mark.parametrize('declared',[False,True])
def test_gpu_state_refuses_unprojected_input_alias_contract(declared):
    from tessera.compiler.trace import trace
    from tessera.compiler.source_control_flow import to_native_source_ir
    from tessera.compiler.native_source_state import materialize_source_state
    def two(x,y):
        return x+y
    ir=to_native_source_ir(trace(two,np.ones(1,np.float32),np.ones(1,np.float32),
        source_control_flow=True,source_state_groups=((0,),) if declared else ()))
    with pytest.raises(ValueError,match='one declared state input'):
        materialize_source_state(ir,compiler='/missing',llvm_bin='/missing',
                                 backend='rocm',chip='gfx1151',capacity=1)
