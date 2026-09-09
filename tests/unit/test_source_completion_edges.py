"""Native values and effects survive return/exception completion edges."""
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from tessera.compiler.trace import trace
from tessera.compiler.source_control_flow import to_native_source_ir, SourceControlFlowError
from tessera.compiler.scheduled_matmul import find_tessera_opt


def loop_return(x,step,limit):
    total=x-x
    while x < limit:
        x=x+step
        total=total+x
        if total > limit:
            return total,x*x
    return x,total


def nested_return(x,step,limit):
    y=x
    while x < limit:
        x=x+step
        y=x
        while y < limit:
            y=y+step
            if y > x+step:
                return y*y
    return x*x


def handled_raise(x,step,limit):
    y=x
    try:
        if x < limit:
            raise ValueError('below limit')
        y=x+step
    except ValueError:
        y=x-step
    else:
        y=y+step
    finally:
        y=y*y
    return y


def finally_return(x,step,limit):
    y=x
    while x < limit:
        try:
            x=x+step
            if x > step:
                return x*x
        finally:
            y=y+step
    return y*y


def caught_assertion(x,step,limit):
    try:
        assert x > limit, 'too small'
        return x*x
    except AssertionError:
        return x+step


def finally_overrides_return(x,step,limit):
    while x < limit:
        try:
            return x+step
        finally:
            return x*x  # noqa: B012 - exercise Python finally overriding a return
    return step*step


def nested_handler(x,step,limit):
    y=x
    try:
        try:
            if x < limit:
                raise RuntimeError('outer handler')
            y=x+step
        except ValueError:
            y=x-step
        finally:
            y=y+step
    except RuntimeError:
        y=y*y
    return y+step


def exception_crosses_loop(x,step,limit):
    y=x
    try:
        while x < limit:
            x=x+step
            if x > step:
                raise ValueError('leave loop')
            y=y+step
    except ValueError:
        y=x*x
    finally:
        y=y+step
    return y


@pytest.mark.parametrize('fn,values,bound',[
    (exception_crosses_loop,(0.,1.,3.),4),(exception_crosses_loop,(4.,1.,3.),4),
    (finally_overrides_return,(2.,1.,3.),4),(finally_overrides_return,(4.,1.,3.),4),
    (nested_handler,(0.,1.,3.),4),(nested_handler,(4.,1.,3.),4),
    (loop_return,(0.,1.,3.),4),(loop_return,(5.,1.,3.),4),
    (nested_return,(0.,1.,3.),3),(nested_return,(4.,1.,3.),3),
    (handled_raise,(0.,1.,3.),4),(handled_raise,(4.,1.,3.),4),
    (finally_return,(0.,1.,3.),4),(finally_return,(4.,1.,3.),4),
    (caught_assertion,(0.,1.,3.),4),(caught_assertion,(4.,1.,3.),4),
])
def test_completion_edges_execute_native(fn,values,bound):
    compiler=find_tessera_opt()
    if compiler is None:pytest.skip('native compiler required')
    arrays=[np.array([v],np.float32) for v in values]
    ir=to_native_source_ir(trace(fn,*arrays,source_control_flow=True,max_steps=bound))
    result=subprocess.run([str(compiler),'--tessera-to-linalg'],input=ir,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    if Path(os.environ.get('TESSERA_JIT_LIB','/nonexistent')).is_file():
        from tessera import _jit_boundary as jit
        expected=fn(*[a.copy() for a in arrays])
        expected=expected if isinstance(expected,tuple) else (expected,)
        outputs=[np.empty_like(a) for a in expected]
        handle=jit.compile_module(ir)
        try:
            jit.invoke(handle,'source_program',arrays,outputs if len(outputs)>1 else outputs[0])
            for actual,want in zip(outputs,expected,strict=True):np.testing.assert_array_equal(actual,want)
        finally:jit.destroy(handle)


def test_uncaught_exception_and_exception_objects_refuse():
    def uncaught(x):
        raise ValueError('no handler')
    def object_escape(x):
        try:
            raise ValueError('bad')
        except ValueError as error:
            return error
    for fn in (uncaught,object_escape):
        with pytest.raises(SourceControlFlowError):trace(fn,np.ones(1,np.float32),source_control_flow=True)
