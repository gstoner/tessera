"""Native exception class transport preserves preceding declared writes."""
import os
from pathlib import Path
import numpy as np
import pytest
from tessera.compiler.native_source_state import compile_source_state
from tessera.compiler.jit import jit


def escape(x,limit):
    try:
        x[:]=x+x
        if x < limit:
            raise ValueError('small')
        return x*x
    finally:
        x[:]=x+x


def pure(x):
    return x*x


@pytest.fixture(autouse=True)
def native():
    if not Path(os.environ.get('TESSERA_JIT_LIB','/nonexistent')).is_file():
        pytest.skip('native LLVM JIT required')


@pytest.mark.parametrize('value',[1.,4.])
def test_native_exception_commits_preceding_state(value):
    x=np.array([value],np.float32);limit=np.array([3.],np.float32)
    with compile_source_state(escape,x,limit,mutable=(0,),error_specs=(((1,),'f32'),)) as program:
        if value==1:
            with pytest.raises(ValueError,match='small'):program.run(x,limit)
        else:np.testing.assert_array_equal(program.run(x,limit),[64.])
        np.testing.assert_array_equal(x,[value*4])


def test_strided_state_commits_to_owning_array():
    base=np.array([1.,99.,1.,99.],np.float32);view=base[::2]
    def update(x):
        x[:]=x*x+x
        return x+x
    with compile_source_state(update,view,mutable=(0,)) as program:
        np.testing.assert_array_equal(program.run(view),[4.,4.])
        np.testing.assert_array_equal(base,[2.,99.,2.,99.])


def test_public_jit_cache_is_bounded_and_owned():
    with jit(source_control_flow=True)(pure) as program:
        for n in range(1,7):
            x=np.arange(n,dtype=np.float32)
            np.testing.assert_array_equal(program(x=x),x*x)
        assert len(program._programs)==4
    assert not program._programs
    with pytest.raises(ValueError,match='closed'):program(np.ones(1,np.float32))


def test_public_jit_transports_exception():
    with jit(source_control_flow=True,source_mutable=(0,),
             source_error_specs=(((1,),'f32'),))(escape) as program:
        x=np.ones(1,np.float32)
        with pytest.raises(ValueError):program(x,np.array([3.],np.float32))
        np.testing.assert_array_equal(x,[4.])


def test_negative_stride_state_and_overlap_refusal():
    base=np.arange(1,5,dtype=np.float32)
    view=base[::-1]
    def update(x):
        x[:]=x+x
        return x*x
    with compile_source_state(update,view,mutable=(0,)) as program:
        program.run(view)
    np.testing.assert_array_equal(base,[2.,4.,6.,8.])
    overlap=np.lib.stride_tricks.as_strided(base,shape=(2,2),strides=(4,4))
    with pytest.raises(ValueError,match='self-overlapping'):
        compile_source_state(update,overlap,mutable=(0,))


def test_uncaught_assertion_is_a_host_exception_not_abort():
    def check(x):
        assert x > x+x, 'failed'
        return x*x
    with jit(source_control_flow=True,source_error_specs=(((1,),'f32'),))(check) as program:
        with pytest.raises(AssertionError):program(np.ones(1,np.float32))


def test_jit_refuses_incompatible_source_options():
    with pytest.raises(ValueError,match='incompatible'):
        jit(source_control_flow=True,target='nvidia')(pure)
