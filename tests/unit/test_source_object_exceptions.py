"""Declared object fields and static exception identity use native completion SSA."""
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from tessera.compiler.jit import jit


@pytest.fixture(autouse=True)
def native():
    if not Path(os.environ.get('TESSERA_JIT_LIB','/nonexistent')).is_file():pytest.skip('native CPU JIT required')


def field_step(state):
    state.x[:]=state.x+state.y
    return state.x*state.x


def mapping_step(state):
    state['x'][:]=state['x']+state['y']
    return state['x']*state['x']


@pytest.mark.parametrize('fn,container',[(field_step,SimpleNamespace),(mapping_step,dict)])
def test_declared_object_fields_preserve_native_writes(fn,container):
    state=container(x=np.ones(2,np.float32),y=np.full(2,3,np.float32))
    with jit(source_control_flow=True,source_mutable=(0,),source_fields=((0,('x','y')),))(fn) as program:
        np.testing.assert_array_equal(program(state),[16,16])
        x=state['x'] if type(state) is dict else state.x
        np.testing.assert_array_equal(x,[4,4])


def payload(x):
    try:
        if x > x+x:raise ValueError('negative')
        raise ValueError('positive')
    except (TypeError,ValueError):
        raise


@pytest.mark.parametrize('value,message',[(-1.,'negative'),(1.,'positive')])
def test_payload_and_reraise_survive_native_transport(value,message):
    with jit(source_control_flow=True,source_error_specs=(((1,),'f32'),))(payload) as program:
        with pytest.raises(ValueError,match=message):program(np.array([value],np.float32))


def test_handler_matches_builtin_inheritance():
    def f(x):
        try:raise KeyError('key')
        except Exception:return x+x
    with jit(source_control_flow=True)(f) as program:
        np.testing.assert_array_equal(program(np.ones(1,np.float32)),[2])


def test_custom_accessors_are_not_executed():
    class Custom:
        @property
        def x(self):raise AssertionError('must not execute')
    with jit(source_control_flow=True,source_fields=((0,('x','y')),))(field_step) as program:
        with pytest.raises(ValueError,match='custom accessors'):program(Custom())


def test_source_jit_vjp_uses_native_products():
    def cube(x):
        return x*x*x
    x=np.array([2.,3.],np.float32)
    with jit(source_control_flow=True)(cube) as program:
        values,grads=program.vjp(x,cotangents=(np.array([1.,2.],np.float32),))
    np.testing.assert_array_equal(values[0],[8.,27.])
    np.testing.assert_array_equal(grads[0],[12.,54.])


def test_finally_reraises_pending_exception():
    def f(x):
        try:raise ValueError('pending')
        finally:raise
    with jit(source_control_flow=True,source_error_specs=(((1,),'f32'),))(f) as program:
        with pytest.raises(ValueError,match='pending'):program(np.ones(1,np.float32))


def test_read_only_overlapping_inputs_are_snapshot_values():
    def add(x,y):
        return x+y
    base=np.arange(5,dtype=np.float32)
    with jit(source_control_flow=True)(add) as program:
        np.testing.assert_array_equal(program(base[:-1],base[1:]),[1,3,5,7])
    np.testing.assert_array_equal(base,np.arange(5,dtype=np.float32))


def test_overlapping_writes_still_refuse():
    def update(x,y):
        x[:]=x+y
        return x+x
    base=np.arange(5,dtype=np.float32)
    with jit(source_control_flow=True,source_mutable=(0,))(update) as program:
        with pytest.raises(ValueError,match='partial'):program(base[:-1],base[1:])
    np.testing.assert_array_equal(base,np.arange(5,dtype=np.float32))
