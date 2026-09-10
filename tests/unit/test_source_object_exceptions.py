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


def test_read_only_overlap_beside_disjoint_mutable_state():
    def update(state,x,y):
        state[:]=state+x+y
        return state+state
    base=np.arange(5,dtype=np.float32)
    state=np.ones(4,np.float32)
    with jit(source_control_flow=True,source_mutable=(0,))(update) as program:
        np.testing.assert_array_equal(program(state,base[:-1],base[1:]),[4,8,12,16])
        np.testing.assert_array_equal(program(state,base[:-1],base[1:]),[6,14,22,30])
    np.testing.assert_array_equal(base,np.arange(5,dtype=np.float32))


def test_plain_custom_object_storage_is_projected():
    class State:
        def __init__(self):
            self.x=np.ones(2,np.float32)
            self.y=np.full(2,3,np.float32)
    state=State()
    with jit(source_control_flow=True,source_mutable=(0,),source_fields=((0,('x','y')),))(field_step) as program:
        np.testing.assert_array_equal(program(state),[16,16])
    np.testing.assert_array_equal(state.x,[4,4])


def test_custom_getattribute_is_refused_without_execution():
    class State:
        def __getattribute__(self,name):raise AssertionError('must not execute')
    with jit(source_control_flow=True,source_fields=((0,('x','y')),))(field_step) as program:
        with pytest.raises(ValueError,match='custom accessors'):program(State())


def test_state_vjp_differentiates_next_state_without_committing():
    def evolve(x):
        x[:]=x*x
        return x*x
    x=np.array([2.,3.],np.float32)
    with jit(source_control_flow=True,source_mutable=(0,))(evolve) as program:
        values,grads=program.vjp(x,cotangents=(np.ones_like(x),np.ones_like(x)))
    np.testing.assert_array_equal(values[0],[16,81])
    np.testing.assert_array_equal(values[1],[4,9])
    np.testing.assert_array_equal(grads[0],[36,114])
    np.testing.assert_array_equal(x,[2,3])


def test_containing_input_orders_overlapping_writes_and_reads():
    def update(base,left,right):
        left[:]=left+left
        right[:]=right+left
        return base+base
    base=np.arange(1,6,dtype=np.float32)
    expected=base.copy()
    oracle=update(expected,expected[:-1],expected[1:])
    with jit(source_control_flow=True,source_mutable=(1,2))(update) as program:
        result=program(base,base[:-1],base[1:])
    np.testing.assert_array_equal(base,expected)
    np.testing.assert_array_equal(result,oracle)


def test_dynamic_exception_snapshots_payload_after_preceding_write():
    def fail(x):
        x[:]=x+x
        try:raise ValueError(x*x)
        finally:x[:]=x+x
    x=np.array([3.],np.float32)
    with jit(source_control_flow=True,source_mutable=(0,),source_error_specs=(((1,),'f32'),))(fail) as program:
        with pytest.raises(ValueError) as caught:program(x)
    np.testing.assert_array_equal(caught.value.args[0],[36])
    np.testing.assert_array_equal(x,[12])


def test_nested_dynamic_exception_reraise_preserves_outer_payload():
    def fail(x):
        try:raise ValueError(x+x)
        except ValueError:
            try:raise RuntimeError(x*x)
            except RuntimeError:pass
            raise
    with jit(source_control_flow=True,source_error_specs=(((1,),'f32'),))(fail) as program:
        with pytest.raises(ValueError) as caught:program(np.array([3.],np.float32))
    np.testing.assert_array_equal(caught.value.args[0],[6])


def test_object_state_vjp_returns_field_gradients_without_writing():
    def evolve(state):
        state.x[:]=state.x*state.y
        return state.x*state.x
    state=SimpleNamespace(x=np.array([2.],np.float32),y=np.array([3.],np.float32))
    with jit(source_control_flow=True,source_mutable=(0,),source_fields=((0,('x','y')),))(evolve) as program:
        values,gradients=program.vjp(state,cotangents=(np.ones(1,np.float32),np.zeros(1,np.float32),np.zeros(1,np.float32)))
    np.testing.assert_array_equal(values[0],[36])
    np.testing.assert_array_equal(gradients[0],[36])
    np.testing.assert_array_equal(gradients[1],[24])
    np.testing.assert_array_equal(state.x,[2])
    np.testing.assert_array_equal(state.y,[3])


def test_dynamic_exception_payload_survives_loop_exit():
    def fail(x,limit):
        while x < limit:
            x=x+x
            if x > limit:raise ValueError(x)
        return x*x
    with jit(source_control_flow=True,source_max_steps=4,source_error_specs=(((1,),'f32'),))(fail) as program:
        with pytest.raises(ValueError) as caught:program(np.array([2.],np.float32),np.array([5.],np.float32))
    np.testing.assert_array_equal(caught.value.args[0],[8])


def test_overlap_program_rejects_changed_view_offsets():
    from tessera.compiler.native_source_state import compile_source_state
    def update(base,left,right):
        left[:]=left+right
        return base+base
    base=np.arange(6,dtype=np.float32)
    with compile_source_state(update,base,base[:3],base[1:4],mutable=(1,)) as program:
        with pytest.raises(ValueError,match='alias topology'):
            program.run(base,base[:3],base[2:5])


def test_dynamic_exception_does_not_silently_snapshot_mutable_alias():
    from tessera.compiler.source_control_flow import SourceControlFlowError
    def fail(x):
        try:raise ValueError(x)
        finally:x[:]=x+x
    with jit(source_control_flow=True,source_mutable=(0,),source_error_specs=(((1,),'f32'),))(fail) as program:
        with pytest.raises(SourceControlFlowError,match='aliases mutable state'):
            program(np.array([2.],np.float32))


def test_positive_strided_writes_share_containing_root():
    def update(base,even,odd):
        even[:]=even+even
        odd[:]=odd+even
        return base+base
    base=np.arange(1,9,dtype=np.float32)
    expected=base.copy();oracle=update(expected,expected[::2],expected[1::2])
    with jit(source_control_flow=True,source_mutable=(1,2))(update) as program:
        np.testing.assert_array_equal(program(base,base[::2],base[1::2]),oracle)
    np.testing.assert_array_equal(base,expected)


def test_exact_alias_state_vjp_accumulates_at_root():
    def update(x,y):
        x[:]=x*y
        return y*y
    x=np.array([2.],np.float32)
    with jit(source_control_flow=True,source_mutable=(0,))(update) as program:
        values,gradients=program.vjp(x,x,cotangents=(np.ones(1,np.float32),np.zeros(1,np.float32)))
    np.testing.assert_array_equal(values[0],[16])
    np.testing.assert_array_equal(gradients[0],[32])
    np.testing.assert_array_equal(gradients[1],[0])
    np.testing.assert_array_equal(x,[2])


def test_explicit_exception_cause_crosses_native_completion():
    def fail(x):
        raise ValueError('outer') from RuntimeError('inner')
    with jit(source_control_flow=True,source_error_specs=(((1,),'f32'),))(fail) as program:
        with pytest.raises(ValueError,match='outer') as caught:program(np.ones(1,np.float32))
    assert type(caught.value.__cause__) is RuntimeError
    assert caught.value.__cause__.args==('inner',)
    assert caught.value.__suppress_context__


def test_explicit_cause_suppression_crosses_native_completion():
    def fail(x):
        raise ValueError('outer') from None
    with jit(source_control_flow=True,source_error_specs=(((1,),'f32'),))(fail) as program:
        with pytest.raises(ValueError,match='outer') as caught:program(np.ones(1,np.float32))
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__


def test_local_strided_state_views_preserve_alias_updates():
    def update(base):
        even=base[::2]
        even[:]=even+even
        odd=base[1::2]
        odd[:]=odd+even
        return base+base
    base=np.arange(1,9,dtype=np.float32)
    expected=base.copy();oracle=update(expected)
    with jit(source_control_flow=True,source_mutable=(0,))(update) as program:
        np.testing.assert_array_equal(program(base),oracle)
    np.testing.assert_array_equal(base,expected)


@pytest.mark.parametrize('selection',[(slice(None,None,-1),),(slice(None),slice(None,None,-1)),(slice(None,None,-1),slice(None,None,2))])
def test_negative_and_multidimensional_external_views(selection):
    def update(base,view):
        view[:]=view*view
        return base+base
    shape=(6,) if len(selection)==1 else (3,4)
    base=np.arange(1,np.prod(shape)+1,dtype=np.float32).reshape(shape)
    expected=base.copy();oracle=update(expected,expected[selection])
    with jit(source_control_flow=True,source_mutable=(1,))(update) as program:
        np.testing.assert_array_equal(program(base,base[selection]),oracle)
    np.testing.assert_array_equal(base,expected)


def test_local_negative_multidimensional_view_vjp():
    def update(base):
        part=base[::-1,::2]
        part[:]=part*part
        return base+base
    base=np.arange(1,13,dtype=np.float32).reshape(3,4)
    with jit(source_control_flow=True,source_mutable=(0,))(update) as program:
        values,grads=program.vjp(base,cotangents=(np.ones_like(base),np.zeros_like(base)))
        expected=np.full_like(base,2);expected[:,::2]=4*base[:,::2]
        np.testing.assert_array_equal(grads[0],expected)
        actual=program(base.copy())
    np.testing.assert_array_equal(values[0],actual)


def test_overlapping_slice_adjoint_accumulates_and_masks_overwrites():
    def update(base,left,right):
        left[:]=left*left
        right[:]=right+left
        return base*base
    base=np.arange(1,6,dtype=np.float64)
    seed=np.array([1.,-2.,3.,-4.,5.])
    with jit(source_control_flow=True,source_mutable=(1,2))(update) as program:
        values,grads=program.vjp(base,base[:-1],base[1:],cotangents=(seed,np.zeros_like(base)))
    finite=[]
    for i in range(base.size):
        plus=base.copy();minus=base.copy();plus[i]+=1e-5;minus[i]-=1e-5
        finite.append(np.sum((update(plus,plus[:-1],plus[1:])-update(minus,minus[:-1],minus[1:]))*seed)/(2e-5))
    np.testing.assert_allclose(grads[0],finite,rtol=1e-8,atol=1e-7)
    np.testing.assert_array_equal(grads[1],np.zeros(4))
    np.testing.assert_array_equal(grads[2],np.zeros(4))


def test_caught_exception_identity_cause_and_context():
    def fail(x):
        try:raise ValueError('inner')
        except ValueError as original:
            alias=original
            if alias is original:
                raise RuntimeError('outer') from original
            return x+x
    with jit(source_control_flow=True,source_error_specs=(((1,),'f32'),))(fail) as program:
        with pytest.raises(RuntimeError,match='outer') as caught:program(np.ones(1,np.float32))
    assert caught.value.__cause__ is caught.value.__context__
    assert caught.value.__cause__.args==('inner',)
    assert 'Native source raise at' in caught.value.__notes__[0]
    assert 'no Python frame executed' in caught.value.__notes__[0]


def test_named_reraise_preserves_exception_identity():
    def fail(x):
        try:
            try:raise ValueError('original')
            except ValueError as original:
                saved=original
                raise original
        except ValueError as again:
            if saved is again:raise
            return x+x
    with jit(source_control_flow=True,source_error_specs=(((1,),'f32'),))(fail) as program:
        with pytest.raises(ValueError,match='original'):program(np.ones(1,np.float32))


def test_implicit_context_is_retained_even_when_suppressed():
    def fail(x):
        try:raise ValueError('context')
        except ValueError:raise RuntimeError('outer') from None
    with jit(source_control_flow=True,source_error_specs=(((1,),'f32'),))(fail) as program:
        with pytest.raises(RuntimeError) as caught:program(np.ones(1,np.float32))
    assert caught.value.__context__.args==('context',)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__


def test_new_loop_exception_context_survives_completion():
    def fail(x,limit):
        while x<limit:
            try:raise ValueError('inner')
            except ValueError:raise RuntimeError('outer')
        return x+x
    with jit(source_control_flow=True,source_max_steps=2,source_error_specs=(((1,),'f32'),))(fail) as program:
        with pytest.raises(RuntimeError,match='outer') as caught:
            program(np.ones(1,np.float32),np.full(1,2,np.float32))
        assert caught.value.__context__.args==('inner',)


def test_transposed_writable_alias_projects_to_root():
    def update(base,view):
        view[:]=view*view
        return base+base
    base=np.arange(1,7,dtype=np.float32).reshape(2,3)
    with jit(source_control_flow=True,source_mutable=(1,))(update) as program:
        np.testing.assert_array_equal(program(base,base.T),2*np.arange(1,7,dtype=np.float32).reshape(2,3)**2)


def test_mapped_view_codegen_bound_refuses_without_mutation():
    def update(base):
        view=base[::-1,::2]
        view[:]=view*view
        return base+base
    from tessera.compiler.source_control_flow import SourceControlFlowError
    base=np.ones((17,17),np.float32)
    with jit(source_control_flow=True,source_mutable=(0,))(update) as program:
        with pytest.raises(SourceControlFlowError,match='256'):
            program(base)
    np.testing.assert_array_equal(base,np.ones_like(base))


def test_large_rectangular_slice_has_constant_ir_size_and_native_adjoint():
    from tessera.compiler.source_control_flow import to_native_source_ir
    from tessera.compiler.trace import trace
    def update(base):
        view=base[1::2,::3]
        view[:]=view*view
        return base+base
    base=np.arange(1,4097,dtype=np.float32).reshape(64,64)
    source=to_native_source_ir(trace(update,base,source_control_flow=True,source_state_groups=((0,),)))
    assert source.count('tensor.extract_slice')<8
    assert source.count('tensor.insert_slice')==1
    expected=base.copy();oracle=update(expected)
    with jit(source_control_flow=True,source_mutable=(0,))(update) as program:
        values,grad=program.vjp(base,cotangents=(np.ones_like(base),np.zeros_like(base)))
        np.testing.assert_array_equal(values[0],oracle)
        derivative=np.full_like(base,2);derivative[1::2,::3]=4*base[1::2,::3]
        np.testing.assert_array_equal(grad[0],derivative)


def test_dynamic_context_payloads_have_distinct_retained_slots():
    def fail(x):
        try:raise ValueError(x+x)
        except ValueError as inner:
            x[:]=x+x
            raise RuntimeError(x*x) from inner
    x=np.array([3.],np.float32)
    with jit(source_control_flow=True,source_mutable=(0,),source_error_specs=(((1,),'f32'),))(fail) as program:
        with pytest.raises(RuntimeError) as caught:program(x)
    np.testing.assert_array_equal(caught.value.args[0],[36])
    np.testing.assert_array_equal(caught.value.__context__.args[0],[6])
    assert caught.value.__cause__ is caught.value.__context__
    np.testing.assert_array_equal(x,[6])


def test_exception_aware_vjp_only_differentiates_successful_forward():
    def guarded(x):
        if x>x+x:raise ValueError(x*x)
        return x*x*x
    with jit(source_control_flow=True,source_error_specs=(((1,),'f32'),))(guarded) as program:
        values,grads=program.vjp(np.array([2.],np.float32),cotangents=(np.ones(1,np.float32),))
        np.testing.assert_array_equal(values[0],[8]);np.testing.assert_array_equal(grads[0],[12])
        with pytest.raises(ValueError) as caught:
            program.vjp(np.array([-2.],np.float32),cotangents=(np.ones(1,np.float32),))
        np.testing.assert_array_equal(caught.value.args[0],[4])


def test_dynamic_loop_context_generations_have_distinct_storage():
    from tessera.compiler.trace import trace
    from tessera.compiler.source_control_flow import to_native_source_ir
    import json
    from tessera.compiler.native_persistent_tape import _attribute
    def fail(x,limit):
        while x<limit:
            try:raise ValueError(x+x)
            except ValueError as inner:
                if x+x<limit:
                    x=x+x
                    continue
                raise RuntimeError(x*x) from inner
        return x+x
    sample=np.ones(1,np.float32)
    ir=to_native_source_ir(trace(fail,sample,sample,source_control_flow=True,max_steps=2,source_error_specs=(((1,),'f32'),)))
    contract=json.loads(_attribute(ir,'tessera.source_state'))
    assert len(contract['error_payload_sites'])==4
    assert len(set(contract['error_payload_sites']))==4
    with jit(source_control_flow=True,source_max_steps=2,source_error_specs=(((1,),'f32'),))(fail) as program:
        with pytest.raises(RuntimeError) as caught:program(sample,np.full(1,3,np.float32))
        error=caught.value
        np.testing.assert_array_equal(error.args[0],[4])
        np.testing.assert_array_equal(error.__cause__.args[0],[4])
        assert error.__cause__ is error.__context__


def test_nested_exception_slot_budget_checked_before_expansion():
    from tessera.compiler.trace import trace
    from tessera.compiler.source_control_flow import SourceControlFlowError
    def fail(x):
        while x<x+x:
            while x<x+x:
                while x<x+x:
                    raise ValueError(x+x)
        return x
    with pytest.raises(SourceControlFlowError,match='32 payload slots'):
        trace(fail,np.ones(1,np.float32),source_control_flow=True,max_steps=4,source_error_specs=(((1,),'f32'),))


def test_exception_reference_survives_later_loop_generations():
    def fail(x,limit):
        saved=None
        while x<limit:
            try:raise ValueError(x+x)
            except ValueError as error:
                if saved is None:saved=error
            x=x+x
        raise saved
    with jit(source_control_flow=True,source_max_steps=2,source_error_specs=(((1,),'f32'),))(fail) as program:
        with pytest.raises(ValueError) as caught:program(np.ones(1,np.float32),np.full(1,3,np.float32))
        np.testing.assert_array_equal(caught.value.args[0],[2])


def test_saved_exception_can_be_a_cause_after_later_iterations():
    def fail(x,limit):
        saved=None
        while x<limit:
            try:raise ValueError(x+x)
            except ValueError as error:
                if saved is None:saved=error
            x=x+x
        raise RuntimeError(x*x) from saved
    with jit(source_control_flow=True,source_max_steps=2,source_error_specs=(((1,),'f32'),))(fail) as program:
        with pytest.raises(RuntimeError) as caught:program(np.ones(1,np.float32),np.full(1,3,np.float32))
        np.testing.assert_array_equal(caught.value.args[0],[16])
        np.testing.assert_array_equal(caught.value.__cause__.args[0],[2])
        assert caught.value.__context__ is None


def test_native_exception_does_not_fabricate_python_execution_frames():
    def native_failure(x):
        raise ValueError(x+x)
    with jit(source_control_flow=True,source_error_specs=(((1,),'f32'),))(native_failure) as program:
        with pytest.raises(ValueError) as caught:program(np.ones(1,np.float32))
    error=caught.value
    tb=error.__traceback__
    frames=[]
    while tb is not None:
        frames.append(tb.tb_frame.f_code)
        tb=tb.tb_next
    assert frames and native_failure.__code__ not in frames
    assert any('Native source raise' in note for note in error.__notes__)


def test_explicit_custom_exception_binding_preserves_host_constructor():
    from tessera.compiler.native_source_state import decode_source_exception
    class DomainError(Exception):
        def __init__(self,message):
            super().__init__(message)
            self.domain='solver'
    contract={'error_table':[['DomainError',['bad state']]]}
    values=[np.array([1],np.float32)]
    with pytest.raises(RuntimeError,match='payload'):
        decode_source_exception(contract,values)
    error=decode_source_exception(contract,values,exception_types={'DomainError':DomainError})
    assert type(error) is DomainError
    assert error.domain=='solver'
    assert error.args==('bad state',)


def test_native_custom_exception_runs_constructor_only_at_completion():
    calls=[]
    class DomainError(Exception):
        def __init__(self,message):
            calls.append(message)
            super().__init__(message)
            self.domain='solver'
    def source(x):
        if x<x-x:raise DomainError('bad state')
        return x*x
    with jit(source_control_flow=True,source_error_specs=(((1,),'f32'),))(source) as program:
        np.testing.assert_array_equal(program(np.array([2],np.float32)),[4])
        assert not calls
        with pytest.raises(DomainError) as caught:program(np.array([-2],np.float32))
        assert caught.value.domain=='solver'
        assert calls==['bad state']


def test_equal_literal_exceptions_in_distinct_iterations_keep_identity():
    def source(x):
        limit=x+x+x+x
        saved=None
        while x<limit:
            try:raise ValueError('same payload')
            except ValueError as error:
                if saved is None:saved=error
                else:
                    if saved is error:raise RuntimeError('collapsed generations')
            x=x+x
        return x
    with jit(source_control_flow=True,source_max_steps=2,source_error_specs=(((1,),'f32'),))(source) as program:
        np.testing.assert_array_equal(program(np.ones(1,np.float32)),[4])


def test_handled_custom_constructor_effects_refuse():
    from tessera.compiler.trace import trace
    from tessera.compiler.source_control_flow import SourceControlFlowError
    calls=[]
    class DomainError(Exception):
        def __init__(self,message):calls.append(message)
    def source(x):
        try:raise DomainError('effect')
        except Exception:return x*x
    with pytest.raises(SourceControlFlowError,match='handled native paths'):
        trace(source,np.ones(1,np.float32),source_control_flow=True)
    assert not calls
