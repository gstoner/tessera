"""Nested native map owners preserve scalar signatures and logical prefixes."""
import copy
import numpy as np
import pytest
import tessera as ts
from tessera.autodiff import vmap
from tessera._jit_boundary import TesseraJitError
from tessera.compiler.rocm_typed_scaled_native import contract
from tests.unit.test_native_typed_scaled_vmap import case

def nested_case(policy,fmt,nk,shape=(2,3,7,19,256), *, jvp=False):
    b0,b1,m,n,k=shape
    scalar,inner,flat,expected=case(policy,fmt,nk,(b0*b1,m,n,k))
    axes=inner._frontend_batch_axes
    if jvp:
        scalar=ts.jit(target="rocm_gfx1201",autodiff="forward",wrt=("sa","sb"))(scalar._fn)
        inner=vmap(scalar,in_axes=axes)
    outer=vmap(inner,in_axes=axes)
    values=tuple(value.reshape(b0,b1,*value.shape[1:]) if axis==0 else value
                 for value,axis in zip(flat,axes,strict=True))
    return scalar,inner,outer,values,expected.reshape(b0,b1,m,n)

@pytest.mark.parametrize("policy",["shared_rhs_rows","independent_rhs","shared_lhs"])
@pytest.mark.parametrize("fmt",["fp32","e8m0"])
@pytest.mark.parametrize("nk",[False,True])
def test_nested_frontend_projection_and_reference_certificate(policy,fmt,nk):
    scalar,inner,outer,values,expected=nested_case(policy,fmt,nk)
    scalar_before=copy.deepcopy(scalar.graph_ir)
    inner_before=copy.deepcopy(inner.graph_ir)
    graph=outer._specialized_autodiff_module(values,{})
    assert contract(graph) is not None
    assert graph.functions[0].result_types[0].shape==("2","3","7","19")
    assert outer._frontend_batch_depth==2 and inner._frontend_batch_depth==1
    certificate=outer.frontend_differential(*values)
    assert certificate is outer.frontend_differential(*values)
    assert scalar.graph_ir==scalar_before and inner.graph_ir==inner_before
    if fmt=="e8m0":
        assert graph.functions[0].args[2].dtype_status=="planned_gated"

@pytest.mark.parametrize("policy",["shared_rhs_rows","independent_rhs","shared_lhs"])
@pytest.mark.parametrize("nk",[False,True])
def test_nested_scale_jvp_preserves_independent_request(policy,nk):
    scalar,inner,outer,values,_=nested_case(policy,"fp32",nk,jvp=True)
    assert outer.differentiation_request==inner.differentiation_request==scalar.differentiation_request
    assert outer.differentiation_request is not inner.differentiation_request
    assert contract(outer._specialized_autodiff_module(values,{})) is not None

def test_nested_equal_product_prefix_mismatch_is_rejected_before_capture():
    _,_,outer,values,_=nested_case("independent_rhs","fp32",False)
    changed=(values[0],values[1].reshape(3,2,*values[1].shape[2:]),values[2],values[3])
    with pytest.raises(TesseraJitError,match="batch extents differ"):
        outer._specialized_autodiff_module(changed,{})

def test_nested_map_keeps_distinct_batch_symbols_and_scalar_axes():
    scalar,inner,outer,_,_=nested_case("independent_rhs","fp32",False)
    names=outer._constraint_ir_args[0].dim_names
    assert len(names)==4 and names[0]!=names[1] and names[-2:]==("M","K")
    assert scalar._constraint_ir_args[0].dim_names==("M","K")
    assert len(inner._constraint_ir_args[0].dim_names)==3

def test_mixed_nested_scale_jvp_preserves_derivative_roles():
    scalar,inner,_,_,_=nested_case("shared_lhs","fp32",False,jvp=True)
    owner = vmap(inner,in_axes=(0,0,0,0))
    assert owner.differentiation_request.wrt_indices == (2,3)
    assert owner.differentiation_request is not inner.differentiation_request

@pytest.mark.parametrize("policy",["shared_rhs_rows","independent_rhs","shared_lhs"])
def test_nested_scale_jvp_preserves_source_bounds_before_capture(policy,monkeypatch):
    from tessera.compiler.constraints import Range,TesseraConstraintError
    scalar,inner,_,values,_=nested_case(policy,"fp32",False)
    forward=ts.jit(target="rocm_gfx1201",autodiff="forward",wrt=("sa",))(scalar._fn)
    forward.constraints.add(Range("M",1,6))
    first=vmap(forward,in_axes=inner._frontend_batch_axes)
    second=vmap(first,in_axes=first._frontend_batch_axes)
    def forbidden(*args,**kwargs):raise AssertionError("invalid nested bound reached capture")
    monkeypatch.setattr(second,"_specialized_autodiff_module",forbidden)
    with pytest.raises(TesseraConstraintError,match="M"):
        second.native_jvp(*values,tangents=np.ones_like(values[2]))
    assert forward._frontend_batch_axes is None and first._frontend_batch_depth==1


@pytest.mark.parametrize("policy", ["shared_rhs_rows", "independent_rhs", "shared_lhs"])
@pytest.mark.parametrize("nk", [False, True])
def test_ragged_nested_scale_extents_and_frontend_certificate(policy, nk):
    scalar, inner, _, values, expected = nested_case(policy, "fp32", nk, shape=(2,3,7,19,129))
    reverse = ts.jit(target="rocm_gfx1201", autodiff="reverse", wrt=("sa","sb"))(scalar._fn)
    owner = vmap(vmap(reverse,in_axes=inner._frontend_batch_axes),in_axes=inner._frontend_batch_axes)
    assert values[2].shape[-1] == 2
    assert values[3].shape[-2] == 2
    graph = owner._specialized_autodiff_module(values, {})
    from tessera.compiler.rocm_typed_scaled_native import supports_scale_transpose, supports_typed_scaled
    assert supports_scale_transpose(graph)
    assert supports_typed_scaled(graph)  # Partial-group primal lowering is now implemented.
    assert graph.functions[0].result_types[0].shape == ("2","3","7","19")
    assert owner.frontend_differential(*values)
    assert expected.shape == (2,3,7,19)


@pytest.mark.parametrize("role", [2,3])
def test_ragged_scale_reverse_rejects_missing_tail_group_before_capture(role):
    scalar, inner, _, values, _ = nested_case("independent_rhs", "fp32", False, shape=(2,3,7,19,129))
    reverse = ts.jit(target="rocm_gfx1201",autodiff="reverse",wrt=("sa","sb"))(scalar._fn)
    owner = vmap(vmap(reverse,in_axes=inner._frontend_batch_axes),in_axes=inner._frontend_batch_axes)
    changed = list(values)
    changed[role] = changed[role][...,:1] if role==2 else changed[role][...,:1,:]
    with pytest.raises(TesseraJitError,match="exact matrix/scale contract"):
        owner._specialized_autodiff_module(tuple(changed), {})
