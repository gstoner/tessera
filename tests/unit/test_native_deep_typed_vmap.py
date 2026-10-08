"""Static leading prefixes remain semantic types; native IR owns flattening."""
import copy
import math
import numpy as np
import pytest
import tessera as ts
from tessera.autodiff import vmap
from tessera.compiler.native_vmap import batch_specs
from tessera.compiler.rocm_typed_scaled_native import contract
from tests.unit.test_native_typed_scaled_vmap import case

def deep_case(policy,fmt,nk,prefix=(2,1,3),mode=None):
    scalar,first,values,expected=case(policy,fmt,nk,(math.prod(prefix),7,19,256))
    axes=first._frontend_batch_axes
    if mode:
        scalar=ts.jit(target="rocm_gfx1201",autodiff=mode,wrt=("sa","sb"))(scalar._fn)
    owner=scalar
    parents=[]
    for _ in prefix:
        parents.append(owner)
        owner=vmap(owner,in_axes=axes)
    values=tuple(v.reshape(*prefix,*v.shape[1:]) if axis==0 else v
                 for v,axis in zip(values,axes,strict=True))
    return scalar,owner,values,expected.reshape(*prefix,7,19),parents

@pytest.mark.parametrize("prefix",[(2,1,3),(1,2,1,3)])
@pytest.mark.parametrize("policy",["shared_rhs_rows","independent_rhs","shared_lhs"])
@pytest.mark.parametrize("fmt",["fp32","e8m0"])
@pytest.mark.parametrize("nk",[False,True])
def test_deep_primal_graph_certificate_and_independent_owners(prefix,policy,fmt,nk):
    scalar,owner,values,expected,parents=deep_case(policy,fmt,nk,prefix)
    before=[copy.deepcopy(p.graph_ir) for p in parents]
    graph=owner._specialized_autodiff_module(values,{})
    assert contract(graph)
    assert graph.functions[0].result_types[0].shape==tuple(map(str,expected.shape))
    assert owner._frontend_batch_depth==len(prefix)
    assert owner.frontend_differential(*values)
    assert [p.graph_ir for p in parents]==before
    names=owner._constraint_ir_args[0 if policy!="shared_lhs" else 1].dim_names
    assert len(set(names[:-2]))==len(prefix)

@pytest.mark.parametrize("mode",["forward","reverse"])
@pytest.mark.parametrize("policy",["shared_rhs_rows","independent_rhs","shared_lhs"])
@pytest.mark.parametrize("nk",[False,True])
def test_deep_scale_ad_preserves_request_and_certificate(mode,policy,nk):
    scalar,owner,values,_,parents=deep_case(policy,"fp32",nk,mode=mode)
    assert owner.differentiation_request==scalar.differentiation_request
    assert owner.differentiation_request is not scalar.differentiation_request
    assert contract(owner._specialized_autodiff_module(values,{}))
    assert owner.frontend_differential(*values)

@pytest.mark.parametrize("depth",[0,-1,True,1.5])
def test_map_depth_must_be_positive_integer(depth):
    with pytest.raises(ValueError,match="positive leading"):
        batch_specs((),(),depth=depth)

def test_deep_equal_product_prefixes_do_not_hide_mismatch():
    from tessera._jit_boundary import TesseraJitError
    _,owner,values,_,_=deep_case("independent_rhs","fp32",False)
    changed=(values[0],values[1].reshape(3,1,2,*values[1].shape[3:]),values[2],values[3])
    with pytest.raises(TesseraJitError,match="batch extents differ"):
        owner._specialized_autodiff_module(changed,{})


@pytest.mark.parametrize("rank",[0,1,2,3,5,6,12])
def test_gfx1201_scaled_rank_capability_has_explicit_minimum(rank):
    from tessera.compiler.capabilities import supports_op
    assert supports_op("rocm_gfx1201","scaled_matmul",dtype="fp8_e4m3",rank=rank).supported==(rank>=2)

def test_rank_minimum_intersects_explicit_set(monkeypatch):
    from tessera.compiler import capabilities as caps
    from dataclasses import replace
    row=caps.TARGET_CAPABILITIES["rocm_gfx1201"]
    changed=replace(row,supported_ops={**row.supported_ops,
        "tessera.scaled_matmul":caps.OpCapability("tessera.scaled_matmul","ready",
            dtypes=("fp8_e4m3",),ranks=(1,3),min_rank=2)})
    monkeypatch.setitem(caps.TARGET_CAPABILITIES,"rocm_gfx1201",changed)
    assert not caps.supports_op("rocm_gfx1201","scaled_matmul",dtype="fp8_e4m3",rank=1).supported
    assert caps.supports_op("rocm_gfx1201","scaled_matmul",dtype="fp8_e4m3",rank=3).supported
    assert not caps.supports_op("rocm_gfx1201","scaled_matmul",dtype="fp8_e4m3",rank=4).supported


@pytest.mark.parametrize("mode",["forward","reverse"])
def test_deep_source_bound_is_checked_before_capture(mode,monkeypatch):
    from tessera.compiler.constraints import Range,TesseraConstraintError
    _,owner,values,_,_=deep_case("independent_rhs","fp32",False,mode=mode)
    owner.constraints.add(Range("M",1,6))
    def forbidden(*a,**kw):raise AssertionError("invalid source bound reached capture")
    monkeypatch.setattr(owner,"_specialized_autodiff_module",forbidden)
    with pytest.raises(TesseraConstraintError,match="M"):
        if mode=="forward":
            owner.native_jvp(*values,tangents=(np.ones_like(values[2]),np.ones_like(values[3])))
        else:
            owner.native_backward(*values,out_cotangents=np.ones((2,1,3,7,19),np.float32))

def test_deep_native_batch_capacity_rejects_before_image_projection():
    from types import SimpleNamespace
    from tessera.compiler.native_vmap import project_batch
    from tessera.compiler.graph_ir import specialize_module_from_values
    scalar,first,values,_=case("independent_rhs","fp32",False)
    axes=first._frontend_batch_axes
    scalar_values=tuple(v[0] for v in values)
    graph=specialize_module_from_values(scalar.graph_ir,
        dict(zip(scalar.arg_names,scalar_values,strict=True)))
    huge=tuple(SimpleNamespace(shape=(2**31,1,1,*v.shape),dtype=v.dtype) for v in scalar_values)
    with pytest.raises(ValueError,match="exact matrix/scale contract"):
        project_batch(graph,huge,axes,depth=3)
