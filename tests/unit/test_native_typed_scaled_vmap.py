"""Public typed map projects one native Graph batch and preserves scalar owners."""
import numpy as np
import pytest
import tessera as ts
from tessera._jit_boundary import TesseraJitError
from tessera.autodiff import vmap
from tessera.compiler.jit import JitFn
from tessera.compiler.rocm_typed_scaled_native import contract
from tests.device.rocm.test_public_scaled_jvp import scaled, scaled_nk
from tests.device.rocm.test_public_typed_scaled_primal import mxfp8, mxfp8_nk
from tests.unit.test_rocm_independent_scaled_batch import batch_inputs

def case(policy, fmt, nk, shape=(3,7,19,256)):
    batch,m,n,k=shape
    values, expected = batch_inputs(shape=shape,fmt=fmt, nk=nk,
        policy="shared_lhs" if policy == "shared_lhs" else "independent_rhs")
    axes = {"shared_lhs": (None,0,None,0), "independent_rhs": (0,0,0,0),
            "shared_rhs_rows": (0,None,0,None)}[policy]
    if policy == "shared_rhs_rows":
        values = (values[0], values[1][0], values[2], values[3][0])
        # Independent float64 oracle with the same shared RHS in each plane.
        sk,sn = (128,128) if fmt == "fp32" else (32,1)
        a,b,sa,sb = values
        b = b.T if nk else b
        da = sa.astype(np.float64) if fmt == "fp32" else np.exp2(sa.astype(np.float64)-127)
        db = sb.astype(np.float64) if fmt == "fp32" else np.exp2(sb.astype(np.float64)-127)
        expected = np.zeros((batch,m,n), np.float64)
        for g in range((k+sk-1)//sk):
            expected += (a[...,g*sk:(g+1)*sk].astype(np.float64) @
                b[g*sk:(g+1)*sk].astype(np.float64)) * da[...,g,None] * db[g,np.arange(n)//sn]
    source = (mxfp8_nk if nk else mxfp8) if fmt == "e8m0" else (scaled_nk if nk else scaled)
    scalar = ts.jit(target="rocm_gfx1201")(source)
    return scalar, vmap(scalar,in_axes=axes), values, expected

@pytest.mark.parametrize("policy",["shared_rhs_rows","independent_rhs","shared_lhs"])
@pytest.mark.parametrize("fmt",["fp32","e8m0"])
@pytest.mark.parametrize("nk",[False,True])
def test_public_map_projects_checked_native_contract(policy,fmt,nk):
    scalar,owner,values,_ = case(policy,fmt,nk)
    assert isinstance(owner,JitFn) and owner is not scalar
    before = scalar.graph_ir.to_mlir(target="rocm_gfx1201")
    graph = owner._specialized_autodiff_module(values,{})
    assert contract(graph) is not None
    assert graph.functions[0].body[0].kwargs["batching"] == policy
    assert graph.functions[0].result_types[0].shape == ("3","7","19")
    assert scalar.graph_ir.to_mlir(target="rocm_gfx1201") == before
    assert scalar._frontend_batch_axes is None
    if fmt == "e8m0":
        assert graph.functions[0].args[2].dtype_status == "planned_gated"

def test_map_checks_scale_shape_and_extent_before_native_launch():
    _,owner,values,_=case("independent_rhs","fp32",False)
    with pytest.raises(TesseraJitError,match="batch extents differ"):
        owner._specialized_autodiff_module((values[0],values[1],values[2][:2],values[3]),{})
    with pytest.raises(TesseraJitError,match="exact matrix/scale contract"):
        owner._specialized_autodiff_module((values[0],values[1],values[2],values[3][...,:0]),{})

@pytest.mark.parametrize("policy",["shared_rhs_rows","independent_rhs","shared_lhs"])
@pytest.mark.parametrize("nk",[False,True])
def test_mapped_scale_jvp_preserves_intent_and_certifies_scalar_map(policy,nk):
    scalar,primal,values,_=case(policy,"fp32",nk)
    forward=ts.jit(target="rocm_gfx1201",autodiff="forward",wrt=("sa","sb"))(scalar._fn)
    owner=vmap(forward,in_axes=primal._frontend_batch_axes)
    assert owner.differentiation_request == forward.differentiation_request
    assert owner.differentiation_request is not forward.differentiation_request
    graph=owner._specialized_autodiff_module(values,{})
    assert graph.functions[0].body[0].kwargs["batching"]==policy
    certificate=owner.frontend_differential(*values)
    assert certificate is owner.frontend_differential(*values)
    assert forward._frontend_batch_axes is None

@pytest.mark.parametrize("fmt,wrt,mode",[("e8m0",("sa",),"forward"),("fp32",("a",),"forward"),("fp32",("a",),"reverse"),("e8m0",("sa",),"reverse")])
def test_mapped_jvp_rejects_discrete_or_unimplemented_derivatives(fmt,wrt,mode):
    scalar,primal,_,_=case("independent_rhs",fmt,False)
    forward=ts.jit(target="rocm_gfx1201",autodiff=mode,wrt=wrt)(scalar._fn)
    with pytest.raises(ValueError,match="admitted scale-JVP"):
        vmap(forward,in_axes=primal._frontend_batch_axes)


@pytest.mark.parametrize("policy",["shared_rhs_rows","independent_rhs","shared_lhs"])
@pytest.mark.parametrize("nk",[False,True])
def test_mapped_encoded_primal_frontend_certificate_preserves_gated_storage(policy,nk):
    _,owner,values,_=case(policy,"e8m0",nk)
    certificate=owner.frontend_differential(*values)
    assert certificate is owner.frontend_differential(*values)
    graph=owner._specialized_autodiff_module(values,{})
    assert graph.functions[0].args[2].dtype_status=="planned_gated"

def test_legacy_specialization_does_not_infer_byte_opt_in():
    from tessera.compiler.graph_ir import specialize_module_from_values
    scalar,_,values,_=case("independent_rhs","e8m0",False)
    args=(values[0][0],values[1][0],values[2][0],values[3][0])
    graph=scalar._ensure_legacy_graph_ir()
    import copy
    graph=copy.deepcopy(graph)
    graph.functions[0].args[2].dtype_status=None
    with pytest.raises(TypeError,match="unsupported specialization dtype uint8"):
        specialize_module_from_values(graph,dict(zip(scalar.arg_names,args,strict=True)))


@pytest.mark.parametrize("policy",["shared_rhs_rows","independent_rhs","shared_lhs"])
def test_native_mapped_jvp_checks_source_bounds_before_frontend_or_compile(policy,monkeypatch):
    from tessera.compiler.constraints import Range,TesseraConstraintError
    scalar,primal,values,_=case(policy,"fp32",False)
    forward=ts.jit(target="rocm_gfx1201",autodiff="forward",wrt=("sa",))(scalar._fn)
    forward.constraints.add(Range("M",1,6))
    owner=vmap(forward,in_axes=primal._frontend_batch_axes)
    def forbidden(*args,**kwargs):
        raise AssertionError("invalid bounded call reached frontend")
    monkeypatch.setattr(owner,"_specialized_autodiff_module",forbidden)
    with pytest.raises(TesseraConstraintError,match="M"):
        owner.native_jvp(*values,tangents=np.ones_like(values[2]))
    assert forward._frontend_batch_axes is None
