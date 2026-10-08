"""Public independent-prefix primal/JVP admission and native image binding."""
import itertools
import numpy as np
import pytest
import tessera as ts
from tessera.autodiff import vmap
from tessera.compiler.rocm_typed_scaled_native import contract
from benchmarks.rocm.record_independent_scaled_primal import inputs,expected as native_expected
from tests.device.rocm.test_public_typed_scaled_primal import mxfp8,mxfp8_nk

def scaled(a:ts.Tensor["M","K","fp8_e4m3"],b:ts.Tensor["K","N","fp8_e4m3"],
           sa:ts.Tensor["M","G","fp32"],sb:ts.Tensor["G","C","fp32"]):
    return ts.ops.scaled_matmul(a,b,sa,sb,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[3,32],"format":"fp32"})

def scaled_nk(a:ts.Tensor["M","K","fp8_e4m3"],b:ts.Tensor["N","K","fp8_e4m3"],
              sa:ts.Tensor["M","G","fp32"],sb:ts.Tensor["G","C","fp32"]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=True,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[3,32],"format":"fp32"})

def scaled_ta(a:ts.Tensor["K","M","fp8_e4m3"],b:ts.Tensor["K","N","fp8_e4m3"],
              sa:ts.Tensor["M","G","fp32"],sb:ts.Tensor["G","C","fp32"]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeA=True,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[3,32],"format":"fp32"})

def scaled_tatb(a:ts.Tensor["K","M","fp8_e4m3"],b:ts.Tensor["N","K","fp8_e4m3"],
                sa:ts.Tensor["M","G","fp32"],sb:ts.Tensor["G","C","fp32"]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeA=True,transposeB=True,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[3,32],"format":"fp32"})

from tessera.dtype import Dtype
encoded_byte=Dtype("uint8",allow_planned_gated=True)
def mxfp8_ta(a:ts.Tensor["K","M","fp8_e4m3"],b:ts.Tensor["K","N","fp8_e4m3"],
             sa:ts.Tensor["M","G",encoded_byte],sb:ts.Tensor["G","N",encoded_byte]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeA=True,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,32],"format":"e8m0"})

def mxfp8_tatb(a:ts.Tensor["K","M","fp8_e4m3"],b:ts.Tensor["N","K","fp8_e4m3"],
               sa:ts.Tensor["M","G",encoded_byte],sb:ts.Tensor["G","N",encoded_byte]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeA=True,transposeB=True,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,32],"format":"e8m0"})

def expected(values,row):
    if row.get("transposeA"):
        values=[np.ascontiguousarray(values[0].swapaxes(-1,-2)),*values[1:]]
    return native_expected(values,row)

def case(mask,tb=False,encoded=False,prefix=(2,3),jvp=False,seed=1007,ta=False):
    axes=tuple(0 if mask&(1<<i) else None for i in range(4))
    row={"prefixes":[list(prefix) if axis==0 else [] for axis in axes],
         "output_prefix":list(prefix),"transposeB":tb,"encoded":encoded,
         "kind":"paired_jvp" if jvp else "primal","transposeA":ta}
    source=(mxfp8_nk if tb else mxfp8) if encoded else (scaled_nk if tb else scaled)
    if ta:source=(mxfp8_tatb if tb else mxfp8_ta) if encoded else (scaled_tatb if tb else scaled_ta)
    scalar=ts.jit(target="rocm_gfx1201",**({"autodiff":"forward","wrt":("sa","sb")} if jvp else {}))(source)
    owner=scalar
    for _ in prefix:owner=vmap(owner,in_axes=axes)
    values=inputs(row,seed)
    if ta:values[0]=np.ascontiguousarray(values[0].swapaxes(-1,-2))
    return scalar,owner,values,row

@pytest.mark.parametrize("mask",range(1,16))
@pytest.mark.parametrize("tb,encoded",tuple(itertools.product((False,True),repeat=2)))
@pytest.mark.parametrize("prefix",[(2,),(2,3),(2,1,3)])
def test_public_primal_preserves_independent_graph(mask,tb,encoded,prefix):
    scalar,owner,values,row=case(mask,tb,encoded,prefix)
    before=scalar.graph_ir.to_mlir(target="rocm_gfx1201")
    graph=owner._specialized_autodiff_module(values,{})
    shape,fmt,names,out=contract(graph)
    logical_m=3*np.prod(prefix) if graph.functions[0].body[0].kwargs.get("batching")=="shared_rhs_rows" else 3
    assert (shape.m,shape.n,shape.k)==(logical_m,5,64)
    assert graph.functions[0].result_types[0].shape==tuple(map(str,(*prefix,3,5)))
    assert scalar.graph_ir.to_mlir(target="rocm_gfx1201")==before
    assert scalar._frontend_batch_axes is None
    assert owner.frontend_differential(*values) is owner.frontend_differential(*values)

@pytest.mark.parametrize("mask",range(1,16))
@pytest.mark.parametrize("tb",[False,True])
@pytest.mark.parametrize("prefix",[(2,),(2,3),(2,1,3)])
def test_public_scale_jvp_preserves_independent_graph(mask,tb,prefix):
    scalar,owner,values,row=case(mask,tb,prefix=prefix,jvp=True)
    graph=owner._specialized_autodiff_module(values[:4],{})
    assert contract(graph) is not None
    assert owner.differentiation_request==scalar.differentiation_request
    assert owner.frontend_differential(*values[:4]) is owner.frontend_differential(*values[:4])

@pytest.mark.parametrize("encoded",[False,True])
def test_independent_package_binds_actual_native_member_image(encoded):
    import json
    from tessera.compiler.rocm_typed_scaled_native import lower_typed_scaled,package_typed_scaled
    from tessera.compiler.native_scaled_program import NativeScaledProgram
    _,owner,values,_=case(4,True,encoded)
    graph=owner._specialized_autodiff_module(values,{})
    package=package_typed_scaled(graph,lower_typed_scaled(graph),pipeline_name="tessera-lower-to-rocm")
    native=NativeScaledProgram.from_manifest(package.descriptor.provenance["native_scaled_primal_program"])
    member=json.loads(native.members_json[0])
    assert package.image.payload==native.images[0]
    assert package.descriptor.entry_symbol==member["entry"]
    assert package.descriptor.geometry.grid==tuple(member["geometry"][:3])
    assert package.descriptor.geometry.grid[2]==6
    assert member["entry"] in package.target_ir


@pytest.mark.parametrize("mask",range(1,16))
@pytest.mark.parametrize("tb,encoded",tuple(itertools.product((False,True),repeat=2)))
def test_transposed_a_public_projection(mask,tb,encoded):
    scalar,owner,values,row=case(mask,tb,encoded,ta=True)
    graph=owner._specialized_autodiff_module(values,{})
    assert contract(graph) is not None
    assert graph.functions[0].body[0].kwargs["batching"]=="broadcast"
    assert graph.functions[0].body[0].kwargs["transposeA"] is True
    assert graph.functions[0].args[0].ir_type.shape[-2:]==("64","3")
