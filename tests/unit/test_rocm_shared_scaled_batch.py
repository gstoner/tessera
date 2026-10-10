"""Typed shared-RHS batching keeps logical ranks and native physical rows."""
import copy,os
import numpy as np
import ml_dtypes
import pytest
import tessera as ts
from tessera.dtype import Dtype
from tessera.compiler.rocm_typed_scaled_native import contract,lower_typed_scaled
encoded_byte=Dtype("uint8",allow_planned_gated=True)

def shared_fp32_kn(a:ts.Tensor["B","M","K","fp8_e4m3"],
            b:ts.Tensor["K","N","fp8_e4m3"],
            sa:ts.Tensor["B","M","G","fp32"],
            sb:ts.Tensor["G","C","fp32"]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=False,
        batching="shared_rhs_rows",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[128,128],"format":"fp32"})

def shared_fp32_nk(a:ts.Tensor["B","M","K","fp8_e4m3"],
            b:ts.Tensor["N","K","fp8_e4m3"],
            sa:ts.Tensor["B","M","G","fp32"],
            sb:ts.Tensor["G","C","fp32"]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=True,
        batching="shared_rhs_rows",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[128,128],"format":"fp32"})

def shared_e8m0_kn(a:ts.Tensor["B","M","K","fp8_e4m3"],
            b:ts.Tensor["K","N","fp8_e4m3"],
            sa:ts.Tensor["B","M","G",encoded_byte],
            sb:ts.Tensor["G","C",encoded_byte]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=False,
        batching="shared_rhs_rows",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,32],"format":"e8m0"})

def shared_e8m0_nk(a:ts.Tensor["B","M","K","fp8_e4m3"],
            b:ts.Tensor["N","K","fp8_e4m3"],
            sa:ts.Tensor["B","M","G",encoded_byte],
            sb:ts.Tensor["G","C",encoded_byte]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=True,
        batching="shared_rhs_rows",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,32],"format":"e8m0"})

def batch_inputs(shape=(3,7,19,256),fmt="fp32",nk=False):
    batch,m,n,k=shape;rng=np.random.default_rng(851)
    a=rng.choice([-.5,0,.25,1],(batch,m,k)).astype(ml_dtypes.float8_e4m3fn)
    logical_b=rng.choice([-1,0,.5,2],(k,n)).astype(ml_dtypes.float8_e4m3fn)
    b=np.ascontiguousarray(logical_b.T) if nk else logical_b
    scale_k=128 if fmt=="fp32" else 32
    scale_n=128 if fmt=="fp32" else 1
    groups=k//scale_k;columns=(n+scale_n-1)//scale_n
    if fmt=="fp32":
        sa=rng.uniform(.2,1,(batch,m,groups)).astype(np.float32)
        sb=rng.uniform(.2,1,(groups,columns)).astype(np.float32)
        da,db=sa.astype(np.float64),sb.astype(np.float64)
    else:
        sa=rng.integers(125,130,(batch,m,groups),dtype=np.uint8)
        sb=rng.integers(125,130,(groups,columns),dtype=np.uint8)
        da=sa.view(ml_dtypes.float8_e8m0fnu).astype(np.float64)
        db=sb.view(ml_dtypes.float8_e8m0fnu).astype(np.float64)
    oracle=np.zeros((batch,m,n),np.float64)
    for g in range(groups):
        oracle+=(a[...,g*scale_k:(g+1)*scale_k].astype(np.float64)@
                 logical_b[g*scale_k:(g+1)*scale_k].astype(np.float64))*da[...,g,None]*db[g,np.arange(n)//scale_n][None,:]
    return (a,b,sa,sb),oracle

@pytest.mark.parametrize("fmt",["fp32","e8m0"])
@pytest.mark.parametrize("nk",[False,True])
def test_shared_batch_original_graph_native_flat_rows(fmt,nk,monkeypatch):
    monkeypatch.setenv("TESSERA_ROCM_CHIP","gfx1201")
    values,expected=batch_inputs(fmt=fmt,nk=nk)
    source=globals()[f"shared_{fmt}_{'nk' if nk else 'kn'}"]
    np.testing.assert_allclose(source(*values),expected,rtol=4e-5,atol=1e-4)
    fn=ts.jit(target="rocm_gfx1201")(source)
    graph=fn._specialized_autodiff_module(values,{})
    before=copy.deepcopy(graph).to_mlir(target="rocm_gfx1201")
    shape,*_=contract(graph)
    assert (shape.m,shape.n,shape.k)==(21,19,256)
    assert graph.functions[0].result_types[0].shape==("3","7","19")
    assert graph.to_mlir(target="rocm_gfx1201")==before
    if not os.environ.get("TESSERA_OPT"):pytest.skip("matching compiler required")
    program=lower_typed_scaled(graph)
    assert "tensor<3x7x256xf8E4M3FN>" in program.graph_ir
    assert "schedule.matmul" in program.schedule_ir
    assert "tile.scaled_matmul_kernel" in program.tile_ir
    assert "m = 21" in program.tile_ir

@pytest.mark.parametrize("change",["scale_batch","scale_rows","rhs_batch","transpose_a","policy"])
def test_shared_batch_refuses_unmatched_storage(change,monkeypatch):
    from tessera.compiler.graph_ir import tensor_ir_type
    monkeypatch.setenv("TESSERA_ROCM_CHIP","gfx1201")
    values,_=batch_inputs()
    graph=ts.jit(target="rocm_gfx1201")(shared_fp32_kn)._specialized_autodiff_module(values,{})
    op=graph.functions[0].body[0]
    if change=="scale_batch":graph.functions[0].args[2].ir_type=tensor_ir_type((2,7,2),"fp32")
    if change=="scale_rows":graph.functions[0].args[2].ir_type=tensor_ir_type((3,6,2),"fp32")
    if change=="rhs_batch":graph.functions[0].args[1].ir_type=tensor_ir_type((3,256,19),"fp8_e4m3")
    if change=="transpose_a":op.kwargs["transposeA"]=True
    if change=="policy":op.kwargs["numeric_policy"]["execution_mode"]="approximate"
    assert contract(graph) is None
