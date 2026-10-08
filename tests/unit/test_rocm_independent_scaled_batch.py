"""Independent operand batches preserve native matrix and scale offsets."""
import copy,os
import numpy as np
import ml_dtypes
import pytest
import tessera as ts
from tessera.dtype import Dtype
from tessera.compiler.rocm_typed_scaled_native import contract,lower_typed_scaled
encoded_byte=Dtype("uint8",allow_planned_gated=True)

def independent_fp32_kn(a:ts.Tensor["B","M","K","fp8_e4m3"],
            b:ts.Tensor["B","K","N","fp8_e4m3"],
            sa:ts.Tensor["B","M","G","fp32"],
            sb:ts.Tensor["B","G","C","fp32"]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=False,
        batching="independent_rhs",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[128,128],"format":"fp32"})

def independent_fp32_nk(a:ts.Tensor["B","M","K","fp8_e4m3"],
            b:ts.Tensor["B","N","K","fp8_e4m3"],
            sa:ts.Tensor["B","M","G","fp32"],
            sb:ts.Tensor["B","G","C","fp32"]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=True,
        batching="independent_rhs",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[128,128],"format":"fp32"})

def independent_e8m0_kn(a:ts.Tensor["B","M","K","fp8_e4m3"],
            b:ts.Tensor["B","K","N","fp8_e4m3"],
            sa:ts.Tensor["B","M","G",encoded_byte],
            sb:ts.Tensor["B","G","C",encoded_byte]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=False,
        batching="independent_rhs",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,32],"format":"e8m0"})

def independent_e8m0_nk(a:ts.Tensor["B","M","K","fp8_e4m3"],
            b:ts.Tensor["B","N","K","fp8_e4m3"],
            sa:ts.Tensor["B","M","G",encoded_byte],
            sb:ts.Tensor["B","G","C",encoded_byte]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=True,
        batching="independent_rhs",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,32],"format":"e8m0"})

def lhs_shared_fp32_kn(a:ts.Tensor["M","K","fp8_e4m3"],
            b:ts.Tensor["B","K","N","fp8_e4m3"],
            sa:ts.Tensor["M","G","fp32"],
            sb:ts.Tensor["B","G","C","fp32"]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=False,
        batching="shared_lhs",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[128,128],"format":"fp32"})

def lhs_shared_fp32_nk(a:ts.Tensor["M","K","fp8_e4m3"],
            b:ts.Tensor["B","N","K","fp8_e4m3"],
            sa:ts.Tensor["M","G","fp32"],
            sb:ts.Tensor["B","G","C","fp32"]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=True,
        batching="shared_lhs",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[128,128],"format":"fp32"})

def lhs_shared_e8m0_kn(a:ts.Tensor["M","K","fp8_e4m3"],
            b:ts.Tensor["B","K","N","fp8_e4m3"],
            sa:ts.Tensor["M","G",encoded_byte],
            sb:ts.Tensor["B","G","C",encoded_byte]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=False,
        batching="shared_lhs",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,32],"format":"e8m0"})

def lhs_shared_e8m0_nk(a:ts.Tensor["M","K","fp8_e4m3"],
            b:ts.Tensor["B","N","K","fp8_e4m3"],
            sa:ts.Tensor["M","G",encoded_byte],
            sb:ts.Tensor["B","G","C",encoded_byte]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=True,
        batching="shared_lhs",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,32],"format":"e8m0"})

def batch_inputs(shape=(3,7,19,256),fmt="fp32",nk=False,policy="independent_rhs"):
    batch,m,n,k=shape;rng=np.random.default_rng(987)
    shared=policy=="shared_lhs"
    a=rng.choice([-.5,0,.25,1],(m,k) if shared else (batch,m,k)).astype(ml_dtypes.float8_e4m3fn)
    logical_b=rng.choice([-1,0,.5,2],(batch,k,n)).astype(ml_dtypes.float8_e4m3fn)
    b=np.ascontiguousarray(logical_b.swapaxes(-1,-2)) if nk else logical_b
    sk,sn=(128,128) if fmt=="fp32" else (32,1)
    g=(k+sk-1)//sk;c=(n+sn-1)//sn
    shape_a=(m,g) if shared else (batch,m,g);shape_b=(batch,g,c)
    if fmt=="fp32":
        sa=rng.uniform(.2,1,shape_a).astype(np.float32)
        sb=rng.uniform(.2,1,shape_b).astype(np.float32)
        da,db=sa.astype(np.float64),sb.astype(np.float64)
    else:
        sa=rng.integers(125,130,shape_a,dtype=np.uint8)
        sb=rng.integers(125,130,shape_b,dtype=np.uint8)
        da=sa.view(ml_dtypes.float8_e8m0fnu).astype(np.float64)
        db=sb.view(ml_dtypes.float8_e8m0fnu).astype(np.float64)
    oracle=np.zeros((batch,m,n),np.float64)
    for group in range(g):
        oracle+=(a[...,group*sk:(group+1)*sk].astype(np.float64)@
                 logical_b[...,group*sk:(group+1)*sk,:].astype(np.float64))*da[...,group,None]*db[...,group,np.arange(n)//sn][...,None,:]
    return (a,b,sa,sb),oracle

@pytest.mark.parametrize("policy",["independent_rhs","shared_lhs"])
@pytest.mark.parametrize("fmt",["fp32","e8m0"])
@pytest.mark.parametrize("nk",[False,True])
def test_native_independent_batch_contract(policy,fmt,nk,monkeypatch):
    monkeypatch.setenv("TESSERA_ROCM_CHIP","gfx1201")
    values,oracle=batch_inputs(fmt=fmt,nk=nk,policy=policy)
    prefix="independent" if policy=="independent_rhs" else "lhs_shared"
    source=globals()[f"{prefix}_{fmt}_{'nk' if nk else 'kn'}"]
    np.testing.assert_allclose(source(*values),oracle,rtol=4e-5,atol=1e-4)
    fn=ts.jit(target="rocm_gfx1201")(source)
    graph=fn._specialized_autodiff_module(values,{})
    before=copy.deepcopy(graph).to_mlir(target="rocm_gfx1201")
    shape,*_=contract(graph)
    assert (shape.m,shape.n,shape.k)==(7,19,256)
    assert graph.functions[0].result_types[0].shape==("3","7","19")
    if not os.environ.get("TESSERA_OPT"):pytest.skip("matching compiler required")
    program=lower_typed_scaled(graph)
    assert f'batching = "{policy}"' in program.tile_ir
    assert "batch_count = 3" in program.tile_ir
    assert graph.to_mlir(target="rocm_gfx1201")==before
