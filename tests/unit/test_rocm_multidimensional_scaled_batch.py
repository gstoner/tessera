"""Static two-axis batches retain logical shape through native scaled products."""

# ruff: noqa: F821 -- Tensor shape/dtype strings are public annotation DSL symbols.
import copy
import os
import numpy as np
import pytest
import tessera as ts
from tessera.compiler.rocm_typed_scaled_native import contract, lower_typed_scaled
from tests.unit.test_native_typed_scaled_vmap import case
from tests.unit.test_rocm_independent_scaled_batch import encoded_byte

def nested_shared_rhs_rows_fp32_kn(a: ts.Tensor["B0","B1","M","K","fp8_e4m3"],
        b: ts.Tensor["K","N","fp8_e4m3"],
        sa: ts.Tensor["B0","B1","M","G","fp32"],
        sb: ts.Tensor["G","C","fp32"]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=False,
        batching="shared_rhs_rows",numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[128,128],"format":"fp32"})

def nested_shared_rhs_rows_fp32_nk(a: ts.Tensor["B0","B1","M","K","fp8_e4m3"],
        b: ts.Tensor["N","K","fp8_e4m3"],
        sa: ts.Tensor["B0","B1","M","G","fp32"],
        sb: ts.Tensor["G","C","fp32"]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=True,
        batching="shared_rhs_rows",numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[128,128],"format":"fp32"})

def nested_shared_rhs_rows_e8m0_kn(a: ts.Tensor["B0","B1","M","K","fp8_e4m3"],
        b: ts.Tensor["K","N","fp8_e4m3"],
        sa: ts.Tensor["B0","B1","M","G",encoded_byte],
        sb: ts.Tensor["G","C",encoded_byte]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=False,
        batching="shared_rhs_rows",numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,32],"format":"e8m0"})

def nested_shared_rhs_rows_e8m0_nk(a: ts.Tensor["B0","B1","M","K","fp8_e4m3"],
        b: ts.Tensor["N","K","fp8_e4m3"],
        sa: ts.Tensor["B0","B1","M","G",encoded_byte],
        sb: ts.Tensor["G","C",encoded_byte]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=True,
        batching="shared_rhs_rows",numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,32],"format":"e8m0"})

def nested_independent_rhs_fp32_kn(a: ts.Tensor["B0","B1","M","K","fp8_e4m3"],
        b: ts.Tensor["B0","B1","K","N","fp8_e4m3"],
        sa: ts.Tensor["B0","B1","M","G","fp32"],
        sb: ts.Tensor["B0","B1","G","C","fp32"]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=False,
        batching="independent_rhs",numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[128,128],"format":"fp32"})

def nested_independent_rhs_fp32_nk(a: ts.Tensor["B0","B1","M","K","fp8_e4m3"],
        b: ts.Tensor["B0","B1","N","K","fp8_e4m3"],
        sa: ts.Tensor["B0","B1","M","G","fp32"],
        sb: ts.Tensor["B0","B1","G","C","fp32"]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=True,
        batching="independent_rhs",numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[128,128],"format":"fp32"})

def nested_independent_rhs_e8m0_kn(a: ts.Tensor["B0","B1","M","K","fp8_e4m3"],
        b: ts.Tensor["B0","B1","K","N","fp8_e4m3"],
        sa: ts.Tensor["B0","B1","M","G",encoded_byte],
        sb: ts.Tensor["B0","B1","G","C",encoded_byte]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=False,
        batching="independent_rhs",numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,32],"format":"e8m0"})

def nested_independent_rhs_e8m0_nk(a: ts.Tensor["B0","B1","M","K","fp8_e4m3"],
        b: ts.Tensor["B0","B1","N","K","fp8_e4m3"],
        sa: ts.Tensor["B0","B1","M","G",encoded_byte],
        sb: ts.Tensor["B0","B1","G","C",encoded_byte]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=True,
        batching="independent_rhs",numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,32],"format":"e8m0"})

def nested_shared_lhs_fp32_kn(a: ts.Tensor["M","K","fp8_e4m3"],
        b: ts.Tensor["B0","B1","K","N","fp8_e4m3"],
        sa: ts.Tensor["M","G","fp32"],
        sb: ts.Tensor["B0","B1","G","C","fp32"]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=False,
        batching="shared_lhs",numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[128,128],"format":"fp32"})

def nested_shared_lhs_fp32_nk(a: ts.Tensor["M","K","fp8_e4m3"],
        b: ts.Tensor["B0","B1","N","K","fp8_e4m3"],
        sa: ts.Tensor["M","G","fp32"],
        sb: ts.Tensor["B0","B1","G","C","fp32"]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=True,
        batching="shared_lhs",numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[128,128],"format":"fp32"})

def nested_shared_lhs_e8m0_kn(a: ts.Tensor["M","K","fp8_e4m3"],
        b: ts.Tensor["B0","B1","K","N","fp8_e4m3"],
        sa: ts.Tensor["M","G",encoded_byte],
        sb: ts.Tensor["B0","B1","G","C",encoded_byte]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=False,
        batching="shared_lhs",numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,32],"format":"e8m0"})

def nested_shared_lhs_e8m0_nk(a: ts.Tensor["M","K","fp8_e4m3"],
        b: ts.Tensor["B0","B1","N","K","fp8_e4m3"],
        sa: ts.Tensor["M","G",encoded_byte],
        sb: ts.Tensor["B0","B1","G","C",encoded_byte]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=True,
        batching="shared_lhs",numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,32],"format":"e8m0"})

def nested_case(policy,fmt,nk,shape=(2,3,7,19,256)):
    b0,b1,m,n,k=shape
    _,_,values,oracle=case(policy,fmt,nk,(b0*b1,m,n,k))
    axes={"shared_rhs_rows":(0,None,0,None),"independent_rhs":(0,0,0,0),
          "shared_lhs":(None,0,None,0)}[policy]
    values=tuple(v.reshape(b0,b1,*v.shape[1:]) if axis==0 else v
                 for v,axis in zip(values,axes,strict=True))
    source=globals()[f"nested_{policy}_{fmt}_{'nk' if nk else 'kn'}"]
    return ts.jit(target="rocm_gfx1201")(source),values,oracle.reshape(b0,b1,m,n)

@pytest.mark.parametrize("policy",["shared_rhs_rows","independent_rhs","shared_lhs"])
@pytest.mark.parametrize("fmt",["fp32","e8m0"])
@pytest.mark.parametrize("nk",[False,True])
def test_two_axis_frontend_schedule_contract(policy,fmt,nk,monkeypatch):
    monkeypatch.setenv("TESSERA_ROCM_CHIP","gfx1201")
    owner,values,expected=nested_case(policy,fmt,nk)
    np.testing.assert_allclose(owner._fn(*values),expected,rtol=4e-5,atol=1e-4)
    graph=owner._specialized_autodiff_module(values,{})
    assert graph.functions[0].result_types[0].shape==("2","3","7","19")
    info=contract(graph)
    assert info is not None
    assert info[0].m==(42 if policy=="shared_rhs_rows" else 7)
    before=copy.deepcopy(graph).to_mlir(target="rocm_gfx1201")
    if not os.environ.get("TESSERA_OPT"):pytest.skip("matching native compiler required")
    program=lower_typed_scaled(graph)
    assert "tile.scaled_matmul_kernel" in program.tile_ir
    if policy!="shared_rhs_rows":
        assert "batch_count = 6" in program.tile_ir
    assert graph.to_mlir(target="rocm_gfx1201")==before

def test_reject_equal_product_but_different_batch_prefix():
    owner,values,_=nested_case("independent_rhs","fp32",False)
    wrong=(values[0],values[1].reshape(3,2,*values[1].shape[2:]),values[2],values[3])
    with pytest.raises(ValueError,match="batch extent differs"):
        owner._specialized_autodiff_module(wrong,{})

def test_checked_target_admission_keeps_sibling_rank_four_gated():
    from tessera.compiler.capabilities import supports_op
    assert supports_op("rocm_gfx1201","tessera.scaled_matmul",dtype="fp8_e4m3",rank=4).supported
    for target in ("rocm_gfx1151","apple_gpu","x86","nvidia_sm120"):
        assert not supports_op(target,"tessera.scaled_matmul",dtype="fp8_e4m3",rank=4).supported

def test_native_graph_rejects_permuted_equal_product_batch_prefix():
    import subprocess
    from tessera.compiler.graph_ir import tensor_ir_type
    from tessera.compiler.scheduled_matmul import find_tessera_opt
    compiler=find_tessera_opt()
    if compiler is None:pytest.skip("matching native compiler required")
    owner,values,_=nested_case("independent_rhs","fp32",False)
    graph=owner._specialized_autodiff_module(values,{})
    fn=graph.functions[0]
    rhs=fn.args[1].ir_type
    fn.args[1].ir_type=tensor_ir_type((3,2,*rhs.shape[-2:]),rhs.dtype)
    fn.body[0].operand_types=[str(a.ir_type) for a in fn.args]
    result=subprocess.run([str(compiler)],input=graph.to_mlir(target="rocm_gfx1201",canonical=True),
                          capture_output=True,text=True)
    assert result.returncode!=0
    assert "typed batch leading extents differ" in result.stderr
