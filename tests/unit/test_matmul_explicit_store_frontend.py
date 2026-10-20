"""Explicit matmul stores agree in eager, trace and semantic Graph typing."""
import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera.compiler.graph_ir import _shape_matmul_2d,tensor_ir_type

@pytest.mark.parametrize("storage,dtype",[("fp16",np.float16),("bf16",ml_dtypes.bfloat16),("fp32",np.float32)])
def test_explicit_matmul_store_occurs_after_complete_epilogue(storage,dtype):
    a=np.array([[1.0007]],np.float32)
    b=np.array([[1.]],np.float32)
    bias=np.array([-1.0004],np.float32)
    residual=np.array([[.004]],np.float32)
    accum=(a.astype(np.float64)@b.astype(np.float64)).astype(np.float32)
    expected=(np.maximum(accum+bias,np.float32(0))+residual).astype(dtype)
    actual=ts.ops.matmul(a,b,bias=bias,activation="relu",residual=residual,output_dtype=storage)
    assert actual.dtype==np.dtype(dtype)
    np.testing.assert_array_equal(actual,expected)
    if storage!="fp32":
        premature=(np.maximum(accum.astype(dtype).astype(np.float32)+bias,np.float32(0))+residual).astype(dtype)
        assert not np.array_equal(actual,premature)

def bf16_product(a,b):
    return ts.ops.matmul(a,b,output_dtype="bf16")

def test_bf16_explicit_store_survives_frontend_trace():
    a=np.zeros((3,5),np.float32)
    b=np.zeros((5,7),np.float16)
    fn=ts.jit(target="nvidia_sm120")(bf16_product)
    graph=fn._traced_autodiff_module((a,b),{})
    body=graph.functions[0].body
    assert len(body)==1 and body[0].op_name=="tessera.matmul"
    assert body[0].kwargs["output_dtype"]=="bf16"
    assert graph.functions[0].result_types[0].dtype=="bf16"

def test_bf16_explicit_store_has_target_neutral_graph_shape():
    result=_shape_matmul_2d([tensor_ir_type((3,5),"fp32"),tensor_ir_type((5,7),"fp16")],
        {"output_dtype":"bf16"})
    assert result.dtype=="bf16" and tuple(int(v) for v in result.shape)==(3,7)

@pytest.mark.parametrize("storage",["fp64","fp8_e4m3","tf32"])
def test_explicit_output_contract_still_rejects_unsupported_store_modes(storage):
    a=np.zeros((3,5),np.float32)
    b=np.zeros((5,7),np.float32)
    with pytest.raises(ValueError,match="output_dtype"):
        ts.ops.matmul(a,b,output_dtype=storage)


def test_textual_bf16_store_preserves_semantic_result_type():
    from tessera.compiler.frontend import lower_text_to_graph_ir
    module=lower_text_to_graph_ir("""
    module demo {
      func main(A: tensor<3x5xfp32>, B: tensor<5x7xfp16>) -> tensor<3x7xbf16> {
        C = op.matmul(A, B) @{output_dtype = "bf16"};
        return C;
      }
    }
    """)
    function=module.functions[0]
    assert function.result_types[0].dtype=="bf16"
    assert function.body[0].kwargs["output_dtype"]=="bf16"
    assert "tensor<3x7xbf16>" in module.to_mlir()


@pytest.mark.parametrize("storage",["fp16","bf16","fp32"])
def test_fp32_pair_selects_existing_sm120_scalar_store_contract(storage):
    from tessera.compiler.scheduled_matmul import _graph_contract
    a=np.zeros((3,5),np.float32)
    b=np.zeros((5,7),np.float32)
    def product(a,b):
        return ts.ops.matmul(a,b,output_dtype=storage)
    fn=ts.jit(target="nvidia_sm120")(product)
    module=fn._traced_autodiff_module((a,b),{})
    contract=_graph_contract(module,"nvidia_sm120")
    assert contract[9:12]==("fp32","fp32",storage)
    assert contract[12:14]==("f32","f32")
