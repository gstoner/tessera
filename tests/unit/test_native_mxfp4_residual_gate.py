"""Typed exact MXFP4 residual capture; native cast/program integration follows."""
import os
import subprocess
import numpy as np
import pytest
import tessera as ts
from tessera.dtype import Dtype
from tessera.compiler.graph_ir import _shape_scaled_matmul, tensor_ir_type

byte = Dtype("uint8",allow_planned_gated=True)

def gated(a:ts.Tensor["M","K",byte],b:ts.Tensor["P","N",byte],  # noqa: F821
          sa:ts.Tensor["M","fp32"],sb:ts.Tensor["G","N",byte],  # noqa: F821
          gate:ts.Tensor["M","N","fp32"]):  # noqa: F821
    product=ts.ops.scaled_matmul(a,b,sa,sb,
        physical_contract="rocm_mxfp4_w4a8_exact_v1",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,32],"format":"e8m0"})
    coefficient=ts.ops.cast(product,"fp32")
    return coefficient*gate

def typed_case(shape=(17,64,19)):
    m,k,n=shape
    values=(np.full((m,k),0x38,np.uint8),np.full((k//2,n),0x11,np.uint8),
            np.ones(m,np.float32),np.full((k//32,n),127,np.uint8),np.ones((m,n),np.float32))
    owner=ts.jit(target="rocm_gfx1201",autodiff="reverse",wrt=("gate",))(gated)
    return owner,values


@pytest.mark.parametrize("shape",[(17,64,19),(256,128,80)])
def test_exact_mxfp4_cast_gate_keeps_bf16_product_and_fp32_cast(shape):
    owner,values=typed_case(shape)
    graph=owner._specialized_autodiff_module(values,{})
    m,_,n=shape
    assert [(op.op_name,str(op.inferred_type)) for op in graph.functions[0].body]==[
        ("tessera.scaled_matmul",f"tensor<{m}x{n}xbf16>"),
        ("tessera.cast",f"tensor<{m}x{n}xf32>"),
        ("tessera.mul",f"tensor<{m}x{n}xf32>")]


@pytest.mark.skipif(not os.environ.get("TESSERA_OPT"),reason="matching native compiler required")
def test_exact_mxfp4_typed_frontend_is_valid_native_graph():
    owner,values=typed_case()
    graph=owner._specialized_autodiff_module(values,{})
    graph.module_attrs.update({"tessera.target":'"rocm"',"tessera.arch":'"gfx1201"'})
    result=subprocess.run([os.environ["TESSERA_OPT"]],
        input=graph.to_mlir(target="rocm_gfx1201",canonical=True),
        text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert "tensor<17x19xbf16>" in result.stdout


@pytest.mark.parametrize("shapes,dtypes",[
    (((17,64),(31,19),(17,),(2,19)),("uint8","uint8","fp32","uint8")),
    (((17,64),(32,19),(18,),(2,19)),("uint8","uint8","fp32","uint8")),
    (((17,64),(32,19),(17,),(3,19)),("uint8","uint8","fp32","uint8")),
    (((17,64),(32,19),(17,),(2,19)),("fp32","uint8","fp32","uint8")),
])
def test_exact_mxfp4_shape_rule_rejects_invalid_physical_storage(shapes,dtypes):
    operands=[tensor_ir_type(shape,dtype) for shape,dtype in zip(shapes,dtypes,strict=True)]
    with pytest.raises(ValueError,match="exact MXFP4"):
        _shape_scaled_matmul(operands,{"physical_contract":"rocm_mxfp4_w4a8_exact_v1"})


@pytest.mark.parametrize("positional",[False,True])
@pytest.mark.parametrize("dtype,element",[("fp32","f32"),("bf16","bf16"),("fp16","f16"),("fp64","f64")])
def test_value_less_cast_tracing_uses_catalog_result_dtype(positional,dtype,element):
    from tessera.compiler.trace import trace
    def cast_value(value):
        return ts.ops.cast(value,dtype) if positional else ts.ops.cast(value,dtype=dtype)
    traced=trace(cast_value,((2,3),"bf16"),evaluate_catalog_outputs=False)
    assert traced.body[-1].result_type==f"tensor<2x3x{element}>"
