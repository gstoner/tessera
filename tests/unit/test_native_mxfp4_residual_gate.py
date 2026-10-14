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

@pytest.mark.skipif(not os.environ.get("TESSERA_OPT"),reason="matching native compiler required")
@pytest.mark.parametrize("mode",["primal","forward","reverse"])
def test_native_mxfp4_gate_program_preserves_bf16_intermediate_and_cast(mode):
    import json
    from tessera.compiler.native_scaled_program import (
        package_native_scaled_primal,package_native_scaled_jvp,package_native_scaled_vjp)
    owner,values=typed_case()
    if mode=="forward":
        owner=ts.jit(target="rocm_gfx1201",autodiff="forward",wrt=("gate",))(gated)
    graph=owner._specialized_autodiff_module(values,{})
    graph.module_attrs.update({"tessera.target":'"rocm"',"tessera.arch":'"gfx1201"'})
    builder={"primal":package_native_scaled_primal,
             "forward":package_native_scaled_jvp,
             "reverse":package_native_scaled_vjp}[mode]
    package=builder(graph.to_mlir(target="rocm_gfx1201",canonical=True))
    manifest=json.loads(package.program_json)
    product=next(step for step in manifest["steps"] if step["operation"]=="tessera.scaled_matmul")
    cast=next(step for step in manifest["steps"] if step["operation"]=="tessera.cast")
    source=manifest["buffers"][product["output"]]
    output=manifest["buffers"][cast["output"]]
    assert source["storage"]=="bf16" and source["bytes"]==17*19*2
    assert output["storage"]=="f32" and output["bytes"]==17*19*4
    assert cast["inputs"]==[product["output"]]
    assert source["last_read"]>=cast["step"]
    package.validate()



def test_public_admission_keeps_exact_mxfp4_rounding_roots_inactive():
    import copy
    from tessera.compiler.rocm_typed_scaled_native import (
        requests_composed_typed_scaled, supports_composed_scaled_primal,
        supports_composed_scale_jvp, supports_scaled_reverse)
    owner, values = typed_case()
    graph = owner._specialized_autodiff_module(values, {})
    before = copy.deepcopy(graph)
    assert requests_composed_typed_scaled(graph)
    assert supports_composed_scaled_primal(graph)
    assert supports_composed_scale_jvp(graph, (4,))
    assert supports_scaled_reverse(graph, (4,))
    for role in range(4):
        assert not supports_composed_scale_jvp(graph, (role,))
        assert not supports_scaled_reverse(graph, (role,))
    assert graph == before


@pytest.mark.parametrize("attribute,value", [
    ("physical_contract", "rocm_mxfp4_w4a8_packed_folded_prefill_v1"),
    ("numeric_policy", {"accum": "fp32", "execution_mode": "approximate"}),
    ("scale_layout", {"granularity": "per_tensor", "block": [1, 32], "format": "e8m0"}),
])
def test_public_mxfp4_admission_rejects_conflicting_physical_semantics(attribute, value):
    from tessera.compiler.rocm_typed_scaled_native import supports_composed_scaled_primal
    owner, values = typed_case()
    graph = owner._specialized_autodiff_module(values, {})
    graph.functions[0].body[0].kwargs[attribute] = value
    assert not supports_composed_scaled_primal(graph)


@pytest.mark.skipif(not os.environ.get("TESSERA_OPT"), reason="matching native compiler required")
@pytest.mark.parametrize("mode", ["forward", "reverse"])
def test_native_mxfp4_export_rejects_active_row_scale_through_rounding(mode):
    from tessera.compiler.native_scaled_program import package_native_scaled_jvp, package_native_scaled_vjp
    _, values = typed_case()
    owner = ts.jit(target="rocm_gfx1201", autodiff=mode, wrt=("sa",))(gated)
    graph = owner._specialized_autodiff_module(values, {})
    graph.module_attrs.update({"tessera.target": '"rocm"', "tessera.arch": '"gfx1201"'})
    builder = package_native_scaled_jvp if mode == "forward" else package_native_scaled_vjp
    with pytest.raises(RuntimeError, match="cast|derivative|scaled JVP"):
        builder(graph.to_mlir(target="rocm_gfx1201", canonical=True))


def test_exact_mxfp4_oracle_preserves_bf16_store_before_widening():
    import ml_dtypes
    from tessera.compiler.rocm_nvfp4_program import reference_scaled_matmul
    owner, values = typed_case((1, 32, 1))
    args = list(values[:4])
    args[2] = np.array([1.003], np.float32)
    attrs = owner._specialized_autodiff_module(values, {}).functions[0].body[0].kwargs
    result = reference_scaled_matmul(*args, **attrs)
    assert result.dtype == np.dtype(ml_dtypes.bfloat16)
    expected = np.float32(16) * args[2][0]
    assert result.astype(np.float32)[0, 0] != expected
    assert result[0, 0] == ml_dtypes.bfloat16(expected)
