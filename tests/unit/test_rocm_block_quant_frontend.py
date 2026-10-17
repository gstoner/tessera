"""Real frontend quantizer capture and immutable upstream tensor literals."""
import numpy as np
import pytest
import tessera as ts
from tessera.compiler.trace import trace,to_graph_ir_module,TesseraTraceError


def make_quantizer(m=3,k=32):
    groups=m*k//16
    divisor=np.full((groups,1),448,np.float32);divisor.flags.writeable=False
    floor=np.full((groups,1),np.finfo(np.float32).tiny,np.float32);floor.flags.writeable=False

    def quant(x):
        blocks=ts.ops.reshape(x,shape=(groups,16))
        wide=ts.ops.cast(blocks,"fp32")
        absolute=ts.ops.absolute(wide)
        peak=ts.ops.reduce(absolute,op="max",axis=1,keepdims=True)
        raw=ts.ops.div(peak,divisor)
        scales=ts.ops.maximum(raw,floor)
        expanded=ts.ops.expand(scales,shape=(groups,16))
        normalized=ts.ops.div(wide,expanded)
        codes=ts.ops.cast(normalized,"fp8_e4m3")
        return ts.ops.reshape(codes,shape=(m,k)),ts.ops.reshape(scales,shape=(m,k//16))
    return quant


def captured(m=3,k=32):
    result=trace(make_quantizer(m,k),((m,k),"bf16"),arg_names=["x"],evaluate_catalog_outputs=False)
    assert result.output_values==(None,None)
    return to_graph_ir_module(result,name="quant",target="rocm_gfx1201")


def test_real_quantizer_capture_has_explicit_constants_and_two_storage_results():
    module=captured()
    fn=module.functions[0]
    assert len(fn.body)==13
    assert [op.op_name for op in fn.body].count("arith.constant")==2
    assert fn.body[3].result_type=="tensor<6x1xf32>"
    assert list(map(str,fn.result_types))==["tensor<3x32xf8E4M3FN>","tensor<3x2xf32>"]


@pytest.mark.parametrize("axis",[None,0,1,-1])
@pytest.mark.parametrize("keep",[False,True])
def test_abstract_reduce_axes_share_canonical_catalog_shape(axis,keep):
    def fn(x):return ts.ops.reduce(x,op="max",axis=axis,keepdims=keep)
    traced=trace(fn,((3,7),"fp32"),evaluate_catalog_outputs=False)
    module=to_graph_ir_module(traced,name="reduce")
    expected=np.max(np.ones((3,7),np.float32),axis=axis,keepdims=keep)
    assert tuple(map(int,module.functions[0].result_types[0].shape))==expected.shape


@pytest.mark.parametrize("values",[[1.,2.],[0.,-0.],[float("inf")]*2,[float("nan")]*2])
def test_immutable_literal_refuses_unproved_splat_encodings(values):
    constant=np.asarray(values,np.float32);constant.flags.writeable=False
    def fn(x):return ts.ops.div(x,constant)
    with pytest.raises(TesseraTraceError,match="finite bit-identical"):
        trace(fn,((2,),"fp32"),evaluate_catalog_outputs=False)


def test_mutable_capture_is_not_silently_frozen_as_a_tensor_constant():
    constant=np.ones((2,),np.float32)
    def fn(x):return ts.ops.div(x,constant)
    with pytest.raises(TesseraTraceError,match="non-Tracer"):
        trace(fn,((2,),"fp32"),evaluate_catalog_outputs=False)


@pytest.mark.usefixtures("production_compiler")
def test_authored_quantizer_projects_without_mutating_its_graph():
    from tessera.compiler.rocm_block_quant_native import lower_block_quant_graph
    module=captured()
    before=module.to_mlir(target="rocm_gfx1201",canonical=True)
    recipe=lower_block_quant_graph(module,"rocm_gfx1201")
    assert "schema = 4" in recipe.tile_ir
    assert 'tessera.launch_bindings = ["x", "v11", "v12"]' in recipe.graph_ir
    assert module.to_mlir(target="rocm_gfx1201",canonical=True)==before


def test_jit_frontend_identifies_authored_quantizer_without_execution():
    from tessera.compiler.rocm_block_quant_native import requests_block_quant
    fn=ts.jit(target="rocm_gfx1201",native_required=True)(make_quantizer())
    assert requests_block_quant(fn.graph_ir)


@pytest.mark.usefixtures("production_compiler")
def test_canonical_frontend_artifact_restores_without_a_compiler(monkeypatch):
    from tessera.compiler.canonical_compile import canonical_compile
    from tessera.compiler.rocm_block_quant_native import validate_block_quant_runtime_artifact
    from tessera import runtime as rt
    compiled=canonical_compile(captured(),target="rocm_gfx1201",enable_tool_validation=False)
    assert compiled.executable,compiled.reason
    assert compiled.program_executable,compiled.component_blockers
    assert "arith.constant" in compiled.component_ops
    artifact=compiled.to_runtime_artifact()
    assert artifact.metadata["compiler_path"]=="rocm_block_quant_native_descriptor"
    serialized=artifact.to_json()
    monkeypatch.setenv("TESSERA_OPT","/missing/compiler")
    monkeypatch.setenv("PATH","/missing/tools")
    restored=rt.RuntimeArtifact.from_json(serialized)
    package=validate_block_quant_runtime_artifact(restored)
    assert package.images[0].startswith(b"\x7fELF")
    assert len(package.members_json)==1
