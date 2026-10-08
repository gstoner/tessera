"""Frontend tracing and ordinary gfx1201 JIT checkpoint conversion."""
import os
import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera.compiler.rocm_nvfp4_ingest import nvfp4_requantization_policy
from tests.unit.test_rocm_nvfp4_ingest_package import operands

@ts.jit(target="rocm_gfx1201")
def checkpoint(codes,scales,globals_):
    return ts.ops.nvfp4_requantize(codes,scales,globals_,
        row_offsets=[0,3,7],numeric_policy=nvfp4_requantization_policy())

def test_frontend_infers_three_results_without_host_conversion(monkeypatch):
    from tessera.compiler import rocm_nvfp4_ingest as ingest
    from tessera.compiler.trace import trace,to_graph_ir_module
    def forbidden(*a,**k):
        pytest.fail("tracing ran the host converter")
    monkeypatch.setattr(ingest,"reference_nvfp4_requantize",forbidden)
    module=to_graph_ir_module(trace(checkpoint._fn,*operands()),target="rocm_gfx1201")
    assert [str(t) for t in module.functions[0].result_types]==[
        "tensor<7x32xui8>","tensor<2x7xui8>","tensor<7x2x2xf64>"]
    assert [arg.ir_type.dtype for arg in module.functions[0].args]==[
        "uint8","fp8_e4m3","fp64"]

@pytest.mark.skipif(os.getenv("TESSERA_GFX1201_DEVICE_PROOF")!="1",reason="exact gfx1201")
def test_ordinary_jit_executes_cached_native_image_and_replays(monkeypatch):
    from tessera.compiler import rocm_nvfp4_ingest as ingest
    from tessera import runtime as rt
    args=operands()
    expected=ingest.reference_nvfp4_requantize(*args,row_offsets=[0,3,7],
        numeric_policy=nvfp4_requantization_policy())
    def forbidden(*a,**k):
        pytest.fail("native JIT ran the host converter")
    monkeypatch.setattr(ingest,"reference_nvfp4_requantize",forbidden)
    monkeypatch.setattr(ingest,"ingest_nvfp4_projections",forbidden)
    actual=checkpoint(*args)
    artifact=checkpoint.runtime_artifact()
    image=artifact.native_image.image_digest
    assert checkpoint.execution_kind=="native_gpu"
    assert artifact.metadata["canonical_executable"]
    assert artifact.launch_descriptor is not None
    again=checkpoint(globals_=args[2],codes=args[0],scales=args[1])
    assert checkpoint.runtime_artifact().native_image.image_digest==image
    for result in (actual,again):
        np.testing.assert_array_equal(result[0],expected[0])
        np.testing.assert_array_equal(result[1],expected[1])
        np.testing.assert_allclose(result[2],expected[2],rtol=1e-13,atol=1e-30)
    restored=rt.RuntimeArtifact.from_json(artifact.to_json())
    bindings=sorted(restored.launch_descriptor.buffers,key=lambda b:b.ordinal)
    arrays=list(args)+[np.empty_like(x) for x in expected]
    receipt=rt.launch(restored,{"buffers":dict(zip((b.name for b in bindings),arrays)),"scalars":{}})
    assert receipt.get("ok"),receipt
    assert receipt["execution_kind"]=="native_gpu"
    np.testing.assert_array_equal(arrays[3],expected[0])
    np.testing.assert_array_equal(arrays[4],expected[1])
    np.testing.assert_allclose(arrays[5],expected[2],rtol=1e-13,atol=1e-30)


@pytest.mark.skipif(os.getenv("TESSERA_GFX1201_DEVICE_PROOF")!="1",reason="exact gfx1201")
@pytest.mark.parametrize("kind",["bad_global","negative_scale","wrong_dtype"])
def test_ordinary_jit_rejects_bad_inputs_before_hip(kind,monkeypatch):
    from tessera import runtime as rt
    args=list(operands())
    if kind=="bad_global":
        args[2][0]=np.inf
    elif kind=="negative_scale":
        args[1][0,0]=-1
    else:
        args[0]=args[0].astype(np.int8)
    def forbidden():
        pytest.fail("invalid JIT inputs reached HIP")
    monkeypatch.setattr(rt,"_load_hip_for_launch",forbidden)
    with pytest.raises((ValueError,RuntimeError),match="native|NVFP4|LEGALITY_TARGET_CAPABILITY"):
        checkpoint(*args)


def test_frontend_captures_projection_boundaries_as_literal_attributes():
    from tessera.compiler.graph_ir import GraphIRBuilder
    offsets=[0,3,7]
    def convert(codes,scales,globals_):
        return ts.ops.nvfp4_requantize(codes,scales,globals_,
            row_offsets=offsets,numeric_policy=nvfp4_requantization_policy())
    builder=GraphIRBuilder()
    builder.lower(convert)
    assert builder.module().functions[0].body[0].kwargs["row_offsets"]==[0,3,7]
    offsets[1]=2
    assert builder.module().functions[0].body[0].kwargs["row_offsets"]==[0,3,7]
