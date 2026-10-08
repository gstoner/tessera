"""Short-M packed native contract and end-to-end owner proof."""
import os
import subprocess
import numpy as np
import pytest
from tessera.compiler.rocm_nvfp4_program import build_packed_consumer_module
from tessera.compiler.rocm_nvfp4_resident import package_resident_packed_consumer
from tests.device.rocm.test_nvfp4_resident_jit import make_function
from tests.unit.test_rocm_nvfp4_resident import inputs_and_oracle

@pytest.mark.parametrize("m",[1,16,32,64,65,128])
def test_short_m_typing(m):
    graph=build_packed_consumer_module(m,32,64)
    assert str(graph.functions[0].result_types[0])==f"tensor<{m}x32xbf16>"

@pytest.mark.parametrize("m",[0,-1])
def test_zero_or_negative_m_refuses(m):
    with pytest.raises(ValueError):
        build_packed_consumer_module(m,32,64)

@pytest.mark.compiler_route
@pytest.mark.usefixtures("rocm_image_toolchain")
@pytest.mark.parametrize("shape",[(m,n,k) for m in (1,16,32,64) for n,k in ((32,64),(80,256))])
@pytest.mark.parametrize("runtime_mn",[False,True])
def test_short_m_native_package(shape,runtime_mn):
    m,n,k=shape
    consumer=package_resident_packed_consumer(m,n,k,runtime_mn=runtime_mn)
    consumer.validate()
    assert consumer.package.image.payload.startswith(b"\x7fELF")
    assert "tile.scaled_matmul_kernel" in consumer.package.tile_ir
    assert "tessera_rocm.scaled_wmma_gemm" in consumer.package.target_ir
    assert consumer.package.descriptor.geometry.grid==((n+63)//64,1,1)

@pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF")!="1",reason="owning gfx1201 required")
@pytest.mark.parametrize("shape",[(m,n,k) for m in (1,16,32,64) for n,k in ((32,64),(80,256))])
@pytest.mark.parametrize("reordered",[False,True])
def test_short_m_public_resident_chain(shape,reordered,monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler import rocm_nvfp4_ingest,rocm_mxfp4_storage,rocm_nvfp4_program
    assert rt._rocm_live_arch()=="gfx1201"
    m,n,k=shape
    args,_,_,_,expected=inputs_and_oracle(m,n,k)
    function=make_function(n,k,reordered)
    named=dict(zip(("codes","scales","projection_globals","a","a_scale"),args,strict=True))
    def forbidden(*args,**kwargs):
        raise AssertionError("compiled resident chain invoked host arithmetic")
    monkeypatch.setattr(rocm_nvfp4_ingest,"reference_nvfp4_requantize",forbidden)
    monkeypatch.setattr(rocm_mxfp4_storage,"reference_mxfp4_folded_storage",forbidden)
    monkeypatch.setattr(rocm_nvfp4_program,"reference_scaled_matmul",forbidden)
    actual=function(**named)
    np.testing.assert_allclose(actual.astype(np.float32),expected,rtol=.008,atol=.015625)
    assert function.execution_kind=="native_gpu"
    artifact=rt.RuntimeArtifact.from_json(function.runtime_artifact().to_json())
    monkeypatch.setattr(subprocess,"run",forbidden)
    named["a_scale"]=named["a_scale"]*.5
    changed=function(**named)
    np.testing.assert_allclose(changed.astype(np.float32),expected*.5,rtol=.008,atol=.015625)
    receipt=rt.launch(artifact,named)
    assert receipt["ok"] and receipt["execution_kind"]=="native_gpu"
    assert all(row["native_call_binding"]=="native_cpp_nvfp4" for row in receipt["component_receipts"])
    np.testing.assert_array_equal(receipt["output"],changed)


@pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF")!="1",reason="owning gfx1201 required")
@pytest.mark.parametrize("m",[3,7,17,31,63])
@pytest.mark.parametrize("runtime_mn",[False,True])
def test_short_m_ragged_static_and_runtime_owner(m,runtime_mn):
    from tessera import runtime as rt
    from tessera.compiler.rocm_nvfp4_resident import package_resident_nvfp4_matmul
    from tessera.compiler.rocm_nvfp4_ingest import nvfp4_requantization_policy
    assert rt._rocm_live_arch()=="gfx1201"
    args,offsets,_,_,expected=inputs_and_oracle(m,80,256)
    program=package_resident_nvfp4_matmul(m,80,256,offsets,
        numeric_policy=nvfp4_requantization_policy(),
        approximate_policy="explicit_allow",runtime_mn=runtime_mn)
    with program.session(*args) as session:
        session.run_combined()
        np.testing.assert_allclose(session.read_output().astype(np.float32),expected,
            rtol=.008,atol=.015625)
