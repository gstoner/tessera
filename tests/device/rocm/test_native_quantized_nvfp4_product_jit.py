"""Exact gfx1201 three-member public native program, without host intermediates."""
from __future__ import annotations
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from benchmarks.rocm.benchmark_public_quantized_nvfp4_product import make_product,inputs,oracle

pytestmark=[pytest.mark.hardware_rocm,pytest.mark.compiler_route]


@pytest.mark.parametrize("m,n,k",[(1,1,16),(3,17,32),(19,33,64),(256,128,12288)])
def test_public_three_member_program_and_portable_replay(m,n,k,monkeypatch):
    assert rt._rocm_live_arch()=="gfx1201","owning gfx1201 required"
    values=inputs(m,n,k);expected=oracle(values)
    fn=ts.jit(target="rocm_gfx1201",native_required=True)(make_product(m,n,k))
    from tessera.compiler import nvfp4_expansion
    def forbidden(*a,**kw):raise AssertionError("compiled program called eager expansion")
    monkeypatch.setattr(nvfp4_expansion,"reference_nvfp4_expand",forbidden)
    first=fn(*values)
    np.testing.assert_allclose(first,expected,rtol=.0002,atol=.0002)
    assert fn.execution_kind=="native_gpu"
    assert len(fn._rocm_quantized_product_last_receipts)==3
    artifact=fn.runtime_artifact()
    assert artifact.metadata["native_program"]["schema"]=="tessera.rocm.quantized_product_frontend.v2"
    restored=rt.RuntimeArtifact.from_json(artifact.to_json())
    monkeypatch.setenv("TESSERA_OPT","/missing/compiler")
    second=fn(*inputs(m,n,k,seed=419))
    np.testing.assert_allclose(second,oracle(inputs(m,n,k,seed=419)),rtol=.0002,atol=.0002)
    replay=rt.launch(restored,dict(zip(("x","packed","weight_scales","globals_"),values)))
    assert replay.get("ok") and replay.get("execution_kind")=="native_gpu",replay
    np.testing.assert_allclose(replay["output"],expected,rtol=.0002,atol=.0002)
    np.testing.assert_allclose(first,expected,rtol=.0002,atol=.0002)
