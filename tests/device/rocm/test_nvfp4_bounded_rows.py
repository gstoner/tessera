"""Exact gfx1201 bounded NVFP4 ingest/product replay and native ownership."""
import os
import subprocess

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.rocm_nvfp4_program import program_from_manifest, runtime_artifact
from tests.device.rocm.test_nvfp4_resident_jit import make_function
from tests.unit.test_rocm_nvfp4_resident import inputs_and_oracle

pytestmark=[pytest.mark.hardware_rocm,
    pytest.mark.skipif(os.getenv("TESSERA_GFX1201_DEVICE_PROOF")!="1",
                      reason="exact gfx1201 compiler/runtime required")]


@pytest.mark.parametrize("shape",[(257,32,64),(513,80,256),(256,64,1024)])
@pytest.mark.parametrize("reordered",[False,True])
def test_bounded_public_and_portable_reuse_without_compiler(shape,reordered,monkeypatch):
    bound,n,k=shape
    frames=[inputs_and_oracle(rows,n,k) for rows in (17,1,bound,200,1,17)]
    order=(4,3,0,2,1) if reordered else tuple(range(5))
    function=make_function(n,k,reordered)
    program=function.compile_native_nvfp4_program(
        *(frames[0][0][i] for i in order),m_bound=bound)
    assert program.native.consumer.m==bound
    program=program_from_manifest(program.manifest())
    artifact=rt.RuntimeArtifact.from_json(runtime_artifact(program).to_json())
    digest=program.manifest()["contract_digest"]
    images=tuple(p["image"]["image_digest"] for p in program.native.to_dict()["stages"])
    def forbidden(*args,**kwargs):pytest.fail("compiler/eager frontend called during bounded replay")
    monkeypatch.setattr(subprocess,"run",forbidden)
    monkeypatch.setattr(function,"_fn",forbidden)
    monkeypatch.setattr(function,"_trace_frontend_capture",forbidden)
    retained=[]
    for arguments,_,_,_,expected in frames:
        actual=function(*(arguments[i] for i in order))
        np.testing.assert_allclose(actual.astype("f4"),expected,rtol=.008,atol=.015625)
        receipt=rt.launch(artifact,tuple(arguments[i] for i in order))
        assert receipt["ok"] and receipt["execution_kind"]=="native_gpu"
        np.testing.assert_array_equal(actual,receipt["output"])
        assert program.manifest()["contract_digest"]==digest
        assert tuple(p["image"]["image_digest"] for p in program.native.to_dict()["stages"])==images
        retained.append((actual,actual.copy()))
    for actual,held in retained:np.testing.assert_array_equal(actual,held)
    invalid=inputs_and_oracle(bound+1,n,k)[0]
    monkeypatch.setattr(rt,"_rocm_live_arch",forbidden)
    with pytest.raises(ValueError,match="capacity"):
        program.execute(*(invalid[i] for i in order))


@pytest.mark.parametrize("bound",[257,513])
def test_one_native_owner_rebinds_rows_and_invalidates_captured_geometry(bound):
    n,k=32,64
    frames=[inputs_and_oracle(rows,n,k) for rows in (17,bound,1,200,17)]
    function=make_function(n,k)
    program=function.compile_native_nvfp4_program(*frames[0][0],m_bound=bound)
    with program.native.native_session(*frames[0][0],reuse=True) as session:
        handle=session.handle
        initial=session.frame_stats()
        assert initial["capacity_m"]==bound and initial["allocation_count"]==11
        for args,_,_,_,expected in frames:
            session.update_inputs(*args)
            session.run_combined_graph()
            actual=session.read_output()
            np.testing.assert_allclose(actual.astype("f4"),expected,rtol=.008,atol=.015625)
            stats=session.frame_stats()
            assert session.handle==handle and stats["active_m"]==args[3].shape[0]
            assert stats["allocation_bytes"]==initial["allocation_bytes"]
            assert stats["allocation_count"]==initial["allocation_count"]
            # Activation-only rebinding preserves the newly ingested weights.
            active=args[3][:1].copy();scale=args[4][:1].copy()
            session.update_activations(active,scale)
            session.launch_matmul_graph()
            np.testing.assert_array_equal(session.read_output(),actual[:1])
