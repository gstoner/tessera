"""Exact gfx1201 full rebinding and native allocation-cache ownership."""
import ctypes as C
import os
import numpy as np
import pytest
from tessera import runtime as rt
from tessera.compiler.rocm_nvfp4_ingest import reference_nvfp4_requantize, nvfp4_requantization_policy
from tessera.compiler.rocm_mxfp4_storage import reference_mxfp4_folded_storage, MXFP4_STORAGE_CONTRACT
from tessera.compiler.rocm_mxfp4 import folded_weights
from tessera.compiler.rocm_mxfp4_folded import prepare_folded_weights

pytest_plugins=("tests.unit.test_rocm_nvfp4_resident",)
pytestmark=pytest.mark.skipif(os.getenv("TESSERA_GFX1201_DEVICE_PROOF")!="1",
                             reason="matching exact gfx1201 compiler/runtime required")


def clear_cache():
    lib=rt._load_rocm_native_movement_runtime()
    lib.tessera_rocm_nvfp4_cache_clear.argtypes=[]
    lib.tessera_rocm_nvfp4_cache_clear.restype=C.c_int
    assert lib.tessera_rocm_nvfp4_cache_clear()==0
    return lib


def changed_inputs(args, offsets):
    codes,scales,globals_,a,a_scale=(value.copy() for value in args)
    codes^=np.uint8(0x88)
    scales[:]=(scales.astype(np.float32)*.5).astype(scales.dtype)
    globals_*=.5
    a^=np.uint8(128)
    a_scale*=2
    converted=reference_nvfp4_requantize(codes,scales,globals_,
        row_offsets=offsets,numeric_policy=nvfp4_requantization_policy())
    stored=reference_mxfp4_folded_storage(*converted[:2],storage_contract=MXFP4_STORAGE_CONTRACT)
    weights=folded_weights(prepare_folded_weights(*converted[:2],allow_approximate=True))
    import ml_dtypes
    expected=((a.view(ml_dtypes.float8_e4m3fn).astype(np.float64)*a_scale[:,None])
              @ weights.astype(np.float64).T).astype(ml_dtypes.bfloat16).astype(np.float32)
    return (codes,scales,globals_,a,a_scale),converted,stored,expected


def verify(session, converted, stored, expected):
    np.testing.assert_allclose(session.read_output().astype(np.float32),expected,rtol=.008,atol=.015625)
    actual=session.diagnostics()
    for name,wanted in zip(("packed","exponents","stats"),converted):
        if name=="stats":
            np.testing.assert_allclose(actual[name],wanted,rtol=1e-13,atol=1e-30)
        else:
            np.testing.assert_array_equal(actual[name],wanted)
    np.testing.assert_array_equal(actual["fragment"],stored[0])
    np.testing.assert_array_equal(actual["plane"],stored[1])


def test_full_inputs_invalidate_derived_state_and_snapshot(compiled):
    program,args,*_=compiled
    offsets=program.ingest.native.descriptor.provenance["row_offsets"]
    changed,converted,stored,expected=changed_inputs(args,offsets)
    with program.native_session(*args) as session:
        session.run_combined()
        session.update_inputs(*changed)
        with pytest.raises(RuntimeError):
            session.read_output()
        with pytest.raises(RuntimeError):
            session.launch_matmul()
        for value in changed:
            value[:]=0
        session.run_combined()
        verify(session,converted,stored,expected)
        # Invalid input rejection preserves the last complete result.
        invalid=[value.copy() for value in args]
        invalid[2][0]=np.nan
        with pytest.raises(ValueError):
            session.update_inputs(*invalid)
        verify(session,converted,stored,expected)


def test_cached_checkout_rebinds_every_input_and_retires_token(compiled):
    program,args,converted,stored,expected=compiled
    lib=clear_cache()
    with program.native_session(*args,reuse=True) as first:
        assert not first.native_cache_hit
        old=first.handle
        first.run_combined()
        verify(first,converted,stored,expected)
    offsets=program.ingest.native.descriptor.provenance["row_offsets"]
    changed,converted,stored,expected=changed_inputs(args,offsets)
    try:
        with program.native_session(*changed,reuse=True) as second:
            assert second.native_cache_hit
            assert second.handle!=old
            generation=C.c_uint64()
            assert lib.tessera_rocm_nvfp4_invoke(old,4,1,C.byref(generation),None)==1
            for value in changed:
                value[:]=0
            second.run_combined()
            verify(second,converted,stored,expected)
            # A concurrent lease cannot acquire the currently checked-out owner.
            with program.native_session(*args,reuse=True) as third:
                assert not third.native_cache_hit
                assert third.handle!=second.handle
                third.run_combined()
                np.testing.assert_allclose(third.read_output().astype(np.float32),
                                           compiled[-1],rtol=.008,atol=.015625)
            verify(second,converted,stored,expected)
    finally:
        clear_cache()


def test_graph_owner_is_cleaned_instead_of_retained(compiled):
    program,args,*_=compiled
    clear_cache()
    try:
        with program.native_session(*args,reuse=True) as first:
            first.run_combined_graph()
        with program.native_session(*args,reuse=True) as second:
            assert not second.native_cache_hit
            second.run_combined()
    finally:
        clear_cache()

@pytest.mark.parametrize("shape",[(128,32,256),(257,80,1024),(256,64,64)])
def test_public_jit_and_portable_cache_consume_mutated_weights(shape,monkeypatch):
    from tests.device.rocm.test_nvfp4_resident_jit import make_function
    from tests.unit.test_rocm_nvfp4_resident import inputs_and_oracle
    arrays,offsets,_,_,expected=inputs_and_oracle(*shape)
    changed,_,_,changed_expected=changed_inputs(arrays,offsets)
    owned=[value.copy() for value in arrays]
    function=make_function(shape[1],shape[2],True)
    clear_cache()
    names=("codes","scales","projection_globals","a","a_scale")
    named=dict(zip(names,owned))
    try:
        actual=function(**named)
        np.testing.assert_allclose(actual.astype(np.float32),expected,rtol=.008,atol=.015625)
        artifact=rt.RuntimeArtifact.from_json(function.runtime_artifact().to_json())
        def forbidden(*args,**kwargs):
            pytest.fail("warm compiled execution used eager code or compilation")
        monkeypatch.setattr(function,"_fn",forbidden)
        monkeypatch.setattr(function,"compile_native_nvfp4_program",forbidden)
        for source,wanted in ((changed,changed_expected),(arrays,expected)):
            for dest,value in zip(owned,source,strict=True):
                np.copyto(dest,value)
            np.testing.assert_allclose(function(**named).astype(np.float32),wanted,rtol=.008,atol=.015625)
            receipt=rt.launch(artifact,named)
            assert receipt["ok"] and receipt["execution_kind"]=="native_gpu",receipt
            assert all(item["native_allocation_cache_hit"] for item in receipt["component_receipts"])
            np.testing.assert_allclose(receipt["output"].astype(np.float32),wanted,rtol=.008,atol=.015625)
    finally:
        clear_cache()


@pytest.mark.parametrize("shape",[(128,32,256),(257,80,1024),(256,64,64)])
def test_warm_static_program_retention_and_invalid_mutation_gate(shape,monkeypatch):
    from copy import deepcopy
    from tests.device.rocm.test_nvfp4_resident_jit import make_function
    from tests.unit.test_rocm_nvfp4_resident import inputs_and_oracle
    from tessera.compiler import prepared_rocm_nvfp4_program as cache
    from tessera.compiler import rocm_nvfp4_program as source
    arrays,*_,expected=inputs_and_oracle(*shape)
    function=make_function(shape[1],shape[2],True)
    named=dict(zip(("codes","scales","projection_globals","a","a_scale"),arrays))
    with cache._LOCK:
        cache._CACHE.clear()
    clear_cache()
    try:
        function(**named)
        artifact=rt.RuntimeArtifact.from_json(function.runtime_artifact().to_json())
        original=source.program_from_manifest
        def forbidden(*args,**kwargs):
            pytest.fail("warm static contract was reconstructed")
        monkeypatch.setattr(source,"program_from_manifest",forbidden)
        np.testing.assert_allclose(function(**named).astype(np.float32),expected,rtol=.008,atol=.015625)
        result=rt.launch(artifact,named)
        assert result["ok"],result
        np.testing.assert_allclose(result["output"].astype(np.float32),expected,rtol=.008,atol=.015625)
        monkeypatch.setattr(source,"program_from_manifest",original)
        bad=deepcopy(artifact.metadata)
        bad["native_program"]["role_indices"][-1]=False
        manifest=bad["native_program"]
        manifest["contract_digest"]=source._program_digest(
            {k:v for k,v in manifest.items() if k!="contract_digest"})
        def no_hip(*args,**kwargs):
            pytest.fail("invalid changed contract reached HIP")
        monkeypatch.setattr(rt,"_load_hip_for_launch",no_hip)
        monkeypatch.setattr(rt,"_load_rocm_native_movement_runtime",no_hip)
        failed=rt.launch(rt.RuntimeArtifact(graph_ir=artifact.graph_ir,metadata=bad),named)
        assert not failed["ok"] and "argument ABI" in failed["reason"],failed
    finally:
        # Restore runtime loading before clearing actual retained native owners.
        monkeypatch.undo()
        clear_cache()
