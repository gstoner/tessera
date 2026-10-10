"""Exact-device native ownership of compiled NVFP4 stage images."""
from dataclasses import replace
import os
import numpy as np
import pytest
pytest_plugins=("tests.unit.test_rocm_nvfp4_resident",)

pytestmark=pytest.mark.skipif(os.getenv("TESSERA_GFX1201_DEVICE_PROOF")!="1",
                             reason="exact gfx1201 and matching native runtime")

def test_native_program_snapshots_stages_and_updates(compiled):
    program,args,converted,stored,expected=compiled
    original=[a.copy() for a in args]
    with program.native_session(*original) as session:
        for value in original:value[:]=0
        with pytest.raises(RuntimeError,match="invocation"):
            session.launch_matmul()
        session.run_combined()
        np.testing.assert_allclose(session.read_output().astype(np.float32),expected,rtol=.008,atol=.015625)
        diagnostic=session.diagnostics()
        for name,wanted in zip(("packed","exponents","stats"),converted):
            if name=="stats":np.testing.assert_allclose(diagnostic[name],wanted,rtol=1e-13,atol=1e-30)
            else:np.testing.assert_array_equal(diagnostic[name],wanted)
        np.testing.assert_array_equal(diagnostic["fragment"],stored[0])
        np.testing.assert_array_equal(diagnostic["plane"],stored[1])
        for stage in ("converter","storage","consumer","combined"):
            session.run_combined()
            assert all(v>0 for v in session.measure(stage,samples=2,repeats=3))
        session.update_activations(args[3],args[4]*np.float32(.5))
        session.launch_matmul()
        np.testing.assert_allclose(session.read_output().astype(np.float32),
            expected*.5,rtol=.008,atol=.015625)
        output=session.read_output()
        assert session.lib.tessera_rocm_nvfp4_read(session.handle,10,session.generation+1,
                                                   output.ctypes.data,output.nbytes)==10
    session.close()
    with pytest.raises(RuntimeError,match="closed"):session.run_combined()

def test_native_program_validation_precedes_native_preparation(compiled,monkeypatch):
    from tessera import runtime as rt
    program,args,*_=compiled
    def forbidden():pytest.fail("malformed program reached native runtime")
    monkeypatch.setattr(rt,"_load_rocm_native_movement_runtime",forbidden)
    bad=list(args);bad[-1]=np.full_like(bad[-1],np.nan)
    with pytest.raises(ValueError,match="finite"):program.native_session(*bad)
    consumer=program.consumer
    package=consumer.package
    descriptor=replace(package.descriptor,provenance={**package.descriptor.provenance,"image_whole_m":not package.descriptor.provenance["image_whole_m"]})
    changed=replace(program,consumer=replace(consumer,package=replace(package,descriptor=descriptor)))
    with pytest.raises(ValueError,match="descriptor"):changed.native_session(*args)

def test_native_portable_replay_avoids_compilation(compiled,monkeypatch):
    import subprocess
    from tessera.compiler.rocm_nvfp4_resident import NVFP4ResidentProgram
    program,args,*_=compiled
    encoded=program.to_json()
    def forbidden(*a,**k):pytest.fail("native replay ran a compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    restored=NVFP4ResidentProgram.from_json(encoded)
    with restored.native_session(*args) as active:
        active.run_combined()
        assert np.isfinite(active.read_output().astype(np.float32)).all()

def test_native_graph_replay_and_cached_activation_binding(compiled):
    program,args,converted,stored,expected=compiled
    with program.native_session(*args) as session:
        session.run_combined_graph()
        np.testing.assert_allclose(session.read_output().astype(np.float32),expected,rtol=.008,atol=.015625)
        for stage in ("converter","storage","consumer","ingest","combined"):
            session.run_combined()
            windows=session.measure_graph(stage,samples=2,repeats=16)
            assert all(w["graph_nodes"]==16*({"ingest":2,"combined":3}.get(stage,1)) for w in windows)
            assert all(w["host_graph_submissions"]==1 and w["per_iteration_ms"]>0 for w in windows)
        session.update_activations(args[3],args[4]*np.float32(.5))
        session.run_combined_graph()
        np.testing.assert_allclose(session.read_output().astype(np.float32),expected*.5,rtol=.008,atol=.015625)
        for r in range(1,11):
            assert session.measure_graph("combined",samples=1,repeats=r)[0]["graph_nodes"]==3*r
        np.testing.assert_allclose(session.read_output().astype(np.float32),expected*.5,rtol=.008,atol=.015625)
        diagnostic=session.diagnostics()
        np.testing.assert_array_equal(diagnostic["fragment"],stored[0])
        np.testing.assert_array_equal(diagnostic["plane"],stored[1])
        with pytest.raises(ValueError,match="4096"):
            session.measure_graph("combined",samples=1,repeats=4097)
