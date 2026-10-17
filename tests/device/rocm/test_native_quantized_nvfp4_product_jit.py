"""Exact gfx1201 three-member public native program, without host intermediates."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from benchmarks.rocm.benchmark_public_quantized_nvfp4_product import make_product,inputs,oracle

pytestmark=[pytest.mark.hardware_rocm,pytest.mark.compiler_route]


@pytest.mark.parametrize("m,n,k",[(1,1,16),(3,17,32),(19,33,64),(256,128,12288)])
def test_public_three_member_program_and_portable_replay(m,n,k,monkeypatch):
    assert rt._rocm_live_arch()=="gfx1201","owning gfx1201 required"
    values=inputs(m,n,k);expected=oracle(values);snapshots=[value.copy() for value in values]
    fn=ts.jit(target="rocm_gfx1201",native_required=True)(make_product(m,n,k))
    from tessera.compiler import nvfp4_expansion
    def forbidden(*a,**kw):raise AssertionError("compiled program called eager expansion")
    monkeypatch.setattr(nvfp4_expansion,"reference_nvfp4_expand",forbidden)
    first=fn(*values)
    np.testing.assert_allclose(first,expected,rtol=.0002,atol=.0002)
    for before,after in zip(snapshots,values,strict=True):
        np.testing.assert_array_equal(before.view(np.uint8),after.view(np.uint8))
    assert fn.execution_kind=="native_gpu"
    assert len(fn._rocm_quantized_product_last_receipts)==3
    report=fn.compile_report()
    assert "3 ordered native packages" in report.target_decision["rocm_gfx1201"]
    assert all(len(report.ir_hashes[layer])==64 for layer in ("graph_ir","schedule_ir","tile_ir","target_ir"))
    artifact=fn.runtime_artifact()
    assert artifact.metadata["compiler_path"]=="canonical_rocm_quantized_nvfp4_product_program"
    assert sum(line.lstrip().startswith("schedule.artifact {") for line in artifact.schedule_ir.splitlines())==3
    assert artifact.metadata["native_program"]["schema"]=="tessera.rocm.quantized_product_frontend.v2"
    restored=rt.RuntimeArtifact.from_json(artifact.to_json())
    monkeypatch.setenv("TESSERA_OPT","/missing/compiler")
    second=fn(*inputs(m,n,k,seed=419))
    np.testing.assert_allclose(second,oracle(inputs(m,n,k,seed=419)),rtol=.0002,atol=.0002)
    replay=rt.launch(restored,dict(zip(("x","packed","weight_scales","globals_"),values)))
    assert replay.get("ok") and replay.get("execution_kind")=="native_gpu",replay
    np.testing.assert_allclose(replay["output"],expected,rtol=.0002,atol=.0002)
    np.testing.assert_allclose(first,expected,rtol=.0002,atol=.0002)


def test_fresh_process_portable_three_member_replay_without_compiler(tmp_path):
    assert rt._rocm_live_arch()=="gfx1201","owning gfx1201 required"
    values=inputs(3,17,32)
    fn=ts.jit(target="rocm_gfx1201",native_required=True)(make_product())
    expected=fn(*values)
    (tmp_path/"artifact.json").write_text(fn.runtime_artifact().to_json())
    np.savez(tmp_path/"roots.npz",x=values[0].astype(np.float32),packed=values[1],scales=values[2].view(np.uint8),globals=values[3])
    script="\n".join([
        "import sys,numpy as np,ml_dtypes",
        "from pathlib import Path",
        "from tessera import runtime as rt",
        "root=Path(sys.argv[1]);p=np.load(root/'roots.npz')",
        "artifact=rt.RuntimeArtifact.from_json((root/'artifact.json').read_text())",
        "values=[p['x'].astype(ml_dtypes.bfloat16),p['packed'],p['scales'].view(ml_dtypes.float8_e4m3fn),p['globals']]",
        "result=rt.launch(artifact,values)",
        "assert result.get('ok') and result.get('execution_kind')=='native_gpu',result",
        "assert len(result['component_receipts'])==3",
        "np.save(root/'result.npy',result['output'])",
    ])
    result=subprocess.run([sys.executable,"-c",script,str(tmp_path)],capture_output=True,text=True,
        env={**os.environ,"PATH":"/missing/tools","TESSERA_OPT":"/missing/compiler"})
    assert result.returncode==0,result.stdout+result.stderr
    np.testing.assert_array_equal(np.load(tmp_path/"result.npy"),expected)


def test_native_owner_rejects_missing_producers_and_invalidates_private_planes_on_update():
    assert rt._rocm_live_arch()=="gfx1201","owning gfx1201 required"
    values=inputs(3,17,32);changed=inputs(3,17,32,seed=911)
    fn=ts.jit(target="rocm_gfx1201",native_required=True)(make_product())
    first=fn(*values);program=fn._rocm_quantized_product_last_program.native
    lib=rt._load_rocm_native_movement_runtime()
    with program.prepare(values,runtime_library=lib._name) as owner:
        with pytest.raises(RuntimeError,match="status 10"):
            owner.invoke_range(2,1)
        owner.invoke_range(0,1)
        with pytest.raises(RuntimeError,match="status 10"):
            owner.invoke_range(2,1)
        owner.invoke_range(1,1)
        generation,_=owner.invoke_range(2,1)
        held=owner.read(generation)[0]
        np.testing.assert_allclose(held,oracle(values),rtol=.0002,atol=.0002)
        owner.update(changed)
        with pytest.raises(RuntimeError,match="status 10"):
            owner.invoke_range(2,1)
        generation,_=owner.invoke()
        np.testing.assert_allclose(owner.read(generation)[0],oracle(changed),rtol=.0002,atol=.0002)
        np.testing.assert_array_equal(held,first)
