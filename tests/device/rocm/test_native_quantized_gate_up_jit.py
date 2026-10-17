"""Owning gfx1201 proof for quantizer/expansion/BF16-store/SiLU composition."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph
from benchmarks.rocm.benchmark_public_quantized_gate_up import make_gate_up,inputs,oracle
from tests.unit.test_native_quantized_gate_up_program import source

pytestmark=[pytest.mark.hardware_rocm,pytest.mark.compiler_route]


def close(got,wanted):
    assert got.dtype==np.dtype(ml_dtypes.bfloat16)
    np.testing.assert_allclose(got.astype(np.float32),wanted.astype(np.float32),rtol=.015625,atol=.00002)


@pytest.mark.parametrize("m,width,k",[(1,1,16),(3,17,48),(256,128,12288)])
def test_public_four_member_call_has_native_math_and_current_roots(m,width,k,monkeypatch):
    assert rt._rocm_live_arch()=="gfx1201","owning gfx1201 required"
    values=inputs(m,width,k);wanted=oracle(values)
    changed=inputs(m,width,k,seed=419);changed[3]*=np.array([1.25,.75]);wanted2=oracle(changed)
    before=[value.copy() for value in values]
    fn=ts.jit(target="rocm_gfx1201",native_required=True)(make_gate_up(m,width,k))
    from tessera.compiler import nvfp4_expansion
    with monkeypatch.context() as patch:
        def forbidden(*a,**kw):raise AssertionError("native four-member path executed eager numerical work")
        for name in ("max","maximum","exp"):patch.setattr(np,name,forbidden)
        patch.setattr(nvfp4_expansion,"reference_nvfp4_expand",forbidden)
        first=fn(*values)
    close(first,wanted)
    assert fn.execution_kind=="native_gpu"
    assert len(fn._rocm_quantized_product_last_receipts)==4
    for a,b in zip(before,values,strict=True):np.testing.assert_array_equal(a.view(np.uint8),b.view(np.uint8))
    report=fn.compile_report()
    assert "4 ordered native packages" in report.target_decision["rocm_gfx1201"]
    assert all(len(report.ir_hashes[key])==64 for key in ("graph_ir","schedule_ir","tile_ir","target_ir"))
    monkeypatch.setenv("TESSERA_OPT","/missing/compiler")
    second=fn(*changed)
    close(second,wanted2);close(first,wanted)


@pytest.mark.parametrize("reverse",[False,True])
@pytest.mark.parametrize("reorder",[False,True])
def test_analytic_gate_orientation_and_original_root_roles(reverse,reorder):
    assert rt._rocm_live_arch()=="gfx1201","owning gfx1201 required"
    text=source(1,1,16,reverse=reverse)
    values=[np.full((1,16),448,dtype=ml_dtypes.bfloat16),
            np.array([[0x22]*8,[0xaa]*8],np.uint8),np.ones((2,1),dtype=ml_dtypes.float8_e4m3fn),
            np.array([1/7168,1/3584],np.float64)]
    if reorder:
        begin=text.index("(",text.index("func.func"))+1;end=text.index(") ->",begin)
        args=text[begin:end].split(", ")
        text=text[:begin]+", ".join(args[i] for i in (3,1,0,2))+text[end:]
        values=[values[i] for i in (3,1,0,2)]
    package=package_quantized_product_graph(text)
    library=rt._load_rocm_native_movement_runtime()
    gate,up=(np.float32(-2),np.float32(1)) if reverse else (np.float32(1),np.float32(-2))
    wanted=np.array([[gate/(np.float32(1)+np.exp(-gate))*up]],dtype=ml_dtypes.bfloat16)
    with package.prepare(values,runtime_library=library._name) as owner:
        generation,_=owner.invoke();got=owner.read(generation)[0]
    np.testing.assert_array_equal(got.view(np.uint16),wanted.view(np.uint16))


def test_parent_readiness_and_update_invalidation_cover_all_five_private_planes():
    assert rt._rocm_live_arch()=="gfx1201","owning gfx1201 required"
    values=inputs(3,17,48);changed=inputs(3,17,48,seed=911);changed[3]*=np.array([.5,1.5])
    package=package_quantized_product_graph(source())
    library=rt._load_rocm_native_movement_runtime()
    with package.prepare(values,runtime_library=library._name) as owner:
        with pytest.raises(RuntimeError,match="status 10"):owner.invoke_range(3,1)
        owner.invoke_range(0,3)
        generation,_=owner.invoke_range(3,1);held=owner.read(generation)[0]
        close(held,oracle(values))
        owner.update(changed)
        with pytest.raises(RuntimeError,match="status 10"):owner.invoke_range(3,1)
        generation,_=owner.invoke();close(owner.read(generation)[0],oracle(changed))
        close(held,oracle(values))


def test_four_member_portable_roundtrip_and_new_named_roots(monkeypatch):
    assert rt._rocm_live_arch()=="gfx1201","owning gfx1201 required"
    values=inputs(3,17,48);changed=inputs(3,17,48,seed=812);changed[3]*=np.array([1.5,.5])
    wanted,wanted2=oracle(values),oracle(changed)
    fn=ts.jit(target="rocm_gfx1201",native_required=True)(make_gate_up())
    first=fn(*values)
    artifact=fn.runtime_artifact()
    assert artifact.metadata["compiler_path"]=="canonical_rocm_quantized_gate_up_program"
    restored=rt.RuntimeArtifact.from_json(artifact.to_json())
    monkeypatch.setenv("TESSERA_OPT","/missing/compiler");monkeypatch.setenv("PATH","/missing/tools")
    result=rt.launch(restored,dict(zip(("x","packed","weight_scales","globals_"),changed,strict=True)))
    assert result.get("ok") and result.get("execution_kind")=="native_gpu",result
    assert len(result["component_receipts"])==4
    close(result["output"],wanted2);close(first,wanted)


def test_fresh_process_four_member_replay_without_compiler(tmp_path):
    assert rt._rocm_live_arch()=="gfx1201","owning gfx1201 required"
    values=inputs(3,17,48)
    fn=ts.jit(target="rocm_gfx1201",native_required=True)(make_gate_up())
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
        "assert len(result['component_receipts'])==4",
        "np.save(root/'result.npy',result['output'])",
    ])
    result=subprocess.run([sys.executable,"-c",script,str(tmp_path)],capture_output=True,text=True,
        env={**os.environ,"PATH":"/missing/tools","TESSERA_OPT":"/missing/compiler"})
    assert result.returncode==0,result.stdout+result.stderr
    np.testing.assert_array_equal(np.load(tmp_path/"result.npy").view(np.uint16),expected.view(np.uint16))


def test_nested_inspection_metadata_cannot_poison_cached_gate_up(monkeypatch):
    assert rt._rocm_live_arch()=="gfx1201","owning gfx1201 required"
    values=inputs(3,17,48);changed=inputs(3,17,48,seed=619);wanted=oracle(changed)
    fn=ts.jit(target="rocm_gfx1201",native_required=True)(make_gate_up())
    fn(*values);artifact=fn.runtime_artifact()
    artifact.metadata["native_program"]["native_program"]["stages"][3]["descriptor"]["provenance"]["native_parent_view"]["member_json"]="{}"
    result=rt.launch(artifact,values)
    assert not result.get("ok"),result
    monkeypatch.setenv("TESSERA_OPT","/missing/compiler")
    close(fn(*changed),wanted)
