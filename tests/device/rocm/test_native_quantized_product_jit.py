"""Exact gfx1201 public quantizer/product route and compiler-free replay."""
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

pytestmark=[pytest.mark.hardware_rocm,pytest.mark.compiler_route]


from benchmarks.rocm.benchmark_public_quantized_product import make_product,inputs,oracle


@pytest.mark.parametrize("shape",[(3,17,32),(19,33,64),(256,128,12288)])
def test_public_jit_composed_quantization_matches_oracle_and_replays_without_compiler(shape,monkeypatch):
    assert rt._rocm_live_arch()=="gfx1201","owning gfx1201 required"
    m,n,k=shape;values=inputs(m,n,k);expected=oracle(values)
    fn=ts.jit(target="rocm_gfx1201",native_required=True)(make_product(m,n,k))
    # The frontend may inspect tensor metadata; it must not perform quantization.
    with monkeypatch.context() as patch:
        def forbidden(*args,**kwargs):raise AssertionError("eager host quantization executed")
        patch.setattr(np,"max",forbidden);patch.setattr(np,"maximum",forbidden)
        output=fn(*values)
    np.testing.assert_allclose(output,expected,rtol=.0002,atol=.0002)
    artifact=fn.runtime_artifact()
    assert artifact.metadata["compiler_path"]=="canonical_rocm_quantized_product_program"
    report=fn.compile_report()
    assert "canonical_rocm_quantized_product_program" in report.target_decision["rocm_gfx1201"]
    assert all(len(report.ir_hashes[layer])==64 for layer in ("graph_ir","schedule_ir","tile_ir","target_ir"))
    portable=rt.RuntimeArtifact.from_json(artifact.to_json())
    monkeypatch.setenv("TESSERA_OPT","/missing/compiler")
    monkeypatch.setenv("PATH","/missing/tools")
    changed=inputs(m,n,k,seed=901);expected2=oracle(changed)
    result=rt.launch(portable,changed)
    assert result.get("ok") and result.get("execution_kind")=="native_gpu",result
    assert len(result["component_receipts"])==2
    named=rt.launch(portable,dict(zip(("x","b","sb"),changed,strict=True)))
    assert named.get("ok"),named
    np.testing.assert_allclose(named["output"],expected2,rtol=.0002,atol=.0002)
    np.testing.assert_allclose(result["output"],expected2,rtol=.0002,atol=.0002)
    warm=fn(*changed)
    np.testing.assert_allclose(warm,expected2,rtol=.0002,atol=.0002)
    np.testing.assert_allclose(output,expected,rtol=.0002,atol=.0002)


def test_public_quantized_product_fresh_process_replay(tmp_path):
    assert rt._rocm_live_arch()=="gfx1201","owning gfx1201 required"
    values=inputs(3,17,32);fn=ts.jit(target="rocm_gfx1201",native_required=True)(make_product())
    expected=fn(*values)
    (tmp_path/"artifact.json").write_text(fn.runtime_artifact().to_json())
    np.savez(tmp_path/"roots.npz",x=values[0].astype(np.float32),b=values[1].view(np.uint8),sb=values[2])
    script='''import sys
from pathlib import Path
import numpy as np,ml_dtypes
from tessera import runtime as rt
root=Path(sys.argv[1]);p=np.load(root/'roots.npz')
artifact=rt.RuntimeArtifact.from_json((root/'artifact.json').read_text())
result=rt.launch(artifact,[p['x'].astype(ml_dtypes.bfloat16),p['b'].view(ml_dtypes.float8_e4m3fn),p['sb']])
assert result.get('ok') and result.get('execution_kind')=='native_gpu',result
np.save(root/'result.npy',result['output'])
'''
    result=subprocess.run([sys.executable,"-c",script,str(tmp_path)],capture_output=True,text=True,
        env={**os.environ,"PATH":"/missing/tools","TESSERA_OPT":"/missing/compiler"})
    assert result.returncode==0,result.stdout+result.stderr
    np.testing.assert_array_equal(np.load(tmp_path/"result.npy"),expected)


def test_inspection_artifact_cannot_rewrite_cached_public_program(monkeypatch):
    assert rt._rocm_live_arch()=="gfx1201","owning gfx1201 required"
    fn=ts.jit(target="rocm_gfx1201",native_required=True)(make_product())
    values=inputs(3,17,32);first=fn(*values)
    artifact=fn.runtime_artifact()
    artifact.metadata["arg_names"].reverse()
    bad=rt.launch(artifact,values)
    assert not bad.get("ok"),bad
    monkeypatch.setenv("TESSERA_OPT","/missing/compiler")
    changed=inputs(3,17,32,seed=419)
    result=fn(*changed)
    np.testing.assert_allclose(result,oracle(changed),rtol=.0002,atol=.0002)
    np.testing.assert_allclose(first,oracle(values),rtol=.0002,atol=.0002)
    assert fn.runtime_artifact().metadata["arg_names"]==["x","b","sb"]
