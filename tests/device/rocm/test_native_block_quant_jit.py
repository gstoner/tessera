"""Exact gfx1201 authored quantizer JIT and portable two-result replay."""
import json
import os
from pathlib import Path
import subprocess
import sys

import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tests.unit.test_rocm_block_quant_frontend import make_quantizer


def oracle(x):
    wide=x.astype(np.float32).reshape(-1,16)
    scale=np.maximum(np.max(np.abs(wide),axis=1,keepdims=True)/np.float32(448),np.finfo(np.float32).tiny)
    return ((wide/scale).astype(ml_dtypes.float8_e4m3fn).reshape(x.shape),
            scale.reshape(x.shape[0],x.shape[1]//16))


def check(actual,expected):
    codes,scales=actual
    assert codes.dtype==np.dtype(ml_dtypes.float8_e4m3fn) and scales.dtype==np.dtype(np.float32)
    np.testing.assert_array_equal(codes.view(np.uint8),expected[0].view(np.uint8))
    np.testing.assert_array_equal(scales.view(np.uint32),expected[1].view(np.uint32))
    assert not np.shares_memory(codes,scales)


@pytest.mark.parametrize("m,k",[(3,32),(19,12288),(256,12288)])
def test_authored_public_quantizer_jit_and_compiler_disabled_replay(m,k,monkeypatch):
    assert rt._rocm_live_arch()=="gfx1201","owning gfx1201 required"
    x=np.random.default_rng(1217).normal(size=(m,k)).astype(ml_dtypes.bfloat16)
    x.reshape(-1)[:16]=ml_dtypes.bfloat16(-0.0)
    expected=oracle(x);original=x.copy()
    fn=ts.jit(target="rocm_gfx1201",native_required=True)(make_quantizer(m,k))
    def forbidden(*a,**kw):raise AssertionError("public quantizer evaluated eager input arithmetic")
    with monkeypatch.context() as local:
        local.setattr(np,"max",forbidden)
        local.setattr(np,"maximum",forbidden)
        actual=fn(x)
    check(actual,expected)
    np.testing.assert_array_equal(x,original)
    assert fn.execution_kind=="native_gpu"
    artifact=fn.runtime_artifact()
    assert artifact.metadata["compiler_path"]=="rocm_block_quant_native_descriptor"
    held=tuple(a.copy() for a in actual)
    changed=np.ascontiguousarray(x[::-1]);wanted=oracle(changed)
    restored=rt.RuntimeArtifact.from_json(artifact.to_json())
    names=[b.name for b in restored.launch_descriptor.buffers]
    outputs=(np.empty_like(actual[0]),np.empty_like(actual[1]))
    groups=m*k//16
    with monkeypatch.context() as local:
        local.setenv("TESSERA_OPT","/missing/compiler")
        local.setenv("PATH","/missing/tools")
        from tessera.compiler import rocm_native,scheduled_matmul
        def forbidden(*a,**kw):raise AssertionError("portable quantizer invoked compiler or Graph executor")
        local.setattr(rocm_native,"_run_opt",forbidden)
        local.setattr(scheduled_matmul,"run_tessera_opt",forbidden)
        receipt=rt.launch(restored,{"buffers":dict(zip(names,(changed,*outputs),strict=True)),"scalars":{"Groups":groups}})
    assert receipt.get("ok") and receipt.get("execution_kind")=="native_gpu",receipt
    check(outputs,wanted)
    for a,b in zip(actual,held,strict=True):np.testing.assert_array_equal(a.view(np.uint8),b.view(np.uint8))


def test_fresh_process_quantizer_replay_without_compiler(tmp_path):
    assert rt._rocm_live_arch()=="gfx1201","owning gfx1201 required"
    x=np.arange(96,dtype=np.float32).reshape(3,32).astype(ml_dtypes.bfloat16)
    fn=ts.jit(target="rocm_gfx1201",native_required=True)(make_quantizer())
    expected=fn(x)
    (tmp_path/"artifact.json").write_text(fn.runtime_artifact().to_json())
    np.save(tmp_path/"x.npy",x.astype(np.float32))
    script='''import json,sys
from pathlib import Path
import numpy as np,ml_dtypes
from tessera import runtime as rt
root=Path(sys.argv[1]);artifact=rt.RuntimeArtifact.from_json((root/'artifact.json').read_text())
x=np.load(root/'x.npy').astype(ml_dtypes.bfloat16)
q=np.empty(x.shape,ml_dtypes.float8_e4m3fn);s=np.empty((3,2),np.float32)
names=[b.name for b in artifact.launch_descriptor.buffers]
r=rt.launch(artifact,{'buffers':dict(zip(names,(x,q,s),strict=True)),'scalars':{'Groups':6}})
assert r.get('ok') and r.get('execution_kind')=='native_gpu',r
np.savez(root/'result.npz',codes=q.view(np.uint8),scales=s)
'''
    process=subprocess.run([sys.executable,"-c",script,str(tmp_path)],capture_output=True,text=True,
        env={**os.environ,"PATH":"/missing/tools","TESSERA_OPT":"/missing/compiler"},check=False)
    assert process.returncode==0,process.stdout+process.stderr
    with np.load(tmp_path/"result.npz") as data:
        np.testing.assert_array_equal(data["codes"],expected[0].view(np.uint8))
        np.testing.assert_array_equal(data["scales"],expected[1])
