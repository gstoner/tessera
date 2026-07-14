from __future__ import annotations
import os,shutil
import numpy as np
import pytest
from tessera import optim

def _live():
    if not (shutil.which("nvcc") or os.path.exists("/usr/local/cuda/bin/nvcc")):return False
    from tessera import runtime as rt
    return rt._nvidia_mma_runtime_available()

def _art(rt,op,vals,extras,kw):
    names=[f"a{i}" for i in range(len(vals))];kw=dict(kw,extras=extras)
    return rt.RuntimeArtifact(metadata={"target":"nvidia_sm120","compiler_path":"nvidia_optimizer_compiled","executable":True,"execution_kind":"native_gpu","arg_names":names,"output_name":"o","ops":[{"op_name":op,"result":"o","operands":names,"kwargs":kw}]})

@pytest.mark.skipif(not _live(),reason="requires nvcc and NVIDIA GPU")
def test_sgd_momentum_nesterov():
    from tessera import runtime as rt
    rng=np.random.default_rng(5);p=rng.standard_normal((3,7)).astype(np.float32);g=rng.standard_normal(p.shape).astype(np.float32);z=np.zeros_like(p)
    s=rt.launch(_art(rt,"tessera.sgd",[p,g],[],{"lr":.1}),(p,g))["output"];np.testing.assert_allclose(s,optim.sgd(p,g,lr=.1),atol=1e-6)
    for name,fn in (("momentum",optim.momentum),("nesterov",optim.nesterov)):
        out=rt.launch(_art(rt,"tessera."+name,[p,g,z],["v"],{"lr":.01,"momentum":.9}),(p,g,z))["output"];rp,st=fn(p,g,None,lr=.01,momentum=.9);np.testing.assert_allclose(out[0],rp,atol=1e-6);np.testing.assert_allclose(out[1],st["velocity"],atol=1e-6)

@pytest.mark.skipif(not _live(),reason="requires nvcc and NVIDIA GPU")
@pytest.mark.parametrize("name",["adam","adamw","lion"])
def test_adaptive_optimizer_contracts(name):
    from tessera import runtime as rt
    rng=np.random.default_rng(13+len(name));p=rng.standard_normal((4,6)).astype(np.float32);g=rng.standard_normal(p.shape).astype(np.float32);z=np.zeros_like(p)
    if name=="lion":vals=[p,g,z];extras=["m"];kw={"lr":1e-4,"beta1":.9,"beta2":.99,"weight_decay":.01};ref=optim.lion(p,g,None,**kw)
    else:vals=[p,g,z,z];extras=["m","v"];kw={"lr":1e-3,"beta1":.9,"beta2":.999,"eps":1e-8};
    if name=="adamw":kw["weight_decay"]=.01
    if name!="lion":ref=getattr(optim,name)(p,g,None,**kw)
    out=rt.launch(_art(rt,"tessera."+name,vals,extras,kw),tuple(vals))["output"]
    np.testing.assert_allclose(out[0],ref[0],atol=1e-5)
