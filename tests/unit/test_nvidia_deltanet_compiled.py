from __future__ import annotations
import os,shutil
import numpy as np
import pytest

def _live():
    if not (shutil.which("nvcc") or os.path.exists("/usr/local/cuda/bin/nvcc")):return False
    from tessera import runtime as rt
    return rt._nvidia_mma_runtime_available()

def _ref(Q,K,V,gate=None,beta=None,decay=None,erase=False,modified=False):
    B,H,S,D=Q.shape;DV=V.shape[-1];st=np.zeros((B,H,D,DV),np.float64);o=np.zeros((B,H,S,DV),np.float64)
    Q=Q.astype(np.float64);K=K.astype(np.float64);V=V.astype(np.float64)
    for t in range(S):
        kt=K[:,:,t];target=V[:,:,t]
        if erase:target=target-(decay[:,:,t,None] if decay is not None else 1)*np.einsum("bhd,bhde->bhe",kt,st)
        if decay is not None:st=decay[:,:,t,None,None]*st
        delta=np.einsum("bhd,bhe->bhde",kt,target)
        if modified:delta/=1+np.linalg.norm(delta,axis=(-2,-1),keepdims=True)
        st+=(beta[:,:,t,None,None] if beta is not None else 1)*delta
        o[:,:,t]=np.einsum("bhd,bhde->bhe",Q[:,:,t],st)
    if gate is not None:o*=1/(1+np.exp(-gate))
    return o.astype(np.float32)

def _artifact(rt,name,vals,kw):
    names=[f"x{i}" for i in range(len(vals))]
    return rt.RuntimeArtifact(metadata={"target":"nvidia_sm120","compiler_path":"nvidia_deltanet_compiled","executable":True,"execution_kind":"native_gpu","arg_names":names,"output_name":"o","ops":[{"op_name":name,"result":"o","operands":names,"kwargs":kw}]}),tuple(vals)

@pytest.mark.skipif(not _live(),reason="requires nvcc and NVIDIA GPU")
@pytest.mark.parametrize("variant",["plain","kimi","gated","erase","modified"])
def test_deltanet_variants_match_oracle(variant):
    from tessera import runtime as rt
    rng=np.random.default_rng(70+len(variant));shape=(1,2,9,16);q=(rng.standard_normal(shape)*.4).astype(np.float32);k=rng.standard_normal(shape);k=(k/np.maximum(np.linalg.norm(k,axis=-1,keepdims=True),1e-6)).astype(np.float32);v=(rng.standard_normal(shape)*.4).astype(np.float32)
    vals=[q,k,v];kw={"causal":True};opts={}
    name=("tessera.modified_delta_attention" if variant=="modified" else
          "tessera.kimi_delta_attention" if variant=="kimi" else
          "tessera.gated_deltanet")
    if variant=="gated":
        gate=rng.standard_normal(shape).astype(np.float32);beta=rng.uniform(.2,.9,shape[:3]).astype(np.float32);decay=rng.uniform(.85,.99,shape[:3]).astype(np.float32);vals += [gate,beta,decay];kw.update(has_gate=True,has_beta=True,has_decay=True);opts.update(gate=gate,beta=beta,decay=decay)
    if variant=="erase":
        beta=rng.uniform(.2,.9,shape[:3]).astype(np.float32);decay=rng.uniform(.85,.99,shape[:3]).astype(np.float32);vals += [beta,decay];kw.update(erase=True,has_beta=True,has_decay=True);opts.update(beta=beta,decay=decay,erase=True)
    if variant=="modified":opts["modified"]=True
    art,args=_artifact(rt,name,vals,kw);res=rt.launch(art,args);assert res["ok"] is True,res.get("reason")
    np.testing.assert_allclose(res["output"],_ref(q,k,v,**opts),rtol=3e-5,atol=3e-5)

def test_deltanet_rejects_noncausal_without_cuda():
    from tessera import runtime as rt
    z=np.zeros((1,1,2,4),np.float32);art,args=_artifact(rt,"tessera.gated_deltanet",[z,z,z],{"causal":False})
    with pytest.raises(ValueError,match="causal-only"):rt._execute_nvidia_deltanet_compiled(art,args)
