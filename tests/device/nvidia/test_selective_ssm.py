from __future__ import annotations
import numpy as np
import pytest
import tessera
from tests._support.nvidia import nvidia_mma_runtime_available

def _artifact(rt,names,extras):
    return rt.RuntimeArtifact(metadata={"target":"nvidia_sm120","compiler_path":"nvidia_ssm_compiled","executable":True,"execution_kind":"native_gpu","arg_names":names,"output_name":"o","ops":[{"op_name":"tessera.selective_ssm","result":"o","operands":names,"kwargs":{"extras":extras}}]})

@pytest.mark.skipif(not nvidia_mma_runtime_available(),reason="requires nvcc and NVIDIA GPU")
@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("with_extras",[False,True])
def test_selective_ssm_matches_reference(with_extras):
    from tessera import runtime as rt
    rng=np.random.default_rng(31+with_extras);bs,s,d,n=2,7,4,10
    x=rng.standard_normal((bs,s,d)).astype(np.float32);A=(-rng.random((d,n))).astype(np.float32)
    B=rng.standard_normal((bs,s,n)).astype(np.float32);C=rng.standard_normal((bs,s,n)).astype(np.float32);dt=(rng.random((bs,s,d))*.5).astype(np.float32)
    vals=[x,A,B,C,dt];extras=[];kw={}
    if with_extras:
        gate=rng.standard_normal(x.shape).astype(np.float32);state=rng.standard_normal((bs,d,n)).astype(np.float32);vals += [gate,state];extras=["gate","state"];kw={"gate":gate,"state":state}
    names=[f"x{i}" for i in range(len(vals))];res=rt.launch(_artifact(rt,names,extras),tuple(vals))
    assert res["ok"] is True,res.get("reason")
    ref=np.asarray(tessera.ops.selective_ssm(x,A,B,C,dt,**kw))
    np.testing.assert_allclose(res["output"],ref,rtol=2e-5,atol=2e-5)
