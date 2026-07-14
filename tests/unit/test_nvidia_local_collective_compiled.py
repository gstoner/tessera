from __future__ import annotations
import os,shutil
import numpy as np
import pytest

def _live():
    if not (shutil.which("nvcc") or os.path.exists("/usr/local/cuda/bin/nvcc")):return False
    from tessera import runtime as rt
    return rt._nvidia_mma_runtime_available()

def _art(rt,op,world=1):
    return rt.RuntimeArtifact(metadata={"target":"nvidia_sm120","compiler_path":"nvidia_local_collective_compiled","executable":True,"execution_kind":"native_gpu","arg_names":["x"],"output_name":"o","ops":[{"op_name":op,"result":"o","operands":["x"],"kwargs":{"world_size":world}}]})

@pytest.mark.skipif(not _live(),reason="requires nvcc and NVIDIA GPU")
@pytest.mark.parametrize("op",["tessera.all_reduce","tessera.reduce_scatter","tessera.all_gather","tessera.all_to_all"])
def test_single_device_collective_identity_runs_on_cuda(op):
    from tessera import runtime as rt
    x=np.arange(60,dtype=np.float32).reshape(3,4,5);res=rt.launch(_art(rt,op),(x,));assert res["execution_kind"]=="native_gpu";np.testing.assert_array_equal(res["output"],x)

def test_multi_rank_collective_is_explicitly_deferred():
    from tessera import runtime as rt
    with pytest.raises(ValueError,match="requires world_size=1"):
        rt._execute_nvidia_local_collective(_art(rt,"tessera.all_reduce",2),(np.ones(4,np.float32),))
