"""sm_120 exact DSA sparse-attention execution proof."""
import os, shutil
import numpy as np
import pytest

def _rt():
    if not (shutil.which("nvcc") or os.path.exists("/usr/local/cuda/bin/nvcc")): pytest.skip("nvcc unavailable")
    from tessera import runtime as rt
    if not rt._nvidia_mma_runtime_available(): pytest.skip("CUDA unavailable")
    return rt

@pytest.mark.slow
def test_live_nvidia_dsa_sparse_attention():
    from tessera import runtime as rt
    from tessera.stdlib.attention import dsa_block_sparse_attention
    rng=np.random.default_rng(971); q=rng.standard_normal((1,2,5,4),dtype=np.float32); k=rng.standard_normal((1,1,7,4),dtype=np.float32); v=rng.standard_normal((1,1,7,3),dtype=np.float32)
    kw={"block_size":2,"top_k_blocks":2,"causal":True,"scale":.5}
    art=rt.RuntimeArtifact(metadata={"target":"nvidia_sm120","compiler_path":"nvidia_sparse_attn_compiled","executable":True,"execution_kind":"native_gpu","arg_names":["q","k","v"],"output_name":"o","ops":[{"op_name":"tessera.dsa_block_sparse_attention","result":"o","operands":["q","k","v"],"kwargs":kw}]})
    got=_rt().launch(art,(q,k,v)); assert got["ok"],got.get("reason")
    np.testing.assert_allclose(got["output"],dsa_block_sparse_attention(q,k,v,**kw),atol=8e-5,rtol=0)
