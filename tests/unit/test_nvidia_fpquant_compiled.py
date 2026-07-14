from __future__ import annotations
import os,shutil
import numpy as np
import pytest
from tessera import ops

def _live():
    if not (shutil.which("nvcc") or os.path.exists("/usr/local/cuda/bin/nvcc")):return False
    from tessera import runtime as rt
    return rt._nvidia_mma_runtime_available()

def _art(rt,op,fmt):
    return rt.RuntimeArtifact(metadata={"target":"nvidia_sm120","compiler_path":"nvidia_fpquant_compiled","executable":True,"execution_kind":"native_gpu","arg_names":["x"],"output_name":"o","ops":[{"op_name":op,"result":"o","operands":["x"],"kwargs":{"format":fmt}}]})

@pytest.mark.skipif(not _live(),reason="requires nvcc and NVIDIA GPU")
@pytest.mark.parametrize("op,fn,formats",[("tessera.quantize_fp8",ops.quantize_fp8,["e4m3","e5m2"]),("tessera.quantize_fp6",ops.quantize_fp6,["e2m3","e3m2"]),("tessera.quantize_fp4",ops.quantize_fp4,["e2m1"])])
def test_quantize_grid_matches_reference(op,fn,formats):
    from tessera import runtime as rt
    rng=np.random.default_rng(91+len(op));x=(rng.standard_normal((8,16))*3).astype(np.float32)
    for fmt in formats:
        out=rt.launch(_art(rt,op,fmt),(x,))["output"];ref,_=fn(x,format=fmt);np.testing.assert_allclose(out,np.asarray(ref,np.float32),rtol=2e-3,atol=2e-3)

@pytest.mark.skipif(not _live(),reason="requires nvcc and NVIDIA GPU")
def test_dequantize_is_native_shape_preserving_passthrough():
    from tessera import runtime as rt
    x=np.linspace(-3,3,33,dtype=np.float32);res=rt.launch(_art(rt,"tessera.dequantize_fp8","e4m3"),(x,));assert res["execution_kind"]=="native_gpu";np.testing.assert_array_equal(res["output"],x)
