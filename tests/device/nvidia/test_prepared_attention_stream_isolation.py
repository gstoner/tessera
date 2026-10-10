"""Prepared attention completes without draining unrelated nonblocking work."""
import ctypes as ct
import json
import os
from pathlib import Path
import numpy as np
import pytest
from tessera.compiler.native_attention_jvp_runtime import PreparedAttentionJVP
from tessera.compiler.prepared_attention_vjp import PreparedAttentionVJP

ROOT=Path(__file__).resolve().parents[3]
pytestmark=pytest.mark.skipif(os.environ.get("TESSERA_NVIDIA_DEVICE_PROOF")!="1",
                              reason="owning RTX5070 proof required")
DELAY=b"""
.version 8.7
.target sm_120
.address_size 64
.visible .entry independent_delay(.param .u64 ticks) {
 .reg .u64 start, now, elapsed, bound;
 .reg .pred done;
 ld.param.u64 bound, [ticks];
 mov.u64 start, %clock64;
loop:
 mov.u64 now, %clock64;
 sub.u64 elapsed, now, start;
 setp.ge.u64 done, elapsed, bound;
 @!done bra loop;
 ret;
}
"""

@pytest.mark.parametrize("kind",["jvp","vjp"])
def test_prepared_attention_does_not_join_unrelated_stream(kind):
    directory=ROOT/"benchmarks/baselines"/("nvidia_public_attention_jvp_20261006" if kind=="jvp" else "nvidia_public_attention_vjp_20261006")/"artifacts"
    raw=json.loads((directory/"qkv_q_5_0.json").read_text())
    metadata=raw["native_jvp"]["steps"][0]["child_metadata"] if kind=="jvp" else raw["metadata"]
    owner=(PreparedAttentionJVP if kind=="jvp" else PreparedAttentionVJP)(metadata)
    driver=ct.CDLL("libcuda.so.1")
    signatures={
        "cuModuleLoadData":([ct.POINTER(ct.c_void_p),ct.c_void_p],ct.c_int),
        "cuModuleGetFunction":([ct.POINTER(ct.c_void_p),ct.c_void_p,ct.c_char_p],ct.c_int),
        "cuStreamCreate":([ct.POINTER(ct.c_void_p),ct.c_uint],ct.c_int),
        "cuEventCreate":([ct.POINTER(ct.c_void_p),ct.c_uint],ct.c_int),
        "cuEventRecord":([ct.c_void_p,ct.c_void_p],ct.c_int),
        "cuEventQuery":([ct.c_void_p],ct.c_int),
        "cuStreamSynchronize":([ct.c_void_p],ct.c_int),
        "cuStreamDestroy_v2":([ct.c_void_p],ct.c_int),
        "cuEventDestroy_v2":([ct.c_void_p],ct.c_int),
        "cuModuleUnload":([ct.c_void_p],ct.c_int),
        "cuLaunchKernel":([ct.c_void_p,*([ct.c_uint]*7),ct.c_void_p,
                           ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_void_p)],ct.c_int),
    }
    for name,(args,result) in signatures.items():
        fn=getattr(driver,name);fn.argtypes=args;fn.restype=result
    module,stream,event,function=(ct.c_void_p() for _ in range(4))
    rng=np.random.default_rng(1291)
    values=tuple(rng.normal(0,.1,shape).astype(np.float32) for shape in owner.shapes)
    try:
        expected=owner.invoke(metadata,values)
        image=ct.create_string_buffer(DELAY)
        assert driver.cuModuleLoadData(ct.byref(module),image)==0
        assert driver.cuModuleGetFunction(ct.byref(function),module,b"independent_delay")==0
        assert driver.cuStreamCreate(ct.byref(stream),1)==0
        assert driver.cuEventCreate(ct.byref(event),2)==0
        ticks=ct.c_uint64(2_000_000_000)
        args=(ct.c_void_p*1)(ct.addressof(ticks))
        assert driver.cuLaunchKernel(function,1,1,1,1,1,1,0,stream,args,None)==0
        assert driver.cuEventRecord(event,stream)==0
        actual=owner.invoke(metadata,values)
        for output,reference in zip(actual,expected,strict=True):
            np.testing.assert_array_equal(output,reference)
        assert driver.cuEventQuery(event)==600, "attention drained unrelated stream"
    finally:
        if stream.value:
            assert driver.cuStreamSynchronize(stream)==0
        if event.value:
            assert driver.cuEventDestroy_v2(event)==0
        if stream.value:
            assert driver.cuStreamDestroy_v2(stream)==0
        if module.value:
            assert driver.cuModuleUnload(module)==0
        owner.close()
