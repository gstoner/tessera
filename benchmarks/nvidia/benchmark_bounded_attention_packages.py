"""Checked bounded saved-LSE package/private-frame timing characterization."""
import argparse,ctypes as ct,hashlib,json,os,subprocess,time
from pathlib import Path
from statistics import median
import numpy as np
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.nvidia_native import package_scheduled_checkpoint_pair
from benchmarks.nvidia.benchmark_jit_attention_vjp import download
from tests.unit.test_bounded_attention_packages import scheduled
from tests.device.nvidia.test_lse_checkpoint_native import _reference
from tests._support.nvidia import nvidia_cuda_host_ready

def record():
    if not nvidia_cuda_host_ready():raise RuntimeError("owning NVIDIA compiler/device required")
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi","--query-gpu=name,uuid,compute_cap","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or gpu.split(",")[-1].strip()!="12.0":raise RuntimeError("SM120 required")
    driver=ct.CDLL("libcuda.so.1");P=ct.c_void_p
    def bind(name,types):
        fn=getattr(driver,name);fn.argtypes=types;fn.restype=ct.c_int;return fn
    def check(x):
        if x:raise RuntimeError("CUDA status "+str(x))
    create=bind("cuEventCreate",[ct.POINTER(P),ct.c_uint])
    event=bind("cuEventRecord",[P,P]);elapsed=bind("cuEventElapsedTime",[ct.POINTER(ct.c_float),P,P])
    destroy=bind("cuEventDestroy_v2",[P])
    rows=[]
    for bias in (False,True):
      pair=package_scheduled_checkpoint_pair(scheduled(bias=bias),scheduled(True,bias),pipeline_name="tessera-nvidia-pipeline-sm120")
      for sq,sk in ((3,4),(9,11)):
        dims=(1,2,1,sq,sk,8,6);rng=np.random.default_rng(sq*5070+sk)
        q,k,v,seed=[rng.normal(0,.2,shape).astype("f4") for shape in
            ((1,2,sq,8),(1,1,sk,8),(1,1,sk,6),(1,2,sq,6))]
        score_bias=rng.normal(0,.1,(1,2,sq,sk)).astype("f4") if bias else None
        wanted,lse,grads=_reference(q,k,v,seed,score_bias,scale=.5)
        with NvidiaDeviceSession() as session:
          inputs=[session.upload(x) for x in (q,k,v)]
          rb=session.upload(score_bias) if bias else None;rs=session.upload(seed);check(session.synchronize())
          for asynchronous in (False,True):
            submission=[];completed=[];device=[];capture=[];error=0.
            for _ in range(5):
              begin=time.perf_counter()
              with pair.capture(*inputs,bias=rb,asynchronous=asynchronous) as frame:
                frame.synchronize();capture.append((time.perf_counter()-begin)*1000)
                frame.wait_on(session.stream);check(session.synchronize())
                got=download(session,frame.primal)
                np.testing.assert_allclose(got,wanted,atol=4e-5,rtol=4e-5)
                first,last=P(),P();check(create(ct.byref(first),0));check(create(ct.byref(last),0))
                try:
                    check(event(first,frame._stream));begin=time.perf_counter();submitted=0.
                    for _ in range(11):
                        start=time.perf_counter();actual=frame.backward(rs);submitted+=time.perf_counter()-start
                    check(event(last,frame._stream));frame.synchronize()
                    completed.append((time.perf_counter()-begin)*1000/11)
                    submission.append(submitted*1000/11)
                    ms=ct.c_float();check(elapsed(ct.byref(ms),first,last));device.append(ms.value/11)
                    frame.wait_on(session.stream);check(session.synchronize())
                    for got,want in zip(actual,grads,strict=True):
                        host=download(session,got);np.testing.assert_allclose(host,want,atol=4e-5,rtol=4e-5)
                        error=max(error,float(np.max(np.abs(host-want))))
                finally:check(destroy(first));check(destroy(last))
            rows.append(dict(shape=list(dims),bias=bias,asynchronous=asynchronous,
                contract_digest=pair.contract_digest,
                forward_image_sha256=hashlib.sha256(pair.forward.image.payload).hexdigest(),
                backward_image_sha256=hashlib.sha256(pair.backward.image.payload).hexdigest(),
                completed_capture_samples_ms=capture,completed_capture_median_ms=median(capture),
                backward_submission_samples_ms=submission,backward_submission_median_ms=median(submission),
                completed_backward_host_samples_ms=completed,completed_backward_host_median_ms=median(completed),
                native_owner_event_samples_ms=device,native_owner_event_median_ms=median(device),max_abs_error=error))
    for bias in (False,True):
        arms=[row for row in rows if row["bias"]==bias]
        assert len({r["forward_image_sha256"] for r in arms})==1
        assert len({r["backward_image_sha256"] for r in arms})==1
    root=Path(__file__).resolve().parents[2]
    sources=("benchmarks/nvidia/benchmark_bounded_attention_packages.py","tests/device/nvidia/test_bounded_attention_packages.py","tests/unit/test_bounded_attention_packages.py","python/tessera/compiler/attention_shape_contract.py","python/tessera/compiler/scheduled_checkpoint.py","python/tessera/compiler/nvidia_native.py","python/tessera/compiler/resident_attention.py","python/tessera/runtime.py","python/tessera/compiler/compact_attention_contract.py","python/tessera/compiler/lse_cotangent_contract.py","src/compiler/programming_model/lib/NativeCheckpoint.h","src/compiler/tile_opt_fa4/lib/Dialect/Attn/AttnOps.cpp","tests/device/nvidia/test_lse_checkpoint_native.py")
    return dict(schema="tessera.sm120.bounded_attention_packages.v1",gpu=gpu,bounds=[1,2,1,9,11,8,6],rows=rows,
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        nvidia_compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_OPT"]).read_bytes()).hexdigest(),
        runtime_library_sha256=hashlib.sha256((root/".build-sm120-w1-1/src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/libtessera_nvidia_ptx_launch.so").read_bytes()).hexdigest(),
        source_sha256={s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources},
        scope="Checked native checkpoint packages and private residual API. Public JIT/automatic AD export still pending. Capture host includes allocation, module load, snapshots, forward and synchronization; backward event includes allocations, seed copies, kernels, frees and driver gaps. No isolated kernel or counterbalanced speedup claim.")
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    packet=record();args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(packet,indent=2)+"\n")
    print("verified",len(packet["rows"]),"bounded checked-package timing arms")
