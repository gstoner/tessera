"""SM120 saved-LSE synchronous/async native-owner timing characterization."""
import argparse,ctypes as ct,hashlib,json,os,subprocess,time
from pathlib import Path
from statistics import median
import numpy as np
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_jit_multiresult_attention_vjp import function,values,reference
from benchmarks.nvidia.benchmark_jit_attention_vjp import download

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if not nvidia_cuda_host_ready():raise RuntimeError("exact SM120 owning host required")
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi","--query-gpu=name,uuid,driver_version,compute_cap","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or gpu.split(",")[-1].strip()!="12.0":raise RuntimeError("requires selected SM120")
    driver=ct.CDLL("libcuda.so.1")
    def bind(name,types):
        fn=getattr(driver,name);fn.argtypes=types;fn.restype=ct.c_int;return fn
    P=ct.c_void_p
    create=bind("cuEventCreate",[ct.POINTER(P),ct.c_uint]);record=bind("cuEventRecord",[P,P])
    elapsed=bind("cuEventElapsedTime",[ct.POINTER(ct.c_float),P,P]);destroy=bind("cuEventDestroy_v2",[P])
    def check(status):
        if status:raise RuntimeError("CUDA event status "+str(status))
    rows=[]
    for shape in ((1,2,1,3,5,4,3),(2,4,2,16,19,8,6)):
      for bias in (False,True):
        inputs,seeds=values(shape,bias,"mixed");wanted,gradients=reference(inputs,seeds,True)
        program=function(bias,True).compile_native_attention_vjp(*inputs,compiler=os.environ["TESSERA_OPT"],compact_gradients=True)
        with NvidiaDeviceSession() as session:
          resident=[session.upload(v) for v in inputs];seed_buffers=tuple(session.upload(v) for v in seeds);session.synchronize()
          for asynchronous in (False,True):
            submission=[];completed=[];device=[];error=0.
            with program.capture(*resident,asynchronous=asynchronous) as frame:
              frame.wait_on(session.stream);session.synchronize()
              for got,want in zip(frame.primal,wanted,strict=True):np.testing.assert_allclose(download(session,got),want,rtol=4e-5,atol=4e-5)
              for _ in range(5):
                first,last=P(),P();check(create(ct.byref(first),0));check(create(ct.byref(last),0))
                try:
                  enqueue=0.;start=time.perf_counter()
                  check(record(first,frame._frame._stream))
                  for _ in range(11):
                    begin=time.perf_counter();actual=frame.backward(seed_buffers)
                    enqueue+=time.perf_counter()-begin
                  check(record(last,frame._frame._stream));frame.synchronize()
                  completed.append((time.perf_counter()-start)*1000/11)
                  submission.append(enqueue*1000/11)
                  ms=ct.c_float();check(elapsed(ct.byref(ms),first,last));device.append(ms.value/11)
                  frame.wait_on(session.stream);session.synchronize()
                  for got,role in zip(actual,program.active,strict=True):
                    result=download(session,got);want=gradients[role]
                    np.testing.assert_allclose(result,want,rtol=4e-5,atol=4e-5)
                    error=max(error,float(np.max(np.abs(result-want))))
                finally:check(destroy(first));check(destroy(last))
            row=dict(shape=list(shape),bias=bias,asynchronous=asynchronous,compact=True,
                gradient_roles=list(program.active),program_digest=program.program_digest,
                forward_image_sha256=hashlib.sha256(program.pair.forward.image.payload).hexdigest(),
                backward_image_sha256=hashlib.sha256(program.pair.backward.image.payload).hexdigest(),
                submission_samples_ms=submission,submission_median_ms=median(submission),
                completed_host_samples_ms=completed,completed_host_median_ms=median(completed),
                native_owned_stream_event_samples_ms=device,native_owned_stream_event_median_ms=median(device),
                max_abs_error=error,correctness="primal_before_and_gradients_after_each_window")
            rows.append(row);print(json.dumps(row),flush=True)
    root=Path(__file__).resolve().parents[2]
    sources=("benchmarks/nvidia/benchmark_attention_async_owner.py","tests/device/nvidia/test_attention_async_owner.py","tests/device/nvidia/test_jit_multiresult_attention_vjp.py","tests/device/nvidia/test_lse_cotangent_native.py","python/tessera/compiler/resident_attention.py","python/tessera/compiler/native_attention_program.py","python/tessera/compiler/nvidia_native.py","python/tessera/compiler/jit.py")
    packet=dict(schema="tessera.sm120.attention_async_owner.v1",gpu=gpu,architecture="sm_120",
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        sources={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},rows=rows,
        timing="Submission is not completed execution. Completed host includes waiting; private-stream event includes native allocation, snapshot copies, kernel, frees and host enqueue gaps. No isolated kernel speedup or automatic promotion claim.")
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(packet,indent=2)+"\n")
if __name__=="__main__":main()
