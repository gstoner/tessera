"""Native bounded checkpoint image proof; public JIT ABI integration is pending."""
import argparse,ctypes as ct,hashlib,json,os,re,subprocess,time
from pathlib import Path
from statistics import median
import numpy as np
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.nvidia_native import _compile_tile_ir
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.unit.test_bounded_attention_checkpoint import graph,lower
from tests.device.nvidia.test_lse_checkpoint_native import _reference

def record():
    if not nvidia_cuda_host_ready():raise RuntimeError("owning CUDA host required")
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi","--query-gpu=name,uuid,compute_cap","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or gpu.split(",")[-1].strip()!="12.0":raise RuntimeError("exact SM120 device required")
    driver=ct.CDLL("libcuda.so.1");P=ct.c_void_p
    def bind(name,args):
        fn=getattr(driver,name);fn.argtypes=args;fn.restype=ct.c_int;return fn
    def check(code):
        if code:raise RuntimeError("CUDA status "+str(code))
    load=bind("cuModuleLoadData",[ct.POINTER(P),P])
    function=bind("cuModuleGetFunction",[ct.POINTER(P),P,ct.c_char_p])
    unload=bind("cuModuleUnload",[P])
    launch=bind("cuLaunchKernel",[P,*([ct.c_uint]*7),P,ct.POINTER(P),P])
    create=bind("cuEventCreate",[ct.POINTER(P),ct.c_uint])
    event=bind("cuEventRecord",[P,P]);elapsed=bind("cuEventElapsedTime",[ct.POINTER(ct.c_float),P,P])
    destroy=bind("cuEventDestroy_v2",[P])
    rows=[];images=[];modules=[]
    with NvidiaDeviceSession() as session:
      try:
        for backward in (False,True):
            source=graph(backward)
            schedule=lower(source,"--tessera-graph-to-schedule")
            tile=lower(schedule,"--tessera-schedule-to-tile")
            entry=re.search(r"llvm.func @([\w]+)",tile)[1]
            target,ptx,*_= _compile_tile_ir(tile,entry)
            payload=ct.create_string_buffer(ptx.encode())
            module=P();kernel=P();check(load(ct.byref(module),ct.cast(payload,P)));modules.append(module)
            check(function(ct.byref(kernel),module,entry.encode()))
            images.append(dict(backward=backward,entry=entry,sha256=hashlib.sha256(ptx.encode()).hexdigest(),graph_sha256=hashlib.sha256(source.encode()).hexdigest(),schedule_sha256=hashlib.sha256(schedule.encode()).hexdigest(),tile_sha256=hashlib.sha256(tile.encode()).hexdigest(),target_sha256=hashlib.sha256(target.encode()).hexdigest()))
            for sq,sk in ((1,1),(3,4),(9,11),(9,1),(1,11),(3,4)):
                dims=(1,2,1,sq,sk,8,6);rng=np.random.default_rng(sq*100+sk+len(rows))
                q=rng.normal(0,.2,(1,2,sq,8)).astype("f4")
                k=rng.normal(0,.2,(1,1,sk,8)).astype("f4")
                v=rng.normal(0,.2,(1,1,sk,6)).astype("f4")
                seed=rng.normal(0,.2,(1,2,sq,6)).astype("f4")
                expected_o,expected_lse,grad=_reference(q,k,v,seed,scale=.5)
                if backward:
                    operands=[seed,q,k,v,expected_o.astype("f4"),expected_lse.astype("f4")]
                    wants=grad
                else:operands=[q,k,v];wants=(expected_o,expected_lse)
                inputs=[session.upload(x) for x in operands]
                outputs=[session.empty(w.shape,np.float32) for w in wants]
                holders=[P(x.ptr) for x in inputs+outputs]+[ct.c_int64(x) for x in dims]
                args=(P*len(holders))(*(ct.cast(ct.byref(x),P) for x in holders))
                total=sum(w.size for w in wants) if backward else wants[0].size
                def submit():
                    check(launch(kernel,(total+127)//128,1,1,128,1,1,0,P(session.stream),args,None))
                check(session.synchronize());submit();check(session.synchronize())
                errors=[]
                def validate():
                    for got,want in zip(outputs,wants,strict=True):
                        actual=session.download(got)
                        np.testing.assert_allclose(actual,want,rtol=4e-5,atol=4e-5)
                        errors.append(float(np.max(np.abs(actual-want))))
                validate()
                device=[];wall=[]
                first,last=P(),P();check(create(ct.byref(first),0));check(create(ct.byref(last),0))
                try:
                    for _ in range(5):
                        check(event(first,P(session.stream)));start=time.perf_counter()
                        for _ in range(101):submit()
                        check(event(last,P(session.stream)));check(session.synchronize())
                        wall.append((time.perf_counter()-start)*1000/101)
                        ms=ct.c_float();check(elapsed(ct.byref(ms),first,last));device.append(ms.value/101)
                        validate()
                finally:check(destroy(first));check(destroy(last))
                rows.append(dict(backward=backward,shape=list(dims),image_sha256=images[-1]["sha256"],max_abs_error=max(errors),device_window_samples_ms=device,device_window_median_ms=median(device),completed_launch_host_samples_ms=wall,completed_launch_host_median_ms=median(wall)))
      finally:
        check(session.synchronize())
        for module in modules:check(unload(module))
    root=Path(__file__).resolve().parents[2]
    sources=["benchmarks/nvidia/benchmark_bounded_attention_images.py","tests/unit/test_bounded_attention_checkpoint.py","src/compiler/programming_model/lib/NativeCheckpoint.h","src/compiler/tile_opt_fa4/lib/Dialect/Attn/AttnOps.cpp","tests/device/nvidia/test_lse_checkpoint_native.py"]
    return dict(schema="tessera.sm120.bounded_checkpoint_native_images.v1",gpu=gpu,architecture="sm_120",bounds=[1,2,1,9,11,8,6],images=images,rows=rows,compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),source_sha256={s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources},scope="Native checkpoint Graph/Schedule/Tile image only. Raw driver test harness; public JIT, checked package ABI and paired residual ownership integration remain pending. Event windows include driver submission gaps, not isolated kernel timing. No speedup claim.")
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    packet=record();args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(packet,indent=2)+"\n")
    print("verified",len(packet["rows"]),"same-image sequence cases")
