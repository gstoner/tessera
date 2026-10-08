"""Native movement HIP graph replay and separate resident/host timing domains."""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler import rocm_native
from benchmarks.rocm.benchmark_native_movement import device_identity
from benchmarks.rocm.benchmark_rocm_e2e_movement import _ResidentDescriptor
from tests.unit.test_public_movement_frontend import paged, dispatched, inputs

def paged_full(pages,table):
    return ts.ops.kv_cache_read(pages,0,1024,page_table=table)

class CapturedMovement:
    """Retain one module/allocation set until captured work has completed."""
    def __init__(self,hip,resident,repeats):
        self.hip,self.resident,self.repeats=hip,resident,repeats
        self.stream,self.graph,self.executable=ct.c_void_p(),ct.c_void_p(),ct.c_void_p()
        self.events=[]
        self.closed=False
        def bind(name,args):
            fn=getattr(hip,name);fn.argtypes=args;fn.restype=ct.c_int
            return fn
        P=ct.c_void_p
        self.create=bind("hipStreamCreateWithFlags",[ct.POINTER(P),ct.c_uint])
        self.begin=bind("hipStreamBeginCapture",[P,ct.c_int])
        self.end=bind("hipStreamEndCapture",[P,ct.POINTER(P)])
        self.instantiate=bind("hipGraphInstantiateWithFlags",[ct.POINTER(P),P,ct.c_ulonglong])
        self.nodes=bind("hipGraphGetNodes",[P,P,ct.POINTER(ct.c_size_t)])
        self.submit=bind("hipGraphLaunch",[P,P])
        self.synchronize=bind("hipStreamSynchronize",[P])
        self.destroy_exec=bind("hipGraphExecDestroy",[P])
        self.destroy_graph=bind("hipGraphDestroy",[P])
        self.destroy_stream=bind("hipStreamDestroy",[P])
        try:
            self.check(self.create(ct.byref(self.stream),1))
            self.check(self.begin(self.stream,0))
            try:
                for _ in range(repeats):resident.launch(self.stream)
            finally:
                self.check(self.end(self.stream,ct.byref(self.graph)))
            count=ct.c_size_t()
            self.check(self.nodes(self.graph,None,ct.byref(count)))
            if count.value!=repeats:raise RuntimeError("captured movement kernel node count differs")
            self.node_count=count.value
            self.check(self.instantiate(ct.byref(self.executable),self.graph,0))
            for _ in range(2):
                event=P();self.check(hip.hipEventCreate(ct.byref(event)));self.events.append(event)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(status):
        if status:raise RuntimeError(f"HIP graph status {status}")

    def measure(self,captured):
        if self.closed:raise ValueError("captured movement is closed")
        start,stop=self.events
        self.check(self.hip.hipEventRecord(start,self.stream))
        wall=time.perf_counter_ns()
        if captured:self.check(self.submit(self.executable,self.stream))
        else:
            for _ in range(self.repeats):self.resident.launch(self.stream)
        self.check(self.hip.hipEventRecord(stop,self.stream))
        self.check(self.synchronize(self.stream))
        wall_ms=(time.perf_counter_ns()-wall)/1e6/self.repeats
        elapsed=ct.c_float()
        self.check(self.hip.hipEventElapsedTime(ct.byref(elapsed),start,stop))
        if elapsed.value<=0:raise RuntimeError("HIP graph event duration is not positive")
        return elapsed.value/self.repeats,wall_ms

    def close(self):
        if self.closed:return
        if self.stream.value:self.check(self.synchronize(self.stream))
        for event in reversed(self.events):self.check(self.hip.hipEventDestroy(event))
        self.events.clear()
        if self.executable.value:self.check(self.destroy_exec(self.executable));self.executable=ct.c_void_p()
        if self.graph.value:self.check(self.destroy_graph(self.graph));self.graph=ct.c_void_p()
        if self.stream.value:self.check(self.destroy_stream(self.stream));self.stream=ct.c_void_p()
        self.closed=True

def record(hip,arch,family,large,repeats,directory):
    if family=="full":
        rng=np.random.default_rng(61006)
        pages=rng.normal(size=(32,16,8,128)).astype(np.float32)
        table=rng.integers(0,32,64,dtype=np.int32)
        args=(pages,table);expected=pages[table].reshape(-1,8,128)
        source=paged_full
    else:
        args,expected=inputs(family,large)
        source=paged if family=="paged" else dispatched
    # Movement must preserve nonfinite payload bits, not only numerical equality.
    values=(args[1][args[0][0]] if family=="dispatched" else
            args[0][args[1][0],0 if family=="full" else 1])
    values.view(np.uint32).flat[:4]=[0x7fc12345,0x80000000,0x7f800000,0xff800000]
    if family=="full":expected=args[0][args[1]].reshape(-1,8,128)
    elif family=="paged":expected=args[0][args[1]].reshape(-1,*args[0].shape[2:])[1:6]
    else:expected=args[1][args[0]]
    fn=ts.jit(target="rocm_"+arch,native_required=True)(source)
    start=time.perf_counter_ns();actual=fn(*args);first_ms=(time.perf_counter_ns()-start)/1e6
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
    assert fn.execution_kind=="native_gpu"
    module,_=fn._trace_frontend_capture(args,{})
    artifact=fn.runtime_artifact()
    contract=rocm_native._moe_dispatch_contract(module) if family=="dispatched" else rocm_native._paged_kv_contract(module)
    named=dict(zip((a.name for a in module.functions[0].args),args,strict=True))
    operands=tuple(named[n] for n in contract[:2])
    package=SimpleNamespace(image=artifact.native_image,descriptor=artifact.launch_descriptor)
    resident=_ResidentDescriptor(hip,package,operands,np.empty_like(expected),contract[3],expected.size)
    owner=None
    label=family+("_large" if large else "_small")
    try:
        start=time.perf_counter_ns();owner=CapturedMovement(hip,resident,repeats)
        prepare_ms=(time.perf_counter_ns()-start)/1e6
        samples={n:[] for n in ("loop","captured")}
        for trial in range(9):
            for name in (("loop","captured") if trial%2==0 else ("captured","loop")):
                event,wall=owner.measure(name=="captured")
                np.testing.assert_array_equal(resident.read().view(np.uint32),expected.view(np.uint32))
                samples[name].append(dict(device_event_ms_per_kernel=event,
                    resident_submission_completion_wall_ms_per_kernel=wall))
        # A captured program must read current resident values, not bake
        # host arrays or index contents into its graph.
        changed=operands[0].view(np.uint32).__xor__(np.uint32(0x80000000)).view(np.float32)
        changed_indices=operands[1][::-1].copy()
        for device,array in zip(resident.devices[:2],(changed,changed_indices),strict=True):
            owner.check(hip.hipMemcpy(device,array.ctypes.data_as(ct.c_void_p),array.nbytes,1))
        if family=="dispatched":
            changed_expected=changed[changed_indices]
        else:
            start,tokens=contract[3][-2:]
            changed_expected=changed[changed_indices].reshape(-1,*changed.shape[2:])[start:start+tokens]
        owner.measure(True)
        np.testing.assert_array_equal(resident.read().view(np.uint32),changed_expected.view(np.uint32))
        for device,array in zip(resident.devices[:2],operands,strict=True):
            owner.check(hip.hipMemcpy(device,array.ctypes.data_as(ct.c_void_p),array.nbytes,1))
        owner.measure(True)
        np.testing.assert_array_equal(resident.read().view(np.uint32),expected.view(np.uint32))
        bundle=fn.compile_bundle
        stages=(bundle.graph,bundle.schedule,bundle.tile,bundle.target_ir,bundle.backend)
        if bundle.schedule.producer!="tessera-opt.tessera-graph-to-schedule":
            raise RuntimeError("captured movement requires native Schedule ancestry")
        for before,after in zip(stages[:-1],stages[1:],strict=True):
            if after.input_digest!=before.output_digest:
                raise RuntimeError("captured movement adjacent compiler lineage differs")
        lineage={stage.level:dict(producer=stage.producer,input_digest=stage.input_digest,
                                 output_digest=stage.output_digest) for stage in stages}
        for name,stage in (("graph",fn.compile_bundle.graph),("schedule",fn.compile_bundle.schedule),
            ("tile",fn.compile_bundle.tile),("target",fn.compile_bundle.target_ir),("backend",fn.compile_bundle.backend)):
            (directory/(label+"."+name+".mlir")).write_text(stage.text)
        (directory/(label+".hsaco")).write_bytes(artifact.native_image.payload)
        medians={name:statistics.median(x["device_event_ms_per_kernel"] for x in rows)
                 for name,rows in samples.items()}
        return dict(family=family,large=large,dims=list(contract[3]),
            image_digest=artifact.native_image.image_digest,entry=artifact.launch_descriptor.entry_symbol,
            compiler_lineage=lineage,
            first_compile_host_call_ms=first_ms,graph_prepare_ms=prepare_ms,
            graph_kernel_nodes=owner.node_count,launch_block_threads=256,
            correctness="bit_exact_before_and_after_each_arm",resident_rebind_bit_exact=True,samples=samples,event_medians_ms=medians,
            captured_over_loop_event_ratio=medians["captured"]/medians["loop"])
    finally:
        if owner is not None:owner.close()
        resident.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--architecture",choices=("gfx1151","gfx1201"),required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--repeats",type=int,default=256)
    args=parser.parse_args()
    if not 2<=args.repeats<=4096:raise ValueError("capture repeats must be in [2,4096]")
    hip=rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):raise RuntimeError("usable owning HIP device required")
    identity=device_identity(hip,args.architecture)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    families=("paged","full","dispatched") if args.architecture=="gfx1151" else ("paged","full")
    rows=[record(hip,args.architecture,f,large,args.repeats,args.output.parent)
        for f in families for large in ((False,True) if f!="full" else (True,))]
    paths=("benchmarks/rocm/benchmark_captured_movement.py","benchmarks/rocm/benchmark_rocm_e2e_movement.py",
        "python/tessera/compiler/jit.py","python/tessera/compiler/rocm_native.py",
        "python/tessera/compiler/scheduled_paged_kv.py","python/tessera/runtime.py")
    from tessera.compiler.scheduled_matmul import find_tessera_opt
    compiler=Path(find_tessera_opt())
    args.output.write_text(json.dumps(dict(device=identity,architecture=args.architecture,rows=rows,
        compiler_sha256=hashlib.sha256(compiler.read_bytes()).hexdigest(),
        fingerprints={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        timing_scope="HIP graph replay versus Python driver loop on identical resident buffers; event windows include device dispatch, not isolated instruction execution; wall separate"),indent=2)+"\n")

if __name__=="__main__":main()
