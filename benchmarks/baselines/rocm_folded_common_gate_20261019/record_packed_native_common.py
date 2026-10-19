"""Scratch A/B of native packed packages; reuses checked resident/clock owners."""
from pathlib import Path
from contextlib import ExitStack
from types import SimpleNamespace
import ctypes as C
import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
import numpy as np
from tessera import runtime as rt
from tessera.compiler import rocm_mxfp4_packed_folded as packed
from benchmarks.rocm import benchmark_gfx1201_mxfp4_production as base
from benchmarks.rocm.ablate_gfx1201_folded_load_schedule import package_engine
from benchmarks.rocm.folded_native_resident_frame import FoldedNativeFramePool
from benchmarks.rocm.folded_graph_windows import FoldedGraphWindows
from benchmarks.rocm.record_gfx1201_mxfp4_folded_load_schedule import DeviceClock
from benchmarks.rocm.record_gfx1201_folded_native_package import independent_output_snapshots, SOURCES, ROOT

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidate-opt",required=True,type=Path)
parser.add_argument("--control-opt",required=True,type=Path)
parser.add_argument("--llvm-bin",required=True,type=Path)
parser.add_argument("--output",required=True,type=Path)
args=parser.parse_args()
tools={"candidate":args.candidate_opt.resolve(),"control":args.control_opt.resolve()}
args.output.parent.mkdir(parents=True,exist_ok=True)
if rt._rocm_live_arch() != "gfx1201":
 raise RuntimeError("owning gfx1201 device required")
hip=rt._load_hip_for_launch()
if hip is None or hip.hipInit(0) != 0:
 raise RuntimeError("HIP initialization failed")
device_name = base._selected_device_name(hip)
result=[]
shapes=[(256,1024,1024),(256,8192,1536),(256,8192,5120),(200,2048,2048)]
for order in ["forward","reverse"]:
 for shape in (shapes if order=="forward" else reversed(shapes)):
  case=base.Case("prefill",*shape);inputs=base._logical_inputs(case)
  weights=packed.prepare_packed_folded_payload(inputs["packed_row_major"],inputs["b_scale"],allow_approximate=True)
  packages={}
  names=list(tools) if order=="forward" else list(reversed(tools))
  for name in names:
   os.environ["TESSERA_OPT"]=str(tools[name])
   packages[name]=packed.compile_packed_folded_scaled_matmul(inputs["a"],inputs["a_scale"],weights,tessera_opt=tools[name]).package
  assert packages["candidate"].descriptor.abi_id==packages["control"].descriptor.abi_id
  assert packages["candidate"].descriptor.geometry==packages["control"].descriptor.geometry
  assert packages["candidate"].tile_ir==packages["control"].tile_ir
  # Target source is the same physical directive; only native lowering changes.
  assert packages["candidate"].target_ir==packages["control"].target_ir
  frame_arrays=SimpleNamespace(weight_bytes=weights.weight_bytes,row_reference=weights.scale_plane)
  with ExitStack() as stack:
   pool=FoldedNativeFramePool(lambda arrays,count:base._copies(hip,arrays,count));stack.callback(pool.close)
   engines=[]
   for name in names:
    package=packages[name]
    assert package.descriptor.provenance["native_compiler_owned"] is True
    engine=package_engine(hip,case,inputs,frame_arrays,package,name=name,
        metadata={"route":package.descriptor.provenance,"physical_plane_role":"packed scale_plane, not unpacked row_reference"},copies=3,resident_pool=pool)
    engines.append(engine);stack.callback(engine.close)
   outputs=independent_output_snapshots(engines)
   np.testing.assert_array_equal(outputs["candidate"].view(np.uint16),outputs["control"].view(np.uint16))
   rows,cols,want=base._sampled_exact_reference(case,inputs)
   for output in outputs.values():np.testing.assert_array_equal(output[np.ix_(rows,cols)],want)
   public={}
   for name in names:
    package=packages[name]
    artifact=rt.RuntimeArtifact(tile_ir=package.tile_ir,target_ir=package.target_ir,
        metadata={"target":"rocm_gfx1201"},native_image=package.image,launch_descriptor=package.descriptor)
    samples=[]
    for _ in range(3):
     output=np.full_like(inputs["output"],np.nan)
     start=time.perf_counter_ns()
     receipt=rt.launch(artifact,{"buffers":{"a":inputs["a"],"b_packed":weights.weight_bytes,
       "a_scale":inputs["a_scale"],"scale_plane":weights.scale_plane,"output":output},
       "scalars":dict(zip(("M","N","K"),shape,strict=True))})
     samples.append((time.perf_counter_ns()-start)/1e6)
     assert receipt["ok"] and receipt["execution_kind"]=="native_gpu",receipt
     np.testing.assert_array_equal(output.view(np.uint16),outputs[name].view(np.uint16))
    public[name]=samples
   clock=DeviceClock(hip,tools["candidate"],args.llvm_bin);stack.callback(clock.close)
   graph=FoldedGraphWindows(clock);stack.callback(graph.close)
   for engine in engines:base._warmup(hip,engine,6)
   estimates=[clock.window(e,3,bracketed=False)["event_window_ms"]/3 for e in engines]
   count=3*math.ceil(max(12,math.ceil(12/min(estimates)))/3)
   for engine in engines:
    for bracketed in [True,False]:
     for bundle in engine.copies:
      poison=np.empty_like(outputs[engine.name]);poison.view(np.uint8).fill(255)
      clock._check(hip.hipMemcpy(bundle.device[4],poison.ctypes.data_as(C.c_void_p),poison.nbytes,1),"poison")
     graph.window(engine,12,bracketed=bracketed)
     for bundle in engine.copies:np.testing.assert_array_equal(bundle.download(4).view(np.uint16),outputs[engine.name].view(np.uint16))
   windows={name:[] for name in names}
   for trial in range(11):
    for engine in (engines if trial%2==0 else reversed(engines)):
     for bracketed in ([True,False] if trial%2==0 else [False,True]):
      n=count
      for attempt in range(5):
       sample=graph.window(engine,n,bracketed=bracketed)
       if not bracketed or sample["device_window_ms"]>=5:break
       n=3*math.ceil(max(n*2,n*8/max(sample["device_window_ms"],.001))/3)
      if bracketed:
       assert sample["device_window_ms"]>=5 and sample["device_event_disagreement"]<=.05,sample
      windows[engine.name].append(sample)
   for engine in engines:
    for bundle in engine.copies:np.testing.assert_array_equal(bundle.download(4).view(np.uint16),outputs[engine.name].view(np.uint16))
   medians={name:statistics.median(w["device_window_ms"]/w["launches"] for w in windows[name] if w["bracketed"]) for name in names}
   row={"order":order,"shape_mnk":list(shape),"candidate_over_control_graph":medians["candidate"]/medians["control"],
       "correctness":"public checked launches, full bitwise, sampled independent oracle, poisoned bracketed/plain all-three-frame graph replay",
       "graph_medians_ms":medians,"windows":windows,"complete_public_call_samples_ms":public,
       "images":{e.name:e.metadata for e in engines},"resident_pool":pool.report}
  assert pool.report["closed"] and pool.report["leases"]==0
  result.append(row)
  record={"architecture":rt._rocm_live_arch(),"device":device_name,"kernel_release":platform.release(),
      "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "dirty_worktree":bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()),
      "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES},
      "recorder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      "scope":"Graph/Schedule/Tile/native packed ABI; GPU graph dispatch plus kernels, not profiler isolated kernel; public complete calls separate",
      "compiler_sha256":{name:hashlib.sha256(tool.read_bytes()).hexdigest() for name,tool in tools.items()},"rows":result}
  args.output.write_text(json.dumps(record,indent=2)+"\n")
  print(order,shape,row["candidate_over_control_graph"],flush=True)
