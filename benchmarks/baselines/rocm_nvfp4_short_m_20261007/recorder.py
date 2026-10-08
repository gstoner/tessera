import hashlib,json,os,time
from pathlib import Path
from statistics import median
import numpy as np
from tessera import runtime as rt
from tessera.compiler.rocm_nvfp4_resident import package_resident_nvfp4_matmul
from tessera.compiler.rocm_nvfp4_ingest import nvfp4_requantization_policy
from tests.unit.test_rocm_nvfp4_resident import inputs_and_oracle
assert rt._rocm_live_arch()=="gfx1201"
rows=[]
for m in (1,16,32,64,65,128):
 for n,k in ((32,64),(80,256)):
  args,offsets,_,_,expected=inputs_and_oracle(m,n,k)
  program=package_resident_nvfp4_matmul(m,n,k,offsets,numeric_policy=nvfp4_requantization_policy(),approximate_policy="explicit_allow",runtime_mn=True)
  with program.session(*args) as session:
   session.run_combined()
   np.testing.assert_allclose(session.read_output().astype(np.float32),expected,rtol=.008,atol=.015625)
   events={stage:session.measure(stage,samples=7,repeats=5) for stage in ("converter","storage","consumer","combined")}
   graph={stage:session.measure_graph(stage,samples=7,repeats=32) for stage in ("converter","storage","consumer","combined")}
   walls=[]
   for _ in range(7):
    start=time.perf_counter_ns();session.run_combined();output=session.read_output();walls.append((time.perf_counter_ns()-start)/1e6)
    np.testing.assert_allclose(output.astype(np.float32),expected,rtol=.008,atol=.015625)
   np.testing.assert_allclose(session.read_output().astype(np.float32),expected,rtol=.008,atol=.015625)
  rows.append(dict(shape_mnk=[m,n,k],correctness="passed_before_and_after_timing",resident_event_samples_ms=events,resident_event_median_ms={s:median(v) for s,v in events.items()},graph_windows=graph,graph_per_iteration_median_ms={s:median(x["per_iteration_ms"] for x in v) for s,v in graph.items()},prepared_run_readback_samples_ms=walls,prepared_run_readback_median_ms=median(walls),compiler_receipt=program.receipt))
  Path("/home/angstorms/scratch/short-m-nvfp4-timing-20261007.json").write_text(json.dumps(dict(architecture=rt._rocm_live_arch(),timing_scope="HIP event loops include host enqueue gaps; native graph samples include GPU graph dispatch; prepared walls include execution/readback",rows=rows),indent=2))
  print(m,n,k,rows[-1]["graph_per_iteration_median_ms"],flush=True)
