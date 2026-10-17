from pathlib import Path
import json
import math
import os
import time
import statistics
import hashlib
import numpy as np
from tessera import runtime as rt
from tessera.compiler.rocm_nvfp4_expansion_program import package_native_nvfp4_expansion_program
from tests.unit.test_native_nvfp4_parent_view_program import frontend_activation
from tessera.compiler.trace import trace, to_graph_ir_module
from tests.device.rocm.test_prepared_nvfp4_expansion_program import inputs_and_oracle
from tests.device.rocm.test_native_parent_view_bf16_jit import oracle,assert_result
assert rt._rocm_live_arch()=="gfx1201"
v=Path("/home/angstorms/scratch/rocm-native-gate-up-views-20261016")
rows=[]
for m,width,k in [(3,17,48),(16,64,80),(256,128,256)]:
 roots,parent=inputs_and_oracle(m,2*width,k,(0,1,2,3,4),"bf16")
 source=to_graph_ir_module(trace(frontend_activation(m,width),*roots),target="rocm_gfx1201")
 started=time.perf_counter()
 package=package_native_nvfp4_expansion_program(source)
 compile_ms=(time.perf_counter()-started)*1000
 roots,parent=inputs_and_oracle(m,2*width,k,(0,1,2,3,4),"bf16")
 expected=oracle(parent,width,False)
 started=time.perf_counter()
 with package.prepare(roots,runtime_library=os.environ["TESSERA_ROCM_MULTI_OUTPUT_LIB"]) as owner:
  prepare_ms=(time.perf_counter()-started)*1000
  owner.import_weights()
  generation,_=owner.invoke_consumer()
  assert_result(owner.read(generation)[0],expected)
  generation,pilot=owner.profile_members(repeats=64)
  assert_result(owner.read(generation)[0],expected)
  repeats=min(65536,max(64,math.ceil(20/max(min(pilot),1e-6))))
  for calibration in range(5):
   generation,calibrated=owner.profile_members(repeats=repeats)
   assert_result(owner.read(generation)[0],expected)
   window=repeats*min(calibrated)
   if window >= 18: break
   if repeats == 65536: raise RuntimeError("cannot establish stable captured windows")
   repeats=min(65536,max(repeats+1,math.ceil(repeats*22/max(window,1e-6))))
  else: raise RuntimeError("captured window calibration did not converge")
  captured=[];program=[];host=[]
  for sample in range(5):
   generation,stage=owner.profile_members(repeats=repeats)
   assert_result(owner.read(generation)[0],expected)
   captured.append(list(stage))
   generation,elapsed=owner.invoke_consumer(repeats=4096,timed=True)
   assert_result(owner.read(generation)[0],expected)
   program.append(elapsed)
   started=time.perf_counter()
   owner.update_activations(roots[3],roots[4])
   generation,_=owner.invoke_consumer()
   result,=owner.read(generation)
   host.append((time.perf_counter()-started)*1000)
   assert_result(result,expected)
  minimum_window=repeats*min(min(x) for x in captured)
  if minimum_window < 10:
   raise RuntimeError(f"captured window below required 10 ms: {minimum_window}")
  row={"shape_m_width_k":[m,width,k],"compile_package_ms":compile_ms,
   "prepare_upload_ms":prepare_ms,"member_order":["expansion","bf16_product","bf16_activation"],
   "captured_member_samples_ms_per_iteration":captured,"captured_repeats":repeats,
   "captured_minimum_window_ms":minimum_window,
   "resident_product_activation_event_samples_ms":program,
   "resident_event_repeats":4096,
   "completed_activation_update_invoke_read_samples_ms":host,
   "completed_activation_update_invoke_read_median_ms":statistics.median(host),
   "correctness":"independent_e2m1_f64_product_bf16_parent_f64_silu_bf16_store_before_and_after_every_window",
   "program_sha256":hashlib.sha256(package.program_json.encode()).hexdigest(),
   "image_digests":[p.image.image_digest for p in [package.producer.native,package.consumer,package.activation]]}
  rows.append(row)
  print(json.dumps({"shape":row["shape_m_width_k"],"host_ms":row["completed_activation_update_invoke_read_median_ms"],"captured_min_window_ms":minimum_window}),flush=True)
packet={"scope":"synthetic Python-authored Graph resident characterization; no model-quality or performance promotion",
 "architecture":rt._rocm_live_arch(),"device_probe":json.loads((v/"bf16-device-probe.json").read_text()),
 "timing_scope":{"captured_members":"device graph dispatch included; capture/instantiation/host enqueue excluded; grouped members are not interleaved whole-program time",
 "resident_event":"HIP event range includes host enqueue gaps; average returned by native API; not divided again",
 "completed_host":"retained owner/weights; activation upload, product+activation invocation and readback; package compilation/preparation excluded"},
 "model_quality_acceptance":False,"rows":rows}
(v/"bf16-store-source-characterization.json").write_text(json.dumps(packet,indent=2)+"\n")
