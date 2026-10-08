"""Controlled public and native timings across equivalent leading prefixes."""
import argparse,json,os,subprocess,time
from pathlib import Path
from statistics import median
import numpy as np
from tessera import runtime
from tessera.compiler.native_scaled_program import PreparedScaledProgram
from tests.unit.test_native_deep_typed_vmap import deep_case
from tests.support.scaled_product_transpose_oracle import scale_adjoint

parser=argparse.ArgumentParser()
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
assert runtime._rocm_live_arch()=="gfx1201"
probe=subprocess.check_output(["/opt/rocm/bin/rocminfo"],text=True)
schedules=("serial_per_scale_element","wave_per_scale_element")
rows=[]
for prefix in ((2,3),(2,1,3),(1,2,1,3)):
 for policy in ("shared_rhs_rows","independent_rhs","shared_lhs"):
  for nk in (False,True):
   owners={};gold=None
   for schedule in schedules:
    os.environ["TESSERA_ROCM_SCALE_VJP_SCHEDULE"]=schedule
    _,owner,values,expected,_=deep_case(policy,"fp32",nk,prefix,mode="reverse")
    dy=np.random.default_rng(1910).uniform(-.5,.5,expected.shape).astype(np.float32)
    gold=scale_adjoint(*values,dy,scale_k=128,scale_n=128,batching=policy,transpose_b=nk)
    def check(outputs,factor=1):
     for output,want in zip(outputs,gold,strict=True):
      np.testing.assert_allclose(output,want*factor,rtol=4e-5,atol=2e-4)
    check(owner.native_backward(*values,out_cotangents=dy))
    owners[schedule]=owner
   samples={s:[] for s in schedules}
   native={s:[] for s in schedules}
   original_run=subprocess.run
   def forbidden(*a,**kw):raise AssertionError("warm deep-map measurement invoked compiler")
   subprocess.run=forbidden
   try:
    for window in range(6):
     for schedule in schedules if window%2==0 else schedules[::-1]:
      os.environ["TESSERA_ROCM_SCALE_VJP_SCHEDULE"]=schedule
      start=time.perf_counter_ns()
      for _ in range(3):result=owners[schedule].native_backward(*values,out_cotangents=dy)
      samples[schedule].append((time.perf_counter_ns()-start)/3e6)
      check(result)
    for schedule in schedules:
     package=owners[schedule].native_backward_runtime_artifact()
     with PreparedScaledProgram(package,[*values,dy],runtime_library=os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]) as prepared:
      generation,_=prepared.invoke();check(prepared.read(generation))
      for _ in range(7):
       generation,elapsed=prepared.invoke(repeats=32,timed=True)
       native[schedule].append(elapsed/32)
      check(prepared.read(generation))
     os.environ["TESSERA_ROCM_SCALE_VJP_SCHEDULE"]=schedule
     check(owners[schedule].native_backward(*values,out_cotangents=dy*-.5),-.5)
   finally:subprocess.run=original_run
   row=dict(prefix=list(prefix),policy=policy,rhs_nk=nk,output_shape=list(expected.shape),
    correctness="independent_f64_before_after_and_changed_cotangent",
    public_samples_ms=samples,native_event_window_samples_ms=native,
    public_medians_ms={s:median(v) for s,v in samples.items()},
    native_event_window_medians_ms={s:median(v) for s,v in native.items()},
    artifact_hashes={s:owners[s].last_backward_execution["artifact_hash"] for s in schedules})
   rows.append(row)
   args.output.write_text(json.dumps(dict(architecture="gfx1201",device_probe=probe,rows=rows,
    timing_scope="public walls include frontend/ABI/uploads/native/readback; native event windows include native multi-kernel submission, no Python enqueue loop"),indent=2))
   print(prefix,policy,nk,row["public_medians_ms"],row["native_event_window_medians_ms"],flush=True)
