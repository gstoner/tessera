from pathlib import Path
import time
from statistics import median
root=Path(__file__).parent
exec((root/"check_chain_device.py").read_text().split("\nrows=[]")[0])
live=subprocess.check_output(["pgrep","-af","[p]ytest|[g]raphify update"],text=True) if subprocess.run(["pgrep","-f","[p]ytest|[g]raphify update"],capture_output=True).returncode==0 else ""
if live: raise RuntimeError("validation or graph extraction is active: "+live)
rows=[]
for dtype,storage in (("fp16",np.float16),("bf16",ml_dtypes.bfloat16)):
 for count,function in ((2,chain),(3,three_chain)):
  m,k,n=128,4096,64
  rng=np.random.default_rng(5070128);x=rng.normal(0,.2,(m,k)).astype(storage)
  rhs=np.array(rng.normal(0,.2,(k,n)),dtype=storage,order="F")
  graph=lhs.project_rhs_storage(function._traced_autodiff_module((x,rhs),{}),[x,rhs])
  value=x.astype(np.float64)
  if count==3:
   centered=value-value.mean(axis=-1,keepdims=True)
   value=(centered/np.sqrt(np.mean(centered*centered,axis=-1,keepdims=True)+1e-5)).astype(storage).astype(np.float64)
  value=(value/np.sqrt(np.mean(value*value,axis=-1,keepdims=True)+1e-5)).astype(storage).astype(np.float64)
  exp=np.exp(value-value.max(axis=-1,keepdims=True));soft=(exp/exp.sum(axis=-1,keepdims=True)).astype(storage)
  expected=soft.astype(np.float64)@rhs.astype(np.float64)
  programs={mode:lhs.package_traced_lhs(graph,softmax_schedule=mode) for mode in ("serial","cooperative_128")}
  owners={mode:prepared.PreparedLhsCall(p) for mode,p in programs.items()}
  samples={mode:[] for mode in programs}
  try:
   for mode,owner in owners.items():
    actual,receipt=owner([x,rhs]);np.testing.assert_allclose(actual,expected,rtol=.015,atol=.002)
   for window in range(5):
    for mode in (("serial","cooperative_128") if window%2==0 else ("cooperative_128","serial")):
     start=time.perf_counter()
     for _ in range(11):actual,receipt=owners[mode]([x,rhs])
     samples[mode].append((time.perf_counter()-start)*1000/11)
     np.testing.assert_allclose(actual,expected,rtol=.015,atol=.002)
   row=dict(dtype=dtype,shape_mkn=[m,k,n],producer_count=count,prepared_host_wall_samples_ms=samples,host_speedup=median(samples["serial"])/median(samples["cooperative_128"]),consumer_image=programs["serial"].edge.consumer.image.image_digest,correctness="before_and_after_timing")
   assert programs["serial"].edge.consumer.image.image_digest==programs["cooperative_128"].edge.consumer.image.image_digest
   rows.append(row);print(json.dumps(row),flush=True)
  finally:
   for owner in owners.values():owner.close()
packet=json.loads((root/"chain_device_proof.json").read_text());packet["rows"]=rows;packet["timing"]="alternating 5 windows of 11 prepared whole-chain host calls; copies and completion included; no device-kernel claim"
(root/"chain_ab.json").write_text(json.dumps(packet,indent=2)+"\n")

