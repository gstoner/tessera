"""Fresh-process native softmax staging A/B; requires an idle test host."""
import argparse,hashlib,json,os
from pathlib import Path
from statistics import median
import subprocess,sys,tempfile

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--control",type=Path,required=True)
 p.add_argument("--candidate",type=Path,required=True)
 p.add_argument("--control-source",type=Path,required=True)
 p.add_argument("--candidate-source",type=Path,required=True)
 p.add_argument("--output",type=Path,required=True)
 p.add_argument("--windows",type=int,default=5)
 args=p.parse_args()
 if args.windows<3:p.error("at least three alternating windows required")
 active=subprocess.run(["pgrep","-af","[p]ytest"],text=True,capture_output=True)
 if active.returncode==0 and active.stdout.strip():
  raise RuntimeError("aggregate/test process is active; matched GPU timing must wait")
 root=Path.cwd()
 recorder=root/"benchmarks/nvidia/benchmark_public_softmax_alias.py"
 libraries={"control":args.control.resolve(),"candidate":args.candidate.resolve()}
 sources={"control":args.control_source.resolve(),"candidate":args.candidate_source.resolve()}
 trials=[]
 with tempfile.TemporaryDirectory(prefix="softmax-staging-ab-",dir="/home/angstorms/scratch") as temporary:
  for window in range(args.windows):
   order=("control","candidate") if window%2==0 else ("candidate","control")
   paired={}
   for arm in order:
    output=Path(temporary)/f"{window}-{arm}.json"
    env=dict(os.environ,TESSERA_NVIDIA_PTX_LAUNCH_LIB=str(libraries[arm]))
    subprocess.run([sys.executable,str(recorder),"--output",str(output)],env=env,check=True,capture_output=True,text=True)
    packet=json.loads(output.read_text())
    if packet["runtime_sha256"]!=sha(libraries[arm]):
     raise RuntimeError("child did not load the selected native runtime")
    packet["compiled_runtime_source_sha256"]=sha(sources[arm])
    packet["runtime_source_scope"]="isolated native runtime source; child source tree hashes identify the common frontend checkout"
    paired[arm]=packet
   trials.append({"window":window,"order":list(order),"arms":paired})
 comparisons=[]
 for index,row in enumerate(trials[0]["arms"]["control"]["rows"]):
  wall=[];device=[]
  for trial in trials:
   a,b=(trial["arms"][arm]["rows"][index] for arm in ("control","candidate"))
   if any(a[key]!=b[key] for key in ("operation","dtype","shape","image_digest","descriptor_digest")):
    raise RuntimeError("matched arms differ in native package or profile identity")
   if a["correctness"]!="passed_before_and_after_timing" or b["correctness"]!=a["correctness"]:
    raise RuntimeError("timed pair lacks numerical proof")
   wall.append(a["public_host_wall_median_ms"]/b["public_host_wall_median_ms"])
   device.append(a["resident_device_event_median_ms"]/b["resident_device_event_median_ms"])
  comparisons.append({"operation":row["operation"],"dtype":row["dtype"],"shape":row["shape"],
                      "image_digest":row["image_digest"],"paired_control_over_candidate_public_wall":wall,
                      "median_public_wall_ratio":median(wall),"paired_control_over_candidate_device":device,
                      "median_device_ratio":median(device)})
 packet={"schema":"tessera.sm120.softmax_staging_ab.v1","recorder_sha256":sha(__file__),
         "timing_scope":"alternating fresh-process native runtime arms; public-wall and resident kernel-event metrics separate",
         "trials":trials,"comparisons":comparisons,"promotion":False}
 args.output.parent.mkdir(parents=True,exist_ok=True)
 args.output.write_text(json.dumps(packet,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"comparisons":len(comparisons),"median_public_wall_ratio":median(r["median_public_wall_ratio"] for r in comparisons)}))
if __name__=="__main__":main()
