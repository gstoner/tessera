"""Fresh-process counterbalanced A/B for prepared native attention staging."""
from pathlib import Path
import argparse
import hashlib
import json
import os
from statistics import median
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
CASES={"jvp":("qkv_q_5_0","vqk_v_129_1","kvq_k_q_5_0","vkq_q_k_v_129_1"),
       "vjp":("qkv_q_5_0","vkq_q_k_v_129_1","biasvqk_bias_v_k_q_5_1_1x4x1x1")}

def child(runtime,output,repetitions):
    from tessera import runtime as rt
    from benchmarks.nvidia.benchmark_prepared_attention_jvp import run as run_jvp
    from benchmarks.nvidia.benchmark_prepared_attention_vjp import run as run_vjp
    if rt._nvidia_device_name()!="sm_120":
        raise RuntimeError("owning SM120 GPU required")
    lib=rt._load_nvidia_ptx_launch()
    if lib is None or Path(lib._name).resolve()!=runtime.resolve():
        raise RuntimeError("requested native library was not loaded")
    rows=[]
    for kind,runner in (("jvp",run_jvp),("vjp",run_vjp)):
        directory=ROOT/"benchmarks/baselines"/f"nvidia_public_attention_{kind}_20261006"/"artifacts"
        for name in CASES[kind]:
            row=runner(directory/(name+".json"),repetitions)
            row["derivative"]=kind
            rows.append(row)
    output.write_text(json.dumps({"runtime_sha256":hashlib.sha256(runtime.read_bytes()).hexdigest(),
                                 "profiles":rows},indent=2)+"\n")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--control",type=Path)
    parser.add_argument("--candidate",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--windows",type=int,default=3)
    parser.add_argument("--repetitions",type=int,default=11)
    parser.add_argument("--child-runtime",type=Path)
    args=parser.parse_args()
    if args.repetitions<3:
        parser.error("at least three correctness-gated repetitions required")
    if args.child_runtime:
        child(args.child_runtime,args.output,args.repetitions)
        return
    if not args.control or not args.candidate or args.windows<3:
        parser.error("both libraries and at least three counterbalanced windows required")
    processes=subprocess.check_output(["ps","-eo","args"],text=True)
    if any("python" in line and "-m pytest" in line for line in processes.splitlines()):
        raise RuntimeError("timing refused while pytest validation is running")
    libraries={"control":args.control.resolve(),"candidate":args.candidate.resolve()}
    if any(not path.is_file() for path in libraries.values()):
        raise ValueError("native runtime library missing")
    device=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,driver_version","--format=csv,noheader"],text=True).strip()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    trials=[]
    for window in range(args.windows):
        for arm in (("control","candidate") if window%2==0 else ("candidate","control")):
            file=args.output.parent/f"window_{window}_{arm}.json"
            env=dict(os.environ,TESSERA_NVIDIA_PTX_LAUNCH_LIB=str(libraries[arm]))
            subprocess.run([sys.executable,str(Path(__file__).resolve()),
                "--child-runtime",str(libraries[arm]),"--output",str(file),
                "--repetitions",str(args.repetitions)],env=env,cwd=ROOT,check=True)
            trials.append({"window":window,"arm":arm,**json.loads(file.read_text())})
    comparisons=[]
    for kind,names in CASES.items():
        for name in names:
            rows=[(trial,next(row for row in trial["profiles"]
                if row["case"]==name and row["derivative"]==kind)) for trial in trials]
            if len({row["artifact_hash"] for _,row in rows})!=1:
                raise RuntimeError("runtime arms did not use identical native artifacts")
            ratios=[]
            for window in range(args.windows):
                samples={trial["arm"]:row for trial,row in rows if trial["window"]==window}
                ratios.append(samples["control"]["medians_ms"]["prepared"]/
                              samples["candidate"]["medians_ms"]["prepared"])
            comparisons.append({"case":name,"derivative":kind,"artifact_hash":rows[0][1]["artifact_hash"],
                "paired_control_over_candidate_wall_ratios":ratios,"median_ratio":median(ratios)})
    packet={"schema":"tessera.prepared_attention_staging_ab.v1","device":device,
            "target":"nvidia_sm120","method":"fresh processes; alternating library order; identical native artifacts; independent oracle and retained-output checks inside every timed trial; prepared host wall and separate CUDA-event forward/derivative times",
            "trials":trials,"comparisons":comparisons,
            "recorder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "promotion":"none; named-envelope matched comparison"}
    args.output.write_text(json.dumps(packet,indent=2)+"\n")
    print(json.dumps({"comparisons":len(comparisons),"median_wall_ratio":median(c["median_ratio"] for c in comparisons)}))

if __name__=="__main__":
    main()
