
#!/usr/bin/env python3
import argparse, yaml, json, time, os, subprocess, sys, pathlib, math
from collect_env import collect as collect_env
from probes import collect_all as collect_probes
from trace import Trace

def run_cpp_binary(path, args):
    if not os.path.exists(path):
        return {"ok": False, "skip_reason": f"Missing binary: {path}"}
    try:
        start = time.time()
        out = subprocess.check_output([path] + args, text=True, stderr=subprocess.STDOUT, timeout=1800)
        dur = time.time() - start
        last = out.strip().splitlines()[-1]
        row = json.loads(last)
        row.setdefault("latency_ms", dur * 1000.0)
        row["ok"] = True
        return row
    except subprocess.CalledProcessError as e:
        return {"ok": False, "skip_reason": f"nonzero exit: {e.returncode}", "stdout": e.output}
    except Exception as e:
        return {"ok": False, "skip_reason": str(e)}

def run_python_module(path, args):
    if not os.path.exists(path):
        return {"ok": False, "skip_reason": f"Missing module: {path}"}
    try:
        start = time.time()
        out = subprocess.check_output([sys.executable, path] + args, text=True, stderr=subprocess.STDOUT, timeout=1800)
        dur = time.time() - start
        last = out.strip().splitlines()[-1]
        row = json.loads(last)
        row.setdefault("latency_ms", dur * 1000.0)
        row["ok"] = True
        return row
    except subprocess.CalledProcessError as e:
        return {"ok": False, "skip_reason": f"nonzero exit: {e.returncode}", "stdout": e.output}
    except Exception as e:
        return {"ok": False, "skip_reason": str(e)}

def maybe_efficiency(row, peaks):
    # Try to compute roofline efficiency if peaks provided in suite dict
    if not peaks: return
    bytes_per_s = row.get("bytes_per_s")
    flops = row.get("throughput_flops")
    if bytes_per_s:
        row["mem_efficiency_pct"] = 100.0 * bytes_per_s / peaks["memory_peak_bytes_per_s"]
    if flops:
        row["compute_efficiency_pct"] = 100.0 * flops / peaks["compute_peak_flops"]
        if bytes_per_s:
            I = flops / max(bytes_per_s, 1e-12)
            ridge = peaks["compute_peak_flops"] / peaks["memory_peak_bytes_per_s"]
            row["arithmetic_intensity"] = I
            row["ridge_point"] = ridge

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    peaks = None
    peaks_file = cfg.get("peaks_file")
    if peaks_file and os.path.exists(peaks_file):
        import yaml as y
        with open(peaks_file,"r") as pf:
            peaks = y.safe_load(pf)

    trace = Trace()
    trace.begin("suite")

    results = {
        "suite": cfg.get("suite", {}),
        "env": collect_env(),
        "probes": collect_probes(),
        "timestamp": time.time(),
        "rows": []
    }

    for task in cfg.get("tasks", []):
        rid = task["id"]
        runner = task["runner"]
        path = task["path"]
        targs = [str(a) for a in task.get("args", [])]

        trace.begin(rid, cat="task", args={"path": path, "args": targs})
        if runner == "cpp_binary":
            row = run_cpp_binary(path, targs)
        elif runner == "python_module":
            row = run_python_module(path, targs)
        else:
            row = {"ok": False, "skip_reason": f"unknown runner: {runner}"}
        trace.end()

        row["bench_id"] = rid
        row["variant"] = task.get("variant","default")
        maybe_efficiency(row, peaks)
        results["rows"].append(row)
        print(f"[{rid}] -> ok={row.get('ok')} reason={row.get('skip_reason')}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    trace.end()
    trace.save(os.path.join(args.out, "trace.json"))
    print(f"Wrote {os.path.join(args.out,'results.json')} and trace.json")

if __name__ == "__main__":
    main()
