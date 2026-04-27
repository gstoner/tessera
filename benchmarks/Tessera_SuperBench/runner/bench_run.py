#!/usr/bin/env python3
import argparse, copy, json, os, pathlib, subprocess, sys, time
try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required for benchmark configs. Install project dependencies with `python3 -m pip install -r requirements.txt`.") from exc
from collect_env import collect as collect_env
from probes import collect_all as collect_probes
from trace import Trace

def _last_json_line(out):
    for line in reversed(out.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise ValueError("child process did not emit a JSON result line")

def run_cpp_binary(path, args):
    if not path.exists():
        return {"ok": False, "skip_reason": f"Missing binary: {path}"}
    try:
        start = time.time()
        out = subprocess.check_output([str(path)] + args, text=True, stderr=subprocess.STDOUT, timeout=1800)
        dur = time.time() - start
        row = _last_json_line(out)
        row.setdefault("latency_ms", dur * 1000.0)
        row.setdefault("ok", True)
        return row
    except subprocess.CalledProcessError as e:
        return {"ok": False, "skip_reason": f"nonzero exit: {e.returncode}", "stdout": e.output}
    except Exception as e:
        return {"ok": False, "skip_reason": str(e)}

def run_python_module(path, args):
    if not path.exists():
        return {"ok": False, "skip_reason": f"Missing module: {path}"}
    try:
        start = time.time()
        out = subprocess.check_output([sys.executable, str(path)] + args, text=True, stderr=subprocess.STDOUT, timeout=1800)
        dur = time.time() - start
        row = _last_json_line(out)
        row.setdefault("latency_ms", dur * 1000.0)
        row.setdefault("ok", True)
        return row
    except subprocess.CalledProcessError as e:
        return {"ok": False, "skip_reason": f"nonzero exit: {e.returncode}", "stdout": e.output}
    except Exception as e:
        return {"ok": False, "skip_reason": str(e)}

def maybe_efficiency(row, peaks):
    # Try to compute roofline efficiency if peaks provided in suite dict
    if not peaks: return
    compute_peak = float(peaks["compute_peak_flops"])
    memory_peak = float(peaks["memory_peak_bytes_per_s"])
    bytes_per_s = row.get("bytes_per_s")
    flops = row.get("throughput_flops")
    if bytes_per_s:
        row["mem_efficiency_pct"] = 100.0 * bytes_per_s / memory_peak
    if flops:
        row["compute_efficiency_pct"] = 100.0 * flops / compute_peak
        if bytes_per_s:
            I = flops / max(bytes_per_s, 1e-12)
            ridge = compute_peak / memory_peak
            row["arithmetic_intensity"] = I
            row["ridge_point"] = ridge

def _merge_config(base, overlay):
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in ("extends", "overrides"):
            continue
        merged[key] = value
    overrides = overlay.get("overrides", {})
    for key, value in overrides.items():
        if key == "tasks":
            by_id = {task["id"]: copy.deepcopy(task) for task in merged.get("tasks", [])}
            for task_override in value:
                task_id = task_override["id"]
                by_id.setdefault(task_id, {}).update(task_override)
            merged["tasks"] = list(by_id.values())
        else:
            merged[key] = value
    return merged

def load_config(config_path):
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.get("extends")
    if parent:
        parent_cfg = load_config((config_path.parent / parent).resolve())
        cfg = _merge_config(parent_cfg, cfg)
    return cfg

def resolve_task_path(suite_root, path):
    task_path = pathlib.Path(path)
    if task_path.is_absolute():
        return task_path
    return suite_root / task_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    config_path = pathlib.Path(args.config).resolve()
    suite_root = config_path.parents[1] if config_path.parent.name == "configs" else config_path.parent
    cfg = load_config(config_path)

    peaks = None
    peaks_file = cfg.get("peaks_file")
    if peaks_file:
        peaks_path = resolve_task_path(suite_root, peaks_file)
        if peaks_path.exists():
            with peaks_path.open("r", encoding="utf-8") as pf:
                peaks = yaml.safe_load(pf)

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
        path = resolve_task_path(suite_root, task["path"])
        targs = [str(a) for a in task.get("args", [])]

        trace.begin(rid, cat="task", args={"path": str(path), "args": targs})
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
