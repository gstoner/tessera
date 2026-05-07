#!/usr/bin/env python3
import argparse, csv, itertools, json, os, pathlib, subprocess, sys, time
try:
  import yaml
except ImportError as exc:
  raise SystemExit("PyYAML is required for benchmark configs. Install project dependencies with `python3 -m pip install -r requirements.txt`.") from exc

ARG_ALIASES = {
  "M": "m",
  "N": "n",
  "K": "k",
}

SCRIPT_ROOT = pathlib.Path(__file__).resolve().parents[1]

def grid(sweep):
  keys = list(sweep.keys())
  values = [sweep[k] for k in keys]
  for combo in itertools.product(*values):
    yield dict(zip(keys, combo))

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--config", required=True)
  ap.add_argument("--bin", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--backend", default=None, choices=["reference", "artifact", "tessera-runtime"])
  ap.add_argument("--runtime", default="bridge", choices=["bridge", "native"])
  ap.add_argument("--artifact-root", default=str(SCRIPT_ROOT))
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  with open(args.config) as f:
    cfg = yaml.safe_load(f)
  seed = cfg.get("seed", 123)
  rows=[]
  for run in cfg["runs"]:
    op = run["op"]
    iters = run.get("iters", 50)
    backend = args.backend or run.get("backend", cfg.get("backend", "reference"))
    for params in grid(run.get("sweep", {})):
      cmd = [
        args.bin,
        "--op", op,
        "--backend", backend,
        "--runtime", args.runtime,
        "--artifact-root", args.artifact_root,
        "--json",
        "--iters", str(iters),
        "--seed", str(seed),
      ]
      for k,v in params.items():
        cmd += [f"--{ARG_ALIASES.get(k, k)}", str(v)]
      t0 = time.time()
      env = os.environ.copy()
      env["OPBENCH_PYTHON"] = sys.executable
      python_path = str(SCRIPT_ROOT.parents[1] / "python")
      env["PYTHONPATH"] = python_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
      out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, env=env).strip()
      dur = time.time()-t0
      stats = _parse_output(out)
      if not stats:
        raise RuntimeError(f"benchmark emitted no key=value stats: {out!r}")
      row = _flatten_row(op, backend, iters, params, stats, dur)
      rows.append(row)
      print(row)
  csv_path = os.path.join(args.out, "results.csv")
  if not rows:
    raise RuntimeError("no benchmark rows were produced")
  with open(csv_path,"w",newline="") as f:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader(); w.writerows(rows)
  json_path = os.path.join(args.out, "results.json")
  with open(json_path, "w") as f:
    json.dump({
      "schema": "tessera.operator_bench.v1",
      "rows": rows,
      "telemetry": [row["telemetry"] for row in rows if isinstance(row.get("telemetry"), dict)],
      "telemetry_summary": _telemetry_summary([row["telemetry"] for row in rows if isinstance(row.get("telemetry"), dict)]),
    }, f, indent=2)
  print("Wrote", csv_path)
  print("Wrote", json_path)


def _parse_output(out):
  for line in reversed(out.splitlines()):
    line = line.strip()
    if not line:
      continue
    if line.startswith("{"):
      try:
        return json.loads(line)
      except json.JSONDecodeError:
        pass
  return dict(item.split("=", 1) for item in out.split() if "=" in item)


def _flatten_row(op, backend, iters, params, stats, dur):
  if "operator" not in stats:
    return {"op": op, "backend": backend, "iters": iters, **params, **stats, "wall_s": f"{dur:.3f}"}
  metrics = stats.get("metrics", {})
  profile = stats.get("profile", {})
  correctness = stats.get("correctness", {})
  artifact = stats.get("artifact_levels", {})
  operator = stats.get("operator", {})
  return {
    "op": operator.get("name", op),
    "backend": backend,
    "iters": iters,
    **params,
    "compiler_path": stats.get("compiler_path"),
    "runtime_status": stats.get("runtime_status"),
    "reason": stats.get("reason", ""),
    "avg_ms": profile.get("cpu_wall_ms"),
    "gflops": metrics.get("gflops"),
    "gbps": metrics.get("gbps"),
    "l2_ref": correctness.get("max_error"),
    "artifact_graph": artifact.get("graph"),
    "artifact_schedule": artifact.get("schedule"),
    "artifact_tile": artifact.get("tile"),
    "artifact_target": artifact.get("target"),
    "artifact_hash": artifact.get("artifact_hash"),
    "artifact_graph_hash": artifact.get("graph_hash"),
    "artifact_schedule_hash": artifact.get("schedule_hash"),
    "artifact_tile_hash": artifact.get("tile_hash"),
    "artifact_target_hash": artifact.get("target_hash"),
    "telemetry": stats.get("telemetry"),
    "wall_s": f"{dur:.3f}",
  }


def _telemetry_summary(events):
  counts = {}
  for event in events:
    label = str(event.get("bottleneck") or "unknown")
    counts[label] = counts.get(label, 0) + 1
  return {"schema": "tessera.telemetry.v1", "event_count": len(events), "bottlenecks": counts}

if __name__=="__main__":
  main()
