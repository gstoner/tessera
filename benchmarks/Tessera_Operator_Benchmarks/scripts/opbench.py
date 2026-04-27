#!/usr/bin/env python3
import argparse, csv, itertools, os, subprocess, time
try:
  import yaml
except ImportError as exc:
  raise SystemExit("PyYAML is required for benchmark configs. Install project dependencies with `python3 -m pip install -r requirements.txt`.") from exc

ARG_ALIASES = {
  "M": "m",
  "N": "n",
  "K": "k",
}

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
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  with open(args.config) as f:
    cfg = yaml.safe_load(f)
  seed = cfg.get("seed", 123)
  rows=[]
  for run in cfg["runs"]:
    op = run["op"]
    iters = run.get("iters", 50)
    for params in grid(run.get("sweep", {})):
      cmd = [args.bin, "--op", op, "--iters", str(iters), "--seed", str(seed)]
      for k,v in params.items():
        cmd += [f"--{ARG_ALIASES.get(k, k)}", str(v)]
      t0 = time.time()
      out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
      dur = time.time()-t0
      # parse "avg_ms=... gflops=... gbps=... l2_ref=..."
      stats = dict(item.split("=", 1) for item in out.split() if "=" in item)
      if not stats:
        raise RuntimeError(f"benchmark emitted no key=value stats: {out!r}")
      row = {"op": op, "iters": iters, **params, **stats, "wall_s": f"{dur:.3f}"}
      rows.append(row)
      print(row)
  csv_path = os.path.join(args.out, "results.csv")
  if not rows:
    raise RuntimeError("no benchmark rows were produced")
  with open(csv_path,"w",newline="") as f:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader(); w.writerows(rows)
  print("Wrote", csv_path)

if __name__=="__main__":
  main()
