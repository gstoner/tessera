#!/usr/bin/env python3
import argparse, subprocess, yaml, csv, os, itertools, time

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
        cmd += [f"--{k}", str(v)]
      t0 = time.time()
      out = subprocess.check_output(cmd, text=True).strip()
      dur = time.time()-t0
      # parse "avg_ms=... gflops=... gbps=... l2_ref=..."
      stats = dict(item.split("=") for item in out.split())
      row = {"op": op, "iters": iters, **params, **stats, "wall_s": f"{dur:.3f}"}
      rows.append(row)
      print(row)
  csv_path = os.path.join(args.out, "results.csv")
  with open(csv_path,"w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
  print("Wrote", csv_path)

if __name__=="__main__":
  main()
