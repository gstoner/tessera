
import argparse, json, random, math, subprocess, tempfile, os
from pathlib import Path
from sched_cache import ensure_db, insert_row

def run_microbench(exe, M,N,K, cfg, copy_path, compute_path):
  with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as tf:
    outp = tf.name
  cmd = [
    exe,
    "--M", str(M), "--N", str(N), "--K", str(K),
    "--block_m", str(cfg["block_m"]),
    "--block_n", str(cfg["block_n"]),
    "--block_k", str(cfg["block_k"]),
    "--stages", str(cfg["stages"]),
    "--split_k", str(cfg["split_k"]),
    "--cta_pairs", str(cfg["cta_pairs"]),
    "--swizzle", cfg.get("swizzle","identity"),
    "--tma_cols_per_copy", str(cfg.get("tma_cols_per_copy",0)),
    "--copy_path", copy_path,
    "--compute_path", compute_path,
    "--json_out", outp,
  ]
  subprocess.run(cmd, check=True)
  data = json.load(open(outp))
  os.remove(outp)
  return float(data["ms_avg"]), data

def sample():
  return {
    "block_m": random.choice([64,128,256]),
    "block_n": random.choice([64,128,256]),
    "block_k": random.choice([16,32,64]),
    "stages": random.choice([2,3]),
    "split_k": random.choice([1,2,4]),
    "cta_pairs": random.choice([0,1]),
    "swizzle": random.choice(["identity","xor128b"]),
    "tma_cols_per_copy": random.choice([0,32,64,128]),
  }

def hyperband(M,N,K, exe, copy_path, compute_path, max_iters=18, eta=3):
  smax = int(math.log(max_iters, eta))
  best = (1e9, None, None)
  for s in reversed(range(smax+1)):
    n = int((smax+1)/(s+1) * (eta**s))
    r = max_iters * (eta**(-s))
    configs = [sample() for _ in range(n)]
    scores = []
    for cfg in configs:
      ms, data = run_microbench(exe, M,N,K, cfg, copy_path, compute_path)
      scores.append((ms, cfg, data))
    scores.sort(key=lambda x: x[0])
    if scores[0][0] < best[0]:
      best = (scores[0][0], scores[0][1], scores[0][2])
  return best

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--db", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--arch", default="sm90")
  ap.add_argument("--dtype", default="bf16")
  ap.add_argument("--op", default="gemm")
  ap.add_argument("--shape", default="M=4096,N=4096,K=4096")
  ap.add_argument("--exe", required=True, help="Path to tessera_microbench")
  ap.add_argument("--copy_path", default="cp.async", choices=["cp.async", "tma"])
  ap.add_argument("--compute_path", default="wmma", choices=["wmma", "wgmma"])
  args = ap.parse_args()

  shape_parts = dict(x.split("=") for x in args.shape.split(","))
  M,N,K = int(shape_parts["M"]), int(shape_parts["N"]), int(shape_parts["K"])

  best_ms, best_cfg, bench_json = hyperband(M,N,K, args.exe, args.copy_path, args.compute_path)
  conn = ensure_db(args.db)
  insert_row(conn, args.arch, args.dtype, args.op, args.shape, best_cfg, best_ms)
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  with open(args.out, "w") as f:
    json.dump({"best_ms": best_ms, "knobs": best_cfg, "bench": bench_json, "copy_path": args.copy_path, "compute_path": args.compute_path}, f, indent=2)
  print(json.dumps({"best_ms": best_ms, "knobs": best_cfg}))
