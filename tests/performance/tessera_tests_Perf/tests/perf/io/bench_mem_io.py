import time, json, argparse, torch
from pathlib import Path

def time_copy_bytes(nbytes, iters=50, warmup=10):
    elem = nbytes // 4  # float32 elements
    x = torch.empty(elem, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    stream = torch.cuda.default_stream()
    # warmup
    for _ in range(warmup):
        y.copy_(x, non_blocking=True)
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record(stream)
        y.copy_(x, non_blocking=True)
        end.record(stream)
        end.synchronize()
        ms = start.elapsed_time(end)  # ms
        times.append(ms/1000.0)
    times.sort()
    p50 = times[len(times)//2]; p95 = times[int(len(times)*0.95)]
    gbps_p50 = (nbytes / p50) / 1e9
    gbps_p95 = (nbytes / p95) / 1e9
    return {"latency_p50_s": p50, "latency_p95_s": p95, "gbps_p50": gbps_p50, "gbps_p95": gbps_p95}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes_mb", type=str, default="64,128,256,512,1024")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out_csv", type=str, default="runs/mem_io.csv")
    ap.add_argument("--out_jsonl", type=str, default="runs/mem_io.jsonl")
    args = ap.parse_args()

    Path("runs").mkdir(parents=True, exist_ok=True)
    sizes = [int(s)*1024*1024 for s in args.sizes_mb.split(",") if s]
    import csv
    with open(args.out_csv, "w", newline="") as fcsv, open(args.out_jsonl, "a") as fj:
        wr = csv.writer(fcsv)
        wr.writerow(["bytes","latency_p50_s","latency_p95_s","gbps_p50","gbps_p95"])
        for n in sizes:
            res = time_copy_bytes(n, args.iters, args.warmup)
            wr.writerow([n, res["latency_p50_s"], res["latency_p95_s"], res["gbps_p50"], res["gbps_p95"]])
            fj.write(json.dumps({"bytes":n, **res})+"\n")
            print(n, res)
