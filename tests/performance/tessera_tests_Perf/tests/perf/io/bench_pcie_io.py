import time, json, argparse, torch
from pathlib import Path

def time_copy_h2d(nbytes, pinned=True, iters=50, warmup=10):
    elem = nbytes // 4
    cpu = torch.empty(elem, dtype=torch.float32, pin_memory=pinned)
    gpu = torch.empty(elem, dtype=torch.float32, device='cuda')
    stream = torch.cuda.default_stream()
    # warmup
    for _ in range(warmup):
        gpu.copy_(cpu, non_blocking=True)
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record(stream)
        gpu.copy_(cpu, non_blocking=True)
        end.record(stream)
        end.synchronize()
        ms = start.elapsed_time(end)
        times.append(ms/1000.0)
    times.sort()
    p50 = times[len(times)//2]; p95 = times[int(len(times)*0.95)]
    gbps_p50 = (nbytes / p50) / 1e9
    gbps_p95 = (nbytes / p95) / 1e9
    return {"latency_p50_s": p50, "latency_p95_s": p95, "gbps_p50": gbps_p50, "gbps_p95": gbps_p95}

def time_copy_d2h(nbytes, pinned=True, iters=50, warmup=10):
    elem = nbytes // 4
    cpu = torch.empty(elem, dtype=torch.float32, pin_memory=pinned)
    gpu = torch.empty(elem, dtype=torch.float32, device='cuda')
    stream = torch.cuda.default_stream()
    for _ in range(warmup):
        cpu.copy_(gpu, non_blocking=True)
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record(stream)
        cpu.copy_(gpu, non_blocking=True)
        end.record(stream)
        end.synchronize()
        ms = start.elapsed_time(end)
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
    ap.add_argument("--out_csv", type=str, default="runs/pcie_io.csv")
    ap.add_argument("--out_jsonl", type=str, default="runs/pcie_io.jsonl")
    args = ap.parse_args()

    Path("runs").mkdir(parents=True, exist_ok=True)
    sizes = [int(s)*1024*1024 for s in args.sizes_mb.split(",") if s]
    import csv
    with open(args.out_csv, "w", newline="") as fcsv, open(args.out_jsonl, "a") as fj:
        wr = csv.writer(fcsv)
        wr.writerow(["direction","pinned","bytes","latency_p50_s","latency_p95_s","gbps_p50","gbps_p95"])
        for n in sizes:
            for pinned in (False, True):
                res = time_copy_h2d(n, pinned, args.iters, args.warmup)
                wr.writerow(["H2D", int(pinned), n, res["latency_p50_s"], res["latency_p95_s"], res["gbps_p50"], res["gbps_p95"]])
                fj.write(json.dumps({"direction":"H2D","pinned":int(pinned),"bytes":n, **res})+"\n")
                res = time_copy_d2h(n, pinned, args.iters, args.warmup)
                wr.writerow(["D2H", int(pinned), n, res["latency_p50_s"], res["latency_p95_s"], res["gbps_p50"], res["gbps_p95"]])
                fj.write(json.dumps({"direction":"D2H","pinned":int(pinned),"bytes":n, **res})+"\n")
                print(("pinned" if pinned else "pageable"), n, "OK")
