import os, time, csv, json, argparse, torch, torch.distributed as dist
from pathlib import Path

def barrier():
    if dist.is_initialized():
        dist.barrier()

def setup():
    if dist.is_initialized(): return
    backend = "nccl"
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method=os.environ.get("INIT_METHOD","env://"))
    return rank, world_size, local_rank

def bench_op(buf, op, iters=50, warmup=10):
    times = []
    stream = torch.cuda.current_stream()
    for _ in range(warmup):
        if op=="allreduce":
            dist.all_reduce(buf, op=dist.ReduceOp.SUM, async_op=False)
        elif op=="allgather":
            out = [torch.empty_like(buf) for _ in range(dist.get_world_size())]
            dist.all_gather(out, buf, async_op=False)
        elif op=="broadcast":
            dist.broadcast(buf, src=0, async_op=False)
    torch.cuda.synchronize()
    for _ in range(iters):
        t0 = time.perf_counter()
        if op=="allreduce":
            dist.all_reduce(buf, op=dist.ReduceOp.SUM, async_op=False)
        elif op=="allgather":
            out = [torch.empty_like(buf) for _ in range(dist.get_world_size())]
            dist.all_gather(out, buf, async_op=False)
        elif op=="broadcast":
            dist.broadcast(buf, src=0, async_op=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter()-t0)
    times.sort()
    return times[len(times)//2], times[int(len(times)*0.95)]

def eff_bw_allreduce(bytes_per_rank, world_size, latency_s):
    # Approx ring allreduce payload: 2 * (world_size - 1)/world_size * bytes per rank
    payload = 2.0 * (world_size - 1.0) / world_size * bytes_per_rank
    return (payload / latency_s) / 1e9  # GB/s

def eff_bw_allgather(bytes_per_rank, world_size, latency_s):
    payload = bytes_per_rank * (world_size - 1.0)
    return (payload / latency_s) / 1e9

def eff_bw_broadcast(bytes_per_rank, world_size, latency_s):
    payload = bytes_per_rank * (world_size - 1.0)
    return (payload / latency_s) / 1e9

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes_mb", type=str, default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--dtype", type=str, default="float32")
    ap.add_argument("--out_dir", type=str, default="runs")
    args = ap.parse_args()

    rank, world, local_rank = setup()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    dtype = dict(float32=torch.float32, float16=torch.float16, bfloat16=torch.bfloat16)[args.dtype]
    sizes = [int(x)*1024*1024 for x in args.sizes_mb.split(",") if x]

    results = []
    for n in sizes:
        elems = n // torch.tensor([], dtype=dtype).element_size()
        buf = torch.randn(int(elems.item()), device="cuda", dtype=dtype)
        for op in ["allreduce","allgather","broadcast"]:
            p50, p95 = bench_op(buf, op, args.iters, args.warmup)
            if op=="allreduce": bw = eff_bw_allreduce(n, world, p50)
            elif op=="allgather": bw = eff_bw_allgather(n, world, p50)
            else: bw = eff_bw_broadcast(n, world, p50)
            results.append({"op":op,"bytes":n,"p50_s":p50,"p95_s":p95,"gbps_eff":bw})
        barrier()

    if rank == 0:
        # write CSV/JSONL
        import csv
        csv_path = Path(args.out_dir)/"nccl_collectives.csv"
        jsonl_path = Path(args.out_dir)/"nccl_collectives.jsonl"
        with open(csv_path, "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(["op","bytes","p50_s","p95_s","gbps_eff","world"])
            for r in results: wr.writerow([r["op"], r["bytes"], r["p50_s"], r["p95_s"], r["gbps_eff"], world])
        with open(jsonl_path,"a") as fj:
            for r in results: fj.write(json.dumps({**r,"world":world})+"\n")
        print("Wrote", csv_path, jsonl_path)
