
#!/usr/bin/env python3
import argparse, os, time, json, sys

def worker(rank, world_size, backend, iters, num_bytes, device):
    import torch, torch.distributed as dist
    if backend == "nccl" and not torch.cuda.is_available():
        print(json.dumps({"ok": False, "skip_reason": "CUDA not available"}))
        return
    dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
    if backend == "nccl":
        torch.cuda.set_device(rank % torch.cuda.device_count())
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    t = torch.empty(num_bytes//4, dtype=torch.float32, device=dev).fill_(1.0)

    # Warmup
    for _ in range(5):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        if backend == "nccl": torch.cuda.synchronize()

    # Timed
    start = time.time()
    for _ in range(iters):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        if backend == "nccl":
            torch.cuda.synchronize()
    dur = time.time() - start

    # One-way bandwidth per rank approx: (bytes * iters) / dur
    bw = (num_bytes * iters) / max(dur, 1e-9)
    if rank == 0:
        print(json.dumps({"ok": True, "latency_ms": (dur/iters)*1000.0, "bandwidth_Bps": bw}))
    dist.destroy_process_group()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world_size", type=int, default=2)
    ap.add_argument("--backend", type=str, default="nccl", choices=["nccl","gloo"])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--bytes", type=int, default=1<<26)
    args = ap.parse_args()

    try:
        import torch, torch.multiprocessing as mp
    except Exception as e:
        print(json.dumps({"ok": False, "skip_reason": f"torch not available: {e}"}))
        return

    ctx = mp.get_context("spawn")
    procs = []
    for r in range(args.world_size):
        p = ctx.Process(target=worker, args=(r,args.world_size,args.backend,args.iters,args.bytes,"cuda"))
        p.start()
        procs.append(p)
    # Collect a single JSON line from rank 0 via stdout — for simplicity run with rank0 in same process:
    for p in procs: p.join()

if __name__ == "__main__":
    main()
