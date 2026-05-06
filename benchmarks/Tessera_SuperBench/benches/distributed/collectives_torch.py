#!/usr/bin/env python3
import argparse, json, os, pathlib, sys, time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

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

def run_tessera_facade(world_size, iters, num_bytes):
    try:
        import tessera as ts
    except Exception as exc:
        print(json.dumps({"ok": False, "skip_reason": f"tessera import failed: {exc}"}))
        return

    adapter = ts.collectives.adapter(backend="mock", world_size=world_size, mesh_axes={"dp": world_size})
    values = [np.ones(num_bytes // 4, dtype=np.float32) for _ in range(world_size)]
    for _ in range(2):
        adapter.all_reduce(values)
    start = time.perf_counter()
    for _ in range(iters):
        adapter.all_reduce(values)
    dur = time.perf_counter() - start
    status = adapter.status().to_dict()
    latency_ms = (dur / max(iters, 1)) * 1000.0
    bandwidth = (num_bytes * world_size * iters) / max(dur, 1e-9)
    print(json.dumps({
        "ok": True,
        "operator": "all_reduce",
        "dtype": "f32",
        "shape": f"{world_size}x{num_bytes}",
        "target": "cpu",
        "compiler_path": "runtime_facade",
        "runtime_status": status["status"],
        "reason": status["reason"],
        "latency_ms": latency_ms,
        "bandwidth_Bps": bandwidth,
        "telemetry": {
            "schema": "tessera.telemetry.v1",
            "name": "all_reduce",
            "source": "tessera_superbench",
            "op": "all_reduce",
            "dtype": "f32",
            "arch": "cpu",
            "latency_ms": latency_ms,
            "bandwidth_gbps": bandwidth / 1.0e9,
            "status": "ok",
            "counters": {},
            "metadata": {"backend": "mock", "world_size": world_size, "bytes": num_bytes},
            "bottleneck": "collective_or_overlap",
        },
    }))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world_size", type=int, default=2)
    ap.add_argument("--backend", type=str, default="tessera_mock", choices=["tessera_mock","nccl","gloo"])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--bytes", type=int, default=1<<26)
    args = ap.parse_args()

    if args.backend == "tessera_mock":
        run_tessera_facade(args.world_size, args.iters, args.bytes)
        return

    try:
        import torch, torch.multiprocessing as mp
    except Exception as e:
        print(json.dumps({"ok": False, "skip_reason": f"torch not available: {e}"}))
        return
    if args.backend == "nccl" and not torch.cuda.is_available():
        print(json.dumps({"ok": False, "skip_reason": "CUDA not available for NCCL"}))
        return

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    ctx = mp.get_context("spawn")
    procs = []
    for r in range(args.world_size):
        p = ctx.Process(target=worker, args=(r,args.world_size,args.backend,args.iters,args.bytes,"cuda"))
        p.start()
        procs.append(p)
    # Collect a single JSON line from rank 0 via stdout — for simplicity run with rank0 in same process:
    failed = False
    for p in procs:
        p.join()
        failed = failed or (p.exitcode != 0)
    if failed:
        print(json.dumps({"ok": False, "skip_reason": "one or more distributed workers failed"}))

if __name__ == "__main__":
    main()
