#!/usr/bin/env python3
"""Scheduled GEMM/attention adapter and a single-process mixed capture workload."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import uuid
ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT),str(ROOT/'python')]
from benchmarks.native_matrix_adapter import prepare  # noqa: E402
from benchmarks.profile_ranges import Ranges  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', required=True, choices=['nvidia','rocm'])
    p.add_argument('--workload', choices=['gemm','attention','mixed'], default='mixed')
    p.add_argument('--repeat',type=int,default=3)
    p.add_argument('--nvtx',action='store_true')
    args = p.parse_args()
    if args.repeat < 1 or (args.nvtx and args.backend != 'nvidia'):
        p.error('positive repeat required; NVTX supported only on NVIDIA')
    run_id = uuid.uuid4().hex
    workloads = ['gemm','attention'] if args.workload == 'mixed' else [args.workload]
    prepared = [prepare(args.backend,w) for w in workloads]
    # Compilation and initial warmup intentionally precede the measured ranges.
    for run, _ in prepared:
        run()
    ranges = Ranges() if args.nvtx else None
    rows = []
    try:
        for i in range(args.repeat):
            for run, identity in prepared:
                label = f'tessera:{run_id}:{identity["artifact"]}:{identity["image"]}:{i}'
                start = time.perf_counter_ns()
                with ranges.mark(label) if ranges else nullcontext():
                    error = run()
                rows.append(dict(**identity,label=label,max_abs_err=error,latency_ms=(time.perf_counter_ns()-start)/1e6))
    finally:
        if ranges:
            ranges.close()
    compiler = Path(os.environ['TESSERA_OPT'])
    print(json.dumps(dict(schema=1,ok=True,backend=args.backend,process_id=os.getpid(),run_id=run_id,
        runtime_status='executable',execution_kind='native_gpu',promotion_eligible=False,
        clock='host_wall_checked_package_and_oracle',nvtx=args.nvtx,rows=rows,
        latency_ms=statistics.median(r['latency_ms'] for r in rows),
        compiler_sha256=hashlib.sha256(compiler.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
