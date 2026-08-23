#!/usr/bin/env python3
"""parse_ncu.py — turn one ncu CSV (single N) into counters.jsonl rows.

Called by profile.sh once per size, so N is passed on the command line rather
than parsed from a demangled kernel name. Maps the CUDA function symbol to the
study kernel name (and dtype) used in results.jsonl, so the (kernel, dtype, N)
key matches for the check_consistency cross-check.

Usage: parse_ncu.py <ncu_raw.csv> <N>
"""
from __future__ import annotations
import csv
import json
import sys

# CUDA function symbol substring -> (study kernel name, dtype). Extend as bench.cu
# grows (int8_ptx_mma_k32, int4_wmma, cublasLt, ...).
SYMBOL_MAP = [
    ("wmma_fp16_int4_emulated", ("int4_wmma_preexpanded_fp16", "int4")),
    ("wmma_fp16",       ("fp16_wmma", "fp16")),
    ("int4_native_k64_3stage", ("int4_ptx_3stage", "int4")),
    ("int4_native_k64", ("int4_ptx_mma_k64", "int4")),
    ("int8_native_k32", ("int8_ptx_mma_k32", "int8")),
    ("nvjet_sm120_bii", ("cublaslt_int8", "int8")),
    # cuBLASLt's internal kernel names are toolkit-specific.  This signature is
    # emitted by CUDA 13.3 on SM120; retain it as an explicit parser contract.
    ("nvjet_sm120",     ("cublaslt_fp16", "fp16")),
    ("cutlass_80_tensorop_s16816gemm_f16", ("cublaslt_fp16", "fp16")),
]

KEYMAP = {
    "dram__cycles_active.avg": "dram_cycles_active",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct",
    "lts__t_sector_hit_rate.pct": "l2_hit_pct",
    "l1tex__t_sector_hit_rate.pct": "l1tex_hit_pct",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio": "sectors_per_req",
    "smsp__thread_inst_executed_per_inst_executed.ratio": "active_threads_per_warp",
    "smsp__sass_branch_targets_threads_divergent.sum": "divergent_branches",
    "smsp__inst_executed.sum": "inst_executed",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy_pct",
    "gpu__time_duration.sum": "ncu_duration_ms",
}


def classify(kernel_symbol: str):
    for sub, (name, dt) in SYMBOL_MAP:
        if sub in kernel_symbol:
            return name, dt
    return kernel_symbol, None  # unknown — keep raw, dtype unknown


def main() -> int:
    raw, n = sys.argv[1], int(sys.argv[2])
    with open(raw) as f:
        lines = [l for l in f if l.strip()]
    try:
        hdr_i = next(i for i, l in enumerate(lines) if "Metric Name" in l)
    except StopIteration:
        print(f"ERROR: no header in {raw}", file=sys.stderr)
        return 1
    rows: dict[str, dict] = {}
    for r in csv.DictReader(lines[hdr_i:]):
        sym = r.get("Kernel Name", "")
        m, v = r.get("Metric Name", ""), r.get("Metric Value", "")
        name, dt = classify(sym)
        key = name
        rows.setdefault(key, {"kernel": name, "dtype": dt, "n": n,
                              "profiling_scope": "main_kernel"})
        if m in KEYMAP:
            try:
                val = float(str(v).replace(",", ""))
            except ValueError:
                val = v
            if KEYMAP[m] == "ncu_duration_ms" and isinstance(val, float):
                val = val / 1e6  # ns -> ms (confirm unit on-box via --query-metrics)
            rows[key][KEYMAP[m]] = val
    for v in rows.values():
        print(json.dumps(v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
