# PTX Tensor-Core GEMM Study (sm_120 / SuperBear)

A corrected, sm_120-retargeted reproduction + extension of Borowski & Osinski,
*"Hand-Written PTX Tensor-Core GEMM Kernels: A Multi-Precision Study on NVIDIA
L4"* (arXiv:2608.10103). See [`PLAN.md`](PLAN.md) for the full rationale and the
defect-by-defect correction table.

## TL;DR — run on SuperBear
```bash
ssh super-bear                          # port 5023
cd <tessera>/benchmarks/nvidia/ptx_gemm_study
bash run.sh                             # Phase 0 → 1 → 2 → 3 → 4, gated
```
Outputs: `capability_matrix.json`, `results.jsonl`, `counters.jsonl`, `REPORT.md`.

## What's different from the paper
- **sm_120, not L4/Ada.** No tcgen05/wgmma — `mma.sync`+`cp.async`+`ldmatrix` is
  the right family. The paper's INT4 win depends on native `mma.sync.m16n8k64.s4`,
  which was *removed* on Hopper and is flagged "correctness-first" in Tessera's
  own sm_120 model — **so Phase 0 decides whether it executes here.** That is the
  first thing `run.sh` prints.
- **Every paper defect is a structural correction** (PLAN.md §2): one source of
  truth for durations (no Table-V contradiction), timing separated from profiling,
  same-precision speedup as the primary metric (not the L2-cliff-inflated "vs
  FP16" headline), auto-generated numbers (no "GFLOPS roughly halve"), theoretical
  vs measured AI kept separate, and a correctness gate that disqualifies a fast-
  but-wrong kernel before it is timed.

## Files
| File | Role |
|---|---|
| `probe.cu` | **Phase 0** capability + numeric probe. Decides native `s4` on sm_120. |
| `bench.cu` | **Phase 1+2** correctness-then-timing. CUDA events only. → `results.jsonl` |
| `profile.sh` | **Phase 3** targeted `ncu` counters (separate pass). → `counters.jsonl` |
| `check_consistency.py` | **Phase 4** gate — the anti-Table-V guard. `--selftest` runs offline. |
| `analyze.py` | **Phase 4** report generator (all numbers derived). → `REPORT.md` |
| `run.sh` | end-to-end, INT4-first ordering, gate blocks the report |
| `Makefile` | `-arch=sm_120a`; `make NO_INT4=1` if the s4 MMA is compile-rejected |

## Prerequisites (WSL2)
1. **ncu counters** are off by default → enable on the Windows host (NVIDIA
   Control Panel → Developer → Manage GPU Performance Counters → *Allow access to
   all users*), then `wsl --shutdown`. Verify: `ncu --query-metrics | head`.
2. **Clock locking** usually fails under WSL2; `run.sh` falls back to the CoV gate
   and records `clocks_locked:false` (honest, per PLAN §C7).
3. **Arch:** `make archcheck` proves the toolchain accepts `sm_120a` first.
4. **Metric names drift** across ncu versions — resolve the §7 list against
   `ncu --query-metrics` before trusting `counters.jsonl`.

## Honesty notes (claim integrity)
- The hand-PTX kernels in `bench.cu` are **reference skeletons**; their fragment
  maps are validated on-box by the in-binary correctness check. A kernel that
  fails emits `status:WRONG`/`EXEC_FAIL` and is **not** timed — the study cannot
  report a number for an unvalidated kernel.
- No sm_120 result is asserted for a variant that failed Phase 0.
- A red consistency gate blocks `REPORT.md`.

## Feeding Tessera (Phase 5, after a clean run)
Register measured `dram_bw_gbps`/`llc_bytes`/peak-TOPS into `target_perf.py` with
`Provenance.MEASURED`; add the empirical per-dtype L2-ridge N as the arbiter's
regime feature; record best (K-tile, pipeline depth) per regime into
`mma_selector.py`; turn the DRAM-cycle≈wall-time and coalescing invariants into
Evaluator oracles; and promote each Phase-0-proven precision in `capabilities.py`
from `artifact_only` toward proven-execution with this run as the evidence row.
