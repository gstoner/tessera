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
| `profile.sh` | **Phase 3** targeted `ncu`, one pass per N (separate from timing). → `counters.jsonl` |
| `profile_library_floor.sh` | Dedicated cuBLASLt protocol: public-call CUDA-event and selected-internal-kernel NCU timings, explicitly separate scopes. |
| `parse_ncu.py` | maps each ncu CSV → JSONL keyed by (kernel, dtype, N), matching `results.jsonl` |
| `check_consistency.py` | **Phase 4** gate — the anti-Table-V guard. `--selftest` runs offline. |
| `analyze.py` | **Phase 4** report generator (all numbers derived). → `REPORT.md` |
| `record_selector_decision.py` | Records a study-local native INT4 observation; it never edits production selector policy. |
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
- CUDA-event durations for a hand-written kernel and NCU's same-kernel duration
  are compared strictly.  A cuBLASLt CUDA event instead surrounds its public
  library call while NCU observes one selected internal kernel, so those rows
  carry `timing_scope: library_call` and are retained as separate evidence.
  Reproduce either floor with `bash profile_library_floor.sh fp16` or `int8`.

## Feeding Tessera (Phase 5, after a clean run)
`phase5_ingest.py` produces a proposal only after a complete green packet.  WSL2
packets remain `selector_eligible:false`; reproduce the same protocol bare metal
before changing `target_perf.py`, `mma_selector.py`, or `capabilities.py`.
