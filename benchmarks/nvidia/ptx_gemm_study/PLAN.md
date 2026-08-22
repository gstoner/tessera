# Corrected PTX Tensor-Core GEMM Study — SuperBear (sm_120)

**Status:** plan / not yet run. **Box:** SuperBear (`ssh super-bear`, port 5023) —
Threadripper + RTX 5070 (Blackwell **sm_120**, CC 12.0), Ubuntu 26.04 under WSL2.
**Origin:** a corrected, sm_120-retargeted reproduction + extension of Borowski &
Osinski, *"Hand-Written PTX Tensor-Core GEMM Kernels: A Multi-Precision Study on
NVIDIA L4"* (arXiv:2608.10103), built to feed Tessera's Decision #28 arbiter /
`target_perf` roofline / Evaluator.

This plan encodes every methodological defect found in the review of that paper as
a structural correction (§2), and treats every sm_120 capability as **unproven
until the probe in Phase 0 says otherwise** (claim integrity).

---

## 1. Why this is not just "rerun the paper"

The paper is **L4 / Ada / SM89**. SuperBear is **sm_120 / consumer Blackwell**, a
different architecture, so results do not transfer and two things change the study
substantively:

- **The pivotal INT4 question is open.** The paper's largest win (2.9–4.3×) is
  native `mma.sync.m16n8k64.s4` beating WMMA's *software-emulated* s4 path. On
  Hopper (sm_90) the integer `.s4`/`.u4` MMA was **removed**. Tessera's own model
  (`capabilities.py`) lists sm_120 INT4 matmul but flags it "**correctness-first
  signed-int4**" and `artifact_only` — i.e. it may itself be an emulated/compiler
  path. **Whether native `s4` MMA executes on sm_120 is the thing Phase 0 must
  decide.** We do not assume the L4 result ports.
- **The interesting precision is FP4, not INT4.** sm_120 exposes `nvfp4` /
  `fp4_e2m1` (and MX block-scale) via `mma.sync.block_scale` — Tessera's frontier
  and the natural "beyond the paper" contribution. There is **no tcgen05 / wgmma /
  TMEM** on sm_120, so the paper's `mma.sync` + `cp.async` + `ldmatrix` family is
  exactly the right instruction set (unlike Hopper, where it would be wrong).

So: same methodology, new hardware truth, extended precision set, and the whole
thing wired to produce Tessera cost-model evidence rather than a standalone paper.

---

## 2. Corrections baked in (each maps to a defect in the review)

| # | Paper defect | Structural correction in this study |
|---|---|---|
| C1 | **Table V (FP16 durations) irreconcilable with every other table** — small-N flat, up to ~3700× off, likely `ncu`-instrumented time mislabeled as wall-clock. | **One source of truth.** All durations come from a single clean-timing pass into `results.jsonl`. Every table/plot/prose number is *derived* from it by the report generator — there is no second table that can disagree. Profiling time is never a duration. |
| C2 | **Timing and profiling conflated** (`ncu --set full` replay inflates small-N). | **Two disjoint passes** (Phase 2 timing = CUDA events; Phase 3 counters = `ncu`). Neither borrows the other's numbers. `ncu` durations are used only to *cross-check* event timing, never to report speedup. |
| C3 | **Headline 34.4×/98.7× "vs FP16" rests on one L2-cliff datapoint** (fp16 = 16500 ms, a 161× jump the authors call variance-amplified). | **Same-precision speedup is the primary metric**, with CIs. Cross-precision is shown only as an absolute-TOPS-vs-N chart with the per-precision L2-overflow N annotated — never a single headline ratio. |
| C4 | **Prose numbers wrong/loose** ("GFLOPS roughly halve" ≈ actually 20×; "+22%" unsupported by its own table). | **No hand-typed numbers.** The report is generated from `results.jsonl`; any number in prose is a template field. |
| C5 | **AI absolute basis unreconstructable / conflated.** | Report **both** `ai_theoretical = 2N³ / model_bytes` and `ai_measured = 2N³ / dram_bytes_ncu`, each labeled, never merged. |
| C6 | **No numerical validation** (paper explicitly skips it — a fast wrong kernel counts). | **Correctness gate before timing** (Phase 1): every kernel must match a reference within a per-dtype tolerance or it is disqualified, not benchmarked. |
| C7 | **Variance amplified at large N, unmanaged.** | Lock clocks if WSL2 permits; else raise iteration count and **gate on coefficient of variation** — any (kernel, N) with CoV above threshold is flagged in the output, not silently reported. |
| C8 | **WMMA-as-baseline overstates PTX value** (WMMA is not the library floor). | Add **cuBLASLt** as a candidate so the real library floor is visible; keep WMMA as a *second* reference, not *the* baseline. |
| — | (new) capability claims. | **Capability probe gates the whole matrix** (Phase 0): a candidate that does not assemble *and* execute *and* validate on sm_120 is dropped with a recorded reason (Decision #21, fail-closed). |

---

## 3. Prerequisites & WSL2 gotchas (do these first)

1. **Nsight Compute counters under WSL2.** `ncu` works on WSL2 but performance
   counters are **off by default** and fail with `ERR_NVGPUCTRPERM`. On the
   **Windows host**: NVIDIA Control Panel → Desktop → *Enable Developer Settings* →
   Developer → *Manage GPU Performance Counters* → **Allow access to all users**,
   then reboot WSL (`wsl --shutdown`). Verify inside WSL with
   `ncu --query-metrics | head` returning without permission error.
2. **Clock locking (C7).** Attempt `sudo nvidia-smi -lgc <clk>` and
   `nvidia-smi --lock-memory-clocks`. WSL2 frequently **rejects** these (no
   persistence/management). If it fails, do **not** fake stability — fall back to
   the CoV gate (§C7) and record `clocks_locked: false` in the run metadata.
3. **Toolkit / arch.** CUDA 13.3 toolkit is present (toolkit-only WSL2 CUDA,
   `/dev/dxg`, no driver package). Compile arch-specific: `-arch=sm_120a` (the
   block-scale FP4 path requires the `a` variant). Confirm with a 5-line
   `cudaGetDeviceProperties` probe that `major.minor == 12.0`.
4. **Metric-name drift.** The paper itself warns Nsight counter semantics shift
   across versions. **Resolve every metric name against `ncu --query-metrics` on
   this box** before the profiling pass; the names in §7 are candidates, not gospel.
5. **Isolation.** Nothing here touches ROCm/x86. Pure CUDA + Python stdlib +
   numpy. No PyTorch/JAX at runtime (Decision #23) — cuBLASLt is called through the
   C driver, not a framework.

---

## 4. Phase plan

### Phase 0 — Environment + capability probe (GATING)
Small CUDA program `probe.cu` that, for each candidate MMA variant, (a) compiles
for `sm_120a`, (b) launches a one-warp fragment test, (c) checks it runs without
`illegal instruction` (catch via `cudaGetLastError`), and (d) validates the tile
result against a CPU reference. Emits `capability_matrix.json`:

| Variant | PTX | must decide |
|---|---|---|
| FP16 | `mma.sync.m16n8k16.f32.f16.f16.f32` | expected OK |
| INT8 | `mma.sync.m16n8k32.s32.s8.s8.s32` | expected OK |
| **INT4** | `mma.sync.m16n8k64.s32.s4.s4.s32` | **PIVOTAL — native or absent/emulated?** |
| FP4 (nvfp4/mxfp4) | `mma.sync.block_scale …` (sm_120a) | extension target |
| BF16 | `mma.sync.m16n8k16.f32.bf16.bf16.f32` | expected OK |

Rule: a variant that fails any of (a)–(d) is **dropped from the study with the
recorded reason**, and any Tessera claim about that precision on sm_120 stays
`artifact_only`. This is the "prove native execution, don't assume it" step; it
also reconciles or corrects `capabilities.py`'s sm_120 INT4 entry with silicon.

### Phase 1 — Correctness oracle (C6)
For every (kernel, dtype, N) that survives Phase 0, validate output against a
reference **before** any timing:
- FP16/BF16: vs fp64 CPU GEMM, relative Frobenius error ≤ dtype tolerance.
- INT8/INT4: exact INT32 accumulation vs a CPU integer GEMM (bit-exact; quant
  kernels compute unscaled INT32 as in the paper — no in-kernel dequant).
- FP4: vs a reference emulating the block-scale semantics.
A kernel that fails is **disqualified, not slow** — it never enters Phase 2.

### Phase 2 — Clean timing rig (C1, C2, C7) → primary numbers
- CUDA events; `warmup` (≥20) then `iters` (≥200 small N, scaled down for the
  largest); report **median, P05, P95, CoV**.
- Square N ∈ {512,1024,2048,4096,8192} (+ optional 12288/16384 — sm_120 has less
  DRAM than L4, so the L2/DRAM cliff will land at a *different* N; find it, don't
  assume 8192).
- One row per (candidate, dtype, N) appended to `results.jsonl` — **the single
  source of truth** (C1). Conforms to the Decision #12 schema: `backend, op,
  shape, dtype, latency_ms, tflops, memory_bw_gbps, device, tessera_version` plus
  `p05_ms,p95_ms,cov,clocks_locked,commit`.

### Phase 3 — Profiling rig (C2, C5) → mechanism attribution
Separate `ncu` pass over a **targeted metric set** (§7), NOT `--set full`. Emit
`counters.jsonl`. Cross-check: `ncu` kernel duration must agree with Phase-2 event
median within a tolerance; a mismatch is reported, not hidden (this is the guard
that would have caught Table V).

### Phase 4 — Analysis + auto-generated report + consistency gate (C1–C5)
`analyze.py` reads *only* `results.jsonl` + `counters.jsonl` and:
- Emits same-precision speedup tables with CIs (primary) and the absolute-TOPS-vs-N
  quantization chart with per-precision cliff annotation (C3).
- Computes `ai_theoretical` and `ai_measured` separately (C5).
- **Internal-consistency gate `check_consistency.py` (the anti-Table-V guard):**
  fails CI if `tflops ≠ 2N³/latency` (±0.5%), if any reconstructed speedup ≠
  reported, if AI ratios across precisions ≠ byte ratios, or if any (kernel,N)
  duration appears with two different values. Report generation is blocked on green.

### Phase 5 — Feed Tessera (the payoff)
- **`target_perf.py`:** register SuperBear's measured `dram_bw_gbps`, `llc_bytes`
  (from `cudaGetDeviceProperties`), and per-dtype achieved peak TOPS with
  `Provenance.MEASURED`. Compute the empirical **L2-residency ridge N per dtype**
  and add it as the regime feature the review flagged missing.
- **Arbiter / `mma_selector.py`:** record measured `(dtype → best K-tile, pipeline
  depth)` per regime so K-tile choice (which the footprint model ignores) becomes a
  cost-model-backed candidate axis.
- **Evaluator:** register two oracles from the data — (i) DRAM-active-cycles ≈
  wall-time above the ridge; (ii) global-load coalescing (bytes/sector) predicts
  ranking. Both were near-1:1 in the paper and are cheap invariants to assert.
- **`capabilities.py`:** promote every Phase-0-proven sm_120 precision from
  `artifact_only` toward proven-execution with this run as the evidence row; leave
  unproven ones exactly where they are.

---

## 5. Candidate matrix (one variable at a time, per surviving precision)

Keep the paper's discipline. Per precision: WMMA reference, cuBLASLt reference
(C8), then PTX variants changing **one** axis: SRAM→reg loader (`ldmatrix` width
vs scalar pack), MMA K-tile (k8/k16/k32/k64 as the dtype allows), accumulator
type, pipeline depth (2- vs 3-stage), `cp.async` policy (`.ca`/`.cg`), B layout
(non-transposed vs transposed — expected to collapse coalescing). Add Tessera's
own emitted candidate (`emit/nvidia_cuda.py`) as a peer so the arbiter sees
compiled-vs-hand-vs-library side by side.

## 6. (moved into §4)

## 7. Nsight metric set (confirm names via `ncu --query-metrics`)

| Signal | Candidate metric |
|---|---|
| Duration (cross-check only) | `gpu__time_duration.sum` |
| DRAM active cycles | `dram__cycles_active.avg` |
| DRAM throughput % | `dram__throughput.avg.pct_of_peak_sustained_elapsed` |
| L2 hit rate | `lts__t_sector_hit_rate.pct` |
| L1/TEX hit rate | `l1tex__t_sector_hit_rate.pct` |
| Global-load coalescing (bytes/sector) | `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` |
| Active threads / warp | `smsp__thread_inst_executed_per_inst_executed.ratio` |
| Divergent branches | `smsp__sass_branch_targets_threads_divergent.sum` |
| Instr / scheduler | `smsp__inst_executed.sum` (÷ scheduler count) |
| Achieved occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` |

## 8. Deliverables
`capability_matrix.json`, `results.jsonl`, `counters.jsonl`, generated `REPORT.md`
(+ charts), `consistency.log` (must be green), and the Phase-5 patches to
`target_perf.py` / `mma_selector.py` / `capabilities.py` with this run as
provenance.

## 9. Claim-integrity guardrails (non-negotiable)
- No sm_120 result is asserted for a variant that failed Phase 0.
- No number is hand-typed into the report; all derive from the two JSONL files.
- `reference_cpu` / emulated fallbacks are labeled as such and never counted as
  native sm_120 execution.
- If clocks can't be locked, that is stated and the CoV gate carries the honesty.
- A red consistency gate blocks the report.
