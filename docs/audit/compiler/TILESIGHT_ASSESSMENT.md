---
last_updated: 2026-07-30
audit_role: reference
---

# TileSight assessment — and what it says about *our* cost model

> **Source:** Mo et al., *TileSight: A First-Principles Tile-Centric Analytical
> GPU Performance Model from Cores to Clusters*, arXiv:2607.22432v1 (Imperial /
> PKU / SJTU / Tile-AI / MSR / Edinburgh). Assessed 2026-07-28 against the tree
> at `8358004`.
>
> **This is a `reference` doc, not a status surface.** It records an external
> survey and the internal finding it surfaced. Status truth stays in
> [`MASTER_AUDIT.md`](../MASTER_AUDIT.md) + `generated/` (Decision #26).
>
> **Cited from:** [`COMPILER_THEORY_OF_OPERATION.md`](COMPILER_THEORY_OF_OPERATION.md)
> §4 (the arbiter's scoring step) and W7 (absolute performance truth).

---

## 0. Why this doc exists

The paper is a good paper. But its main value to Tessera was diagnostic: reading
it forced an audit of our own analytical cost model, and **that model is a
mock**. The durable content here is §2 (the finding) and §4 (the two items it
opened), not the paper summary.

---

## 1. What TileSight is

An analytical, white-box, **pre-execution** GPU performance model that treats the
tile as the unit of *analysis*, not just of programming. No training, no kernel
profiling — one-shot per-architecture microbenchmark calibration (minutes), then
prediction. Three composed layers, all sharing one abstraction:

| Layer | What it models | Mechanism |
|---|---|---|
| **Intra-tile** | One tile's work | A 9-entry **resource vector** `⟨TC, CUDA, SFU, TMEM, SMEM, L1.5, L2, DDR, Net⟩` — per-tile *time* on independently schedulable pipelines, derived from op + footprint + src/dst placement + calibrated rates |
| **Inter-tile** | Overlap and locality | Tile-action DAG → search legal topological orders for best overlap; **tile reuse distance** → multi-level cache hit rates |
| **Cross-device** | Multi-GPU | A remote placement just populates the `Net` entry via a routed α–β stage cost; same envelope applies |

The unifying trick is **placement as one abstraction for both fusion and
distribution**: marking an intermediate register/SMEM-scope removes a global
store (fusion); marking a load source as a remote shard makes it a transfer
(distribution). Both flow through the same envelope.

Everything is evaluated through a recursive **prologue–steady–epilogue envelope**:

```
T = T_pro + max(N − d, 0) · T_steady + T_epi
d = pipeline_stages × resident_tiles_per_SM − 1
T_steady(σ) = max over resources r of Σ over actions o in σ of u_r(o)
T_steady    = min over σ ∈ Topo(DAG) of T_steady(σ)
```

**Reported accuracy** (their numbers, four NVIDIA archs + MI210):

| Claim | Result |
|---|---|
| Single-GPU GEMM latency, 703 shapes | **12.35% pooled MAPE** (roofline 33.85%, GenZ 34.89%, NeuSight 32.95%, PipeWeave 21.97%) |
| MI210 (CDNA2, cross-vendor) | 23.4% MAPE — their worst, and they explain it: CK exposes no rasterization/swizzle control, so the model runs blind |
| L2 hit rate, 4,680 persistent-kernel cases | **~1pp mean abs error** (0.78pp B6000, 0.88 H200, 1.05 B200, 1.46 A100) |
| Fused FA-3 on H100 vs NCU | 5.73 ms predicted vs 5.58 ms measured (2.7%) |
| As a TileLang cost model | Prune 95% of schedules, keep predicted top 5% → **99.66%** of exhaustive-search best |
| Implementation size | ~6K lines Python |

Not open source yet ("upon publication"). Taking anything means reimplementing
from the paper.

---

## 2. The finding: our analytical cost model is a mock

This is the part that matters. Three independent pieces of evidence:

**(a) The schedule planner has no memory term.**
[`schedule_planner.py:134`](../../../python/tessera/compiler/schedule_planner.py#L134):

```python
def _estimate_latency_ms(workload, cfg, peak_tflops):
    occupancy   = min(1.0, (cfg.tile_m * cfg.tile_n) / (128 * 128))
    stage_bonus = min(1.15, 1.0 + 0.04 * max(0, cfg.num_stages - 1))
    warp_bonus  = min(1.1, 0.85 + 0.05 * cfg.num_warps)
    effective_peak = max(1e-6, peak_tflops * occupancy * stage_bonus * warp_bonus)
    return workload.flops() / (effective_peak * 1e12) * 1_000.0
```

Latency is FLOPs ÷ a fudge-factored peak. No bytes, no bandwidth, no cache, no
pipeline. It cannot distinguish a compute-bound from a memory-bound shape, which
is the *first* thing any cost model must do.

**(b) The Bayesian autotuner's objective is explicitly synthetic.**
[`autotune_v2.py:380`](../../../python/tessera/compiler/autotune_v2.py#L380) —
`_mock_latency`, whose own docstring says *"Optimal point: tile_m=128,
tile_n=128, tile_k=32, num_warps=4, num_stages=2. Deviations incur
multiplicative penalties to simulate realistic behaviour."* It is honest (the
`on_device` path is tagged `status="unmeasured"`), but it means TPE/Hyperband is
searching a hand-drawn bowl whose minimum was placed by hand — on every arch,
for every shape.

**(c) Target profiles carry no performance parameters at all.**
[`gpu_target.py`](../../../python/tessera/compiler/gpu_target.py) and
[`rocm_target.py`](../../../python/tessera/compiler/rocm_target.py) are rich
*capability* matrices (`lds_async_copy: ready`, ISA gates, dtype support) with
zero performance data — no peak TFLOPS, no DRAM bandwidth, no L2 size, no CU/SM
count. The only peaks file in the tree is a 12-line
`tools/roofline_tools/tools/roofline/peaks/sm90_with_links.yaml` example
consumed by *post-hoc* roofline reporting.

**Why this matters for Decision #28.**
[`COMPILER_THEORY_OF_OPERATION.md`](COMPILER_THEORY_OF_OPERATION.md) §4 step 3
says: *"Without silicon, score by the Tier 2 cost model (roofline /
`MmaDescriptor` footprint)."* Given (a)–(c), that fallback ranks nothing — it
returns 128×128 for every shape on every architecture. The measured arbiter
([`emit/candidate.py`](../../../python/tessera/compiler/emit/candidate.py),
[`emit/autotune.py`](../../../python/tessera/compiler/emit/autotune.py)) is
sound and is doing all the real work; the analytical tier underneath it is
decorative. **Every candidate must be run to be ranked.** That is the cost the
paper's approach removes.

---

## 3. Verdict — take, adapt, skip

### 3.1 Take

| # | Idea | Why | Lands in |
|---|---|---|---|
| **T1** | **Tile reuse distance** (§3.5) — reuse distance over the *symbolic tile order* with tile-sized blocks as the reuse universe, Gaussian + Zelen–Severo approximation of the SDCM binomial, sampling along reduction axes | Landed 2026-07-30 as a deliberately smaller first adaptation: symbolic GEMM tile order + capacity-bounded exact LRU over A/B tiles, bounded deterministic sampling, real dtype widths, and a compute/DRAM roofline. It replaces both mock estimators for pruning; the paper's SDCM approximation and multi-level hierarchy are not claimed. | `reuse_distance_cost.py` |
| **T2** | **Prune, don't select** (§5.6) | 12% MAPE is far too coarse to pick between two close candidates but plenty to discard the bottom 95%. Slots in front of the arbiter's existing `measure` seam with **no API change** and no threat to Decision #28 lead-safety — the measured loop still decides, the model only decides what gets measured. Cuts device runs per shape bucket during fleet bring-up | pre-filter before `arbitrate()` |
| **T3** | **Resource vector + tile-action DAG ordering search** (§3.4) | Per-action time on independent pipelines, then `min over legal topological orders of max over resources`. Tractable because real DAGs are constrained — their MLA-decode example: 11 actions, 11! permutations → **132** legal orders | new cost-model module |
| **T4** | **`d = stages × resident_tiles_per_SM − 1`**, and separate prologue/steady/epilogue | Occupancy changes *overlap structure*, not just utilization: two blocks per SM deepens the pipeline. Our current `occupancy = tile_area/(128·128)` term gets this exactly backwards. 4 of their 7 diagnosis wins (Table 5) are "Not Overlapped → more blocks per SM / smaller tile", 1.17×–2.0× | same module |

**We are structurally better placed than TileSight is** on T3. Their §4 requires
users to hand-write tile execution plans or scrape them from Triton/TileLang. We
have a Tile IR that already carries the dependency edges, scratchpad levels, and
a `num_stages` knob
([`schedule_ir.py:361`](../../../python/tessera/compiler/schedule_ir.py#L361),
[`tile_ir.py:404`](../../../python/tessera/compiler/tile_ir.py#L404)). Our cost
model can read the IR directly rather than asking a human to describe it.

### 3.2 The corollary finding — we have no block-rasterization knob

The paper reports block swizzle moving L2 hit rate **35% → 72%**. In our tree:

- Threadblock swizzle exists on **exactly one** backend —
  [`apple_gemm_schedules.py:128`](../../../python/tessera/compiler/apple_gemm_schedules.py#L128),
  an MLX-inherited hardcoded heuristic (`swizzle_log = 0 if tm <= 3 else 1`),
  not a tuned axis.
- ROCm's "swizzle" ([`rocm_lds.py`](../../../python/tessera/compiler/rocm_lds.py))
  is an LDS bank-conflict XOR — a different concept at a different level.
- NVIDIA and x86 have none. A grep for `rasteriz|group_size_m|z_order|
  block_swizzle|tile_order|launch_order` across `python/` and `src/` returns
  nothing.

This is the cheapest large lever in GEMM codegen — a few lines in the block-index
computation — and we are not pulling it on either lead target. **This finding
stands on its own regardless of whether the cost model is ever built.**

### 3.3 Skip / defer

- **Never in the selection path.** 12.35% pooled (23.4% on MI210) cannot rank
  near-neighbors. Their own §7 admits systematic optimism for deep-K GEMMs
  (82% predicted vs 43% measured L2 hit at K=28672) because the model assumes SMs
  advance in lockstep; real SMs desynchronize across K-slices.
- **Weakest exactly where some of our targets live.** §7: *"latency-bound cases
  such as small-batch decode attention"* need finer latency/topology parameters.
  That is dflash / MLA-decode territory.
- **Don't build the x86 lane from this.** The abstraction is SM/SMEM/wave-shaped.
  Tile-granular reuse distance ports fine to AMX/AVX-512 cache blocking (that is
  Lam/Rothberg/Wolf 1991 — a CPU paper, which they cite); the pipeline-envelope
  machinery does not.
- **Distributed (§3.6) — file, don't build.** A competent α–β model, and we have
  no communication cost model at all today
  ([`distributed_planner.py`](../../../python/tessera/compiler/distributed_planner.py)
  has no cost terms;
  [`comm_overlap.py`](../../../python/tessera/compiler/comm_overlap.py) is
  SC-HRF scope/ordering *correctness* metadata, not cost). But multi-GPU is
  behind Phase G/H. Revisit with W6.

### 3.4 Per-backend read

| Backend | Value | Why |
|---|---|---|
| **ROCm gfx1151** | **Highest** | Strix Halo is a unified-memory APU — LPDDR5X bandwidth is the binding constraint, so a model with a real *memory* term is worth far more than one with a compute peak. Their MI210 caveat (no rasterization control in CK) does not bind us: we own the emitter |
| **NVIDIA sm_120** | **High** | They validate on B6000 = RTX PRO 6000 Blackwell — the same consumer-Blackwell family as the NR2 Pro's RTX 5070 Ti, and their **best** L2 result (0.78pp). Their Table 3 measured-vs-spec gaps on B6000 (FP32 88.6 vs 117 spec; DDR 1.4 vs 1.8 TB/s) are exactly the delta a calibration sweep recovers and a spec sheet hides |
| **Apple** | **High, different reason** | No NCU equivalent; MPS/MPSGraph are black boxes. A white-box pre-execution model is the only way to reason about *why* an MPS lane beats our MSL lane. Maps cleanly (resident tiles → threadgroups/core, SMEM → threadgroup memory); the DDR/L2 cascade needs rework for unified memory + SLC |
| **x86** | **Low** | Take reuse distance, skip the envelope |

---

## 4. What this opened

Three foundation items have landed — see
[`COMPILER_AUDIT.md`](COMPILER_AUDIT.md) for the current detail.

1. **`compiler/target_perf.py`** — calibrated, per-device performance parameters
   with per-field provenance (`MEASURED` > `DERIVED` > `SPEC` > `UNKNOWN`) and a
   hard rule that an absent number stays absent. Hard prerequisite for anything
   analytical, and independently the missing input for **W7**. Consumed by
   `SchedulePlanner.for_target()`, which refuses rather than falling back to the
   A100-shaped default.
2. **`compiler/tile_rasterization.py`** — block rasterization as a schedule knob
   (`schedule.knob` + `schedule.tile` + `TuningConfig` axes), with the bijection
   oracle and a test that compiles the emitted C and checks it against the
   reference for every block id.
3. **`compiler/reuse_distance_cost.py`** — the T1 first slice. It replaces the
   FLOPs-only planner estimator and hand-shaped autotuner mock with symbolic
   tile reuse, explicit cache capacity, cache-derived DRAM traffic, target
   compute/bandwidth inputs, and deterministic sampling. It is a pruning model:
   exact measured latency remains the only promotion authority.

**Still open, and the honest limits of what landed:**

- **No emitter consumes the rasterization knob yet.** The NVIDIA `mma.sync` GEMM
  and the ROCm GEMM path still compute `blockIdx` directly. Wiring them needs a
  measurement on the NR2 Pro / Strix Halo boxes to mean anything, and neither was
  available at the time. The default is the identity, so nothing changed.
- **The calibration sweeps have not been run.** `target_perf.py` ships the
  mechanism and the spec/derived baseline; the `measured` overlay is empty. Every
  fleet box's bf16/fp8 *matrix* peak and all Zen5 peaks are deliberately absent
  rather than guessed — which means `SchedulePlanner.for_target(..., "bf16")`
  currently raises on all three boxes. That is the gap made visible, not a
  regression.
- **Only T1 v1 is built.** T3 resource/action-DAG ordering and T4
  resident-tile/prologue/steady/epilogue overlap remain follow-ons. The T1
  implementation intentionally assigns no warp/stage bonus and does not select
  an unvalidated raster order. `ROCM-CALIB-1` rejected the separate
  step-distance approach; the new cache model must earn its own retain verdict
  against gfx1151 and sibling-backend corpora rather than receive tuned
  coefficients.

---

## 5. References worth chasing (from their bibliography)

Ranked by what is genuinely new to Tessera.

| # | Work | Why it matters here |
|---|---|---|
| 1 | **tritonBLAS** — Swann et al. 2025 (AMD), arXiv:2512.04226 | "Triton-based **analytical** approach for GEMM kernel parameter selection." AMD-authored, GEMM-specific, analytical. The closest existing thing to what `schedule_planner.plan_gemm` pretends to be — from the vendor of our lead target. **Read before writing any ROCm cost model** |
| 2 | **TileLink** — Zheng et al. 2025, arXiv:2503.20313 | *Generating* compute-communication overlapping kernels from tile-centric primitives. Our `comm_overlap.py` has the correctness contract (release/acquire, SC-HRF scopes) but no codegen. TileLink is the missing half. Ties to W6 |
| 3 | **KPerfIR** — Guan et al. 2025, arXiv:2505.21661 | Compiler-centric GPU performance tooling: profiling as an IR/compiler pass, not an external profiler. Fits our IR-first stance better than anything we do now, and it is the answer to "how do we get tile-level attribution on Apple and gfx1151, where NCU does not exist." Pairs with **Neutrino** (Huang & Wu, OSDI'25) on programmable probing |
| 4 | **NonGEMM** — Karami et al. 2025 (ISPASS) | Non-GEMM ops are up to **74%** of inference latency. Validates our broad-primitive-coverage bet, *and* warns that a GEMM-only cost model models a quarter of the problem. Relevant to Apple GPU op-gap prioritization |
| 5 | **TileFlow** — Zheng et al. 2023 (MICRO) | Tree-based analysis of **fusion** dataflow. We have fusion synthesis (`fusion_core.py`) and an F4 correctness oracle, but no answer to *"should* I fuse this?" TileFlow is the modeling framework for that question |
| 6 | **FractalTensor** — Liu et al. 2024 (SOSP) | Nested data parallelism and data reuse in DNN computation. Likely relevant to the `linear_recurrence` normal form in [`SEQUENCE_MIXER_THEORY.md`](SEQUENCE_MIXER_THEORY.md) |
| 7 | **Nugteren et al. 2014 (HPCA); Arafa et al. 2019, 2020** | The GPU reuse-distance / SDCM literature TileSight builds on. If T1 is implemented, **these are the primary sources**, not the TileSight summary |
| 8 | **Thakur, Rabenseifner & Gropp 2005** | Collective algorithm selection by message size (ring / recursive-doubling / Rabenseifner). Concrete, well-tested rules our `ChunkPlanner` can adopt when multi-GPU comes forward |
| 9 | **CUDA Tile / cuTile** — NVIDIA 2026 | Reinforces the tile-VM direction already tracked in the `nvidia-tile-ir` note: a better NVIDIA lowering target than hand-rolled PTX |

---

## 6. Honest limits of this assessment

- Assessed from the paper text, not from running the artifact — there is no
  artifact yet. Every accuracy number above is **their claim**, unreproduced.
- No Tessera row here is a proof row. The two items in §4 are implemented and
  gated by tests; nothing else in this doc has been built, and §4 lists what
  those two items still do not do.
- The `~1pp L2` and `12.35% MAPE` figures are on NVIDIA datacenter/workstation
  parts plus one CDNA2 card. Neither Apple silicon nor RDNA3.5 appears anywhere
  in their evaluation, so transfer to gfx1151 and M-series is **unvalidated
  extrapolation**, not a result.
