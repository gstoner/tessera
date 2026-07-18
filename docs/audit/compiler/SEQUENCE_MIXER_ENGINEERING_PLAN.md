---
last_updated: 2026-07-17
audit_role: plan
plan_state: open
---

# Sequence Mixer Engineering Plan

> **Status:** engineering plan (Track L / compiler direction). Pairs with
> [`SEQUENCE_MIXER_THEORY.md`](SEQUENCE_MIXER_THEORY.md) (read first — facet
> vocabulary and the layer map come from there). Extends Decision #28.
> `MASTER_AUDIT.md` + generated dashboards stay status truth (Decision #26);
> this plan is the build sequence, not a status claim.

---

## 1. Scope, goals, non-goals

**Goal.** Land the Sequence Mixer abstraction so that KDA, Gated DeltaNet, GLA,
Mamba-2 SSD, sliding-window attention, short causal conv, and MLA are **one
Graph-IR op × facet knobs**, cache-planned by one N-way planner, lowered by one
tag-parameterized chunkwise scan, and arbiter-selected per shape/dtype/target —
with NVFP4/MXFP8 low precision as a parallel track.

**In scope:** the abstraction (TSOL protocol), faithful channel-wise KDA, the
schedule + N-way cache planner, the `linear_recurrence` Graph-IR op + transition
attribute, the shared chunkwise-scan Tile-IR lowering + `AbsorbChannelDecayIntoKeys`
+ reassociation canonicalization, Apple-GPU bring-up first, and the low-precision
operand-type + arbiter track.

**Non-goals (this plan):** datacenter Hopper/CDNA bring-up (stays Phase G/H);
training-time optimizer/sharding changes; new autodiff rules beyond what the
mixers need; MoE routing internals (orthogonal — `hybrid.py` already covers it).

**Guiding constraint.** Every increment is **host-free-provable** first (numpy
reference + an equivalence oracle from Theory §7) before any hardware kernel.
Leads (ROCm/CUDA) set the perf ceiling; the shared framework must never cap them
(Decision #28) — hand-emitted kernels stay first-class arbiter candidates.

---

## 2. Workstreams

Eight workstreams. W1–W4 are the critical path; W5 is per-backend bring-up; W6
(low precision) and W7 (arbiter) are semi-independent and can start early; W8
(backward) follows the forward path. **W5/W6/W7 do not open new Apple items —
they thread into the live Apple queue (items 8–14); see §2.9.**

### W1 — TSOL reference unification & faithful KDA  *(critical path, host-free)*

The reference tier. No IR, no hardware — numpy + oracles only.

1. **`SequenceMixer` protocol** in `python/tessera/stdlib/` — formalize the
   stringly-typed `_SPAN` dict in `hybrid.py` into a protocol carrying the four
   facets: `transition_tag`, `state_type`, `reassociable: bool`, `numeric_policy`,
   plus `span_forward(x, state) -> (y, state)` and both lowerings where they
   apply. Existing span functions (`_delta_span`, `_ssm_span`, `_liv_span`,
   `_full_span`) become protocol instances; **behavior must be byte-identical**
   (the `hybrid` streaming≡recompute oracle is the guard).
2. **Faithful channel-wise KDA** — extend `stdlib/delta_rule.py`'s `decay` from
   `[B,H,S]` to `[B,H,S,d_k]` (channel-diagonal / `dplr_bound`). Implement the
   `AbsorbChannelDecayIntoKeys` reweighting in the chunked form
   (`gamma` becomes `[B,H,C,d_k]`; scalar `ratio` becomes the `k̂/k̃` fold).
   Scalar path = `Γ` broadcast → **exactly today's code** (the scalar-reduction
   oracle). Make `tessera.ops.kimi_delta_attention` route to this.
3. **First-class `short_conv`** — lift `hybrid.py`'s `_causal_dwconv` to a
   standalone TSOL primitive with a `conv_state(k−1)` carry (Inkling `use_sconv`,
   `k=4`).
4. **First-class `sliding_window` mixer** — wrap `attn_sliding_window` /
   `lsa.py` as a mixer with a `windowed_kv(W)` ring-buffer state (Inkling
   `window=512`, GQA `H_kv=8`).
5. **Oracles** (Theory §7): `chunk≡recurrent` (KDA channel-wise),
   `scalar-reduction` (KDA↓GDN), `streaming≡recompute` (all mixers), and the
   `structure-canon` `dplr_bound == dplr_general` check.

**Exit:** channel-wise KDA + short_conv + sliding_window are TSOL primitives with
green oracles; `hybrid.py` runs unchanged through the new protocol.

### W2 — Schedule + N-way cache planner  *(critical path, host-free)*

1. **`HybridSchedule` explicit-set mode** — add `explicit(layer_ids→tag)`
   alongside `periodic(period, offset)`; Kimi 3:1 and Inkling `local_layer_ids`
   both express cleanly.
2. **N-way cache planner** (Python model first) — allocate the right state type
   per layer (Theory §3) and **normalize to a uniform physical block class**
   (the vLLM lesson): size the logical `growing_kv`/`windowed_kv` block so it and
   the constant-size states share a physical footprint. Model quantized blocks
   (NVFP4/MXFP8) as the same class at reduced physical size.
3. **`windowed_kv` + `latent_kv` handles** — the two missing cache-handle types
   (Theory §3 gaps), matching the `DeltaNetStateHandle`/`SSMStateHandle` ABI
   shape.
4. **Streaming≡recompute verifier over mixed stacks** — generalize the
   `hybrid` oracle to any schedule × mixer-set × cache-type mix.

**Exit:** a hybrid stack with `{windowed_kv, growing_kv, conv_state}` (Inkling
shape) and one with `{recurrent_matrix, latent_kv}` (Kimi shape) both pass
streaming≡recompute on the uniform-block planner.

### W3 — Graph IR: the `linear_recurrence` normal form  *(critical path)*

1. **`tessera.linear_recurrence` op** (`TesseraOps.td`) carrying `transition`
   (the Facet-A tag as an enum attr), `numeric_policy`, and symbolic-`S` policy.
   GLA/DeltaNet/KDA/Mamba lower to this op + tag.
2. **Reassociation canonicalization** — `(QKᵀ)V → Q(KᵀV)` guarded by the
   softmax-free predicate (Theory §4); rewrites softmax-free attention into
   `linear_recurrence`.
3. **Catalog + coverage** — update **both** `op_catalog.py` (acceptor) and
   `primitive_coverage.py` (audit truth) per Decision #24; register the mixer
   tags; keep `kimi_delta_attention`/`gated_deltanet`/`selective_ssm` as sugar
   over the normal form.

**Exit:** a `.mlir` fixture round-trips `linear_recurrence` with each tag;
coverage dashboard shows the new rows (drift-gated).

### W4 — Tile IR lowering  *(critical path)*

1. **Tag-parameterized chunkwise-scan pass** — one lowering; the tag selects the
   pre-fold. Emits GEMM tiles + the triangular-solve tile primitive.
2. **`AbsorbChannelDecayIntoKeys` legalization** — the `k̂/k̃` reweighting
   (Theory §2) with the fp32-accum + chunk-bound precondition attached as a
   `numeric_policy` check (Facet D).
3. **Triangular-solve tile primitive** — the `(I+Ã)⁻¹` within-chunk solve as a
   named Tile-IR op (generalize `_forward_substitution`).
4. **Windowed-attention lowering** — the non-reassociable branch for
   `sliding_window` (FlashAttn family + window mask + ring KV).
5. **Lit + FileCheck** fixtures under `tests/tessera-ir/` per pass.

**Exit:** `tessera-opt` lowers `linear_recurrence{dplr_bound}` end-to-end; lit
fixtures + FileCheck green.

### W5 — Backends: thread mixer candidates into the live per-target queues

**W5 opens no new backend items on any target.** It threads mixer candidates +
state types into the three live backend queues and inherits each queue's evidence
discipline (bindings: §2.9 Apple / §2.10 NVIDIA / §2.11 ROCm). Apple is the
**fastest oracle loop**; **ROCm and CUDA are the performance ceiling** (Decision
#28) — their hand-emitted MFMA / WMMA / `wgmma` / `mma.sync` kernels stay
first-class arbiter candidates from day one and are never capped by the shared
framework. Every candidate, on every target, proves **native placement** (a
fallback / stub / host recompute cannot earn a native pass; a forced binding-miss
returns the reference route) and shares one CUDA/ROCm/Apple numerical oracle.

1. **Apple** (oracle speed; existing scaffolding):
   * channel-wise KDA / GDN **decode** → extend the **ReplaySSM /
     `SSMStateHandle` / `DeltaNetStateHandle`** path (**APPLE-REPLAY-1**).
   * `sliding_window` / full-attn → online-softmax forward path
     (**APPLE-ATTN-FWD-1**; window/GQA/softcap are already its named cases).
   * `windowed_kv`/`growing_kv` + uniform-block planner → **APPLE-PAGED-KV-1**.
   * chunkwise-scan inner GEMMs → **APPLE-RETUNE-1**; `short_conv` its own MSL lane.
2. **NVIDIA** (sm_120 verified lane; sm_90/sm_100 stay Phase G/H gated) — mixer
   candidates into the **NVIDIA-TEST-3** correctness families (attention,
   KV/ReplaySSM, GEMM/Tile) and **NVIDIA-TEST-5** performance families. Inner
   GEMMs emit `wgmma` (sm_90) / `mma.sync` (sm_120), preferably via the **NVIDIA
   Tile IR** lowering target rather than hand-rolled PTX. NVFP4 low precision is
   already in the **NVIDIA-TEST-4** numerical policy (this is the executing FP4
   lane).
3. **ROCm** (gfx1151 verified lane) — **extend the already-complete
   ROCM-REPLAY-1 (ReplaySSM decode) and ROCM-9 (paged-KV)** with channel-wise KDA
   decode and `windowed_kv`; inner GEMMs emit WMMA via **ROCM-TILE-1** fragments
   (f16/bf16/int8/int4); attention fwd/bwd compose with **ROCM-6 G6-B/G6-C**.
   **ISA guard:** gfx1151 (RDNA3.5) has **no FP8/FP4 WMMA** — low-precision mixer
   GEMMs stay bf16/f16 here; FP8 needs RDNA4 (gfx1201) or CDNA4 (gfx950), FP4 the
   CDNA4 descriptor table (access-gated P0: ROCM-1/2/3). Never route an RDNA WMMA
   fragment map into MFMA.

**Exit:** channel-wise KDA executes + F4-verifies on Apple (oracle) with native
placement; the *same op* has lead-backend `wgmma`/`mma.sync` (sm_120) and WMMA
(gfx1151) arbiter candidates with native provenance.

### W6 — Low-precision track  *(semi-independent — can start with W1)*

1. **Microscale operand type** — NVFP4 (FP4 + per-16 FP8 block scale + per-tensor
   FP32) and MXFP8 (FP8 + per-32 E8M0) as IR operand-type metadata (Theory §5),
   not side tensors. Flip the relevant `planned_gated` dtype rows (Decision #15a).
2. **`numeric_policy` extension** — W4A4 / W4A16 modes on matmul/attention ops.
3. **FP4/FP8 emit is per-arch — do not conflate the tensor-core paths:**
   * **NVIDIA sm_120** (consumer Blackwell, the verified box) — **native
     block-scaled FP4 Tensor Core execution IS available** via warp-level
     **`mma.sync.aligned.kind::mxf4nvf4.block_scale`** (E2M1 + UE4M3, target
     `sm_120a`/`sm_120f`). The accurate framing is **not** "FP4 unavailable/fragile"
     — it is a **software kernel-selection gap**: sm_120 has no `tcgen05`/`TMEM`,
     so an SM100 CUTLASS tactic **cannot be retargeted** and fails init. But
     **TMA exists on sm_120** and feeds a **sm_120-specific mainloop**
     (`TMA → 99 KB SMEM → ldmatrix/registers → mma.sync.block_scale`) under the
     SM120 CUTLASS schedules `KernelTmaWarpSpecialized{Cooperative,Pingpong}`.
     Constraints (CUTLASS 4.4.1 SM120 doc): **TN layout only** (A row-major, B
     col-major), **cluster fixed 1×1×1** (no multicast), `EpilogueScheduleAuto`,
     tile `128×128×128`. When a framework ships no valid SM120 tactic (or picks an
     SM100 one that fails init) it falls back to **Marlin W4A16/W4A8-FP8**
     (dequant — correct, not native-FP4 throughput). The native candidate is a
     **directly-authored sm_120 `mma.sync.block_scale` kernel** — a textbook
     Decision-#28 case (the generic library path doesn't run; the hand-emitted
     kernel is the ceiling). Refs: `lna-lab/blackwell-geforce-nvfp4-gemm` (reports
     *working native NVFP4*, not Marlin fallback), `VincentKaufmann/fp4-cuda-kernel`,
     NVIDIA CUTLASS SM120 doc.
   * **NVIDIA sm_100** (datacenter Blackwell) — `tcgen05` FP4 MMA; **Phase G/H
     gated** (no datacenter silicon on the fleet).
   * **ROCm** — gfx1151 (RDNA3.5) has **no FP8/FP4 WMMA** (bf16/f16 only); FP8 on
     RDNA4 gfx1201 (E4M3/E5M2) and CDNA4 gfx950; FP4 only from the CDNA4
     descriptor table — all access-gated P0 (ROCM-1/2/3).
   * **Apple** — **no enabled NVFP4 cooperative-matrix route** (`NVFP4-TILE-SCALES`
     sync); **APPLE-DTYPE-1** is scale-layout round-trip + registered
     toolchain-gated state until the macOS 27 public Metal tensor path ships. It
     must never fall through to a hardware claim.
4. **Accuracy-budgeted selection** — W4A4 vs W4A16 vs bf16 as an arbiter budget
   decision (the metamorphic within-budget oracle, Theory §7). On sm_120 this
   fork is concrete: **W4A4 = the directly-authored `mma.sync.block_scale`
   kernel** (true native FP4); **W4A16 = the Marlin-style dequant fallback**
   (portable, correct, higher precision, no native-FP4 throughput). The arbiter
   must carry the hand-authored native kernel as its own candidate — the generic
   FP4 grouped-GEMM tactics do not run on consumer Blackwell.
5. **Bind to the existing sm_120 NVFP4 op — do not reinvent.** The NVIDIA backend
   already ships **`tessera_nvidia.nvfp4_block_scale_mma`** (E2M1 + UE4M3), whose
   **fixed-tile execution proof already passes on sm_120a**; its open gate is a
   *general-shape runtime ABI* ("blocked at runtime-dispatch gate" in
   `SM120_DIFFERENTIATION_DASHBOARD.md`). The mixer low-precision GEMM **rides that
   op** and inherits that gate. Implementation is `m16n8k64` `mma.sync…block_scale`
   (extend the existing `m16n8k16` `mma.sync` emitter, not a new path); 99 KB smem
   forces smaller tiles than sm_100. Grounded external corroboration:
   `lna-lab/blackwell-geforce-nvfp4-gemm` (SM80-lineage `mma.sync` + `ldmatrix`,
   scale applied post-MMA) and `VincentKaufmann/fp4-cuda-kernel` (128×128×128 tile,
   ~1.2% mean rel. error / Pearson 0.991; **1.4–2.4× vs bf16 at M≤2048, but bf16
   cuBLAS wins at M=4096** — a real arbiter shape-bucket rule).

**Exit:** an MXFP8/NVFP4 GEMM lowers with correct scale metadata and passes the
metamorphic-budget oracle; W4A4 emits `mma.sync … kind::mxf4nvf4.block_scale` on
sm_120a (not a Marlin W4A16 fallback); ROCm stays bf16/f16 on gfx1151; Apple
returns the gated state cleanly (APPLE-DTYPE-1).

### W7 — Arbiter integration  *(after W4/W5 land candidates)*

Wire mixer + precision candidates into the measured arbiter (Decision #28):
per `(op, shape-bucket, dtype, target)` pick fastest-in-budget; encode the
`dplr_bound` < `dplr_general` cost fork; decode(`S=1`) vs prefill(large-S) bucket
selection.

**FP4 native-route-proof rule (non-negotiable).** The three FP4 lanes are
distinct routes, and the selector must prove *which one ran* — a successful
Marlin W4A16 must **never** be recorded as native NVFP4 (the FP4 analog of
"a `reference_cpu` result cannot earn `native_gpu`"):

```
sm_100 (datacenter):  TMA → TMEM → tcgen05.mma                         [Phase G/H gated]
sm_120 (consumer):    TMA → 99-KB SMEM → ldmatrix/regs
                          → mma.sync.kind::mxf4nvf4.block_scale         [native — prove this]
fallback (any):       NVFP4 weights → dequantize to FP16/BF16 → Marlin W4A16   [report explicitly]
```

The arbiter carries the native sm_120 kernel and the Marlin fallback as
**separate candidates**, proves the native instruction route (assembled
`mma.sync…block_scale`, not a dequant), and reports Marlin as an explicit
fallback row — never folding W4A16 throughput into an "NVFP4" claim.

On Apple this **is APPLE-ROUTE-1** — adopt its device-keyed corpus fields
(physical device/family, OS/SDK/compiler fingerprint, route, **timing domain**,
native proof, correctness, resources, cold/warm, cache behavior) and its
rejection rules verbatim: **kernel/GPU time and end-to-end time are separate
timing-domain keys** (different winners allowed; production domain is
end-to-end), and a promotion requires **two-run stability + ≥5% win in both runs
+ native placement + retained resource evidence**. Stale / reference /
mismatched-device / wrong-domain rows are rejected.

### W8 — Mixer backward  *(new — closes a gap; composes with APPLE-ATTN-BWD-1)*

Training the mixers needs backward, which the mixer plan must own:

1. **Linear-recurrence backward** — the **chunked backward scan** (dQ/dK/dV +
   d(gate)/d(β)), the reverse-mode dual of the W4 chunkwise forward; register the
   VJP/JVP so `primitive_coverage.py` auto-flips the (V/J)VP axes (Decision #24).
2. **Softmax-family backward** — composes with **APPLE-ATTN-BWD-1** (native
   dQ/dK/dV, split-workspace vs atomics, f32 accum, determinism/workspace policy)
   for the `sliding_window` / full / MLA mixers.
3. **Oracle** — gradients match the shared CUDA/ROCm/numpy autodiff oracle across
   dtype, ragged, causal, GQA/MQA, boundary; `check_grad` clean.

**Exit:** channel-wise KDA + sliding-window backward match the shared gradient
oracle; coverage rows flip to (V/J)VP-complete.

---

### 2.9 Binding to the live Apple backend queue (items 8–14)

The Apple backend queue (`docs/audit/backend/apple/todo.md`) already carries the
measured-proof discipline this plan needs. The mixer work **binds into it** — new
candidates and state types under existing items, inheriting their evidence
contract — rather than reimplementing bring-up:

| Mixer piece | Apple item | Contribution |
|-------------|-----------|--------------|
| Channel-wise KDA/GDN **decode** (Theory §6 sequential) | **11 · APPLE-REPLAY-1** | extend ReplaySSM / `SSMStateHandle` / `DeltaNetStateHandle` |
| `sliding_window` / full-attn forward | **8 · APPLE-ATTN-FWD-1** | window/GQA/softcap already named; add mixer candidates |
| `windowed_kv` + uniform-block planner (W2) | **10 · APPLE-PAGED-KV-1** | paged-KV block = the uniform physical block |
| Chunkwise-scan inner GEMMs (W4) | **12 · APPLE-RETUNE-1** | ride the retuned GEMM/grouped-GEMM routes |
| Arbiter (W7) | **13 · APPLE-ROUTE-1** | its corpus + rejection rules *are* the arbiter contract |
| Low precision (W6) | **14 · APPLE-DTYPE-1** | SDK-gated; no NVFP4 cooperative-matrix on Apple |
| Mixer backward (W8) | **9 · APPLE-ATTN-BWD-1** | compose softmax-family bwd; own the linear-recurrence bwd |

**Inherited evidence contract (non-negotiable on Apple):** native-`native_gpu`
placement proof per candidate; shared CUDA/ROCm oracle; GPU/kernel vs end-to-end
as separate timing-domain keys; two-run stability + ≥5% win + retained
resource/counter evidence before any production route change; forced binding-miss
returns `reference_cpu`.

### 2.10 Binding to the NVIDIA (CUDA) queue

The NVIDIA queue (`docs/audit/backend/nvidia/todo.md`) is organized as test-family
groups, not per-op items. The mixer work threads candidates into those families
under the sm_120 verified lane (`BLACKWELL_SM120_EXECUTION_PLAN.md`); sm_90
(`wgmma`) and sm_100 (`tcgen05`) datacenter paths stay Phase G/H gated per
MASTER_AUDIT P2.

| Mixer piece | NVIDIA family | Contribution |
|-------------|---------------|--------------|
| Channel-wise KDA/GDN decode | **NVIDIA-TEST-3/-5 · KV/ReplaySSM** | add the channel-diagonal transition candidate |
| `sliding_window` / full-attn fwd | **NVIDIA-TEST-3/-5 · attention** | shares the online-softmax methodology (ROCM-6 G6-B origin) |
| `windowed_kv` + planner | **NVIDIA-TEST-3/-5 · paged KV** | window ring + uniform block |
| Chunkwise-scan inner GEMMs | **NVIDIA-TEST-3/-5 · GEMM/Tile** | emit `wgmma`(sm_90)/`mma.sync`(sm_120), prefer NVIDIA Tile IR target |
| Low precision (W6) | **NVIDIA-TEST-4** numerical policy | native FP4 = directly-authored `mma.sync … kind::mxf4nvf4.block_scale`; CUTLASS grouped-GEMM tactics fail on sm_120 (no `tcgen05`) → frameworks fall back to Marlin W4A16 (the arbiter's W4A4-vs-W4A16 fork) |
| Arbiter (W7) | **NVIDIA-TEST-5** + shared autotune corpus | `dplr_bound`<`dplr_general` fork; kernel-only vs end-to-end |
| Mixer backward (W8) | **NVIDIA-TEST-3/-5 · attention** | split/reduced dK/dV (G6-C-style) |

**Inherited contract:** NVIDIA-TEST-3 native provenance + execute/compare, two
collections, fallback-injection negative; NVIDIA-TEST-5 kernel-only vs end-to-end
with registers/shared-memory/occupancy/spills and selected route.

### 2.11 Binding to the ROCm queue

The ROCm queue (`docs/audit/backend/rocm/todo.md`) is the most mature: **paged-KV
(ROCM-9) and ReplaySSM (ROCM-REPLAY-1) are already complete on gfx1151**, so the
mixer work *extends completed vehicles*. gfx1151 (RDNA3.5) is the verified lane;
CDNA4/RDNA4 low-precision is access-gated P0 (ROCM-1/2/3).

| Mixer piece | ROCm item | Contribution / state |
|-------------|-----------|----------------------|
| Channel-wise KDA/GDN decode | **ROCM-REPLAY-1** *(complete)* | add channel-diagonal transition to the proven persistent/flush/rollback/async-ring path |
| `windowed_kv` + planner | **ROCM-9** *(complete)* | add window ring to the proven direct + gather paged-KV routes |
| Chunkwise-scan inner GEMMs | **ROCM-TILE-1** *(complete on gfx1151)* | WMMA f16/bf16/int8/int4 fragments |
| `sliding_window` / full-attn fwd | **ROCM-6 G6-B** | two-wave online-softmax forward attention (the origin methodology) |
| Mixer backward (W8) | **ROCM-6 G6-C** | split/reduced dK/dV backward attention |
| Low precision (W6) | **ROCM-1/2/3** *(access-gated)* | **ISA guard: gfx1151 has no FP8/FP4 WMMA** → bf16/f16 only; FP8 on gfx1201/gfx950, FP4 on CDNA4 descriptor table |
| Arbiter (W7) | **ROCM-6 revalidation / ROCM-8** | valid paired device timing needs bare-metal gfx1151 (WSL HIP events currently return zero durations) |

**Inherited contract:** exact-device execute/compare with native provenance;
aligned + ragged oracles; retained resources + device/E2E timing; the RDNA↛MFMA
fragment guard.

## 3. Phasing & dependencies

```
P0  (foundation, host-free)       P1  (compiler)               P2  (backend + precision + bwd)
──────────────────────────────    ──────────────────────────   ──────────────────────────────
W1 TSOL protocol + KDA + mixers   W3 linear_recurrence op      W5 Apple (into items 8/10/11/12)
W2 schedule + N-way planner       W4 chunkwise-scan lowering   W5 lead-backend candidates
                                  (needs W1 facets)            W6 low-precision (item 14 / sm_120)
W6 microscale operand type ───────┘ (dtype model, parallel)    W7 arbiter (into item 13)
                                                               W8 backward (into item 9)
```

* **W1, W2, W6-step1 have no interdependency** — start in parallel.
* **W3 needs W1** (facet vocabulary must be settled before it becomes IR attrs).
* **W4 needs W3** (op) and W1 (reference to check the lowering against).
* **W5 needs W4**; **W7 needs W4/W5** (candidates to choose among) + W6 (budget).
* **W6 gates W7's precision budget** but its dtype model can land in P0.
* **W8 needs W4** (forward chunk form to dualize) and composes with W5 backends.

---

## 4. Verification gates (per Theory §7)

Each phase advances only when its oracles are green:

| Phase | Gate |
|-------|------|
| P0 / W1 | `chunk≡recurrent` (channel-wise KDA), `scalar-reduction` (KDA↓GDN), `structure-canon` (`dplr_bound==dplr_general`) |
| P0 / W2 | `streaming≡recompute` for Inkling-shape and Kimi-shape stacks on the uniform-block planner |
| P1 / W3–W4 | `.mlir` round-trip + FileCheck per pass; `tessera-opt` end-to-end lowering; coverage drift gate |
| P2 / W5 | DESIL backend≡numpy per kernel (Apple first) |
| P2 / W6 | metamorphic within-accuracy-budget (W4A4/W4A16 vs bf16); FP4 MMA emit verified on sm_120 |
| P2 / W7 | arbiter picks the known-fastest in-budget candidate on a fixed shape sweep; Apple rows satisfy APPLE-ROUTE-1 rejection rules |
| P2 / W8 | KDA + sliding-window gradients match the shared CUDA/ROCm/numpy oracle (dtype/ragged/causal/GQA/boundary); coverage (V/J)VP axes flip complete |

Generated dashboards (`docs/audit/generated/`) carry the counts; update the
theme audit (`compiler/COMPILER_AUDIT.md`) as rows flip, not this plan.

---

## 5. Fleet ownership

Per the fleet coordination model (Decision #28 / `WORKSTREAM_C_HANDOFF.md`):

* **Mac (M-series)** — W1, W2, W3, W4, and the Apple lane of W5/W6/W7/W8
  (into `apple/todo.md` items 8–14). The host-free + Apple-GPU critical path;
  fastest oracle loop.
* **Strix Halo (gfx1151)** — the ROCm lane, **extending completed vehicles**
  (ROCM-REPLAY-1 decode, ROCM-9 paged-KV) + WMMA GEMMs via ROCM-TILE-1, attention
  fwd/bwd via ROCM-6 G6-B/G6-C. No FP8/FP4 WMMA (RDNA3.5) — low precision stays
  bf16/f16; FP8/FP4 are the access-gated CDNA4/RDNA4 boxes (ROCM-1/2/3). Valid
  paired device timing needs bare-metal (WSL HIP events return zero durations —
  ROCM-8).
* **NR2 Pro (sm_120)** — the NVIDIA lane (into NVIDIA-TEST-3/-5 families) +
  **the executing W6 FP4 path** via `mma.sync` (NVFP4 in the NVIDIA-TEST-4
  policy). sm_100/`tcgen05` datacenter Blackwell is the Phase-G/H analog, not on
  the fleet.

---

## 6. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| Channel-decay `1/Γ` numerical blow-up | chunk-bound + fp32 accum as a `numeric_policy` precondition (W4); oracle catches drift |
| Abstraction over-fits KDA, can't hold Mamba/MLA | facets are validated against **both** Kimi and Inkling shapes in W1/W2 before IR (W3) freezes them |
| Uniform-block planner fragments under quantized + windowed mix | model quantized block as same class / smaller size (W2-step2) and prove on Inkling shape before backend |
| Low-precision scope creep stalls the mixer work | W6 is a **separate track**; the mixer path (W1–W5) does not depend on it except at W7 |
| Leads' hand-tuned kernels capped by shared infra | mixer GEMMs stay first-class arbiter candidates (Decision #28); framework raises the floor only |
| Bespoke ops (`kimi_delta_attention` etc.) drift from normal form | keep them as **sugar over `linear_recurrence`** (W3-step3), not parallel implementations |

---

## 7. First concrete slice (the opening PR)

Smallest end-to-end-meaningful, fully host-free increment — proves the
abstraction on the highest-value gap without touching IR or hardware:

1. `SequenceMixer` protocol in `stdlib/` + port `hybrid.py`'s span functions to
   it (behavior-identical; the `hybrid` oracle is the guard). **[W1.1]**
2. Channel-wise `decay=[B,H,S,d_k]` in `stdlib/delta_rule.py` with
   `AbsorbChannelDecayIntoKeys` in the chunked form; scalar path preserved as the
   reduction oracle. **[W1.2]**
3. New tests: `chunk≡recurrent` and `scalar-reduction` for channel-wise KDA
   (extend `tests/unit/test_stdlib_delta_rule.py`).
4. Route `tessera.ops.kimi_delta_attention` to the faithful path.

This lands faithful KDA as a TSOL primitive with green oracles — a shippable,
self-contained unit — and establishes the protocol every later workstream builds
on. Everything after is "register another tag / another facet," each its own
oracle-gated PR.
