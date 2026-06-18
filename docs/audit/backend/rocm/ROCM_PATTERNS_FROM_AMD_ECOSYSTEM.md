# ROCm Backend — Patterns from the AMD ROCm Ecosystem

> **Purpose.** A source-grounded survey of eight production AMD ROCm projects, read for
> *transferable patterns* that improve Tessera's ROCm backend and the compiler at large.
> This is an **ideas + design-vocabulary** document, not a status surface — for status see
> [`ROCM_AUDIT.md`](ROCM_AUDIT.md); for the hardware bring-up ladder see
> [`STRIX_HALO_EXECUTION_PLAN.md`](STRIX_HALO_EXECUTION_PLAN.md).
>
> Surveyed: **AITER**, **ATOM**, **hipBLASLt**, **rocWMMA**, **Mori**, **Iris**, **XIO**,
> and the **AMD Gluon GEMM tutorial**.
>
> _Researched 2026-06-18. Provenance is flagged per claim: **[V]** verified from
> repo/docs/paper source, **[D]** DeepWiki summary (high-confidence, not source-exact),
> **[I]** inference / recommendation about Tessera internals._

---

## 0. The frame — where Tessera's ROCm backend actually stops today

The patterns below only make sense against the current frontier (mapped from the repo):

- **Real today:** [`python/tessera/compiler/rocdl_emit.py`](../../../../python/tessera/compiler/rocdl_emit.py)
  emits legal WMMA LLVM-IR for RDNA 3 / 3.5 / 4 and verifies it lowers to real `v_wmma_*`
  instructions via `llc -mcpu=gfx1151` on the host. Per-arch feature/dtype and
  MFMA/WMMA shape tables are grounded in
  [`rocm_target.py`](../../../../python/tessera/compiler/rocm_target.py) and the C++
  `mfma_table.inc` (generated from the Python `_MFMA_VARIANTS` by
  [`generate_mfma_table.py`](../../../../scripts/generate_mfma_table.py)).
- **Artifact-only:** the C++ backend stops at *placeholder marker ops* in
  [`TesseraTargetToROCDL.cpp`](../../../../src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TesseraTargetToROCDL.cpp)
  — it never emits real `rocdl.mfma.*` / `rocdl.wmma.*` intrinsics. The Python emitter
  bypasses the C++ spine entirely.
- **Missing:** HIP launcher isn't registered into `tsrRegisterGpuLauncher`; zero
  `execute_compare_fixture`s for ROCm; the RCCL adapter is a version-pin scaffold
  (`AdapterVersionPin.h`) with no collective kernels; `backend_kernel = complete` is **0/all**.
- **The unblock:** Strix Halo (gfx1151, RDNA 3.5) — first real non-Apple silicon, on an
  *open LLVM→AMDGPU* path that generalizes to NVIDIA (unlike Apple's MSL-only route).

The patterns therefore split into **(A) hardware-free wins to adopt now**,
**(B) the GEMM perf ladder to wire at Strix Halo bring-up**, and
**(C) the distributed / GPU-initiated-comm track**.

The throughline across all eight projects: **AMD's stack has converged on "make the
hardware concept a first-class object"** — layouts, fragments, symmetric heaps, epilogues,
tuned-config rows. That is Tessera's founding thesis. Most of what follows is not new
architecture; it is validation of the IR design plus a concrete vocabulary (exact enums,
shapes, schemas) to fill it in.

---

## 1. Project briefs

### 1.1 AITER — AI Tensor Engine for ROCm  ([repo](https://github.com/ROCm/aiter))

A multi-backend GPU **operator library** (the kernel substrate under vLLM-ROCm, SGLang-ROCm,
ATOM). A thin Python dispatch/JIT shell over six kernel backends: Composable Kernel (CK),
CKTile, hand-written ASM, Triton, FlyDSL (a mixed-precision GEMM/MoE DSL), and
hipBLAS/hipBLASLt fallback. **[V]**

- **Table-driven dispatch.** Kernel selection is data, not heuristics: CSV tuned-config DBs
  in `aiter/configs/` (`a8w8_tuned_gemm.csv`, `bf16_tuned_gemm.csv`, `tuned_fmoe.csv`, …),
  each paired with an `*_untuned_*.csv` worklist. **[V]** Schema (**[D]**, confirm against
  repo before hard-coding): `gfx, cu_num, M, N, K, libtype, solidx, splitK, kernelName, us`
  where `libtype ∈ {hipblaslt, asm, triton, flydsl, ck, cktile}`. Runtime lookup keys on
  **(M, N, K, gfx, cu_num)**, de-duped on load for deterministic dispatch, with a default
  kernel on a miss. **`gfx` and `cu_num` are first-class dispatch keys** — per-arch
  specialization is rows, not branches.
- **Correctness-gated autotuning.** `untune → tune → CSV`. Tuners (`GemmTuner`, `FmoeTuner`,
  `GemmA8W8BpreShuffleTuner`) enumerate variants; `checkAllclose` vs `torch.mm` reference
  (rtol/atol/tol_err_ratio) is a **gate before** `@perftest` median-latency ranking; winner
  row persisted. Split-K is computed from CU count + tile dims for occupancy. CI-driven
  (`operators-tuning.yaml`). **[V mechanism]**
- **Ops/fusions.** MLA decode (≤17×), MHA prefill (≤14×), fused MoE (≤3×), RoPE+KVCache
  fusion, fused gated-RMSNorm+group-quant, and a fused **collective**:
  `reduce_scatter → RMSNorm → quant → all_gather`. **[V]**
- **`aiter_tensor_t`** — a `{data, shape[8], strides[8], dtype}` framework-agnostic tensor
  ABI so kernels avoid PyTorch overhead. **[D]**
- **`@compile_ops`** — lazy first-call JIT: cache → build params from `optCompilerConfig.json`
  → codegen → Ninja `.so` → import. **[V]**

### 1.2 ATOM — AiTer Optimized Model  ([repo](https://github.com/ROCm/ATOM))

A vLLM-style **inference server** built natively on AITER (the dispatch/orchestration layer
deciding *which AITER kernel to call* from a HF model config). OpenAI-compatible API. **[V]**

- **Plugin layer abstraction.** Op-specific wrappers route to AITER: `Column/RowParallelLinear
  → gemm_*`, attention → `flash_attn_varlen_func`/`mla_attention`, RMSNorm →
  `fused_rms_fp8_group_quant`, `FusedMoE` with `expert_map`. **[V]**
- **Quant auto-detection.** `get_quant_config()` parses FP8-E4M3 (per-tensor/token/block),
  MXFP4 (per-1×32 / 1×128 block), INT4 (per-group 32/128) by regex on
  `quant_method`/`group_size`; online quant at forward time; KV-cache quant via
  `kv_cache_dtype` + separate `k_scale`/`v_scale`. **[V]**
- **Piecewise compilation.** Graph split at `splitting_ops` (default
  `["aiter.unified_attention_with_output", "aiter.mla_attention"]`) — **attention is the
  cut-point, fuse aggressively between cuts.** CUDA-graph modes incl. "full-decode +
  piecewise-prefill". **[V]**

### 1.3 hipBLASLt — fused GEMM library  ([repo](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt))

cuBLASLt-equivalent flexible/fused GEMM. Canonical op
`D = Activation(α·op(A)·op(B) + β·op(C) + bias)`. Generates its own kernels via *tensilelite*
(not layered on rocBLAS). **[V]**

- **Epilogue as composable bit-flags** — `hipblasLtEpilogue_t` values OR together
  (`RELU=2 | BIAS=4 → RELU_BIAS=6`): activations `RELU/GELU/SIGMOID/SWISH_EXT(SiLU)/CLAMP_EXT`;
  `BIAS`; **aux output** storing pre-activation for backward (`RELU_AUX`, `GELU_AUX`);
  activation-backward `DRELU/DGELU`; fused act-bwd+bias-grad `DRELU_BGRAD/DGELU_BGRAD`;
  operand bias-grad `BGRADA/BGRADB`. **Scale, amax and dtype-cast are NOT epilogue values** —
  they go through descriptor attributes / layout dtypes. **[V, exact values verified]**
- **Heuristic selection.** `AlgoGetHeuristic` returns N candidates ordered by estimated time,
  each with `workspaceSize`; `Preference{MAX_WORKSPACE_BYTES}` filters. Algos are opaque
  indexed integers — `getAllAlgos`/`getAlgosFromIndex` — **non-portable across release/arch**. **[V]**
- **Two-tier offline tuning.** (1) runtime env path, no recompile: `HIPBLASLT_TUNING_FILE`
  stores best **solution indices** per observed problem, `HIPBLASLT_TUNING_OVERRIDE_FILE`
  loads/overrides at runtime; (2) source-merge: `hipblaslt-bench` → tensilelite YAML logic
  files merged into the build. **[V]**
- **Grouped GEMM.** `GemmType::{HIPBLASLT_GEMM, HIPBLASLT_GROUPED_GEMM}`. *Batched* = one
  shape, strided. *Grouped* = vector of independently-sized problems in one launch, with
  **device-resident user arguments** (`run(void* deviceUserArgs, stream)`) — the MoE path. **[V]**
- **FP8 scaling.** `hipblasLtMatmulMatrixScale_t`: `SCALAR_32F`, `VEC16_UE4M3`,
  `VEC32_UE8M0` (OCP MX), `BLK128x128_32F` (DeepSeek 128×128). **FNUZ-vs-OCP arch split:**
  gfx942 uses `E4M3FNUZ`/`E5M2FNUZ`; gfx950 uses OCP `E4M3FN`/`E5M2` — *same code, different
  bits by arch.* **[V]**

### 1.4 rocWMMA — cooperative-matrix header library  ([repo](https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocwmma))

Header-only C++ template lib modeled on `nvcuda::wmma`; lowers to `amdgcn_mfma_*` (CDNA) or
`amdgcn_wmma_*` (RDNA) chosen at compile time. **[V]**

- **Fragment model** — `fragment<MatrixT, FragM, FragN, FragK, DataT, DataLayoutT, Scheduler>`
  with context tags `matrix_a`/`matrix_b`/`accumulator`. The `Scheduler` (2.0) encapsulates
  cooperative wave coordination. A fragment is distributed across a whole wavefront's
  registers; **element ordering in registers is not exposed.** **[V]**
- **Per-arch shape table** keyed on `(InputT, OutputT, ComputeT, arch)`: f16/bf16/i8 →
  16×16×16 (CDNA+RDNA); fp8/bf8 → 16×16×32+ (gfx942/950, gfx1200/1201); f32 → 16×16×4+
  (CDNA only); f64 → 16×16×4+ (gfx90a/942/950). BlockM/N ≥ 16 else pad; BlockK power-of-2 ≥ min.
  Accumulator = f32 for low-precision floats, i32 for int8. **[V]**
- **One code path for MFMA vs WMMA.** Same `fragment` + `mma_sync(d,a,b,c)` transparently
  lowers per arch; wave width auto (64 CDNA / 32 RDNA). API: `fill_fragment`,
  `load_matrix_sync`, `store_matrix_sync`, `load_matrix_coop_sync`. **[V]**

### 1.5 Mori — modular RDMA framework  ([repo](https://github.com/ROCm/mori))

"Bottom-up, modular, composable framework for high-performance RDMA + GPU comm," explicitly
inspired by MLIR's role in compiler infra. **[V]**

- **Library suite:** MORI-SHMEM (OpenSHMEM symmetric memory), MORI-IO (P2P GPU-Direct RDMA
  read/write for KVCache transfer), MORI-CCL (AllGather/AllReduce), MORI-EP (MoE
  dispatch/combine), MORI-UMBP (tiered KV). **[V]**
- **Symmetric heap** — GPU memory at identical virtual offset across all PEs, so RDMA
  addresses remote memory without translation tables. `shmem_malloc`/`free`,
  `shmem_buffer_register`, `shmem_mype/npes`, `shmem_barrier_all`. **[V]** (device-side
  put/get/atomic signatures referenced but not shown — **[I/gap]**).
- **MORI-IO engine model** — `IOEngine` + `IOEngineSession` (pre-established pair context);
  one-sided `read/write/batch_read/batch_write`; **Transport Store** multiplexing
  **RDMA/XGMI/TCP with failover**; split slow-control-plane (TCP metadata) + fast-control-plane
  (RDMA notifications); knobs `qp_per_transfer` (≥2/NIC), `num_nics_per_transfer` (NUMA-local
  striping), `chunk_bytes` pipelining (~28 GB/s single-outstanding → NIC line rate);
  GPU-initiated via **IBGDA**. **[V]**
- **MORI-IR — compiler integration.** *Not* an MLIR dialect: a **bitcode library of
  `extern "C"` device functions** (P2P/RDMA/IBGDA/SDMA) `llvm-link`ed into the kernel,
  explicitly "for FlyDSL, MLIR-based compilers, or any framework supporting LLVM bitcode." **[V]**

### 1.6 Iris — SHMEM-like RMA inside Triton  ([repo](https://github.com/ROCm/iris) · [paper](https://arxiv.org/html/2511.12500v1))

AMD RAD's multi-GPU framework: SHMEM-like APIs *in Triton*, pure Python+Triton. Intra-node /
IPC today (not cross-node RDMA). **[V]**

- **Symmetric heap via IPC.** `hipMalloc` + `hipIpcGetMemHandle`/`OpenMemHandle`; remote
  pointer = offset arithmetic in `__translate()` (`offset = local - bases[src];
  remote = bases[dst] + offset`); 64-byte bases array stays L1-cached → **measured no
  overhead.** **[V]**
- **Device primitives** — `load/store` (reg↔mem), `get/put/copy` (buf↔buf, any rank pair),
  atomics (`add/and/or/xor/min/max/cas/xchg`), non-blocking, relaxed by default. **[V]**
- **Memory model** — AMD SC-HRF, scopes **wavefront/workgroup/agent/system** ×
  `relaxed/acquire/release/acq_rel`. **[V]**
- **Tile-granular overlap** — three patterns: sequential-fused store in the GEMM loop
  (1.5–1.8×); **workgroup specialization** (compute WGs signal via `atomic_cas(release)`,
  scatter WGs `acquire`-spin, ≤2.5×); unfused producer/consumer on separate streams +
  CU partitioning. GEMM+all-scatter avg 1.21×. **[V]**

### 1.7 XIO — GPU-initiated IO  ([repo](https://github.com/ROCm/rocm-xio) · beta-0.1.0)

"Accelerator-Initiated IO": AMD GPUs perform direct IO to NVMe SSDs, RDMA NICs, SDMA engines
from `__device__` code, no CPU. Early-access, not production. **[V, sparse]**

- **Unified endpoint model** — all derive from `xio::XioEndpoint` + `XioEndpointConfig`;
  runtime `createEndpoint()`; types `nvme-ep / rdma-ep / sdma-ep / test-ep`. **[V]**
- **Device submission/completion** — `XIO_MEM_MODE_SQ_DEVICE` / `_CQ_DEVICE` place queues in
  VRAM so a kernel posts+polls IO itself (exact device `submit/poll` names not in reachable
  docs — **[I/gap]**).
- **Relevance:** the checkpoint + data-pipeline + KV-offload transport (S12/S15), *not* a
  collective. Forward-looking pattern, not yet a stable dependency. **[I]**

### 1.8 AMD Gluon & the GEMM tutorial  ([tutorial](https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html))

Gluon is a **lower-level, tile-level frontend inside OpenAI/Triton** — the Python frontend to
Triton's `ttg` (TritonGPU) IR, skipping the higher-level `tt`. **It is essentially a
hand-written Tile IR exposed as a language** — the single most important framing for Tessera.
Thesis (verbatim): *"Layouts are explicit. Pipeline stages are explicit. Register budgeting
becomes part of the kernel design, not something discovered after the backend lowers the
code."* **[V]**

First-class objects (direct Tessera parallels) **[V]**:
- **Register layouts:** `BlockedLayout(size_per_thread, threads_per_warp, warps_per_cta,
  order)`, `SliceLayout(dim, parent)`, `DistributedLinearLayout(reg/lane/warp/block_bases,
  shape)` (bit-basis → zero-cost split/join/reshape/permute), `AutoLayout`, `CoalescedLayout`.
- **Shared-memory layouts:** `SwizzledSharedLayout(vec, per_phase, max_phase, order)` (XOR
  swizzle, bank-conflict-free) **vs** `PaddedSharedLayout` (additive padding — AMD-preferred
  for CDNA4 `GLOBAL_LOAD_LDS_*` which needs consecutive warp-wide writes).
- **MMA layouts:** `AMDMFMALayout(version, instr_shape, transposed, warps_per_cta)` (anchor) +
  `DotOperandLayout(parent, operand_index, k_width)` (A/B derived from the MMA layout; `k_width`
  packs multiple MFMA instructions); scaled `cdna4.mfma_scaled →
  v_mfma_scale_f32_16x16x128_f8f6f4` (MXFP4).
- **Memory movement:** `cdna3.buffer_load` (one 64-bit base/warp + 32-bit/thread offsets,
  native OOB — collapsed **140 branches → 4**), global→LDS `async_copy`, transposing `ds_read_tr`.
- **Warp specialization:** `warp_specialize(...)` with per-partition warp counts + register budgets.

GEMM optimization ladder v0→v9 on MI350/gfx950 (512-VGPR budget), validated against
`rocprofv3 --att` counters, ~99% MFMA utilization **[V]**:

| Ver | Technique | Lesson |
|----|-----------|--------|
| v1 | `buffer_load` | 140 branches → 4 |
| v2 | async global→LDS | removes register staging + inner-loop `ds_write` |
| v3 | LDS layout (raw/swizzle/pad) compared at instruction level | bank-conflict elimination |
| v4→v5 | 2-stage → 3-stage software pipeline | MFMA + LDS reads + global loads all overlap |
| **v6** | register double-buffer | **FAILED −73%** — exceeds 512 VGPRs, spills 99/iter |
| v7 | N-slicing | halves B-tile footprint; spills → 0 |
| v8 | M+N quad-slicing | drops register pressure further |
| v9 | XCD-aware WG remap + `amdgcnas` peephole | L2 locality; kills AGPR↔VGPR copies |

**Key empirical lesson:** the biggest perf decisions are *register-budget-driven tiling
choices*, not micro-opts. The "obvious" double-buffer regressed badly; slicing the output
tile to fit the VGPR budget was the real lever. Tiling and register allocation are coupled
and must be co-designed.

---

## 2. Patterns to adopt — ranked, mapped to Tessera

### A. Hardware-free wins (adopt now)

**A1 — Unify MFMA + WMMA behind one Tile-IR cooperative-matrix type.** *(rocWMMA + Gluon)*
The MMA shape is an *anchor*; A/B operand layouts + packing width derive from it. Tessera
currently selects MFMA shape (`_MFMA_VARIANTS`) independently of operand layout. Make
`tessera_rocm.mma` one typed op carrying role (`matrix_a/matrix_b/accumulator`), shape, dtype,
layout, `k_width` — dispatched to MFMA (CDNA) or WMMA (RDNA) at lowering. This is
Decision #19 done properly and hardens the V3/V6c verifier ceilings. Invariant: "register
element ordering is backend-owned." **[I]**

**A2 — Layout as a first-class typed operand with an explicit, costed `convert_layout`.** *(Gluon)*
Extend `LayoutLegalityPass` so every Tile-IR tensor carries a register-distribution layout;
model layout change as `tessera.tile.convert_layout` with a data-movement cost. Reach goal:
the **linear-layout (bit-basis) representation** so split/join/reshape/permute are free
metadata ops. **[I]**

**A3 — Epilogue as a composable bit-flag attribute.** *(hipBLASLt)*
Generalize the ad-hoc Apple `matmul→{gelu,rmsnorm,softmax}` fusions into one typed
`EpilogueSpec` over the `activation × bias × aux × gradient` lattice. The Tessera-specific
win: `*_AUX` (store pre-activation for backward) and `DGELU`/`BGRAD*` are *autodiff
primitives* — the VJP/JVP registry can request the aux tensor instead of recomputing. Slots
into `primitive_coverage.py` axes. **[V vocab / I mapping]**

**A4 — Table-driven dispatch keyed on `(shape, dtype, gfx, cu_num)`, deterministic fallback.** *(AITER + hipBLASLt)*
Highest-leverage runtime pattern. Adopt the CSV schema (`gfx, cu_num, M, N, K, libtype,
solidx, splitK, kernelName, latency_us`, de-duped on load) into `flywheel.py` (already
device-keyed) + `autotune_v2` SQLite. Add hipBLASLt's two-tier model: runtime
problem→solution override file (no recompile) + build-merged logic. **Hard rule:** key on the
problem *signature*, never opaque solution indices. **[V / I]**

**A5 — Correctness-gated autotuning + the "untuned worklist" artifact.** *(AITER)*
`checkAllclose` vs reference **before** latency ranking (matches the
magellan/alphaevolve/grader "perf gated behind correctness" invariant). Keep a checked-in
`*_untuned_*.csv` worklist of shapes pending tuning, separate from tuned results — an
auditable "tuned vs pending" surface (Decision #26). **[V / I]**

**A6 — Quant taxonomy → `numeric_policy`; arch-keyed FP8 semantics.** *(hipBLASLt + ATOM)*
`hipblasLtMatmulMatrixScale_t` (`SCALAR_32F`/`VEC16_UE4M3`/`VEC32_UE8M0`=MX/`BLK128x128_32F`)
and ATOM's granularity detection are the canonical vocabulary for
`NumericPolicy.scale`/`quant_axis`, confirming "scale lives in numeric_policy, not the dtype
string." **Correctness item:** FNUZ (gfx942) vs OCP (gfx950) FP8 are different bits for the
same op — the per-arch dtype matrix needs an FP8-semantics flag, and any "complete" FP8 claim
is arch-ambiguous without it. Lit fixtures should FileCheck FNUZ vs OCP per target. **[V / I]**

**A7 — Grouped GEMM (device-resident args) as the MoE primitive.** *(hipBLASLt)*
Distinguish *batched* (identical shape, strided) from *grouped* (independently-sized problems,
one launch) in the IR. `run(deviceUserArgs, stream)` — per-expert token counts resolve
on-device — is the right model for `moe_dispatch` / `moe_swiglu_block`. **[V / I]**

### B. The GEMM perf ladder (wire at Strix Halo bring-up)

**B1 — Register budget as a hard design-time constraint; tile-slicing as a first-class transform.** *(Gluon, strongest empirical lesson)*
Add per-arch VGPR/AGPR budget to `ROCmTargetProfile`; the tiling pass / Bayesian autotuner
*prunes* candidates whose live-set exceeds it (same shape as `InsertRecomputePass`). Add
output-tile quad-slicing to the tiling repertoire. Remember v6: double-buffer regressed −73%. **[V / I]**

**B2 — Pipelining as an explicit IR construct; LDS layout arch-selected.** *(Gluon)*
Generalize FA-4 `pipeline_stages` to GEMM/contraction K-loops as `tessera.tile.pipeline
{stages=N}` driving N-buffered LDS allocation (autotuner sweeps N). Model both LDS strategies:
XOR swizzle (general) vs additive padding (gfx950 global→LDS). Connects to the existing
"LDS slot overlap" work. **[V / I]**

**B3 — AMD buffer-load addressing as a Target-IR op.** *(Gluon)*
Add `tessera_rocm.buffer_load` (64-bit base/warp + 32-bit/thread offsets, OOB attr) +
global→LDS `async_copy` + transposing `ds_read_tr` so masking lowers to OOB-handling, not
control flow. **[V / I]**

**B4 — Real intrinsic emission to close the C++ spine.** *(the actual gap, not a steal)*
The C++ backend must emit real `rocdl.wmma.*`/`rocdl.mfma.*` instead of markers, and the
gfx1151 D-accumulator VGPR mapping (RDNA3.5 ISA §7.9 Table 33) must be grounded numerically.
rocWMMA's fragment ABI and Gluon's `DotOperandLayout` are the operand→VGPR reference. See
[`STRIX_HALO_EXECUTION_PLAN.md`](STRIX_HALO_EXECUTION_PLAN.md). **[I]**

**B5 — Counter-driven "derive-validates-declare."** *(Gluon process pattern)*
When hardware lights up, score MFMA/WMMA kernels in the conformance/flywheel evaluator
against the same `rocprofv3 --att` counters Gluon uses (MFMA-issue utilization, VGPR spills,
`ds_read`/`ds_write` rates), not just numerical correctness — the evaluator program extended
to perf rungs. **[V / I]**

### C. Distributed / GPU-initiated comm track

Tessera's collectives are **two-sided, host-orchestrated** (RCCL adapter + credit scheduler +
chunk planner). Iris/Mori/XIO are all moving to **one-sided, GPU-initiated** comm.

**C1 — Symmetric heap as a first-class sharding/IR concept.** *(Iris + Mori converge)*
Allocate shards at identical virtual offsets across ranks → remote address is pure offset
arithmetic (L1-cached bases, measured zero overhead). The substrate that makes one-sided
collectives expressible in IR. Pairs with Sprint-D `MemoryShardSpec` / `KEY_HASH` banks. **[V / I]**

**C2 — Comm device functions as linkable LLVM bitcode behind a stable ABI.** *(Mori-IR — directly fits)*
Mori-IR is *not* an MLIR dialect — `extern "C"` device funcs (P2P/RDMA/IBGDA/SDMA) shipped as
bitcode, `llvm-link`ed in, "for MLIR-based compilers." Tessera already links runtime C-ABI
symbols (Apple/x86). Mirror it: a `tessera_rocm_comm` bitcode so collectives lower to
*device-side calls*, not host RCCL. Cleanest path to a real AMD one-sided story without
authoring RDMA verbs in the compiler. **[V / I]**

**C3 — Tile-granular compute/comm overlap + transport abstraction.** *(Iris + Mori-IO)*
Iris's workgroup-specialization (compute WGs `release`-signal, scatter WGs `acquire`-spin,
≤2.5×) is what a tile-first compiler should emit — fits `WarpSpecializationPass` +
`PipelineOverlapPass`. SC-HRF four-scope model becomes explicit attributes on collective/signal
ops. Longer-term: generalize the single RCCL adapter into Mori-IO's Transport Store
(RCCL/XGMI/RDMA/TCP + failover; slow-TCP-control / fast-RDMA-data split) and extend
`ChunkPlanner` with QP-count + NIC-striping knobs. XIO's GPU-initiated NVMe/SDMA endpoints are
the forward-looking transport for S12 checkpointing / S15 data pipeline. **[V / I]**

---

## 2.5 Implementation status — hardware-free batch landed (2026-06-18)

The entire **A (hardware-free)** batch is implemented and green (134 unit tests,
mypy-clean). These are pure Python IR/metadata/dispatch surfaces — no silicon
required — that the ROCm lowering, autotuner, and audit registry can consume now.

| Item | Module | Tests | Notes |
|------|--------|-------|-------|
| **A1** | `python/tessera/compiler/rocm_mma.py` — `MmaDescriptor`/`MmaOperand`, `select_mma`, `mma_for_matmul` | `tests/unit/test_rocm_mma.py` (27) | Shape-as-anchor; MFMA(CDNA)/WMMA(RDNA) unified; operand layout + `k_width` derived; rejects FP8-on-gfx1151 / FP8-on-gfx90a / fp32-on-RDNA with stable diagnostics; `k_width` documented as a packing hint (register order backend-owned). |
| **A2** | `python/tessera/compiler/tile_layout.py` — `BlockedLayout`, `SliceLayout`, `LinearLayout`, `convert_cost`/`ConvertLayout` | `tests/unit/test_tile_layout.py` (30) | Bit-basis linear layout → free reshape/permute/transpose/split/join; `convert_cost`=0 for identity + pure bit-permutation, else LDS round-trip cost. |
| **A3** | `python/tessera/compiler/epilogue.py` — `Epilogue(IntFlag)`, `EpilogueSpec`, `backward_epilogue`, `CANONICAL_EPILOGUES` | `tests/unit/test_epilogue_spec.py` (23) | Exact hipBLASLt bit values compose by OR; autodiff bridge maps forward→backward and flags `requires_aux` (VJP requests the `*_AUX` pre-activation instead of recomputing). |
| **A4+A5** | `python/tessera/compiler/tuned_dispatch.py` — `ProblemSignature`, `TunedConfig`, `TunedDispatchTable`, `tune`, `UntunedWorklist` | `tests/unit/test_tuned_dispatch.py` (24) | CSV keyed on signature (never `solidx`); de-dup keeps min latency; two-tier override; correctness-gated `tune` rejects fast-but-wrong; untuned worklist diff. |
| **A6** | `rocm_target.py` (`fp8_semantics`/`fp8_dtype_flavor` + `_FP8_SEMANTICS`/`_FP8_FLAVOR`, profile property); `grouped_layout.py` (`scale_mode_to_layout`, `HIPBLASLT_SCALE_MODES`) | `tests/unit/test_amd_fp8_and_gemm_dispatch.py` (30, shared with A7) | Arch-keyed FNUZ (gfx940/942) vs OCP (gfx950/1200/125x) FP8; hipBLASLt scale-mode → `ScaleLayout`. **Pre-existing:** `grouped_layout.ScaleLayout` already covered the granularity/block/packing taxonomy. |
| **A7** | `grouped_layout.py` — `classify_gemm_dispatch`, `GemmDispatchClass`, `BATCHED_GEMM_OPS`/`GROUPED_GEMM_OPS` | (shared A6/A7 file) | Batched (uniform, host-known) vs grouped (independently-sized, `device_resident_args=True`). **Pre-existing:** the four DeepGEMM grouped families (incl. masked) + `grouped_gemm`/`batched_gemm` op_catalog rows already existed. |

---

## 2.6 Registry wiring + B4 FP8-flavor emission landed (2026-06-18)

The follow-on pass that the 2.5 batch had deferred is now landed.

**Registry wiring (A1 + A3 + A6 dashboard).**
- `primitive_coverage.py` — `_rocm_mma_for_name` attaches `metadata["rocm_mma"]`
  (per-`(arch, dtype)` unified MFMA/WMMA descriptor: kind, shape, k_blocks, acc,
  k_width, intrinsic) to the GEMM-family ops; `_epilogue_for_name` attaches
  `metadata["epilogue"]` (the canonical epilogue catalog + forward→backward map +
  the autodiff-aux note) to `fused_epilogue`. Mirrors the existing
  `_grouped_layout_for_name` pattern; attached in both `_existing_coverage` and
  `_supplemental_metadata`.
- `gpu_target_map.py` — the **`rocm_target_map.md` dashboard** gained a per-arch
  **FP8 numeric-semantics table** (A6: gfx940/942 = FNUZ, gfx950/1200/125x = OCP,
  rest = none, with the concrete `e4m3fnuz`/`e4m3` flavor spellings). This is the
  visible, drift-gated surface for the FNUZ/OCP distinction.
- Regenerated via `scripts/check_generated_docs.sh --write`; drift gate clean
  (16/16 in sync). Guard: `tests/unit/test_registry_amd_wiring.py` (10).

**B4 (honest slice) — arch-keyed FP8 flavor emission + lit fixtures.**
- `TileToROCM.cpp` now takes an `arch` pass option and emits an arch-keyed
  `fp8_flavor` attribute on `tessera_rocm.mfma` for FP8 operands (base from the
  operand element type, FNUZ/OCP suffix from a C++ semantics table that mirrors
  `rocm_target._FP8_SEMANTICS`). FP8 on a no-FP8 arch (gfx1151) is a hard, named
  error (Decision #21), not a silent guess. `tessera-rocm-opt` rebuilt against
  MLIR 22.1.6.
- Lit fixtures: `test/rocm/fp8_flavor_arch_keyed.mlir` (FNUZ vs OCP) and
  `fp8_unsupported_arch.mlir` (error path) — verified via FileCheck; the existing
  `tile_matmul_to_rocm.mlir` (gfx90a default) still passes (backward compatible).
- Single-source enforcer: `tests/unit/test_rocm_fp8_cpp_python_consistency.py`
  (18) subprocess-runs the binary for every arch × {e4m3, e5m2} and asserts the
  C++-emitted flavor equals `rocm_target.fp8_dtype_flavor` — so the C++ and
  Python FP8 tables can never drift. Skips cleanly if the binary isn't built.

**Lit-runner infra — fixed (2026-06-18).** The backend's lit wiring (which had
never actually run) now works end-to-end: the generated `lit.site.cfg.py`
`load_config`s the source `lit.cfg.py`; `lit.cfg.py` prepends `llvm_tools_dir`
to the RUN-line PATH (self-contained — no `lit.llvm` dependency); the test
`CMakeLists.txt` robustly locates `lit` + the LLVM tools dir (bypassing a
polluted empty-string `LLVM_EXTERNAL_LIT` cache var); the `.txt` suffix that
spuriously matched `CMakeLists.txt` was dropped; and two latent-broken
pre-existing fixtures (`async_and_mfma_realish`, `rocm_target_to_rocdl_contract`)
were fixed with order-robust `CHECK-DAG`. `check-tessera-rocm` now reports
**9/9 passing**, and `tests/unit/test_rocm_lit_suite.py` runs the suite in the
normal pytest lane (skips when the backend isn't built) so the wiring can't
silently regress.

**Still genuinely deferred (hardware-gated):**
- *Full numeric* `rocdl.mfma/wmma` intrinsic emission — `TesseraTargetToROCDL.cpp`
  still emits marker ops; a complete, assemblable, numerically-correct kernel
  needs the offline emitter + real silicon (the rest of B4). The `fp8_flavor`
  attribute is now in place to carry the flavor down to that emitter when it lands.

---

## 2.7 Hardware-free B/C batch landed (2026-06-18)

The hardware-free slices of the **B** (GEMM perf ladder) and **C**
(distributed/comm) tracks are implemented as pure modeling/IR surfaces (170 unit
tests; mypy + ruff clean). They give the compiler the *vocabulary* the real
kernels will need; the executable kernels themselves stay hardware-gated.

| Item | Module | Tests | Notes |
|------|--------|-------|-------|
| **B1** register budget | `rocm_target.py` (`vgpr_budget`/`agpr_budget`/`total_reg_budget`, per-arch tables: CDNA 256+256, RDNA 256+0) + `compiler/rocm_tiling.py` (`TileShape`/`TileCandidate`, `estimate_vgpr_usage`, `prune_candidates`→`PruneResult`, `quad_slice`/`n_slice`) | 30 | The Gluon v6 lesson encoded: a double-buffered over-budget tile is pruned; quad-slice fits it. Never silently drops — `PruneResult` records dropped candidates. |
| **B2** pipeline + LDS layout | `compiler/rocm_lds.py` — `SoftwarePipeline(stages)`→N-buffered LDS; `SwizzledLdsLayout` (XOR) vs `PaddedLdsLayout`; `select_lds_layout(arch, global_to_lds=…)` | 32 | Padding for gfx950 `GLOBAL_LOAD_LDS`, swizzle elsewhere (arch-keyed). |
| **B3** buffer_load / ds_read_tr | `tessera_rocm.buffer_load` (OOB addressing) + `ds_read_tr` ops in the dialect ODS + ROCDL marker lowering | lit | `test/rocm/buffer_load_ds_read_tr.mlir` (roundtrip + lowering); `check-tessera-rocm` 10/10. |
| **C1** symmetric heap | `symmetric_heap.py` — `SymmetricHeap` (offset-arithmetic `remote_address`) + `SymmetricShardSpec` (replicated/partitioned) | 32 | The substrate for one-sided collectives (Iris/Mori): the *offset* is symmetric across ranks; base pointers may differ. |
| **C3** overlap modeling | `compiler/comm_overlap.py` — SC-HRF `MemoryScope`/`MemoryOrdering`, `SignalOp` (producer=release / consumer=acquire), `OverlapPlan` + `plan_overlap` | 38 | Models the three Iris overlap patterns (sequential-fused / workgroup-specialized / unfused). |

**Still hardware-gated (need Strix Halo / MI300 / NICs):** B4 full numeric
`rocdl.mfma/wmma` emission, B5 counter-driven scoring (`rocprofv3 --att`), C2
comm device-function bitcode ABI, and `backend_kernel = complete` (execute-and-
compare on real silicon — see `STRIX_HALO_EXECUTION_PLAN.md`).

---

## 3. Recommended sequencing

1. **Now (hardware-free, highest ROI):** A4 (tuned-config DB) + A3 (epilogue bit-flags) +
   A1 (unified `tessera_rocm.mma` type) — pure IR/metadata/dispatch, each lit-testable and
   guardable without silicon.
2. **Next (still hardware-free):** A2 (layout types), A6 (FP8 FNUZ/OCP + numeric_policy),
   A5 (untuned worklist).
3. **At Strix Halo:** B4 (real intrinsics — the actual blocker) → B1/B2/B3 (GEMM ladder) →
   B5 (counter scoring). Where `backend_kernel` finally moves off 0.
4. **Distributed track (parallel, longer horizon):** C1 → C2 → C3.

---

## 4. Sources

- AITER — https://github.com/ROCm/aiter · DeepWiki https://deepwiki.com/ROCm/aiter ·
  GEMM auto-tuning https://deepwiki.com/ROCm/aiter/5.3-gemm-auto-tuning-infrastructure
- ATOM — https://github.com/ROCm/ATOM · DeepWiki https://deepwiki.com/ROCm/ATOM
- hipBLASLt — https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt
- rocWMMA — https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocwmma
- Mori — https://github.com/ROCm/mori ·
  vLLM+MoRI distributed https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/vllm-mori-distributed.html
- Iris — https://github.com/ROCm/iris · paper https://arxiv.org/html/2511.12500v1
- XIO — https://github.com/ROCm/rocm-xio · https://rocm.docs.amd.com/projects/rocm-xio/en/beta-0.1.0/what-is-xio.html
- Gluon GEMM tutorial — https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html ·
  Triton Gluon intro https://triton-lang.org/main/getting-started/tutorials/gluon/intro.html ·
  layouts https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html

---

*See also: [`ROCM_AUDIT.md`](ROCM_AUDIT.md) (status) ·
[`STRIX_HALO_EXECUTION_PLAN.md`](STRIX_HALO_EXECUTION_PLAN.md) (hardware bring-up) ·
[`../BACKEND_AUDIT.md`](../BACKEND_AUDIT.md) (cross-backend) ·
`../../../rocm_mfma_kernel_inventory.md` (kernel inventory).*
