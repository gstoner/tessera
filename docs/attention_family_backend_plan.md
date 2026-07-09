---
status: Planning
classification: Plan
authority: Consolidated backend plan for the DFlash / MSA / Mamba2 attention family
last_updated: 2026-07-09
---

# Attention Family — ROCm / CUDA / x86 Backend Plan

This is the consolidated, tri-backend execution plan for the **diffusion / DFlash
+ MiniMax Sparse Attention (MSA) + Mamba2** attention family. It supersedes the
NVIDIA-only [`msa_cuda_phase3_plan.md`](msa_cuda_phase3_plan.md) by widening it to
**ROCm (gfx1151, RDNA3.5)**, **CUDA (sm_120, consumer Blackwell)**, and **x86
(AVX-512)** — the three lead/CPU targets per Decision #28 (ROCm and CUDA are the
lead performance targets; x86 is the CPU floor). Apple GPU is already native
across this family and is the reference implementation we mirror.

The plan is **grounded, not aspirational**: every "have today" cell below was
verified against source, and every gap cites the closest reusable kernel and the
reference to mirror. Where MASTER_AUDIT prose overstates a lane, this doc records
the honest status (see NVIDIA flash-attention, §Status).

Related docs: [`dflash.md`](dflash.md) · [`msa.md`](msa.md) ·
[`msa_cuda_phase3_plan.md`](msa_cuda_phase3_plan.md) ·
[`porting_advanced_examples.md`](porting_advanced_examples.md). Status truth stays
in [`docs/audit/MASTER_AUDIT.md`](audit/MASTER_AUDIT.md) + the generated
dashboards (Decision #26); this doc is *direction*.

---

## Scope

Five features span the family:

1. **`flash_attn`** (+ GQA/MQA, sliding-window, Gemma soft-cap) — the shared core.
2. **`attn_bias`** — the optional additive mask operand
   (`O = softmax(scale·Q·Kᵀ + attn_bias)·V`). The **DFlash keystone**: DFlash
   *always* passes a bias, so its backend seam is blocked until each target's
   flash lane accepts the operand.
3. **MSA** (`msa_index_scores` → `msa_select_blocks` → `msa_sparse_attention`) —
   exp-free per-GQA-group Top-k selection + exact block-sparse attention.
4. **DFlash** — the block-diffusion draft; its attention core routes through an
   `attention_fn` seam onto a backend flash lane.
5. **Mamba2 / `selective_ssm`** — the diffusion-LM / Nemotron state-space scan.

---

## Status today (verified 2026-07-09)

Legend: ✅ native & executing · ⚠️ executes but unproven/unrecorded · 🟡 partial ·
❌ absent (numpy-reference or planned).

| Feature | ROCm gfx1151 | CUDA sm_120 | x86 AVX-512 | Apple GPU (ref) |
|---|---|---|---|---|
| `flash_attn` core (MHA, scale+causal) | ✅ compiled (WMMA) | ⚠️ `emit/` arbiter FA lane¹ | ✅ native | ✅ native |
| `flash_attn` variants (GQA/MQA/sliding/softcap) | ✅ compiled (WMMA) | ❌ **not in the emit/ FA lane²** | ✅ native | ✅ native |
| **`attn_bias`** | ✅ compiled (WMMA, #328) | ❌ absent | ✅ native (pre-softmax add) | ✅ native |
| MSA sparse core (`msa_sparse_attention`) | ✅ compiled (block-sparse WMMA + GPU top-k) | ❌ artifact_only contract | ❌ **reference-only** | ✅ fused (scalar/tiled f32/f16) |
| MSA IR-artifact mirror (`kv_outer_sparse`) | ❌ (execution exists, no IR mirror) | ✅ the contract (no kernel body) | ❌ | n/a |
| DFlash `attention_fn` seam | ✅ `rocm_attention_fn` (#330) | ❌ (blocked on bias) | ✅ `x86_attention_fn` | ✅ `apple_gpu_attention_fn` |
| `selective_ssm` (Mamba2) | ✅ compiled (naive f32 fwd) | ❌ planned | ✅ native (f32 fwd) | ✅ (Mamba SSD) |

**¹ NVIDIA flash-attention — honest status.** A real `mma.sync` tensor-core FA
kernel exists in the `emit/` plugin framework
([`python/tessera/compiler/emit/nvidia_cuda.py`](../python/tessera/compiler/emit/nvidia_cuda.py):815
`_synthesize_mma_attn_cuda`, candidate `NvidiaMmaAttnCandidate` :911) — nvcc-compiled
for `sm_120a`, launched on-device by the measured arbiter
([`emit/autotune.py`](../python/tessera/compiler/emit/autotune.py) `measured_arbitrate`),
gated by the F4 oracle. **But** it is *not* `hardware_verified` in the manifest
(`_NVIDIA_HARDWARE_VERIFIED` = matmul only, `backend_manifest.py:1787`), *not* in
the execution matrix, has *no* committed baseline (only matmul rows in
`benchmarks/baselines/nvidia_sm120_hot_paths.json`), does *not* use the PTX launch
bridge (that's GEMM-only), and has *no* `attn_bias`. The MASTER_AUDIT "~2.7×
flash-attention / ~6× GEMM-epilogue" figures are prose estimates, **not** committed
measured results. **Matmul (`mma.sync` GEMM) is the only committed-proven NVIDIA
execution.** Recording this FA lane honestly is a Phase 0 deliverable below.

**² NVIDIA FA variants (GQA/MQA/sliding/softcap) — absent.** The `emit/` FA
candidates implement only the plain attention ABI
(`Q,K,V,O,M,Nk,D,Dv,scale,causal` — `emit/nvidia_cuda.py:140` scalar, `:815` mma):
there are **no GQA/MQA head-group arguments, no sliding-window bounds, and no
logit-softcap transform**. So even the ⚠️ core-FA lane does not cover the
variants; they are new design/kernel work on CUDA (Phase 1/§FA-variants below),
not "already present." (ROCm and x86 carry all four variants today.)

### Key source anchors

| What | Where |
|---|---|
| ROCm flash lane (needs bias) | `ROCM_FlashAttnOp` `TesseraROCMOps.td:88-95`; `GenerateWMMAFlashAttnKernel.cpp`; executor `runtime.py:_execute_rocm_compiled_flash_attn` (~2880, reads operands[0:3] only) |
| ROCm MSA core (already executes) | `_execute_rocm_compiled_sparse_attention` `runtime.py:12924`; block-sparse WMMA + `_rocm_block_sparse_topk_select_native` `:12695` |
| ROCm selective_ssm | `ROCM_SelectiveSsmKernelOp` `TesseraROCMOps.td:425`; `_rocm_selective_ssm` `runtime.py:13167` (naive one-thread-per-(b,d), f32 fwd) |
| CUDA MSA lowering (mirror target) | `schedule_ir.py:416` → `tile_ir.py:248` → `target_ir.py:1334` (`cuda_kernel` `status="artifact_only"`, hardcoded `:1339`) |
| CUDA emit/ FA kernel | `emit/nvidia_cuda.py:815` (mma) / `:140` (scalar); arbiter `emit/candidate.py`, `emit/autotune.py` |
| x86 flash + bias (reference shape) | `avx512_flash_attn_f32.cpp:71` (`tessera_x86_flash_attn_f32`) / `:120` (`_ext_f32`, bias/window/softcap); executor `runtime.py:_execute_x86_compiled_flash_attn:3060` |
| x86 selective_ssm | `avx512_ssm_f32.cpp:70` (Cephes exp512, SIMD over state dim N) |
| x86 NSA (assembly pattern for x86 MSA) | `_execute_x86_compiled_nsa` `runtime.py:3363` |
| DFlash attention seam | `nn/functional.py:700` (`attn_core(..., attn_bias=bias)`); `apple_gpu_attention_fn` `dflash.py:746` |
| Generic Graph-IR `attn_bias` | `FlashAttnOp` `Optional<TensorType>:$attn_bias` (dflash.md §1) |

---

## Correctness invariants (cross-backend anchors)

Every phase is gated by these — they are backend-independent and already proven on
the reference/Apple path:

- **MSA dense-equivalence:** with `top_k == num_blocks`, MSA output must equal
  dense (causal) GQA attention bit-for-bit, independent of the (approximate,
  mean-pooled) index scores. This is the first oracle for any MSA kernel.
- **DFlash greedy-invariant:** greedy speculative-decode output == greedy
  autoregressive decode. Holds regardless of draft quality (target verification
  corrects every divergence). This gates any DFlash backend seam.
- **`attn_bias` parity:** `softmax(scale·Q·Kᵀ + bias)·V` must match the numpy
  reference to the target's tolerance (the x86/Apple bar is ≈3e-7 fp32, ≈1e-3 f16).
- **selective_ssm parity:** the scan output matches the numpy reference; chunked
  variants must match the sequential scan.

---

## Phase 0 — Record the NVIDIA emit/ FA lane honestly (no new kernel)

The `emit/` mma.sync FA lane executes but is invisible to the manifest, the
execution matrix, and the perf ratchet — so MASTER_AUDIT's prose and the recorded
status disagree. Close that gap (this is bookkeeping + one hardware proof, not a
kernel build):

1. Wire the `NvidiaMmaAttnCandidate` / `NvidiaFlashAttnCandidate` lanes into the
   execution matrix + a `flash_attn`/`nvidia_sm120` manifest row (status
   `compiled` when the arbiter can emit+launch; `hardware_verified` only once a
   committed baseline exists).
2. On the RTX 5070 Ti box: run the FA lane, add a `flash_attn` row to
   `benchmarks/baselines/nvidia_sm120_hot_paths.json`, and extend
   `test_nvidia_perf_ratchet.py` to assert it. Only then may the manifest read
   `hardware_verified` and MASTER_AUDIT cite a *measured* number.
3. If it cannot be proven on hardware this cycle, downgrade the MASTER_AUDIT prose
   from "~2.7×" to "emit/-executable, hardware-unproven" so status truth and prose
   agree (Decision #25/#26).

**Where:** buildable-now portions (manifest/matrix wiring) on any box; the proof
row is **sm_120-gated** (RTX 5070 Ti).

---

## Phase 1 — `attn_bias` substrate (the keystone)

`attn_bias` unblocks DFlash on every target and is a general structured-mask
feature (any additive mask flows through `flash_attn`, not just DFlash). x86 is the
worked example to mirror.

| Target | Work | Gated on |
|---|---|---|
| **x86** | ✅ **done** — `tessera_x86_flash_attn_ext_f32` adds bias pre-softmax (`avx512_flash_attn_f32.cpp:120`); runtime normalizes the 4th operand (`runtime.py:3095`). Reference shape for the others. | — |
| **ROCm** | ✅ **done (#328)** — `attn_bias` BoolAttr on `ROCM_FlashAttnOp`; a trailing f32 `[bh,Sq,Sk]` memref arg (LAST, so gqa/window/softcap indices stay stable) with `bias[(bh*Sq+qpos)*Sk+gk]` added to the scaled score after soft-cap and before masking; the executor detects operand[3] and host-broadcasts to `Q.lead+(Sq,Sk)`. Validated on gfx1151 (~2e-4, f16) + a GPU-free codegen gate. | — |
| **CUDA** | Design the bias operand into the `emit/` FA kernels (`_synthesize_mma_attn_cuda` + `_synthesize_attention_cuda`) and their C ABI + region signature (currently `(region, Q, K, V)`). Add to the F4 accuracy budget. | **sm_120-gated** |

**Recipe (ROCm, as landed in #328):** ODS BoolAttr → codegen trailing-memref arg
+ score-stage add → executor operand[3] detect + host-broadcast →
`test_rocm_flash_attn_bias_compiled.py` (gfx1151) + `test_rocm_attn_bias_codegen.py`
(GPU-free gate). The directive op carries attributes only (Q/K/V/O are gpu.func
args the pass builds), so bias is a flag + trailing arg — not an IR operand — and
follows the `logit_softcap` variant precedent (no dedicated manifest row).

---

## Phase 2 — DFlash `attention_fn` seams

DFlash's `block_diffusion_attention` / `dflash_draft_forward` route their core
through an `attention_fn` seam (heads folded into batch → rank-3 `flash_attn`,
`attn_bias` always passed). Only `apple_gpu_attention_fn` exists today; DFlash has
**no manifest presence** at all.

| Target | Work | Gated on |
|---|---|---|
| **x86** | ✅ **done** — `x86_attention_fn` over the f32-native AVX-512 flash lane (`_x86_flash_attn`). No dtype/head-dim gate; matches numpy to f32 epsilon. | — |
| **ROCm** | ✅ **done (#330)** — `rocm_attention_fn`: casts f32→f16 (bf16 preserved), calls `_rocm_flash_attn(..., attn_bias=…)`, falls back to numpy off-silicon / head_dim%16≠0. Whole-draft greedy tokens match numpy on gfx1151. | — |
| **CUDA** | `nvidia_attention_fn` onto the emit/ FA lane — after Phase 1 CUDA bias. | Phase 1 (CUDA), sm_120 |

**Invariant:** greedy spec-decode == greedy AR must hold on each seam (already the
DFlash test bar).

---

## Phase 3 — MSA completion

MSA status is uneven: ROCm executes it, CUDA has an artifact-only contract, x86 is
reference-only, Apple is fully native.

| Target | Work | Gated on |
|---|---|---|
| **x86** | **The one x86 gap.** Build the `msa_sparse_attention` AVX-512 lane by **reusing `tessera_x86_flash_attn_ext_f32`** (dense FA over the selected blocks + additive block/causal mask) plus a host/AVX-512 Top-k select — exactly how NSA (`x86_nsa_compiled`, `runtime.py:3363`) is already assembled. Keep `msa_index_scores` (composite helper) and `msa_select_blocks` (low-FLOP top-k) as reference. | **Buildable now** |
| **CUDA** | Replace the `artifact_only` `msa_kv_outer_sparse` contract (`target_ir.py:1339`) with a **real emit/ mma.sync block-sparse kernel** — mirror the emit/ FA lane, KV-outer over selected blocks, online softmax, dense-equivalence oracle. Promote `target_ir.py` off the hardcoded `artifact_only` + add the manifest/fixture rows. | **sm_120-gated** |
| **ROCm** | *Optional* — the sparse core already **executes** via `rocm_sparse_attn_compiled`; only the IR-visible `kv_outer_sparse` mirror of the CUDA lowering is missing. Add a `schedule/tile/target` ROCm branch (`tessera_rocm.*` target op) if IR parity with CUDA is wanted. Lower priority (no execution gap). | Buildable now (low pri) |

**Invariant:** dense-equivalence (`top_k == num_blocks` == dense GQA) is the first
oracle on every MSA kernel, including the x86 build.

---

## Phase 4 — Mamba2 chunked-scan / dtype / backward

`selective_ssm` is native f32-forward on ROCm + x86 (naive scans) and native on
Apple; NVIDIA is planned. The Phase-G "chunked-scan target kernels" are the
optimized variant + dtype coverage + backward.

| Target | Work | Gated on |
|---|---|---|
| **x86** | Upgrade `avx512_ssm_f32.cpp` from the sequential scan to a **chunked scan**; add bf16/fp16 I/O and the backward pass. | Buildable now |
| **ROCm** | Upgrade `_rocm_selective_ssm` from naive one-thread-per-(b,d) to a **chunked/tiled scan**; add fp16/bf16 + backward. | Buildable now |
| **CUDA** | Build the NVIDIA `selective_ssm` kernel (currently planned/absent) via the emit/ framework — mma-friendly chunked scan. | sm_120-gated |

**Invariant:** chunked output matches the sequential scan; dtype variants within
tolerance.

---

## Sequencing

**Buildable + verifiable now on this box (gfx1151 + AVX-512):**

- ✅ **Phase 1 / ROCm `attn_bias`** — the keystone; **landed (#328)**.
- ✅ **Phase 2 / ROCm DFlash seam** — **landed (#330)**, `rocm_attention_fn` over the bias lane.
- ✅ **Phase 2 / x86 DFlash seam** — **landed**, `x86_attention_fn` over the f32-native flash lane.
1. **Phase 3 / x86 MSA lane** — closes the last x86 gap in the family.
2. **Phase 4 / ROCm + x86 chunked SSM** — optimization pass.
3. **Phase 3 / ROCm MSA IR mirror** — optional IR parity, no execution gap.

**Hardware-gated on the RTX 5070 Ti (sm_120) box:**

- Phase 0 (record + prove the emit/ FA lane), Phase 1 CUDA bias, Phase 2 CUDA
  seam, Phase 3 CUDA MSA kernel, Phase 4 CUDA SSM. These need the NVIDIA box; the
  manifest/matrix wiring parts of Phase 0 are buildable anywhere.

One PR per (phase, backend) cell, following the established recipe, each verified
on real hardware with the invariant above as the numerical gate, then generated
docs regenerated (Decision #26).

---

## The compiled-lane recipe (per cell)

The proven per-op pattern (see `selective_ssm` / the block-sparse pair as templates):

- **ROCm:** ODS op in `TesseraROCMOps.td` → `GenerateROCM<Op>Kernel.cpp` pass
  (registered in `Passes.{h,cpp}` + `CMakeLists.txt`) → runtime
  `_build_compiled_<op>_hsaco` + `_execute_rocm_compiled_<op>` + executor-table
  entry → `backend_manifest` entry + `_NUMERICAL_FIXTURES` → `test_rocm_<op>_compiled.py`
  on real gfx1151 → regen dashboards.
- **x86:** `avx512_<op>_f32.cpp` (scalar reference + AVX-512 kernel pair) added to
  **both** CMake lists (static + `tessera_x86_elementwise` shared) → symbol +
  argtypes in `_load_x86_elementwise` → `_execute_x86_compiled_<op>` + executor
  table → `_X86_KERNELS` status → `test_x86_<op>_compiled.py` → regen.
- **CUDA:** an `emit/` `KernelEmitter` candidate (nvcc CUDA-C or `ptx_emit` PTX) →
  arbiter registration (F4 oracle + measured autotune) → manifest/matrix row +
  committed sm_120 baseline → `test_nvidia_*` + ratchet.

---

## Open questions / decisions to make

1. **CUDA MSA kernel path** — reuse the emit/ FA candidate machinery (nvcc CUDA-C
   + arbiter) rather than the classic `ptx_emit.py`/PTX-bridge lane, since the FA
   precedent lives there. Confirm before building.
2. **ROCm MSA IR mirror priority** — execution already works; is IR-visible parity
   with the CUDA `kv_outer_sparse` contract worth the schedule/tile/target-IR
   branch, or is the execution lane sufficient? (Recommend: defer unless a pass
   needs the IR-visible op.)
3. **DFlash real-checkpoint parity** — still network-gated (downloading a
   `z-lab/*-DFlash` checkpoint); independent of backend kernels.
