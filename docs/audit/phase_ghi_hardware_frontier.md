# Phase G/H/I Hardware-Gated Frontier

**Last updated:** 2026-05-22 (Sprint M — MLA E2E + V3b/V3c verifier closure)

This document is the **single, honest answer** to the question "what
exactly is hardware-blocked vs. still hardware-free in the Tessera
backend coverage frontier?"

It synthesizes:

- The registry's authoritative per-axis status from
  `python/tessera/compiler/primitive_coverage.py` (432 entries × 12
  contract axes).
- The per-target audit docs (`nvidia_execution_audit.md`,
  `nvidia_rocm_execute_and_compare_plan.md`,
  `apple_ga_ebm_native_execution_gap.md`).
- The kernel inventory docs (`nvidia_cuda13_kernel_inventory.md`,
  `rocm_mfma_kernel_inventory.md`, `metalium_kernel_inventory.md`,
  `apple_gpu_kernel_inventory.md`).

The point of having one frontier doc is to give the next person on
real hardware a single-sheet punch list, and to give CPU-only
contributors a single-sheet honest answer to "is this closeable
from my machine?"

---

## 1. What `backend_kernel = complete` means

Per the registry rule documented at
`primitive_coverage.py:351-352`:

> `backend_kernel` stays `partial` until each backend ships a real
> hardware kernel — that's Phase G/H/I work.

"Each backend ships a real hardware kernel" means **every declared
target produces a kernel that has been numerically validated
against a reference oracle on real hardware**, with the
proof checked into the test tree and a benchmark JSON checked
into `benchmarks/`.

By construction, this means:

- **Apple GPU proofs alone are not sufficient** to flip
  `backend_kernel` to `complete`. The registry tracks a primitive,
  not a (primitive, target) pair — so even after Phase 8.4.7 fully
  closes apple_gpu for the canonical-tensor ops, the NVIDIA and
  ROCm slots for those same ops remain `planned` until real
  hardware proof lands.
- **CPU/x86 reference paths are not sufficient either.** They
  satisfy the `reference` status in the backend manifest but the
  registry-level `backend_kernel` axis requires the GPU/accelerator
  surface.

This is intentional: a "complete" primitive is one where every
hardware lane Tessera advertises actually runs.

---

## 2. Current registry truth (2026-05-22)

```
total entries: 432

backend_kernel       partial: 273   planned: 159   complete: 0   not_applicable: 0
sharding_rule        complete: 352  partial: 30    not_applicable: 50
transpose_rule       complete: 288  partial: 40    not_applicable: 104
batching_rule        complete: 346  partial: 36    not_applicable: 50
vjp                  complete: 238  not_applicable: 141  planned: 53
jvp                  complete: 237  not_applicable: 142  planned: 53
math_semantics       complete: 406  not_applicable: 26
shape_rule           complete: 406  not_applicable: 26
dtype_layout_rule    complete: 406  not_applicable: 26
masking_effect_rule  not_applicable: 352  complete: 43  planned: 37
lowering_rule        complete: 382  not_applicable: 50
tests                complete: 432  (no remaining gaps)
```

**The single dominant gap is `backend_kernel`** — zero entries are
`complete`, all 432 carry either `partial` (273 — at least one
target ships a kernel) or `planned` (159 — no target kernel yet).

---

## 3. Hardware-blocked vs. hardware-free closures

### 3a. Hardware-required (not closeable from Apple Silicon)

These all require an actual NVIDIA H100 / A100 / ROCm MI300 /
Tenstorrent Metalium device to flip:

| Slot | Blocker | Reference proof plan |
|------|---------|--------|
| Every entry's `backend_kernel = complete` | NVIDIA H100 + ROCm MI300 + (where declared) Tenstorrent Metalium running the per-target kernel with numerical proof | `nvidia_rocm_execute_and_compare_plan.md` §4–§7 |
| The 159 `backend_kernel = planned` entries | A kernel artifact exists for **zero** target. Needs at minimum one target's kernel landing (any of NVIDIA / ROCm / apple_gpu / x86) | per-target kernel inventory docs §"planned" sections |
| 30 `sharding_rule = partial` entries | Phase G mesh integration: real NCCL/RCCL multi-rank execution with output comparison to the mock-mesh oracle | `nvidia_execution_audit.md` §3.3 |
| Long-tail `transpose_rule = partial` (40) and `batching_rule = partial` (36) | Same as sharding: needs real per-target verification beyond the closed-form rule | per-axis bucket audits in `primitive_coverage_state.md` |
| `masking_effect_rule = planned` (37) | These are mostly stateful KV/memory ops where the effect rule needs runtime-witness verification, only meaningful on the device that actually mutates state | KV/memory op set in `kv_cache_coverage_matrix.md` |

### 3b. Still hardware-free (could be closed from this machine)

After scanning the registry post the 2026-05-22 multi-pass closure
rounds, **the residual hardware-free closure work is small and
covers axes other than `backend_kernel`**:

| Axis | Remaining count | Notes |
|------|----------------|-------|
| `vjp = planned` | 53 | Most are stateful/structural ops where N/A is the eventual right answer — needs case-by-case audit, not a sweep. Trying to write a VJP for `kv_cache_append` is wrong; the right move is `not_applicable` once the audit confirms the gradient is undefined. |
| `jvp = planned` | 53 | Same shape as vjp. |
| `masking_effect_rule = planned` | 37 | Subset reachable by category overrides; the remaining are device-state ops as above. |

The **historically dominant hardware-free axes** (`math_semantics`,
`shape_rule`, `dtype_layout_rule`, `lowering_rule`, `tests`) are
already at zero partial/planned — those were closed by the
multi-axis category-based hardening pass on 2026-05-10 and the
final-stage closure pass on the same day.

---

## 4. Per-target frontier

### NVIDIA (Phase G)

Pinned: **CUDA Toolkit 13.2 Update 1, NCCL 2.22, driver ≥555.85**
(`compiler/gpu_target.py`).

Hardware-free pre-work that has landed:

- G-1 toolchain pin + per-SM feature matrix (12 flags across
  SM_70→SM_120, incl. wgmma_sparse / tcgen05_pair / cluster_launch)
- G-2 50+ planned fused kernels documented in
  `nvidia_cuda13_kernel_inventory.md`
- G-3 `BackendKernelEntry` schema extension (cuda_arch_min,
  wgmma_shape, cluster_size, expected_mfu, roofline_target)
- G-4 8 lit fixtures under `phase3/cuda13/`
- G-5 4 NVIDIATargetPipeline named aliases in `tessera-opt`
- G-6/G-7/G-8 `cmake/TesseraToolchainPins.cmake` +
  `validate_nvcc_compile.py` (covers 8 PTX patterns)
- G-9 `AdapterVersionPin.h` `#error` enforcement of NCCL ≥ 2.22

Still hardware-required:
- Running any of the 8 G-4 lit fixtures' PTX through `nvcc`,
  loading on H100, executing, comparing output to numpy reference.
- All `nvidia_rocm_execute_and_compare_plan.md` §4 entries.

### ROCm (Phase H)

Pinned: **ROCm 7.2.3, HIP 7.2.3, RCCL 2.22**
(`compiler/rocm_target.py`).

Hardware-free pre-work landed:

- H-1 toolchain pin + per-arch feature matrix (gfx90a / 940 / 942 /
  950 / 1100)
- H-2 `mfma_table.inc` X-macro generator + 22-shape C++ table
- H-3 ROCm kernel inventory doc (per-arch MFMA / WMMA surface)
- H-4 6 lit fixtures under `phase8/rocm_7_2/`
- H-6/H-7/H-8 `validate_hipcc_compile.py` (8 AMDGCN intrinsics)
- H-8 `AdapterVersionPin.h` shared NCCL/RCCL ≥ 2.22 gate

Still hardware-required:
- Compiling the H-4 fixtures through `hipcc`, loading on MI300,
  executing, comparing output.

### Tenstorrent Metalium (Phase I)

Hardware-free pre-work landed:

- I-1 3 lit fixtures (softmax / layer_norm / rmsnorm decompose
  through dma + tile-local matmul)
- I-2 `compileable` status + planned/gated `metalium_blockfp`
  target (bfp8 / bfp4)
- I-3 `metalium_kernel_inventory.md` (RISC-V grid mapping,
  BRISC/NCRISC/TRISC0/Packetizer roles)

Still hardware-required:
- TT-Metalium runtime execution of any I-1 fixture against a real
  Tenstorrent device, plus Tenstorrent's blockfp lane.

### Apple GPU (operational from this Mac)

Phases 8.3 → 8.4.7 are landed and tested on this machine. This is
the **only target where backend execution is closed on this
machine.** Sprint M (2026-05-22) added the first model-shaped E2E
proof: `tests/unit/test_apple_gpu_mla_e2e.py` runs an MLA-flavored
single-layer multi-head attention decoder (3 projections + per-head
fused matmul→softmax→matmul + output projection) and validates
output against a numpy reference at fp32 rtol=1e-4 across three
shapes (T=8/D=16/H=2, T=16/D=32/H=4, T=32/D=64/H=8).

This proof exercises:

- Composition of 2 jitted callables (`project`, `per_head_attn`)
  invoked 4× and (num_heads)× respectively in a single test.
- The Phase 8.4.5 fused MSL kernel `matmul_softmax_matmul_f32` for
  per-head attention.
- The Phase 8.3 MPS path for the 3 projections + output
  projection.
- Target IR sanity: per_head_attn lowers to exactly one
  `tessera_apple.gpu.msl_kernel` with `fusion =
  "matmul_softmax_matmul"`.
- Darwin-specific `execution_mode == "metal_runtime"` (the
  portable reference fallback path is skipped on Darwin).

What's still left for Apple GPU before its slot reaches the
formal `complete` definition:

- MLA-specific kernels (decoupled-rope split, compressed-KV
  decode, head absorption) — Phase 8.4.8.
- bf16 fused-attention numerical envelope (this E2E proof is
  fp32 only; the per-kernel f16/bf16 envelope is locked in
  `test_apple_backend_roadmap.py` but the *composed* MLA at f16/
  bf16 is a follow-up).
- Phase G/H closure of the NVIDIA/ROCm equivalents (which is the
  Phase-G/H gate per §1 above).

---

## 5. What "close Phase G/H/I from this machine" means in practice

Given the registry's per-target requirement, **Phase G/H/I cannot
be marked `complete` from this machine.** The closest closure
this machine can produce is what Sprint M just shipped:

1. The single-pass MLA-style E2E proof on apple_gpu
   (`test_apple_gpu_mla_e2e.py`).
2. This frontier audit doc, which makes the hardware boundary
   explicit and gives the next person a concrete punch list.
3. The V3b + V3c verifier closures (separate sprint — already
   landed).

Anything beyond this — flipping `backend_kernel` from `partial`
to `complete` for any entry — requires real NVIDIA/ROCm/Metalium
hardware. The audit docs and validation scripts are in place;
running them is the missing step.

---

## 6. References

- `docs/audit/nvidia_execution_audit.md` — Phase G punch list (8 tasks)
- `docs/audit/nvidia_rocm_execute_and_compare_plan.md` — Phase G/H execute-and-compare matrix
- `docs/audit/standalone_primitive_coverage.md` — generated 12-axis dashboard
- `docs/audit/primitive_coverage_state.md` — historical narrative
- `docs/audit/sharding_partial_audit.md` — sharding-rule partial bucket A/B/C
- `docs/audit/kv_cache_coverage_matrix.md` — KV cache per-target coverage
- `docs/nvidia_cuda13_kernel_inventory.md` — NVIDIA fused kernel surface
- `docs/rocm_mfma_kernel_inventory.md` — ROCm MFMA surface
- `docs/metalium_kernel_inventory.md` — Tenstorrent surface
- `docs/apple_gpu_kernel_inventory.md` — every Apple GPU C ABI symbol
- `tests/unit/test_apple_gpu_mla_e2e.py` — first MLA E2E proof (Sprint M)
- `python/tessera/compiler/primitive_coverage.py` — the source of truth
