# NVIDIA/ROCm Execute-and-Compare Plan — Phase G/H Backlog

> Status (2026-05-21): **deferred — gate on hardware enablement.**  This
> document captures the execution-ready plan for the NVIDIA SM_90 and
> ROCm CDNA/RDNA pass-order + execute-and-compare matrices.  It exists
> so that when Phase G/H lights up, the next engineer doesn't have to
> re-derive scope.

---

## Why this is deferred

Three test classes compose into one "execute-and-compare" lane for a
hardware target:

  1. **Pass-order matrix** — locks the implicit attribute hand-off
     between passes in the NVIDIA / ROCm lowering pipelines.
  2. **Runtime ABI lock** — pins the function signatures of every
     runtime symbol the lowering emits.
  3. **Differential / oracle execution** — runs the emitted kernel on
     real hardware and asserts numerical equivalence to a trusted
     reference.

(1) and (2) can ship today; they're source-structural tests.  But the
**combined** test (the one that actually catches semantic bugs) requires
real hardware.  Doing (1) and (2) alone before Phase G means designing
the harness against synthetic targets and rewriting it when hardware
lights up.

Per the user direction (2026-05-21): "NVIDIA/ROCm hardware matrix paired
with Phase G, because that's where pipeline-order tests should become
execute-and-compare tests."

---

## What ships today (already covered elsewhere)

Both NVIDIA and ROCm tracks already have *some* compiler-correctness
testing:

| Lane | Coverage | Reference |
|---|---|---|
| NVIDIA pass-order matrix (12 passes) | source-structural ✅ planned-for-future-this-doc; existing `phase3/cuda13/*.mlir` lit fixtures verify per-pass IR shape | `docs/audit/compiler/COMPILER_AUDIT.md` |
| NVIDIA toolchain pins | ✅ shipped (CUDA 13.2 U1, PTX 8.6, NCCL 2.22) | `cmake/TesseraToolchainPins.cmake`, `gpu_target.py` |
| NVIDIA kernel inventory | ✅ shipped | `docs/nvidia_cuda13_kernel_inventory.md` |
| NVIDIA lit fixtures | ✅ 10 fixtures across SM_80→SM_120 | `tests/tessera-ir/phase3/cuda13/` |
| ROCm toolchain pins | ✅ shipped (ROCm 7.2.3, HIP 7.2.3, RCCL 2.22) | `cmake/TesseraToolchainPins.cmake`, `rocm_target.py` |
| ROCm MFMA table | ✅ shipped (22 shapes across gfx90a/940/942/950/1100) | `mfma_table.inc` |
| ROCm lit fixtures | ✅ 6 fixtures | `tests/tessera-ir/phase8/rocm_7_2/` |
| Adapter version pin | ✅ shipped (`#error` directives in `AdapterVersionPin.h`) | `src/collectives/include/.../AdapterVersionPin.h` |

So the structural floor is in place.  The gap is the **execute-and-
compare** layer.

---

## NVIDIA SM_90 — execute-and-compare plan

### 1. Pass-order matrix (source-structural)

Lock the 12-pass NVIDIA pipeline:

```
EffectAnnotation
  → Canonicalize
  → SwigluFusion
  → MLA / NSA / Hybrid / Lightning / Delta fusion (5 passes)
  → DistributionLowering
  → TileIRLowering
  → WarpSpec
  → AsyncCopy
  → WGMMA
  → TMA
  → NVFlashAttnEmitter
```

**Acceptance criteria:**
- `tests/unit/test_nvidia_pass_order_matrix.py` enumerating the
  canonical order; mirror the Apple GPU + Spectral pattern.
- ~12 dependency-pair tests covering the "fusion before WGMMA" and
  "TMA before flash-attn emitter" macro contracts.
- ~1 day of work; ships **with** the runtime lane (don't split).

### 2. Runtime ABI lock

For every runtime symbol the NVIDIA pipeline emits (currently
~26 planned across SM_80→SM_120 per `nvidia_cuda13_kernel_inventory.md`):

- Pin the symbol name as a `constexpr llvm::StringLiteral`.
- Pin the function signature (arg count + per-arg dtype) as a
  Python-side ABI lock test counting i64/i32/f32 etc.  Same template
  as `test_attn_local_window_2d_apple_gpu.py::test_runtime_symbol_carries_eleven_arg_signature`.

**Acceptance criteria:**
- `tests/unit/test_nvidia_runtime_abi_lock.py` with one test per
  runtime symbol; each pins its 11-or-so-arg signature.
- ~0.5 day; ships with the pass-order matrix.

### 3. Execute-and-compare oracle

The new piece.  For each canonical kernel (matmul-bf16, FA-4 fwd, MLA
decode, NSA sparse attention, AdamW step):

1. Build a minimal Graph IR module.
2. Lower through the canonical pipeline.
3. JIT-compile to PTX via `tessera-translate-mlir --mlir-to-llvmir`.
4. Execute on real H100/H200 hardware.
5. Compare against a numpy/torch oracle at appropriate tolerance
   (bf16: 1e-2, fp32: 1e-5, bitwise where applicable).

**Hardware requirements:** H100 80GB minimum; CUDA 13.2 U1 driver
≥555.85.

**Acceptance criteria:**
- `tests/hardware/nvidia/test_h100_execute_and_compare.py` (new
  subdirectory; `hardware_*` markers from `pyproject.toml`).
- Five canonical kernels covered end-to-end.
- CI lane gated on `TESSERA_HARDWARE_LANE=nvidia-h100`.
- ~5-7 engineering days when hardware is available; can be
  parallelized across kernels.

---

## ROCm CDNA/RDNA — execute-and-compare plan

Symmetric to NVIDIA.  The differences:

### 1. Pass-order matrix
- Same template as NVIDIA, narrower pipeline (~8 passes today: same
  effect + canonicalize prefix, then ROCm-specific
  TileToROCmMFMAEmitter family).

### 2. Runtime ABI lock
- Per-arch (gfx90a / gfx940 / gfx942 / gfx950 / gfx1100); each arch
  has its own MFMA shape table (`mfma_table.inc`).

### 3. Execute-and-compare oracle
- Hardware: MI300X (gfx942) is the primary target; MI325X (gfx950)
  for CDNA 4 FP4/FP6 lanes.
- Same five canonical kernels.  CDNA 4 adds FP4/FP6 dtype variants
  via the WMMA intrinsics.

**Per-arch effort:** ~5-7 days each; serialize on hardware
availability.

---

## Cost summary

| Phase | Item | Effort | Status |
|---|---|---|---|
| Pre-G structural | NVIDIA pass-order matrix | 1 day | ⏳ planned, paired with #2 |
| Pre-G structural | NVIDIA runtime ABI lock | 0.5 day | ⏳ planned, paired with #1 |
| Phase G | NVIDIA execute-and-compare (5 kernels) | 5-7 days | ⏳ gated on H100 access |
| Pre-H structural | ROCm pass-order matrix | 1 day | ⏳ paired with #2 |
| Pre-H structural | ROCm runtime ABI lock | 0.5 day | ⏳ paired with #1 |
| Phase H | ROCm execute-and-compare per arch | 5-7 days × 2-3 archs | ⏳ gated on MI300X/MI325X access |

**Total minimum to first NVIDIA H100 oracle lane:** ~7 engineering days
after hardware lights up.

---

## What the user gets from this doc today

1. A locked-in deferral with a clear hardware gate (no ambiguity about
   "is this happening now").
2. An explicit list of pre-G structural tests that *could* ship today
   but are intentionally bundled with the execute-and-compare lane
   they pair with — avoiding split work and design debt.
3. A concrete day count so when Phase G enables, scoping is fast:
   ~7 days for NVIDIA, ~10-15 days for ROCm (2-3 archs).
4. A pointer to the existing structural coverage (toolchain pins,
   kernel inventory, MFMA table, lit fixtures, adapter version pin)
   so the next engineer doesn't double-cover.

## How to revive this plan

1. Confirm hardware availability — H100 + MI300X / MI325X.
2. Bundle pre-G structural tests (NVIDIA pass-order matrix + ABI lock)
   into the same PR as the first NVIDIA execute-and-compare kernel.
   Mirror for ROCm.
3. Update `docs/audit/compiler/COMPILER_AUDIT.md` to flip
   the NVIDIA/ROCm rows from ⏳ to ✅ as each lands.
4. Mark this plan document as **superseded** when all six items above
   are shipped.

## Related docs

- `docs/audit/compiler/COMPILER_AUDIT.md` — coverage matrix
  this defers from.
- `docs/audit/roadmap/ROADMAP_AUDIT.md` — Phase G/H roadmap.
- `docs/nvidia_cuda13_kernel_inventory.md` — what NVIDIA kernels need
  oracle coverage.
- `docs/rocm_mfma_kernel_inventory.md` — same for ROCm.
- `tests/unit/test_attn_local_window_2d_apple_gpu.py::test_runtime_symbol_carries_eleven_arg_signature`
  — template for the ABI-lock pattern.
- `tests/unit/test_halo_execution_lane.py` — template for the
  execute-and-compare pattern (mock-collective today; hardware tomorrow).
