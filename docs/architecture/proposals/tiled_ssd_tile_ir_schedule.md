---
status: Decided (deferred, sequenced to NVIDIA/AMD execution)
classification: Architecture Decision
authority: Proposal; defers to docs/architecture/Compiler/Tessera_Compiler_TileIR_Design.md
last_updated: 2026-07-14
---

# Tiled Fused SSD as a Tile-IR Schedule — Architecture Decision

> **Status (2026-06-07): decided, deferred, sequenced to NVIDIA/AMD execution.**
> The Mamba-2 selective-SSD fusion is **not** an Apple-runtime kernel; it is a
> **Tile-IR tiled schedule** with the matmul intrinsic selected per backend.
> Apple is the *executable validation backend* for that schedule, not its owner.
> The current Apple `selective_ssm` path (chunked-parallel, 3 MPS-`bmm` dispatches
> + host elementwise) stays as the correct functional reference until the tiled
> schedule lands.
>
> This document locks the decision so the next engineer does not re-derive scope,
> and so nobody ships a one-off Metal SSD kernel that would have to be reworked
> for the multi-backend path.

---

## TL;DR decision

1. **SSD is matmul-dominant by construction** (that is the entire point of the
   Mamba-2 "SSD" reformulation vs the original selective scan). The fused kernel
   on *every* backend is a **tiled GEMM**, not a monolithic elementwise kernel.
2. Therefore the fusion belongs at **Tile IR** (Decision #19 — hardware-free
   layer), as a tiled schedule that lowers to the backend's tensor-core matmul:
   `simdgroup_matrix` (Apple), WGMMA/`tcgen05` (NVIDIA), MFMA (AMD).
3. **Do not** build the naive one-thread-per-`(b,d)` fused kernel on Apple. It
   recomputes the shared gram `C@Bᵀ` `D` times and is *slower* than the current
   3-`bmm` path. It is also the exact anti-pattern you would never write on
   NVIDIA/AMD.
4. **Apple is the proving ground for this schedule.** Build the tiled schedule
   once, validate it bit-exact on Apple (which executes `simdgroup_matrix`
   today; the tiled-SSD schedule does not yet execute on the NVIDIA sm_120 /
   ROCm gfx1151 lanes), and let the NVIDIA/AMD lowerings inherit the schedule
   and slot in their matmul intrinsic when hardware lights up.

---

## Why SSD = matmul (the load-bearing fact)

Mamba-2's SSD (Dao & Gu, *Transformers are SSMs*, 2024) rewrites the selective
state-space recurrence over a chunk of length `L`, scalar state-decay `a_t`, into
a **masked attention-shaped matmul**:

```
intra-chunk:  Y_intra = ( L_decay ⊙ (C Bᵀ) ) @ X
inter-chunk:  carry the chunk-end state h_L across chunks (a short recurrence)
```

where `L_decay[t,s] = exp(Dcum_t − Dcum_s)` for `s ≤ t` (bounded in `(0,1]`) and
`Dcum = cumsum(delta·A)`. The expensive work — the gram `C@Bᵀ` and the masked
contraction `@ X` — is **GEMM**. That is *why* SSD exists: to run on tensor cores
at high MFU instead of a serial scan. (Tessera's current chunked-parallel
reference, `python/tessera/_mamba_ssd.py::selective_ssm_parallel`, is exactly this
algorithm, validated bit-exact ~1e-15 against the sequential reference.)

**Consequence:** the gram `C@Bᵀ` depends on `(b, t, s)` and is *independent of the
channel `d`*. A correct kernel computes it **once** and reuses it across all `D`
channels. Any kernel that recomputes it per channel is wrong-by-construction for
performance — and that is precisely what tensor-core tiling (shared operands in
fast memory) is built to avoid.

---

## The anti-pattern (do not build this)

A single MSL kernel with **one thread per `(b, d)` channel**, each thread
computing its channel's full chunk output:

- Recomputes `C@Bᵀ` `D` times (loses the cross-channel gram sharing).
- Re-reads `B[b]`, `C[b]` (`L×N` each) from global memory per channel.
- Loses to the current 3-`bmm` path, where MPS computes the gram once on a tuned
  kernel.

This is the kernel my 2026-06-07 analysis flagged as a *pessimization*. It is also
structurally impossible on NVIDIA/AMD (tensor cores require tiled operands in
shared memory). **Skip it.**

---

## The cross-platform tiled design (build this — at Tile IR)

One schedule, three backends. Only the fast-memory keyword, the matmul intrinsic,
and the barrier differ:

| Schedule element | Apple | NVIDIA | AMD |
|---|---|---|---|
| Tile / fast memory | `threadgroup` | `__shared__` | LDS |
| Gram + masked matmul | `simdgroup_matrix` | WGMMA / `tcgen05` | MFMA |
| Tile-carry barrier | threadgroup barrier | `mbarrier` / cluster | `s_barrier` |
| Tile shapes | (see Apple kernel inventory) | `_NVIDIA_KERNEL_TILE_SHAPES` | `_ROCM_KERNEL_MFMA_SHAPES` |

**Schedule (per chunk, scalar-state `A`):**

1. Load `C[b]`, `B[b]` tiles (`L×N`) into fast memory.
2. `M = C@Bᵀ` (`L×L`) via the backend matmul intrinsic → fast memory. *Computed
   once per chunk, shared across all `D` channels.*
3. Compute `Dcum = cumsum(delta·A)` per channel; form the lower-triangular decay
   `L_decay[t,s] = exp(Dcum_t − Dcum_s)` in registers (bounded, stable).
4. `Y_intra = (L_decay ⊙ M) @ (delta·X)` via the matmul intrinsic.
5. `Y_state = exp(Dcum) ⊙ (C @ h0)` (the carried-in state contribution).
6. Carry `h_L = exp(Dcum_L)⊙h0 + Σ_s decay·(delta·X)·B` to the next chunk
   (the inter-chunk recurrence; few iterations).

Steps 2 and 4 are the GEMMs that map to tensor cores. The decay mask (3) and the
chunk-state carry (6) are register/shared-memory elementwise + a short serial
loop. This is the canonical Mamba-2 SSD kernel; it is identical in structure on
all three backends.

**General `(D, N)` `A` (non-scalar state-decay)** does not reduce to a single
masked matmul (the decay couples `(d, n)`); keep it on the sequential reference,
same as today. Real Mamba-2 uses scalar/per-head `A`, which is the matmul case.

---

## Where it lives in the IR stack

Per **Decision #19 (hardware-free Target IR)** and the existing Tile-IR design
(`docs/architecture/Compiler/Tessera_Compiler_TileIR_Design.md`):

- A **Tile-IR tiled-scan schedule** expresses the chunk tiling, the
  gram-tile-in-fast-memory, the masked-matmul + decay, and the chunk-state carry.
- The **backend lowering selects the matmul intrinsic** (`simdgroup_matrix` /
  WGMMA / MFMA) — the same place the WGMMA tile shapes
  (`_NVIDIA_KERNEL_TILE_SHAPES`) and MFMA shapes (`_ROCM_KERNEL_MFMA_SHAPES`)
  already live.
- **Not** an Apple-runtime-only `.mm` kernel. The whole value is that the schedule
  pays off three times.

This makes the SSD fusion *compiler-IR work*, consistent with the rest of the
tile-centric model, rather than a one-backend runtime hack.

---

## Apple as the executable proving ground

Apple/Metal executes this matmul-family schedule today; the NVIDIA sm_120 and
ROCm gfx1151 lanes now have hardware-verified execution for the matmul /
flash-attention family, but not yet for the tiled-SSD schedule, and broader
archs stay hardware-gated. So Apple remains the executable proving ground here:

- Build the tiled SSD **Tile-IR schedule** and prototype its lowering on Apple
  (`simdgroup_matrix` + `threadgroup` memory).
- Validate **bit-exact** against `_mamba_ssd.selective_ssm_parallel` (the
  chunked-parallel reference) and the sequential `selective_ssm`.
- The validated Apple lowering is the **executable oracle** the NVIDIA/AMD
  lowerings mirror — same schedule, swap WGMMA/MFMA — which front-loads the
  hardest part of the eventual NVIDIA/AMD SSD and de-risks it before silicon.

---

## Sequencing

1. **Now → until NVIDIA/AMD execute:** keep `selective_ssm` on the current
   chunked-parallel path (3 MPS-`bmm` + host). It is correct, functional on GPU,
   and a faithful reference. *Do not* ship the naive Apple fused kernel.
2. **When NVIDIA/AMD execute-and-compare begins** (the P0 backend-kernel
   hardware-proof item): co-design the tiled SSD **at Tile IR**, prototype +
   validate on Apple, then add the NVIDIA/AMD matmul-intrinsic lowerings. Built
   once, portable.
3. **Optional accelerator:** building the Apple Tile-IR tiled-SSD lowering *ahead*
   of NVIDIA/AMD is defensible (gives a bit-exact executable template early). The
   cost is a meaty, numerically-delicate kernel; the benefit is it front-loads the
   NVIDIA/AMD SSD. Take this only if SSD MFU is on the critical path.

## Acceptance criteria (when the tiled schedule lands)

- A Tile-IR tiled-scan schedule with chunk tiling + gram-tile-in-fast-memory +
  masked-matmul + decay + chunk-state carry.
- Apple lowering (`simdgroup_matrix`) executes it, **bit-exact** vs
  `selective_ssm_parallel` and the sequential reference, across shapes / chunk
  sizes / gate / initial state / long sequences (the stability case).
- NVIDIA/AMD lowerings select WGMMA/MFMA from the existing tile-shape tables;
  lit fixtures FileCheck the emitted intrinsics (hardware-free), with
  execute-and-compare gated on real silicon.
- Scope guard: scalar-state `A` only on the tiled path; `(D, N)` `A` stays on the
  sequential reference.

---

## Why not just keep the 3-`bmm` Apple path forever?

It is fine *for Apple in isolation* — the contractions already run on tuned MPS
`bmm`. But it is **not** the design that maps to tensor cores, so it does not
inform the NVIDIA/AMD backends at all. The tiled schedule is the one artifact that
serves all three. The decision here is to invest in that artifact *at the right IR
level and the right time*, not to micro-optimize one backend.

---

## Cross-references

- Current reference: `python/tessera/_mamba_ssd.py` (`selective_ssm_parallel`),
  `python/tessera/runtime.py` (`_apple_gpu_dispatch_selective_ssm`).
- Tests: `tests/unit/test_mamba_ssd_gpu.py`.
- IR design: `docs/architecture/Compiler/Tessera_Compiler_TileIR_Design.md`,
  `docs/architecture/Compiler/Tessera_Compiler_TargetIR_Design.md`.
- Backend tile-shape truth: `python/tessera/compiler/backend_manifest.py`
  (`_NVIDIA_KERNEL_TILE_SHAPES`, `_ROCM_KERNEL_MFMA_SHAPES`),
  `docs/backends/nvidia/kernel-inventory.md`, `docs/backends/rocm/kernel-inventory.md`.
- Decision #19 (hardware-free Target IR) — `CLAUDE.md`.
