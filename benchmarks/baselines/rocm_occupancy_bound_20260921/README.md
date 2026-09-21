# An occupancy-bound ROCm kernel, found and measured — 2026-09-21

> **CORRECTED 2026-09-21 — the headline below is wrong. The +25% is real; the
> attribution is not.** Residency never changed: these kernels launch
> `block=32` (one wave per work-group), so LDS binds at 7 resident groups in
> *both* arms and the register lever moved a non-binding ceiling. The model was
> fed `max_flat_workgroup_size` (256, a maximum) instead of the launch size.
> The actual mechanism is **memory-level parallelism** — mem-ops in flight per
> wait 2.26 → 4.80, `vmcnt(0)` full drains 49.8% → 17.3%, with byte-identical
> memory traffic. Raised by the Codex reviewer on PR #791 and confirmed by
> measurement. Read
> [`rocm_mlp_correction_20260921/`](../rocm_mlp_correction_20260921/README.md)
> instead; everything below is retained as the record of what was claimed.


Sync `RDNA-OCCUPANCY-GRANULE-2026-09-20`. Answers the question PR #790 left as
the honest next one: *no occupancy-bound ROCm kernel has been measured*. One
exists. Device work on **Princess-Luna** (gfx1151, RDNA 3.5, ROCm 10.0).

## The finding

**Backward attention, split-reduced, D=128 gains +25% from a spill-free
7 → 8 wave step** — the same lever that cost the forward two-wave kernel 2.2×
in PR #790.

| case | schedule | baseline | `waves_per_eu=8` | delta |
|---|---|---:|---:|---|
| (1,16,16,1024,128) | serial_dkdv *(control)* | 10.0862 ms | 10.0920 ms | −0.1% |
| (1,16,16,1024,128) | **split_reduced** | **10.4289 ms** | **8.3225 ms** | **+25.3%** |
| (1,16,16,1024,128) | serial_dkdv *(control)* | 5.9016 ms | 5.9497 ms | −0.8% |
| (1,16,16,1024,128) | **split_reduced** | **5.9068 ms** | **4.6740 ms** | **+26.4%** |
| (1,16,4,1024,128) | serial_dkdv *(control)* | 6.6324 ms | 6.6515 ms | −0.3% |
| (1,16,4,1024,128) | **split_reduced** | **6.8225 ms** | **6.2947 ms** | **+8.4%** |

Nine interleaved trials per point. The scaffold was gated to the
**split-reduced variant only**, so `serial_dkdv` in the same run is an
untouched control — it moves by at most 0.8% on every row.

Register effect, zero spills:

| kernel | baseline | at `waves_per_eu=8` |
|---|---|---|
| `fa_dkdv` | 209 VGPR / **7 waves** | 192 VGPR / **8 waves**, 0 spills |
| `fa_dq` | 172 VGPR / 8 waves | unchanged |
| `fa_pre` | 104 VGPR / 12 waves | unchanged |

## Why it wins here and lost there

The occupancy number did not predict the sign in PR #790 and does not predict
it here. **What predicts it is whether the register constraint improves or
degrades the schedule** — readable from the compiled code:

| experiment | regs shed | schedule change | occupancy | outcome |
|---|---:|---|---|---:|
| fwd two_wave D=128 (PR #790) | 8 | `s_waitcnt`/100 **2.78 → 3.70 (+33%)** | 10 → 12 | **−55%** |
| bwd split D=64 | 62 + 108 | `fa_dq` wait/100 **4.24 → 4.95 (+17%)**, instrs +8% | 6 → 8/10 | **−23…−33%** |
| **bwd split D=128** | **17** | **`fa_dkdv` 7.44 → 6.56, `fa_dq` 6.04 → 5.50; instrs −65 and −70** | **7 → 8** | **+25%** |

At D=128 the constraint made the schedule *strictly better* — both dominant
kernels lost instructions **and** lost waits — so the extra wave came free.
Everywhere the schedule degraded, occupancy lost despite gaining *more* waves.

**Occupancy pays exactly when it is free.**

## Two explanations excluded by measurement

* **Not spills.** The D=64 build at `waves_per_eu=8` has zero spills and zero
  scratch and still loses. Spilling appears only at `waves_per_eu ≥ 9`, where
  the loss deepens.
* **Not an under-filled grid.** The D=64 regression *worsens* with grid size —
  −23% at 64 q-tiles, −28% at 512, −33% at 2048 — with the serial control flat
  (0.3398/0.3387, 10.62/10.58, 80.72/81.96 ms). An unused-capacity effect would
  shrink as the grid grew; this does the opposite.

## Also learned about the lever

`amdgpu-waves-per-eu` is a *minimum* request and the allocator overshoots it:
asking D=64 for 7 waves produced the same 172/144-VGPR build as asking for 8
(8 and 10 waves). So a kernel cannot always be moved exactly one rung, and the
D=128 and D=64 arms are reported as two separate results rather than a
controlled rung pair.

## Scope

One op family (attention backward), one arch (gfx1151), one direction of the
lever. This establishes that occupancy-bound ROCm kernels exist and that the
gain can be large; it does **not** make occupancy a ranking signal — the same
op at a different head dimension goes the other way by 30%.

The `waves_per_eu` scaffold is deleted (Decision #29). Reproducing means
re-adding a `rocdl.waves_per_eu` stamp to
`GenerateWMMAFlashAttnBwdKernel.cpp`'s `mk` lambda, gated to `splitReduced`.

## Files

| file | content |
|---|---|
| `g6c_bwd_wpe_base.json` | baseline, 7 waves/SIMD on `fa_dkdv` |
| `g6c_bwd_wpe_8.json` | `waves_per_eu=8` — the +25% result |
| `g6c_bwd_wpe_9.json` | `waves_per_eu=9` — 9 waves but spilling; worse |
| `g6c_bwd_wpe_7.json` | `waves_per_eu=7` — no-op at D=128, confirms the baseline |
