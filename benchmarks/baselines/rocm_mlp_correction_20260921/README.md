# Correction to PR #791: the +25% was memory-level parallelism, not occupancy

Sync `RDNA-OCCUPANCY-GRANULE-2026-09-20`. Device work on **Princess-Luna**
(gfx1151) and **Tajasarus** (gfx1201).

PR #791 reported a real, reproducible **+25%** on backward attention D=128 and
attributed it to an occupancy gain (7 → 8 waves/SIMD). **The speedup is real.
The attribution was wrong.** Raised in review by the Codex reviewer, confirmed
here by measurement.

## What was wrong, and why

I fed the occupancy model `max_flat_workgroup_size` (**256**, a *maximum*)
instead of the actual launch geometry. `runtime.py` launches these kernels with
**`block=32` — one wave per work-group**. With one wave per group, LDS binds
long before registers:

| build | VGPR | by_vgpr | by_lds | **resident groups/WGP** | limiter |
|---|---:|---:|---:|---:|---|
| baseline | 209 | 7 | 2 | **7** | **lds** |
| `waves_per_eu=8` | 192 | 8 | 2 | **7** | **lds** |

**Residency never changed.** The register lever moved a ceiling that was not
binding. Feeding a *maximum* where the model wanted the *launch* size inverted
which limiter appeared to bind — the model was right, its input was not.

## What actually produced the +25%

Memory-level parallelism — how many loads are in flight when the wave stalls.
Memory traffic is byte-identical across the arms; only the scheduling changed.

| metric | baseline (209 VGPR) | wpe=8 (192 VGPR) | change |
|---|---:|---:|---|
| memory ops | 338 | 338 | **identical** |
| `s_waitcnt` executed | 149 | **70** | **−53%** |
| **mem ops in flight per wait** | 2.26 | **4.80** | **+112%** |
| `vmcnt(0)` full drains | 49.8% | **17.3%** | **−65%** |
| mean outstanding-load tolerance | 3.20 | **7.44** | **+133%** |

Half the baseline's waits are `vmcnt(0)` — a **full drain**, where the wave
blocks until every outstanding load returns. The constrained build cuts those
to 17% and more than doubles loads in flight.

**Why registers were the lever for something that is not a register limit:**
every outstanding load holds its destination VGPR live until the wait retires
it, so loads-in-flight are *bought with registers*. At 209 VGPRs the registers
were tied up in long-lived values; the constraint forced the scheduler to
rebalance toward in-flight loads. It is not "fewer registers is better" — it is
registers reallocated from long-lived values to outstanding loads.

## The predictor, and its limits

`mem_per_wait` and `vmcnt0_pct` are computable from any compiled kernel with no
device run. Scored against five measured arms:

| arm | ΔMLP | measured | |
|---|---:|---:|---|
| bwd128 split+serial @ w8 | **+112%** | **+25…+32%** | ✓ |
| fwd128_1w @ w8 | −95% | **−58%** | ✓ |
| fwd128_2w @ w12 | −88% | **−55%** | ✓ |
| bwd64 @ w8 | −67% | **−23…−33%** | ✓ |
| **fwd64_1w @ w8** | **+75%** | **≈0%** | **✗** |

**Reliable as a negative predictor (4/4). Unreliable as a positive one (1/2).**
The `fwd64_1w` miss is unexplained; the leading hypothesis is a saturation
point — `bwd128` rose 2.26 → 4.80 from far below latency coverage, while
`fwd64` rose 11.43 → 20 from already above it. That hypothesis is **untested**.

## It does not transfer to gfx1201

| kernel | gfx1151 base → wpe8 | gfx1201 base → wpe8 |
|---|---|---|
| bwd128 `fa_dkdv` | mpw 2.26 → **4.80** | mpw 2.11 → **2.11 (inert)** |
| fwd128 one_wave | 40.0 → 1.67 | 32.0 → **32.0 (inert)** |
| linear_attn D=128 | 1.32 → **1.32 (inert)** | 1.98 → **1.98 (inert)** |

gfx1201 emits far fewer registers for the same kernels (dkdv **149** vs 209 —
the gfx12 fragment layout splits K across half-waves), so the lever has no work
to do. **The +25% is a gfx1151 result and does not carry to gfx1201**, which
has the same problem (2.11 mem/wait, 64.7% drains) and no lever for it.

## LDS, not registers, is the occupancy limiter

On both parts, with the real launch geometry, **every** attention and
linear-attention kernel is LDS-bound. The register ceiling sits 2–8× above the
LDS ceiling. `fa_dkdv` at 17408 B gets 7 resident groups; at ≤9362 B it would
get 14.

A residency-only sweep (dynamic LDS padding — never addressed by the kernel, so
the instruction stream is byte-identical) confirms the kernel is genuinely
residency-sensitive:

| LDS pad | resident groups | measured | |
|---:|---:|---:|---|
| 0 | 7 | 10.17 ms | — |
| 8192 | 5 | 13.39 ms | −32% |
| 16384 | 3 | 18.95 ms | −46% |
| 32768 | 2 | 27.21 ms | −63% |

So residency *does* matter to this kernel — it simply was not what the register
lever changed.

## Two traps recorded

1. **`max_flat_workgroup_size` is a maximum, not the launch size.** Feeding it
   to the occupancy model silently inverts which limiter binds. Read the launch
   site.
2. **RDNA4 split the wait counters.** gfx11 emits `s_waitcnt vmcnt(N)`; gfx12
   emits `s_wait_loadcnt 0xN` plus `s_wait_dscnt`/`s_wait_kmcnt` (ISA Table 4).
   A gfx11-shaped regex reports **zero waits** on gfx1201 rather than failing,
   which is how the first gfx1201 collection in this packet produced `waits=0`
   with `mem_ops=338`.

## Also measured: the pipelining knobs do not reach these families

`lds_copy_depth`, `lds_copy_width`, `lds_double_buffer`, `k_unroll` and
`staging=lds` all produce a **byte-identical hsaco** for both `attention` and
`sequence_linear_attention` (same SHA across seven variants). They are
matmul-body knobs, silently accepted and discarded elsewhere. So
`linear_attn D=128` — the worst-conditioned kernel measured here (1.32
mem/wait, **83.8% full drains** on gfx1151) — currently has **no available
lever**: `waves_per_eu` is inert and the pipelining knobs do not reach it.
Improving it needs a generator change.

## Files

| file | content |
|---|---|
| `evidence_gfx1151.json` | per-kernel VGPR/LDS/spills, launch threads, MLP metrics and the full occupancy verdict at wpe base/8/12 |
| `evidence_gfx1201.json` | the same on gfx1201, with arch-aware `s_wait_loadcnt` parsing |
| `lds_residency_sweep_{0,8192,16384,32768}.json` | residency-only sweep, raw per-trial timings |
| `decompose_arm{A,B,C}.json` | baseline / wpe=8 / wpe=8+LDS-pad arms, raw per-trial timings |

`launch_threads` in the evidence files is read from the `runtime.py` launch
sites, **not** from `max_flat_workgroup_size`.

## What is not claimed

No performance promotion. The +25% stands as measured on gfx1151 for that
kernel and those shapes; it is not a general result, does not transfer to
gfx1201, and its mechanism is scheduling rather than residency. The saturation
hypothesis for the `fwd64_1w` miss is untested.
