# X86-GEMM-ALIGN-1: the x86 f32 GEMM no longer depends on B's alignment — 2026-09-27

`X86-MATMUL-BIMODAL-1` (`../x86_matmul_bimodal_20260926/`) found that
`tessera_x86_avx512_gemm_f32` ran ~1.5x slower at 256³ whenever B was not 64-byte
aligned. The cause: the kernel issued one 64-byte `_mm512_loadu_ps` of B per FMA,
and numpy only guarantees 16-byte alignment. PR #859 aligned the recorder's
buffers. This change fixes the production kernel, so every caller gets the fix:
`runtime.launch` through the x86 descriptor, the matmul-family lane
(`runtime._x86_gemm_2d`) and `TileToX86Pass`'s `func.call`.

Final kernel: `src/compiler/codegen/tessera_x86_backend/src/kernels/avx512_gemm_f32.cpp`
at `86ec9d31` (the before/after sweeps and packets below ran that build).

## What the kernel does

**Two paths, chosen by a measured rule (`packedPathWins`).**

- **Packed.** For each block of up to eight 16-column strips (128 floats) × 512 rows
  of K, the kernel copies B into its own 64-byte-aligned panel (≤ 256 KiB,
  L2-resident).
  - Eight accumulators then run over the panel with aligned loads, sharing each
    broadcast of `A[m,k]`.
  - Between K blocks, the sum continues through C (an exact fp32 store and reload).
  - This is the alignment-insensitive path.
- **Direct.** The same 8-strip, 8-accumulator loop reads B in place, with masked
  unaligned loads.
- **Rule: direct iff M == 1, or M ≤ 4 with B = K·N·4 bytes ≤ 1 MiB; packed
  otherwise.** The rule was measured, not guessed (next section).
- **Allocation fallback.** If the panel cannot be allocated, the direct path runs.
  The result is the same, only slower.

**The strip loops are fully unrolled** (`#pragma GCC unroll 16`), so the
accumulators stay in zmm registers.

- Found during the path measurement: at `-O2`, GCC 15 had left the S ≥ 4 loops
  rolled.
- In that form every FMA step was stack load + FMA + stack store.
- At N = 64, both paths were then 1.4–1.8x slower than the pre-fix kernel (kernel-only
  harness: 16×64×64 at 2.04 µs before, 2.87–2.89 µs after).
- After unrolling, 74 FMAs are inlined and none touch the stack.

**Overlapping C.** The entry point checks whether C's byte range overlaps A's or
B's (`tessera_x86_avx512_gemm_f32_operands_overlap`). The extents are dense
row-major, computed from M/N/K with 128-bit products.

- **On overlap:** the product is computed into an aligned scratch C and copied out.
  The result equals the product of the inputs as they were at entry.
- **If scratch allocation fails:** C is filled with NaN and a message goes to
  stderr (fail-closed).
- **Adjacent-but-disjoint operands** take the fast path.
- **The old kernel was not safe under overlap either.** The same overlap cases,
  linked against the `d8da67f7` kernel, fail 6 of 6. It was right only by accident,
  e.g. C == A with N == K ≤ 16: N == K == 16 passes and 17 fails.
- **Cost of the check:** `princess_luna_overlap_check_cost.jsonl` pairs `de914e51`
  (no check) against `1102a32f` (check). The after/before ratio is 1.00 at 256³, and
  0.86–1.10 (noise) at 32³, 8×64×64 and 1×256×256. That run predates the unrolling
  and path rule; the check itself is unchanged since.

**Numerics: bitwise identical to the pre-fix kernel on both paths, at every
alignment.** The per-element FMA sequence is unchanged: acc = 0, then
`fma(A[m,k], B[k,n], acc)` for k = 0..K−1 in order; masked-off lanes read as zero
and are never stored.

- **`test_gemm_f32.cpp`:**
  - The path table pins `tessera_x86_avx512_gemm_f32_uses_packed_path`.
  - `check_bitwise` compares against the pre-fix kernel, kept as the declared
    oracle, with `memcmp` at every 4-byte B offset. It covers the direct and packed
    paths, tail strips, K = 0, and the K-block boundaries 512 / 513 / 1100.
  - `check_overlap` covers 6 overlapping and 3 adjacent layouts.
  - A mutation that restarts the accumulators at each K block fails 5 cases.
- **Python tests** in `tests/unit/test_x86_matmul_family_compiled.py`: alignment
  independence, overlap, and the path rule.
- **Every probe row** compares the old and new builds' outputs bit for bit
  (`bitwise`), and the `runtime.launch` output against the direct call.

## How the path rule was measured

`run_path_crossover.sh` builds four single-kernel libraries from one checkout:
`old` (`d8da67f7`), `packed` (forced), `direct` (forced) and `shipped`. The forcing
edits only the two `// PATH-PROBE` constants. `probe_gemm_paths.py` then times all
four in one process on the same buffers.

- **Windows:** round-robin interleaved, with the order rotating every sample.
- **Timing source:** each window is timed with the TSC witness.
- **Locking and processes:** every run holds the timing lock, and each cell runs in
  a fresh process.
- **Outputs:** all four are bitwise equal in every cell.

Files, all on Princess-Luna, after the unroll (`0b6a34d2`; `shipped` there still had
the provisional M > 1 rule):

| File | Shapes (N×K) | M | processes per cell |
|---|---|---|---|
| `princess_luna_path_crossover.jsonl` | 64², 256², 1024², 256×4096, 4096×256 | 1–16 | 1 |
| `princess_luna_path_crossover_mid.jsonl` | 128×1024, 1024×128, 512², 384², 256×1024 | 1–16 | 1 |
| `princess_luna_path_crossover_2mib.jsonl` | 1024×512, 512×1024, 2048×256 | 2–4 | 2 |
| `princess_luna_path_crossover_small.jsonl` | 64², 128², 256² | 1–16 | 3 |

What the data says, as packed/direct time ratios:

- **M == 1:** direct wins at every size. Packing was slower by up to 2.1x; the
  64×64 cells sit on the ~4.5 µs ctypes floor and are within noise.
- **M ≤ 4 and B ≤ 1 MiB:** direct was never slower than packed at either B offset
  (for example 512² M=4: 28.0 vs 37.0 µs aligned, 34.2 vs 36.4 µs at B%64 = 16). The
  exception is the 64×64 cells, where the two are within ≤ 6% noise on the ctypes
  floor.
- **B of 2–4 MiB:** packing wins from M = 2 or 3 at B%64 = 16 (1024×512 M=2: 55.2
  vs 52.8 µs; 1024² M=2: 108.5 vs 99.5 µs).
- **B ≤ 1 MiB:** packing wins from M = 6 at B%64 = 16.
- **Where the two are within noise, the rule picks packed**, the alignment-insensitive
  path.

## Before / after (the evidence)

`probe_gemm_align.py` runs in one fresh process per (shape, B%64). It loads the
pre-fix build (`d8da67f7`, origin/main) and the fixed build (`86ec9d31`) of
`libtessera_x86_elementwise.so` into the same process.

- **Timing:** paired interleaved windows on the same A/B/C buffers, with the order
  alternating every sample. Each window is timed with the TSC witness: TSC
  calibrated over separate pinned intervals, region unpinned, checked against
  CLOCK_MONOTONIC_RAW within 5%, and re-verified. The latency reported is the TSC
  one.
- **Production path:** the fixed build is packaged through `package_matmul` →
  `runtime.launch`, and the probe asserts the packaged image payload is the timed
  library.
- **Recorded fields:** each row records which path the fixed build takes
  (`after_path`).
- **Buffers:** A and C are 64-byte aligned; B sits at the stated offset.

`run_probe.sh` drives it and `summarize.py` produces the tables.

**Princess-Luna** (Ryzen AI MAX+ 395, Zen 5, WSL2): `princess_luna_before_after.jsonl`,
204 processes (17 shapes × 4 offsets × 3). All outputs are bitwise equal.

Per-process median, µs, min–max over 3 processes:

| M×N×K | path | B%64 = 0: before → after | B%64 = 16/32/48: before → after |
|---|---|---|---|
| 256×256×256 | packed | 719–722 → 301–312 | 1082–1121 → 302–315 |
| 250×250×250 | packed | 627–679 → 295–296 | 628–686 → 289–314 |
| 256×1000×256 | packed | 2965–3298 → 1198–1271 | 2800–3488 → 1192–1223 |
| 16×1024×1024 | packed | 4766–4880 → 357–363 | 7860–8280 → 360–381 |
| 64×1024×1024 | packed | 19157–19573 → 1222–1246 | 32136–32522 → 1237–1269 |
| 1024×1024×1024 | packed | 305274–308626 → 19102–19406 | 506466–520019 → 19231–19681 |
| 256×256×4096 | packed | 20382–21121 → 4705–4911 | 27003–27847 → 4752–4995 |
| 64×128×16384 | packed | 7718–7915 → 2474–2508 | 9471–9644 → 2498–2531 |
| 2×1024×1024 | packed | 612–619 → 97.4–99.1 | 996–1037 → 97.5–102.6 |
| 4×1024×1024 | packed | 1203–1259 → 137.7–141.3 | 1988–2088 → 136.8–143.8 |
| 8×64×64 | packed | 4.90–5.19 → 4.71–4.89 | 4.84–5.28 → 4.58–5.03 |
| 32×32×32 | packed | 4.83–4.97 → 4.61–4.82 | 4.56–4.96 → 4.60–4.88 |
| 1×256×256 | direct | 6.80–7.09 → 5.25–5.41 | 7.51–8.32 → 5.56–5.92 |
| 1×1024×1024 | direct | 306–316 → 47.4–49.0 | 496–525 → 56.4–69.9 |
| 2×256×256 | direct | 9.73–9.89 → 6.51–6.56 | 12.1–12.3 → 7.1–7.3 |
| 4×256×256 | direct | 15.5–15.6 → 8.7–8.8 | 20.0–20.7 → 10.0–10.4 |
| 4×512×512 | direct | 119–151 → 25.2–25.8 | 157–165 → 34.8–37.4 |

**No shape is slower than before at any alignment, small calls included.**

- **Paired after/before per process:**
  - 0.04–0.77 for every shape above 10 µs.
  - For the ctypes-dominated calls (~4.5 µs floor): 32³ is 0.93–1.05, and 8×64×64 is
    0.87–1.03. The single 1.05 is one process at 32³, B%64 = 48, and is within that
    cell's own before-spread (4.56–4.92 µs).
- **Aligned calls alone (B%64 = 0):** after/before is 0.06–0.77 above 10 µs, and
  0.93–1.00 (32³) / 0.94–0.99 (8×64×64) at the ctypes floor.
- **What changed for small M:** at 2×256×256 the aligned call was 9.9 µs before and
  6.5 µs after. Two earlier states of this branch, from the crossover files, were
  slower:
  - the rolled strip loops with every M > 1 packed: 9.66 µs through this probe. The
    Codex P2 regression was measured kernel-only, in the design harness: 5.42 µs
    before, 6.08 µs after;
  - the unrolled loops with every M > 1 packed: 8.3 µs.

**Alignment effect.** This is the best process at the slowest B%64 divided by the
best process at the fastest B%64:

| Path | Before | After |
|---|---|---|
| Packed, above 10 µs | 1.14–1.68 | 1.01–1.03 |
| Packed, ctypes floor (8×64×64, 32³) | — | 1.04–1.06 |
| Direct, 1×256×256 | 1.18 | 1.07 |
| Direct, 2×256×256 | 1.25 | 1.10 |
| Direct, 4×256×256 | 1.31 | 1.17 |
| Direct, 1×1024×1024 | 1.64 | 1.21 |
| Direct, 4×512×512 | 1.32 | 1.43 |

The direct path keeps a real alignment effect. It is 1.07–1.43x, largest at the
1 MiB boundary (4×512×512, 25.2 vs 35.6 µs). Even so, every direct-path cell is
0.11–0.77x of the pre-fix kernel at the same offset, and it is faster than packing
there.

**Tajasarus** (Ryzen 7 9800X3D, Zen 5, WSL2): `tajasarus_before_after.jsonl`,
56 processes (7 shapes × 4 offsets × 2), at `86ec9d31`. It ran after that box's GPU
worker released the timing lock; loadavg was 8.6 at the start, from work outside the
lock. All outputs are bitwise equal.

| M×N×K | path | aligned: before → after | misaligned: before → after | alignment effect before → after |
|---|---|---|---|---|
| 256×256×256 | packed | 703–706 → 293–295 | 1060–1083 → 294–299 | 1.53 → 1.02 |
| 64×1024×1024 | packed | 7546–7813 → 1204–1214 | 24048–25925 → 1201–1232 | 3.34 → 1.02 |
| 2×1024×1024 | packed | 240–243 → 93.3–94.4 | 760–802 → 94.8–98.5 | 3.32 → 1.04 |
| 32×32×32 | packed | 4.63–4.78 → 4.30–4.44 | 4.38–4.68 → 4.42–4.54 | 1.06 → 1.04 |
| 2×256×256 | direct | 9.29–9.32 → 6.18 | 11.7–12.0 → 6.9–7.0 | 1.28 → 1.12 |
| 4×512×512 | direct | 107–114 → 23.2–23.6 | 131–138 → 31.2–33.0 | 1.29 → 1.36 |
| 1×1024×1024 | direct | 125–133 → 42.1–42.2 | 379–413 → 50.4–58.5 | 3.09 → 1.22 |

## Design history

These files record how the design was reached. They are screens, not the
before/after evidence.

- **`design_variants.cpp` and `princess_luna_design_sweep.txt`** (CLOCK_MONOTONIC_RAW,
  472 rows, all bitwise): whole-B copy, per-strip panels, single against eight
  accumulators, packed against direct. The v7/v8 variants there have rolled strip
  loops (the GCC issue above), so they understate both final paths.
- **`kblock_variants.cpp` and `princess_luna_kblock_sweep.txt`:** a K block of 512
  against an unblocked panel. At K ≥ 4096 the unblocked 2–8 MiB panel ran 20–38%
  slower.
- **Rejected alternatives:**
  - Staging aligned copies in `runtime.launch` would fix only one of the three
    callers.
  - Declaring `alignment = 64` in the descriptor would refuse misaligned callers
    rather than fix them.
  - Aligned loads plus `valignd` need an immediate row offset, and that offset varies
    with k when N % 16 ≠ 0.

## What stays alignment-sensitive

- **The direct path** (M == 1, or M ≤ 4 with B ≤ 1 MiB) has a 1.07–1.43x alignment
  effect, as tabled above.
- **`tessera_x86_avx512_gemm_f32_tiled`** (the T1 cache-model evidence ABI) is
  unchanged. Its committed baselines measure that loop nest.
- **The bf16 / f64 / u8s8 GEMMs** are unchanged, and their alignment sensitivity is
  unmeasured (x86 queue).

## Re-recorded AVX-512 E2E packets

The committed packets' matmul row timed the old kernel. Both hosts were re-recorded
twice at `86ec9d31`, each host in its own worktree and `build/` under the timing
lock (`record_x86_avx512_packet.py`). The second runs are the committed packets
under `docs/audit/evidence/e2e_spine/x86/`; the first runs are `*_run1/` here.

matmul 256³ `kernel_wall`, run 1 → run 2:

| Host | Before (`329fcbf6`) | After (`86ec9d31`) |
|---|---|---|
| Princess-Luna | 731.6 µs | 307.5 → 296.4 µs |
| Tajasarus | 695.3 µs | 293.4 → 290.7 µs |

Other families against the `329fcbf6` packets:

- **Princess-Luna attention:** read 572.8 / 574.1 µs against 540.8 µs (+6%).
  `avx512_flash_attn_f32.cpp` is unchanged and does not call the GEMM, so that move is
  recording-to-recording drift and is not attributed to this change.
- **Everything else** is within 4%.
