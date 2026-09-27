# X86-GEMM-ALIGN-1: the x86 f32 GEMM no longer depends on B's alignment — 2026-09-27

`X86-MATMUL-BIMODAL-1` (`../x86_matmul_bimodal_20260926/`) found that
`tessera_x86_avx512_gemm_f32` ran ~1.5x slower at 256³ whenever B was not 64-byte
aligned. The kernel issued one 64-byte `_mm512_loadu_ps` of B per FMA, and numpy
only guarantees 16-byte alignment. PR #859 aligned the recorder's buffers; this
change fixes the production kernel, so every caller gets the fix: `runtime.launch`
through the x86 descriptor, the matmul-family lane (`runtime._x86_gemm_2d`) and
`TileToX86Pass`'s `func.call`.

## Fix chosen

The kernel packs B itself (`src/compiler/codegen/tessera_x86_backend/src/kernels/avx512_gemm_f32.cpp`):

- For each block of up to eight 16-column strips (128 floats) and each block of up
  to 512 rows of K, it copies `B[k0:k0+kc, n0:n0+128]` into a 64-byte-aligned
  `kc × 128` panel of at most 256 KiB, so the panel stays in L2. The copy reads each
  B row segment once, contiguously.
- Every FMA then reads the panel with an aligned load. Eight independent
  accumulators, one per strip, share each broadcast of `A[m,k]`.
- Between K blocks an accumulator continues through C, as an exact fp32 store and
  reload.
- The strip count is a template parameter. With a runtime count the accumulators
  spilled, and 32³ ran ~4x slower.
- `M == 1` reads B directly with the same 8-strip blocking and no pack. A GEMV
  never reuses a panel, so the copy is pure overhead there.
- If the panel allocation fails, the direct path runs. The result is the same,
  only slower.

**Numerics: bitwise identical to the previous kernel** at every alignment,
because each output element keeps the same FMA sequence: acc = 0, then
`fma(A[m,k], B[k,n], acc)` for k = 0..K−1 in order. Masked-off lanes read as zero
and are never stored. The tests pin this:

- `test_gemm_f32.cpp`: `check_bitwise` compares the kernel with the pre-fix kernel
  (kept in the test as the declared oracle) using `memcmp`, at every 4-byte B offset
  within a cache line. It covers the direct and packed paths, tail strips, K = 0,
  and K = 512 / 513 / 1100 (K-block boundaries). A mutation that restarts the
  accumulators at each K block fails it (2 cases); the tolerance-based `check()`
  did not catch that mutation.
- `tests/unit/test_x86_matmul_family_compiled.py::test_gemm_f32_result_independent_of_b_alignment`
  requires bit-identical output across B%64.
- Every probe row below compares the old and new builds' outputs bit for bit
  (`bitwise`), and the `runtime.launch` output against the direct call
  (`launch=direct`).

**Overlapping C (added after review).** The blocked loop writes C while A and B are
still being read, and it reads C back between K blocks. So the entry point now checks
whether C's byte range overlaps A's or B's, using the dense row-major extents computed
from M/N/K (`tessera_x86_avx512_gemm_f32_operands_overlap`, exported so the rule can
be tested).

- **On overlap:** the product is computed into an aligned scratch C and copied out.
  The result equals the product of the inputs as they were at entry, bit for bit, so
  an in-place `C = A @ B` or any overlapping view is correct.
- **Scratch allocation failure:** the call fails closed. C is filled with NaN and a
  message goes to stderr.
- **Disjoint operands** take the unchanged fast path. That includes C exactly
  adjacent to A or B.
- **Why scratch, not a refusal:** the ABI is `void`, so a refusal could not reach any
  of the three callers. Computing through scratch keeps every caller working and gives
  the mathematically correct answer.

**The old kernel was not safe under overlap either.** With the new test's overlap
cases linked against the old kernel sources:

- The `d8da67f7` kernel failed all 6 overlap cases.
- The `de914e51` kernel (before the entry check) failed 3.
- Both passed the 3 adjacent cases.
- The old kernel was right only by accident, for example C == A with N == K ≤ 16:
  that case passed, and N == K == 17 failed.

So no caller could have been relying on aliasing.

The check's cost was measured with `princess_luna_overlap_check_cost.jsonl`: the same
paired probe, run on de914e51 (no check) against 1102a32f (check), three processes per
cell. The after/before ratio was 1.00 at 256³. At 32³, 8×64×64 and 1×256×256 it read
0.86–1.10, which is noise, with no slowdown. All outputs were bitwise identical.

Tests (all pass on Princess-Luna):
- `check_overlap` in `test_gemm_f32.cpp` covers C == A, C == A on the M == 1 path,
  C == B, C inside A, C's tail over B's head, and in-place C == A with K = 1100
  (K-blocked). It also covers three adjacent-not-overlapping layouts, which must take
  the fast path and leave A and B untouched.
- `test_gemm_f32_overlapping_output_equals_disjoint_product` does the same in Python.

### Why this design (from measurement)

`design_variants.cpp` holds the candidates. Each is checked bitwise against v0 on
the same inputs. `run_design_sweep.sh` produced `princess_luna_design_sweep.txt`
(472 rows, all bitwise). Median µs at B%64 = 0 / the worst of 16, 32, 48
(CLOCK_MONOTONIC_RAW, one fresh process each; a design screen, not the before/after
evidence):

| M×N×K | v0 old | v1 whole-B copy | v2 per-strip panel | v3 8-strip panels, 1 acc | v7 8 acc, packed | v8 8 acc, direct |
|---|---|---|---|---|---|---|
| 32×32×32 | 0.76 / 0.75 | 0.85 / 0.86 | 0.80 / 0.80 | 0.71 / 0.82 | 0.47 / 0.46 | 0.44 / 0.60 |
| 1×256×256 | 2.73 / 4.25 | 5.78 / 6.13 | 4.91 / 5.95 | 5.62 / 5.78 | 4.17 / 4.38 | 2.16 / 2.59 |
| 1×1024×1024 | 309 / 510 | 329 / 328 | 279 / 420 | 112 / 121 | 85.5 / 92.7 | 46.9 / 64.4 |
| 4×1024×1024 | 1223 / 2011 | 1251 / 1263 | 433 / 582 | 267 / 274 | 177 / 184 | 200 / 256 |
| 256×256×256 | 681 / 1096 | 685 / 696 | 622 / 636 | 622 / 623 | 448 / 447 | 503 / 650 |
| 1024×1024×1024 | 307853 / 517512 | 311421 / 316048 | 49872 / 50135 | 49869 / 49934 | 28368 / 30923 | 47962 / 66830 |
| 1×4096×4096 | 14320 / 15415 | — | 22283 / 16176 | 10003 / 9675 | 5593 / 5825 | 5457 / 5952 |

Alternatives that lost:

- **Staging in the launch path** would cover `runtime.launch` only. The matmul-family
  lane and `TileToX86Pass` call the symbol directly.
- **Declaring `alignment = 64`** in the descriptor would fail misaligned callers
  closed instead of fixing them.
- **Aligned loads plus `valignd`** needs an immediate per row offset. The row offset
  varies with k whenever N % 16 ≠ 0.

The K block came from a separate screen, `princess_luna_kblock_sweep.txt`
(`kblock_variants.cpp`: v7 against v9 with KC = 256 / 512, three processes each).
At 64×128×16384 the unblocked 8 MiB panel read 4467–5572 µs, against 3633–3945 µs
with KC = 512. At 256×256×4096 the numbers were 8126–9159 against 7109–7693. At
256³ and 1024³ the variants were equal within the per-process spread.

## Before/after (the evidence)

`probe_gemm_align.py` runs in one fresh process per (shape, B%64). It loads the
pre-fix build (`d8da67f7`, origin/main) and the fixed build (`de914e51`) of
`libtessera_x86_elementwise.so`, and times both on the same A/B/C buffers in paired
interleaved windows. The order alternates every sample, and each window is about
20 ms.

- A and C are 64-byte aligned. B sits at the stated offset.
- Timing: the x86 TSC witness (`profiler_x86_clock`). The TSC is calibrated over
  separate pinned intervals and the region runs unpinned. Each window's TSC time is
  checked against CLOCK_MONOTONIC_RAW within 5% (worst seen: 2.8e-4) and re-verified
  with `verify_witness_sample`. The latency reported is the TSC one.
- The fixed build is also packaged the production way (`package_matmul` →
  `runtime.launch`), and the probe asserts that the packaged image payload is the
  timed library.
- Every run was under `flock /tmp/tessera-timing.lock`. Both builds are `-O2`
  (RUNTIME-LIB-OPT-1), with g++ 15.2.0.
- `run_probe.sh` drives it; `summarize.py` produces the tables; the probe's sha256
  is `67c37092…2e55b`.

**Princess-Luna** (Ryzen AI MAX+ 395, Zen 5, WSL2): `princess_luna_before_after.jsonl`
has 132 processes, three per cell; `princess_luna_repeat_8proc.jsonl` has eight per
cell at 250³ and 256³. Per-process median µs, min–max:

| M×N×K | B%64 | before | after | after/before |
|---|---|---|---|---|
| 256×256×256 | 0 | 709.6–755.6 | 448.7–479.4 | 0.62–0.66 |
| 256×256×256 | 16 / 32 / 48 | 1090.2–1129.5 | 447.9–472.1 | 0.41–0.42 |
| 250×250×250 | 0 / 16 / 32 / 48 | 631.0–700.1 | 428.3–463.1 | 0.63–0.71 |
| 64×1024×1024 | 0 | 19299.9–19595.1 | 1784.2–1831.1 | 0.09 |
| 64×1024×1024 | 16 / 32 / 48 | 31624.1–32463.0 | 1804.5–1879.4 | 0.06 |
| 1024×1024×1024 | 0 | 305728–309182 | 29425–30535 | 0.10 |
| 1024×1024×1024 | 16 / 32 / 48 | 505617–526459 | 29940–31508 | 0.06 |
| 256×256×4096 | 0 / misaligned | 21156–21198 / 26400–27872 | 7339–7594 | 0.26–0.36 |
| 64×128×16384 | 0 / misaligned | 7772–8281 / 9450–9744 | 3652–3802 | 0.39–0.47 |
| 256×1000×256 | all | 2545.8–3369.0 | 1748.6–1876.7 | 0.53–0.69 |
| 16×1024×1024 | 0 / misaligned | 4871–4903 / 7878–8196 | 493–530 | 0.06–0.11 |
| 32×32×32 | all | 4.77–5.00 | 4.63–4.92 | 0.96–1.00 |
| 1×256×256 | 0 / misaligned | 6.88–7.08 / 7.58–8.43 | 6.11–6.23 / 6.66–7.03 | 0.82–0.91 |
| 1×1024×1024 | 0 / misaligned | 301–310 / 499–522 | 50.4–55.2 / 62.9–68.1 | 0.12–0.18 |

**Alignment effect.** The effect is measured as the best process at the slowest B%64
divided by the best process at the fastest B%64:

| M×N×K | before | after |
|---|---|---|
| 256×256×256 | 1.52–1.54 | 1.00–1.01 |
| 64×1024×1024 | 1.66 | 1.03 |
| 1024×1024×1024 | 1.68 | 1.04 |
| 256×256×4096 | 1.31 | 1.01 |
| 64×128×16384 | 1.22 | 1.01 |
| 1×1024×1024 (M = 1, direct) | 1.71 | 1.28 |

The best process is used because a per-process level that does not follow B%64 is
present before and after the fix. At 256³ that level spans 448–479 µs on
Princess-Luna. It is the same ~7% single-level spread recorded (not investigated)
in `X86-MATMUL-BIMODAL-1`.

**Aligned performance did not regress.** At B%64 = 0 every shape with M > 1 is
0.09–0.69x of before. Shapes whose call is under ~10 µs are dominated by the ~4.5 µs
ctypes call, and there aligned is 0.88–1.00x (32³, 1×256×256).

**Tajasarus** (Ryzen 7 9800X3D, Zen 5, WSL2; a short cross-check, taken while that
box's GPU worker was idle on the timing lock):

- `tajasarus_before_after.jsonl`: 48 processes, 6 shapes × 4 offsets × 2.
- `tajasarus_repeat_4proc.jsonl`: 32 processes. It repeats 256³ and 64×1024×1024,
  whose first run was disturbed by concurrent load outside the lock (loadavg ~2.5;
  in that run the pre-fix build's 256³ aligned rows also spread 710–860 µs).

The repeat reads:

| M×N×K | before, B%64 = 0 / misaligned | after, every B%64 | alignment effect before → after |
|---|---|---|---|
| 256×256×256 | 683–708 / 1059–1070 | 435.8–455.2 | 1.56 → 1.01 |
| 64×1024×1024 | 7097–7730 / 22861–25245 | 1756–1963 | 3.29 → 1.03 |

The first run, for the other shapes:

| M×N×K | alignment effect before → after |
|---|---|
| 250×250×250 | 1.07 → 1.03 |
| 256×256×4096 | 1.31 → 1.03 |
| 64×128×16384 | 1.20 → 1.01 |
| 1×1024×1024 | 2.95 → 1.25 |

## What stays alignment-sensitive

- **`M == 1`** (the direct path): misaligned B is 1.25–1.28x slower than aligned
  (1×1024×1024: PL 50–55 µs aligned against 63–68 µs misaligned; before the fix
  301–310 against 499–522). The direct path is faster than packing at every alignment
  (design table: v8 against v7), so packing was not chosen there.
- **`tessera_x86_avx512_gemm_f32_tiled`**, the T1 cache-model evidence ABI, is
  unchanged. Its committed baselines measure that loop nest.
- **The bf16 / f64 / u8s8 GEMMs** are unchanged and their alignment sensitivity is
  unmeasured (x86 queue).

## Re-recorded AVX-512 E2E packets

The committed packets' matmul row timed the old kernel. Both hosts were re-recorded
twice at `de914e51`, each run in its own worktree and `build/` under the timing lock
(`record_x86_avx512_packet.py`). The second runs are the committed packets under
`docs/audit/evidence/e2e_spine/x86/`; the first runs are in `*_run1/` here. See the
x86 queue entry for the medians.
