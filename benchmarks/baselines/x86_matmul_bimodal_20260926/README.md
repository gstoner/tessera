# X86-MATMUL-BIMODAL-1: the x86 AVX-512 f32 GEMM's two levels — 2026-09-26

The E2E-spine x86 AVX-512 matmul family (256³ f32, `tessera_x86_avx512_gemm_f32`) timed
at one of two `kernel_wall` levels, ~0.70 ms or ~1.05 ms. The level was fixed for a
whole process and varied between processes
(`../x86_avx512_unpinned_variance_20260926/`).

**Mechanism: B's address modulo 64.** The kernel does one 64-byte `_mm512_loadu_ps` of B
per FMA. A is a scalar broadcast and C is stored once per 256 FMAs. A 64-byte load from an
address that is not 64-byte aligned spans two cache lines. numpy guarantees only
16-byte alignment. The recorder's 256 KiB B sat at a heap offset that varied per process
in 16-byte steps, so each process got one of the two levels. WSL2 exposes no hardware
counters, so the cost of the split load was not measured directly.

Hosts: **Princess-Luna** (Ryzen AI MAX+ 395, Zen 5, WSL2) and **Tajasarus** (Ryzen 7
9800X3D, Zen 5, WSL2). The library was built `-O2` (RUNTIME-LIB-OPT-1) in a separate
worktree on each host. Every timed run was wrapped in `flock /tmp/tessera-timing.lock`.

## Files

| File | What |
|---|---|
| `probe_matmul.py` | One fresh process: the recorder's packager, bindings (`default_rng(20260926)`), direct C-ABI call, 15 × 100 timing. Prints the median, CPUs, operand addresses, THP state of each mapping and the image base. `--offset-{a,b,o}` places one operand at a chosen byte offset from a 4096-aligned base. `--recorder-bindings` times the recorder's own (now aligned) bindings. |
| `gemm_harness.cpp` | The kernel source linked directly, with no Python and no image load. Takes per-operand offsets and an optional `MADV_HUGEPAGE` request, and runs an ALU dependency-chain frequency probe before and after timing. |
| `*_before_24proc.jsonl` | 24 fresh processes per host, recorder allocation (at `048fac95`). |
| `*_toggle.jsonl` | One variable at a time, three processes per setting (at `048fac95`). |
| `*_after_24proc.jsonl` | 24 fresh processes per host through the fixed recorder's bindings (at `329fcbf6`). |
| `strix_halo_run1/`, `granite_ridge_run1/` | First of the two sealed re-recordings at `329fcbf6`. The second runs are the committed packets under `docs/audit/evidence/e2e_spine/x86/`. |

## Results (per-process median `kernel_wall`, µs)

| | Princess-Luna | Tajasarus |
|---|---|---|
| before, 24 processes | 9 at 698–742, 15 at 1074–1116 | 6 at 682–697, 18 at 1040–1057 |
| before: fast iff B%64 = 0 | 24/24 | 24/24 |
| toggle B offset 0 / 64 / 4096 | 693–740 | 677–696 |
| toggle B offset 16 / 32 / 48 | 1080–1117 | 1051–1054 |
| toggle A or O offset 16 / 32, B aligned | 699–745 | 672–702 |
| after, 24 processes (aligned) | 24 at 693–747 | 24 at 675–696 |

In the C harness, B at offset 0/64 was fast and 16/32/48 was slow on both hosts (PL
718–747 vs 1056–1092; Taj 703–704 vs 982–1004). The frequency probe read 4.5–5.1 GHz
(PL) and 5.1–5.2 GHz (Taj) at both levels.

**Ruled out:**
- CPU/CCD placement: the same CPU numbers appear at both levels.
- THP: B's mapping had `AnonHugePages 0` in every process at both levels, and a
  `MADV_HUGEPAGE` request did not move the level. Whether that request was granted was
  not checked.
- Frequency: see the probe readings above.
- Threading: the kernel is single-threaded by code read.
- Image load address and Python: the C harness reproduces both levels.
- 4K aliasing between operands: all three at page offset 0 is fast.
- Input data: the same data gives both levels, depending only on B's offset.

Denormals were not toggled separately. The inputs are standard normal (probe) and
uniform (harness), and both levels appear with the same data.

**Not explained:** after the fix, Princess-Luna's single level still spans ~7.7% across
processes (Tajasarus ~3%).

## Re-recorded packets (`329fcbf6`, both hosts twice)

Median `kernel_wall`, run 1 → run 2 (µs):

| Family | Princess-Luna | ratio | Tajasarus | ratio |
|---|---|---|---|---|
| matmul 256³ | 720.0 → 731.6 | 1.016 | 696.6 → 695.3 | 0.998 |
| attention | 544.8 → 540.8 | 0.993 | 521.0 → 520.1 | 0.998 |
| linalg (cholesky) | 65.3 → 65.2 | 0.999 | 62.3 → 62.2 | 0.999 |
| softmax | 2.91 → 2.92 | 1.005 | 2.85 → 2.85 | 0.999 |
| reduction | 1.17 → 0.73 | **0.624** | 0.71 → 0.72 | 1.016 |

Matmul is now stable across recordings on both hosts. Two things in this table are
recorded but not investigated:

- **Princess-Luna reduction moved 0.62x between the two runs.** It is a sub-microsecond
  ctypes call, each run was stable within itself, and the `dcdaf2a9` recordings read
  0.76 / 0.75 µs.
- **Attention is faster than at `dcdaf2a9`** (Princess-Luna 625 → ~541 µs; Tajasarus
  708 → ~520 µs). The only recorder change is the aligned placement, but that link is
  not isolated.

**Still open, `X86-GEMM-ALIGN-1`:** `runtime.launch` passes contiguous numpy buffers
through unchanged. A production caller whose B is not 64-byte aligned still gets the
slow level.
