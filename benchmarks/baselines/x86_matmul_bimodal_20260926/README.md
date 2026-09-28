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
| `gemm_harness.cpp` | The kernel source linked directly, with no Python and no image load. Takes per-operand offsets from a 2 MiB-aligned base and a `MADV_HUGEPAGE` / `MADV_NOHUGEPAGE` switch. Runs an ALU dependency-chain frequency probe before and after timing, and prints B's `AnonHugePages` so a granted THP request is visible. The build command is in its header. |
| `run_gemm_harness.sh` | Builds the harness against a checkout's kernel source and runs B offset {0,16,32,48,64} × THP request {0,1}, three fresh processes each, under the timing lock. The output file header records the host, kernel, source commit, compiler, exact compile command, both source sha256s and the THP mode. |
| `gemm_harness_{princess_luna,tajasarus}.txt` | Raw harness output (at `70eee148`, 2026-09-27). |
| `*_before_24proc.jsonl` | 24 fresh processes per host, recorder allocation (at `048fac95`). |
| `*_toggle.jsonl` | One variable at a time, three processes per setting (at `048fac95`). |
| `*_after_24proc.jsonl` | 24 fresh processes per host through the fixed recorder's bindings (at `329fcbf6`). |

**Provenance of the JSONL rows** (the rows carry no commit field themselves):

| Files | Source tree | Probe |
|---|---|---|
| `*_before_24proc.jsonl`, `*_toggle.jsonl` | `048fac95` | An uncommitted scratch copy, identical to the committed probe at `329fcbf6` except that it lacked `--recorder-bindings`. Its digest was not retained. |
| `*_after_24proc.jsonl` | `329fcbf6` | `probe_matmul.py` at `329fcbf6`, sha256 `48baf482549ece5d941afd0734f657a6ab27a99f9cf2b7b8b4781151b369aafa`. The later edit at `70eee148` only splits an import line (ruff E401). |
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

C harness (`gemm_harness_*.txt`, 30 processes per host, µs):

| | Princess-Luna | Tajasarus |
|---|---|---|
| B at 0 or 64, THP off or on | 717.6–740.4 | 702.5–706.3 |
| B at 16, 32 or 48, THP off or on | 1021.3–1089.3 | 991.4–1051.4 |
| ALU-probe GHz, every process | 4.95–5.08 | 5.10–5.17 |
| B `AnonHugePages` with THP requested / not requested | 2048 / 0 kB in every process | 2048 / 0 kB in every process |

The harness was first run on 2026-09-26, before `huge_req` / `AnonHugePages` were
printed. That output was not retained. The table above is the regenerated run at
`70eee148`, and the two runs agree on every level.

**Ruled out:**
- CPU/CCD placement: the same CPU numbers appear at both levels.
- THP:
  - In the recorder configuration, B's mapping had `AnonHugePages 0` in every process at
    both levels.
  - In the harness, a granted huge page (2048 kB `AnonHugePages`) did not move the level
    in either direction.
- Frequency: the ALU probe read the same range at both levels (harness table).
- Threading: the kernel is single-threaded by code read.
- Image load address and Python: the C harness reproduces both levels.
- 4K aliasing between operands: all three at page offset 0 is fast.
- Input data: the same data gives both levels, depending only on B's offset.

Denormals were not toggled separately. The inputs are standard normal (probe) and
uniform (harness), and both levels appear with the same data.

**Not explained:** after the fix, Princess-Luna's single level still spans ~7.7% across
processes (Tajasarus ~3%).

## Re-recorded packets (`329fcbf6`, both hosts twice)

**Superseded 2026-09-27:** the committed packets were re-recorded at `de914e51`
after the kernel fix (`../x86_gemm_align_20260927/`, sync `X86-GEMM-ALIGN-2026-09-27`);
the `329fcbf6` second runs described here are no longer the committed packets.

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
slow level at 256³. Candidate fixes (pack B, stage, or declare 64 in the descriptor's
per-buffer `alignment` so the call fails closed) and the other exposed x86 timers are
listed in the x86 queue.
