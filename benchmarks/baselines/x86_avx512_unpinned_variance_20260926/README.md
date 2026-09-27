# x86 AVX-512 E2E packets: unpinned recordings and cross-recording variance — 2026-09-26

`X86-WITNESS-PIN-1` (x86 queue): the TSC witness used to pin the recorder to one CPU for the
whole run, so `avx512_flash_attn_f32`'s `std::thread`s inherited a one-CPU mask and attention
was timed on one core. From `dcdaf2a9` calibration pins itself and restores the CPU set, the
timed region runs unconfined, and `verify_witness_sample` requires each window's CPU-set size
to equal the host's CPU count.

Both Zen 5 hosts recorded **twice** at clean `dcdaf2a9` with
`benchmarks/e2e_spine/record_x86_avx512_packet.py` (WSL2, `-O2` runtime library, library
force-rebuilt at that commit). The second run of each host is the committed packet
(`docs/audit/evidence/e2e_spine/x86/<key>/`); the first runs are kept here
(`strix_halo_run1/`, `granite_ridge_run1/`, each a sealed packet). Princess-Luna's first
attempt at run 2 was **refused** by the 4% within-recording stability gate (softmax
`kernel_wall` 4.194%); its log is `strix_halo_run2_attempt1_refused.txt`. The retry passed;
there was exactly one retry.

Median `kernel_wall`, run 1 -> run 2 (µs):

| Family | Princess-Luna | ratio | Tajasarus | ratio |
|---|---|---|---|---|
| matmul 256³ | 723.7 -> 1071.3 | **1.48** | 1049.0 -> 1052.0 | 1.003 |
| attention | 629.7 -> 625.3 | 0.993 | 699.9 -> 708.1 | 1.012 |
| linalg (cholesky) | 65.5 -> 63.5 | 0.969 | 62.4 -> 62.3 | 0.999 |
| softmax | 3.37 -> 2.96 | 0.878 | 2.97 -> 2.88 | 0.969 |
| reduction | 0.76 -> 0.75 | 0.98 | 0.75 -> 0.73 | 0.979 |

What this shows:

- **Attention is stable across recordings once unconfined** on both hosts (0.993, 1.012).
  On Princess-Luna it is faster than the pinned recordings (~723 µs pinned vs ~627 µs
  unpinned), consistent with the threads no longer sharing one core. No such comparison is
  claimed for Tajasarus (its pinned recordings were 591 and 709 µs).
- **Matmul is bimodal and the pin did not cause it.** Princess-Luna moved 1.48x between two
  unpinned recordings of identical code, landing on the same two levels (~0.72 / ~1.05 ms)
  seen in the earlier pinned recordings on both hosts. Tajasarus stayed in the slower mode
  both times. Not root-caused; tracked as `X86-MATMUL-BIMODAL-1` in the x86 queue. Until it
  is, a single recording's matmul `kernel_wall` is **not a stable latency**.
- Softmax (~3 µs) varies most within a recording once unpinned; its gate failure is recorded
  above rather than retried away.

## Resolved 2026-09-26: `X86-MATMUL-BIMODAL-1`

The matmul level was decided per process by **B's address modulo 64**: numpy placed the
256 KiB B at a heap offset that varied per process in 16-byte steps, and the f32 GEMM
runs ~1.5x slower when B is not 64-byte aligned. Toggling only B's offset moved the level
on demand, both ways, on both hosts. The recorder now places every timed buffer 64-byte
aligned (resource schema v3; the validator refuses otherwise), and the packets were
re-recorded twice per host at `329fcbf6`. The packets in this directory are the
unaligned `dcdaf2a9` recordings and stay as provenance. Evidence, the probe and the
re-recording's first runs: `../x86_matmul_bimodal_20260926/`.
