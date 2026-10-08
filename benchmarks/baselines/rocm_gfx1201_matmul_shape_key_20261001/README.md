# gfx1201 scheduled matmul image reuse

**Host:** Tajasaurus RX 9070 XT, gfx1201  
**Route:** Graph IR → Schedule IR → Tile IR → ROCm Target IR → HSACO  
**Compiler:** `tessera-opt` SHA-256 `31b835d5c05e1863ab272b98eb52efb932b95d5ea37d6cb2cbbab66b9f66ee7b`  
**Source state:** dirty scratch branch at `58b848cc`; packet is diagnostic and not promotion eligible.

Three static f16 matmul shapes `(M,N,K)=(16,16,16),(32,32,32),(48,32,16)` compiled to one HSACO digest and entry symbol. Shape-specific schedule digests and descriptor guards remained distinct. Each native launch passed the independent NumPy oracle; max absolute error was `1.19e-7`, `3.58e-7`, and `1.79e-7`. The exact-device unit test also passed for both fp16 and bf16 shape reuse.

Cold/warm package medians were 225.9 ms / 52.1 ms / 51.1 ms; Schedule times were 30.0 / 25.7 / 23.8 ms. HIP event medians were 8.56 / 14.64 / 15.36 microseconds. End-to-end medians were 2.239 / 2.409 / 2.480 ms. These micro-shape timings include substantial launch/host overhead and the cold sample has a 40 ms outlier, so use them for stage attribution only.

The validated envelope is static f16/bf16, register scheduled, `k_unroll=1`, `split_k=1`, without fused epilogues, dynamic extents, or LDS staging. Broader scale/layout/codegen-key fields and larger representative shapes remain open. Packet: `gfx1201.json`.

## Tajasaurus recheck — 2026-10-01

The exact-device test run passed 31 cases, including the gfx1201 fp16/bf16
cache reuse cases; one gfx1151-only case was skipped on Tajasaurus. A fresh
three-shape packet again reused one HSACO digest
(`4ce8b11e9f1c66f648908701ce73d2d4dbd59e8119895f678d1b8cbc5a3dc13e`) and
entry symbol while retaining distinct schedule digests and per-shape guards.
All three launches matched the NumPy oracle; maximum absolute errors were
`1.19e-7`, `3.58e-7`, and `1.79e-7`. The packet records cold/warm package
medians 224.08/51.66/51.35 ms, Schedule times 30.19/25.35/23.41 ms, HIP-event
medians 8.68/14.60/10.84 us, and end-to-end medians 2.18/2.40/2.67 ms.
These small-shape measurements are attribution only. The compiler SHA-256 is
`31b835d5c05e1863ab272b98eb52efb932b95d5ea37d6cb2cbbab66b9f66ee7b`; the
source worktree was dirty, and no performance promotion is claimed.

[Recheck packet](gfx1201_recheck_20261001.json).

## Current exact-device recheck

Tajasaurus passed 31 focused unit cases (one gfx1151-only skip). The new
11-sample recorder compiled (16,16,16), (32,32,32), and (48,32,16) to
the same HSACO digest and entry, while keeping distinct schedule digests and
shape guards. Every output passed NumPy; maximum absolute errors were
1.19e-7, 3.58e-7, and 1.79e-7. HIP-event medians were 8.56, 14.52, and
16.20 us. This remains the bounded static register, k_unroll=1, unsplit,
unfused envelope. Packet:
[current recheck](gfx1201_current_recheck_20261001.json).
