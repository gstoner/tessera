# Scheduled ROCm attention cache proof

Exact-device hosts:
- Princess-Luna, gfx1151.
- Tajasaurus, RX 9070 XT (gfx1201).

The five compiler/runtime/test/recorder sources used by both runs were byte-identical: `rocm_native.py`, `scheduled_attention.py`, `runtime.py`, `test_scheduled_attention_consumers.py`, and `measure_scheduled_attention_cache.py`. Per-file SHA-256 values and source revisions are recorded in each packet.

On each GPU, the recorder varied query lengths 17, 23, 31, and 65 while holding K/V length 19. Each architecture compiled exactly one HSACO image for all four shapes. Every package executed natively and matched the independent streaming-attention reference; maximum absolute error was at most 6.51e-5. Schedule digests and descriptor shape guards vary by problem, while each architecture's image digest stays constant. The exact gfx1201 gated test also verifies a changed bias contract causes a cache miss.

| Architecture | Cold first package | Later first-use packages | Repeated package median |
| --- | ---: | ---: | ---: |
| gfx1151 | 557.6 ms | 70.7–72.7 ms | 54.2–56.2 ms |
| gfx1201 | 316.2 ms | 50.5–52.8 ms | 37.1–38.3 ms |

These are host-side graph lowering and compiler/package timings, not GPU kernel timings. Do not compare these cross-host values as a hardware performance result.

Scheduled-attention image reuse is validated on both architectures. This does not close scheduled matmul image identity, whose current shape-free admission remains limited to static gfx1151 f16/bf16 register scheduling without fused epilogues, dynamic extents, split-K, or LDS staging.

## Source-fingerprinted gfx1201 recheck

A fresh run on Tajasaurus (gfx1201) used compiler source revision
58b848ccbc7682db03d3b1e350a5421ded56984d and records SHA-256 fingerprints for
the compiler, shared runtime, exact-device test, and recorder. The Tajasaurus
runtime fingerprint differs from the Super-Bear snapshot because that checkout
contains NVIDIA-scoped runtime changes; this packet records the actual
gfx1201-host source rather than claiming the snapshots are byte-identical.
Across query lengths 17, 23, 31, and 65, one binary image was compiled and
reused, all four native launches matched the streaming-attention reference,
and maximum absolute error was 6.51e-5. Cold package time was 368.34 ms;
warm shape package time was 49.78–52.05 ms; repeated package medians were
36.81–37.86 ms. These measure compiler/host work, not GPU execution time.

[Raw exact-device packet](gfx1201_recheck_20261001.json).
