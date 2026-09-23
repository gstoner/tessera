# GFX1201 MXFP4 prefill shape sweep

Owner `ROCM-MXFP4-W4A8-1 / IKF-1`; sync
`GFX1201-MXFP4-PREFILL-SWEEP-2026-09-23`. The [packet](evidence.json) was
measured on Tajasarus (RX 9070 XT/gfx1201), using the merged PR #825
generator and benchmark bytes verified by SHA-256. The isolated device
worktree retains its original base revision in `source_revision`; that field
alone does not identify the merged source. Pinned Radiance revision
`dfdfa3832922c9a4253133f09c1f5c0d39748fc7` used fragment-order
weights. Every case passed independent sampled FP32-dequantized exact K32
reference output and bitwise BF16 agreement across all six engines before
timing. The folds in this sweep are lossless. Eleven interleaved HIP-event
trials, twelve launches each, measure kernels, not model latency.
The packet also retains the original sweep-recorder hash. PR #826 subsequently
moved optional inventory validation before GPU work; it did not change the
timed generators or relabel this packet as if the rerun used the new recorder.

| M×N×K | Safe TN2 | Expanded TN4 | Packed TN2 | Radiance | Best Tessera gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128×5120×8704 | 157.7 µs | 173.5 µs | 177.5 µs | 134.1 µs | 1.18× |
| 256×5120×8704 | 165.7 µs | 182.1 µs | 180.8 µs | 138.0 µs | 1.20× |
| 1024×5120×8704 | 562.4 µs | 455.5 µs | 589.1 µs | 446.7 µs | 1.02× |
| 128×17408×5120 | 289.7 µs | 414.3 µs | 288.3 µs | 209.7 µs | 1.37× |
| 256×17408×5120 | 310.8 µs | 421.4 µs | 316.0 µs | 243.0 µs | 1.28× |
| 1024×17408×5120 | 1148.4 µs | 1112.4 µs | 1167.6 µs | 886.3 µs | 1.26× |

Two preliminary repeated runs showed the same best-variant crossover at all
six shapes. The 128×17408 packed advantage over safe TN2 is only about 1–2
µs and is not an admission margin. TN4 approaches Radiance at 1024×5120,
but remains 1.26× slower at 1024×17408. Thus BN128/TN4 should be studied
as an M-dependent manual schedule, not promoted wholesale. The exact K32
route remains the correctness oracle; none of these approximate-policy,
folded kernels enter automatic selection.

HIP reported 16,974,905,344 bytes capacity and 15,659,737,088 free before
the sweep, 15,593,648,128 free after cleanup. These are actual device
snapshots, but no model was loaded and no real checkpoint inventory was
available on Tajasarus. They are **not** model headroom. The harness refuses
the budget decision even if an inventory is supplied: it accounts weight
bytes but uses zero authorized extra bytes until a model-owned loader can
snapshot free memory after loading all other resident state and reserve
activations, graph pools, code objects, and fragmentation. This is the
remaining input needed for an expanded-versus-packed production decision.
