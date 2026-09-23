# GFX1201 device-resident MXFP4 graph pipeline — 2026-09-23

Owner `ROCM-MXFP4-W4A8-1` / `IKF-1`; sync
`GFX1201-MXFP4-GRAPH-PIPELINE-2026-09-23`. This packet was recorded on
Tajasarus, selected AMD Radeon RX 9070 XT (gfx1201), from merged main
`f8e754d4` plus the source files pinned by SHA-256 in `evidence.json`.
Automatic selection remains closed.

The manual graph now captures three kernel nodes on one stream: an FP32
activation producer computes a per-token scale and OCP E4M3 bytes, the
packed-folded GEMM consumes those device buffers, and a BF16 ReLU consumer
writes a separate device output. No host transfer, allocation, or sync node
is captured. Five exact-device cases passed: ragged N48/N80 and K64/K128,
lossless/lossy folds, a second replay after a same-stream FP32 input update,
and two simultaneously live M-specialized graphs. Producer FP8 bytes and
scales matched the independent NumPy oracle; final BF16 output matched the
declared folded oracle. Live pointers were distinct across M65 and M129 and
became inaccessible through their leases after pool close.

Dynamic M uses a bounded exact-shape graph cache, not in-place executable
node updates. A third shape is refused when the two-shape budget is full;
the caller must explicitly release a shape. This avoids retargeting a
captured grid or buffer pointer while an executable remains live.

| M×N×K | Direct triple host enqueue | Graph host enqueue | Direct HIP event | Graph HIP event |
| --- | ---: | ---: | ---: | ---: |
| 256×5120×8704 | 15.20 µs | 1.94 µs | 196 µs | 200 µs |
| 1024×17408×5120 | 14.49 µs | 2.00 µs | 1303 µs | 1349 µs |

These same-stream, same-buffer, same-code-object measurements alternate
direct and graph rounds. Each round is seven samples of twenty three-kernel
sequences. Graph replay lowers host enqueue cost, but the device-event
medians are slightly worse; this is not a kernel-speedup claim and does not
close the packed GEMM gap to Radiance. The current producer and consumer
are correctness/lifecycle kernels, not tuned production fusion schedules.

The PR 819 graph executable is now destroyed even if final stream sync
reports an asynchronous error. That cleanup change supersedes PR 819's
historical timing source; its packet remains pinned to the PR 819 commit.

Next: replace the demonstrator producer/consumer with model-owned tensor
routes, prove graph-lifetime behavior under allocator pressure and failed
capture, and investigate device-event overhead. Do not promote the selector
or claim in-place dynamic-M graph updates from this packet.
