# GFX1201 packed MXFP4 device graph — 2026-09-23

Owner `ROCM-MXFP4-W4A8-1` / `IKF-1`; sync
`GFX1201-MXFP4-DEVICE-GRAPH-2026-09-23`. Tajasarus selected its AMD
Radeon RX 9070 XT (gfx1201). The source hashes in `evidence.json` bind
these results to the graph executor and benchmark atop merged main
`f5441371`. The candidate is manual; automatic selection remains closed.

One stable-pointer kernel was captured on the session stream. HIP graph
inspection found exactly one node of type `hipGraphNodeTypeKernel`; no copy,
allocation, or synchronization node was captured. The session owns A, As,
output, packed B, scales, module, stream, and executable graph until close.
An external producer can use the leased A/As pointers on the same stream;
four exact-device tests replayed after such a device-buffer update and
verified BF16 output against the declared folded oracle.

| M×N×K | Direct host enqueue | Graph host enqueue | Direct HIP event | Graph HIP event |
| --- | ---: | ---: | ---: | ---: |
| 256×5120×8704 | 6.33 µs | 1.97 µs | 174 µs | 185 µs |
| 1024×17408×5120 | 6.38 µs | 2.43 µs | 1120 µs | 1122 µs |

The benchmark used the same HSACO, device-resident inputs, and exact K32
sampled oracle, alternating direct and graph rounds. Each round contained
seven samples of twenty launches. Host enqueue time excludes synchronization;
HIP-event time includes device execution and dispatch but excludes transfers.
The first shape varied appreciably across rounds, so these data establish
lower host enqueue cost, not a device-side speedup. Graph replay does not
close the remaining packed-prefill gap to matched Radiance.

Next: prove integration with an upstream device-resident E4M3 producer and
a downstream device-resident BF16 consumer, including graph updates across
dynamic M and buffer lifetime boundaries. Continue packed-decode/LDS tuning
separately; do not infer selector admission from this manual graph proof.
