# Native quantized gate/up — gfx1201, 2026-10-17

Owner: ROCM-NVFP4-INGEST-1 / ROCM-FP8-BLOCKSCALE-1.
Sync: ROCM-QUANTIZED-GATEUP-20261017.

Four native members execute the authored BF16 K16 quantizer, NVFP4 weight
expansion, exact scaled product with BF16 parent store, and SiLU gate/up.
Packets include live GPU identity, source/compiler/provider/recorder hashes,
portable artifacts and four native receipts. The owning device is RX9070XT,
gfx1201, UUID GPU-28d9e7efbf2ef716.

Correctness precedes timing: independent float64 product reference, explicit
BF16 parent rounding and FP32 activation followed by BF16 output. Random
comparison uses rtol 0.015625 and atol 0.00002; it is not bitwise proof. Separate
analytic device tests prove exact orientation and parent rounding.

| M / H / K | Public warm ms | Program event µs | Captured members µs: quant / expand / product / activation | Max absolute error |
|---|---:|---:|---|---:|
| 1 / 1 / 16 | 5.321 | 19.07 | 5.52 / 3.92 / 5.41 / 2.22 | 0 |
| 3 / 17 / 48 | 5.088 | 18.93 | 5.58 / 4.02 / 6.07 / 2.53 | 0 |
| 256 / 128 / 12288 | 6.297 | 166.22 | 27.37 / 15.15 / 117.58 / 1.86 | 0.125 |

Public calls include validation, root copies, owner preparation, execution and
readback. Prepared invoke/read excludes preparation and root upload. Ordinary
HIP events include enqueue gaps; the native API already averages repetitions.
Captured member windows include graph dispatch and exclude capture, copies and
instantiation. Their medians must not be treated as interchangeable or added
to infer public latency. No comparative speedup or selector promotion is claimed.

Device gate: 22 passed, including eleven v3 tests and eleven v1/v2 controls.
Native/registry focused gate: 335 passed. Package gate: 47 passed. Audit/pass-metadata/diagnostic gates pass 311 tests. The rebuilt NVIDIA tool
passes the 85-case existing active-source JVP/runtime fixture on RTX5070; this
is sibling regression evidence, not adoption of the gfx1201 program. Broader
CPU and final generated-document drift gates remain pending. Full resident down projection, model
quality, wider/dynamic layouts, encoded AD and the original five-slice goal
remain open. No sibling physical parity follows from this packet.
