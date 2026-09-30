# Padded host-view ingress for resident RMSNorm → matmul

These packets exercise padded host views at the resident package boundary. The
source and RHS are normalized to the compact layouts required by each existing
backend ABI before upload. No physical padded device allocation or strided
device leading dimension is claimed.

Both exact-device routes checked outputs against independent fp32 RMSNorm and
matmul references before timing. Producer and consumer device-event stages are
measured separately; host packing, upload, and package construction are outside
those intervals. Stable intermediate allocation and package/image reuse are
also checked by the focused device tests.

| Device | Active K (bound 256) | Producer median | Consumer median | Producer CV | Consumer CV |
| --- | ---: | ---: | ---: | ---: | ---: |
| RX 9070 XT, gfx1201 | 128 | 9.88 µs | 10.68 µs | 23.1% | 41.2% |
| RX 9070 XT, gfx1201 | 192 | 10.76 µs | 14.28 µs | 1.7% | 1.2% |
| RX 9070 XT, gfx1201 | 256 | 10.88 µs | 15.40 µs | 8.7% | 158.7% |
| RTX 5070, sm_120 | 128 | 28.71 µs | 10.02 µs | 0.1% | 64.8% |
| RTX 5070, sm_120 | 192 | 46.49 µs | 11.01 µs | 12.4% | 18.6% |
| RTX 5070, sm_120 | 256 | 54.76 µs | 9.87 µs | 0.1% | 1.5% |

High event variation in multiple rows makes this an attribution packet only.
It supports no cross-device comparison, speedup claim, selector decision, or
route promotion. Test evidence is 20/20 gfx1201 resident tests and 28/28 SM120
tensor-program tests, including fp16/bf16 padded-view numerical cases and CUDA
async-upload staging lifetime.

The JSON packets record device/compiler identity, source revision, package and
script fingerprints, numerical errors, buffer addresses, event samples, and
method details:

- [gfx1201_fp16.json](gfx1201_fp16.json)
- [sm120_fp16.json](sm120_fp16.json)
