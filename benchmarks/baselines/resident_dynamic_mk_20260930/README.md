# Paired dynamic M+K resident RMSNorm → matmul

This exact-device slice extends the paired resident Graph → Schedule → Tile
package contract so M and K may vary independently within fixed bounds in the
same package. Dynamic N remains a separate envelope. Both backends accept
padded/sliced host views and normalize them to the existing compact device ABI.
The CUDA uploader keeps staging buffers alive through successful stream sync.

Numerical outputs were checked against an independent fp32 RMSNorm/matmul
reference before device-event measurements. Producer and consumer stages are
measured separately; host packing, upload, and package construction are outside
the device-event intervals. Each test checks stable package images, same-stream
ordering, and resident producer-to-consumer storage.

| Device | Active M,K | Producer median | Consumer median | Producer CV | Consumer CV |
| --- | ---: | ---: | ---: | ---: | ---: |
| RX 9070 XT, gfx1201 | 64, 128 | 9.16 µs | 12.58 µs | 42.6% | 84.0% |
| RX 9070 XT, gfx1201 | 128, 192 | 10.74 µs | 14.16 µs | 9.8% | 16.7% |
| RX 9070 XT, gfx1201 | 128, 256 | 10.76 µs | 15.36 µs | 7.3% | 101.2% |
| RTX 5070, sm_120 | 64, 128 | 16.79 µs | 11.18 µs | 3.4% | 9.2% |
| RTX 5070, sm_120 | 128, 192 | 46.46 µs | 9.82 µs | 0.04% | 70.3% |
| RTX 5070, sm_120 | 128, 256 | 54.75 µs | 10.14 µs | 0.1% | 19.4% |

Variation remains high for several rows, especially on gfx1201. The packets
support correctness and stage attribution only. They do not support
cross-architecture comparisons, speedup claims, selector changes, or route
promotion.

The full packets record exact device identities, compiler fingerprints, image
digests, all event samples, buffer addresses, numerical errors, and source
revision. Both were generated from clean source revision 4e503156.

- [gfx1201.json](gfx1201.json)
- [sm120.json](sm120.json)

Focused exact-device suites passed 22/22 resident cases on gfx1201 and 30/30
tensor-program cases on sm_120. A host-free Graph-projection regression passed
5 cases, including bound mismatch rejection.
