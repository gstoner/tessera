# SM120 RMSNorm to matmul edge

This evidence records the compiler-owned fp16 Graph->Schedule->Tile RMSNorm and matmul packages joined through a caller-owned device buffer on RTX 5070 (sm_120a). Inputs upload once; RMSNorm and matmul enqueue on the same CUDA stream; the exact intermediate device pointer is passed to the consumer. Numerical checks compare the downloaded outputs after stream completion.

| MxKxN | RMSNorm max abs error | Matmul max abs error | RMSNorm us | Matmul us | CV producer / consumer |
|---:|---:|---:|---:|---:|---:|
| 64x64x64 | 0 | 3.33786e-06 | 10.53 | 8.64 | 1.5% / 9.8% |
| 512x256x512 | 0.001953125 | 1.525879e-05 | 59.49 | 12.85 | 0.0% / 2.4% |
| 255x127x129 | 0.0009765625 | 6.67572e-06 | 28.82 | 9.24 | 0.1% / 0.9% |

The timings are medians of five CUDA-event runs with 200 repetitions per sample, measured by the C++ launch bridge over device-resident buffers. The host wall field is separate and includes package/runtime overhead. Resource counts and compiler/toolchain fingerprints are in each JSON packet.

These three shapes validate a small aligned case, the original larger case, and a ragged case. They do not certify all fp16 RMSNorm-to-matmul shapes, promote a selector, or claim kernel fusion. Producer and consumer remain separate kernels; W1.1 still owns the remaining NVIDIA fragment-producer migrations.
