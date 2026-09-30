# gfx1201 resident RMSNorm → matmul: bounded dynamic K

This packet checks one public `from_text` Graph → Schedule → Tile route on the RX 9070 XT (gfx1201). The package bound is M=128, K=256, N=256; the resident edge uses fp16 storage with fp32 accumulation/output. It reuses the same compiled producer and consumer images and the same intermediate allocation for active K prefixes 128, 192, and 256. The RHS is supplied with a matching active K prefix.

The runner compares each output numerically with the independent fp32 RMSNorm/matmul oracle before recording timings. It uses 25 warmups and 100 stage samples per K. Producer and consumer HIP-event samples are reported separately; they exclude package construction, upload, and host launch time.

| Active K | RMSNorm median | Matmul median | Producer CV | Consumer CV |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 17.66 µs | 13.80 µs | 27.4% | 74.2% |
| 192 | 11.24 µs | 14.76 µs | 4.3% | 1.2% |
| 256 | 11.38 µs | 15.88 µs | 1.3% | 152.7% |

The large timing variation means these measurements establish stage attribution only; they do not support a speedup claim or route promotion. Correctness and package reuse passed. The packet records the exact device/architecture, compiler and toolchain fingerprints, image digests, buffer addresses, clean source revision, all event samples, and benchmark script hash in `dynamic_k.json`.

Reproduce on Tajasaurus with the gfx1201 build and environment configured:

```bash
python3 benchmarks/rocm/benchmark_gfx1201_resident_norm_matmul.py \
  --dynamic-k --warmup 25 --iterations 100
```
