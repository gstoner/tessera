# SM120 resident RMSNorm to matmul

Recorded by `benchmarks/nvidia/benchmark_scheduled_rmsnorm_matmul_edge.py`.
This packet contains exact-device RTX 5070 (sm_120) static and dynamic-shape
checks for the resident Schedule -> Tile RMSNorm/matmul contract, including
bounded dynamic M in `dynamic_m_sm120.json`. That packet checks active M=64
and 128 under a bound of 128 on a clean source revision. Its CUDA-event stage
timings have high CV and remain diagnostic only. Values characterize the local
proof envelope; they are not a cross-architecture performance claim.


## Bounded dynamic K

`dynamic_k_sm120.json` records a clean-source RTX 5070 (sm_120) run for the
same M=128, K-bound=256, N=256 fp16 envelope as the gfx1201 packet. It checks
active K=128, 192, and 256 against the numerical oracle, verifies native GPU
receipts, same intermediate pointer, stable package image, and times producer
and consumer separately over 21 CUDA-event samples of 500 launches each.

| Active K | RMSNorm median | Matmul median | Producer CV | Consumer CV |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 28.74 µs | 16.29 µs | 0.7% | 46.2% |
| 192 | 46.44 µs | 15.54 µs | 0.3% | 45.2% |
| 256 | 54.69 µs | 9.89 µs | 0.9% | 53.7% |

Consumer event variance remains high; all timings are diagnostic and support no
performance comparison or route promotion. Exact-device fp16 and bf16 tests trace the producer and consumer from public
`from_text` functions and reuse the package at active K=7, 11, and 16. See the complete samples and
compiler metadata in the JSON packet.
