# SM120 resident RMSNorm to matmul

Recorded by `benchmarks/nvidia/benchmark_scheduled_rmsnorm_matmul_edge.py`.
This packet contains exact-device RTX 5070 (sm_120) static and dynamic-shape
checks for the resident Schedule -> Tile RMSNorm/matmul contract, including
bounded dynamic M in `dynamic_m_sm120.json`. That packet checks active M=64
and 128 under a bound of 128 on a clean source revision. Its CUDA-event stage
timings have high CV and remain diagnostic only. Values characterize the local
proof envelope; they are not a cross-architecture performance claim.
