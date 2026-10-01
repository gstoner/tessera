# NVIDIA SM120 scheduled NVFP4 evidence

- Exact device: Super-Bear, NVIDIA GeForce RTX 5070 (sm_120).
- Compiler: scratch build from this checkout, LLVM/MLIR 23.1.1, CUDA 13.4.59.
- Route: Graph IR -> Schedule IR -> Tile IR -> NVIDIA Target IR -> PTX.
- Correctness: decoded NV_E2M1 values and UE4M3 scales compared to NumPy;
  max absolute error was 0.0 for all three tested shapes. Fractional UE4M3
  scale bytes as well as power-of-two scales were included.
- Shapes: 16x8x64, 33x19x129, and ragged 7x5x31.
- Timing method: seven batches, 100 repetitions per batch, 30 warmup
  invocations. CUDA event timings are separated from end-to-end runtime.launch
  wall timings. End-to-end includes host binding/staging and synchronization.
- Occupancy is queried with the actual one-warp 32-thread launch block.
- These micro-shape results are diagnostic only; no selector or performance
  promotion is made.

See [packet.json](packet.json) and
[benchmark recorder](../../nvidia/benchmark_scheduled_nvfp4.py).
