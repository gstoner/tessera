# SM120 named-pipeline and resident edge evidence

Host: Super-Bear WSL2, NVIDIA GeForce RTX 5070, reported as sm_120a.
Source revision: 58b848ccbc7682db03d3b1e350a5421ded56984d (worktree dirty).

The named `--tessera-nvidia-pipeline-sm120` MLIR fixture now verifies that a registered Graph matmul is projected through Graph -> Schedule -> Tile and emits `tile.view`, typed fragment packs, `tile.mma`, and fragment unpack. The SM90 and SM100 alias checks continue to pass on the same fixture.

The exact-device RMSNorm -> matmul benchmark passed at M=64, K=128, N=128 with fp16 storage. Both packages reported `native_gpu`; producer maximum absolute error was 0 and consumer maximum absolute error was 4.77e-6. The producer and consumer shared the caller-owned intermediate allocation, with no intermediate host copy. The emitted target evidence included `nvvm.mma.sync` and `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.

Five-sample CUDA-event medians were 17.60 us producer and 8.36 us consumer for independent package timings. In the resident stream, medians were 18.09 us and 7.76 us. Coefficients of variation were 6.9%/18.0% and 3.7%/13.3%, respectively; treat these as diagnostic attribution only. The 1.235 s synchronous two-launch host wall includes package/runtime overhead and is not kernel time.

No selector or performance promotion is supported. This proves one static fp16 producer-to-matmul envelope. Generic tensor-valued C++ constructors, other dtypes/layouts, broader schedules, and whole-compiler census closure remain open.
