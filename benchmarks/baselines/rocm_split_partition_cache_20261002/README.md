# gfx1201 split partition image reuse

Owner ROCM-SPLIT-K-1 / E2E-REAL-6-ROCM-MATMUL-CACHE.
Sync ROCM-SPLIT-PARTITION-IMAGE-2026-10-02.

Target IR now preserves Schedule k_blocks. Split-K additionally carries
positive problem_k; native generation validates portable ABI, ordered
reduction, static K and whole-panel slice divisibility. Native MLIR image
projection retains both fields. M/N changes within the same physical recipe
reuse the image; different macro-K panels or K partition extents have distinct
identities. Python does not emit a kernel or bypass native lowering.

Exact RX 9070 XT/gfx1201 tests cover fp16/bf16, plain and bias+ReLU,
M/K/N=(16,2048,256) and (15,2048,200), including masked M/N edges. Both shapes
reuse an eight-slice image. K=4096 creates a cold distinct image and remains
numerically correct. The existing GELU and ordered-replay cases also pass.
Combined split/Target/cache suite: 112 passed, 10 skipped. Separate gfx1151
unsplit Target/cache validation: 47 passed, 12 skipped. No gfx1151 split-K
execution claim is made. Shared package/registry gates: 460 passed, 68 skipped.

plain.json and fused.json contain live device identity, compiler/recorded source
hashes, correctness checks (absolute tolerance 2e-4), native module-cache
counters and independent partial/reduction HIP-event samples. End-to-end
samples disable event instrumentation and include both launches, workspace
allocation, transfers, host dispatch and completion. A partial-plus-reduction
event sum reports kernel work; it does not include interlaunch host latency.
The module loads once, resolves two functions, and hits on later calls.
Timings characterize these two shapes; no speedup or selector promotion.

This corrects earlier register/LDS shape-key packets which inadvertently
compiled different Schedule macro-K recipes as one image because Target IR
dropped k_blocks. Those packets remain historical. Current reuse is explicitly
within a fixed physical recipe; scaled image-key closure remains open.
