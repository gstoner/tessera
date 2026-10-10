# Native paired attention cost and ownership

Owner NVIDIA-LSE-1 / E2E-REAL-6 / AD-RESIDUAL-EVAL-1; sync NVIDIA-PAIRED-COST-2026-10-07. Publication pending.

Recorder: benchmarks/nvidia/benchmark_jit_attention_vjp.py. Public JIT reverse tracing generates paired AD, native Graph/Schedule/Tile, NVIDIA Target/NVVM/PTX and checked resident packages. No production kernel, ABI, physical schedule or selection policy changes.

RTX 5070 / SM120, host WSL, LLVM/MLIR 23.1.1 and CUDA 13.3. Three shapes, full/causal attention and q / k,q / v,q,k requests yield 18 rows. Each row validates capture after caller input mutation and repeated backward against independent FP64 oracles. The 54 complete capture/backward windows also validate their primal and requested gradients before frame release.

The recorder follows native gradient activity: inactive output slots are not claimed as computed gradients. Frame views are borrowed and validated before close. Paired wall includes private allocation, input copies, forward, backward and synchronization; downloads/oracles and frame release are excluded. Separate CUDA-event windows describe kernels and driver gaps, not the complete pair.

Final command: --samples 3 --reps 20 --shapes 1x2x1x3x5x4x3 1x4x2x127x131x64x64 1x4x2x256x256x64x64. The concurrent Graphify process was suspended during the final benchmark and resumed afterward. Device tests run after measurements. Source, both compiler binaries and runtime SHA256 match the packet; Graph/Schedule/Tile/Target digests are present for both stages.

| Shape | Causal | Requested gradients | Capture/backward wall median ms |
| --- | --- | --- | ---: |
| 1x2x1x3x5x4x3 | False | q | 1.7919 |
| 1x2x1x3x5x4x3 | False | k,q | 1.8391 |
| 1x2x1x3x5x4x3 | False | v,q,k | 1.9289 |
| 1x2x1x3x5x4x3 | True | q | 1.7835 |
| 1x2x1x3x5x4x3 | True | k,q | 1.8750 |
| 1x2x1x3x5x4x3 | True | v,q,k | 1.9303 |
| 1x4x2x127x131x64x64 | False | q | 3.5059 |
| 1x4x2x127x131x64x64 | False | k,q | 4.9685 |
| 1x4x2x127x131x64x64 | False | v,q,k | 5.2673 |
| 1x4x2x127x131x64x64 | True | q | 3.5047 |
| 1x4x2x127x131x64x64 | True | k,q | 5.3060 |
| 1x4x2x127x131x64x64 | True | v,q,k | 5.2894 |
| 1x4x2x256x256x64x64 | False | q | 7.5872 |
| 1x4x2x256x256x64x64 | False | k,q | 12.6597 |
| 1x4x2x256x256x64x64 | False | v,q,k | 13.9162 |
| 1x4x2x256x256x64x64 | True | q | 7.7118 |
| 1x4x2x256x256x64x64 | True | k,q | 13.1514 |
| 1x4x2x256x256x64x64 | True | v,q,k | 13.9255 |

Validation: contracts.txt records 39 passed with ten device-gated skips; device-tests.txt records the separately enabled owning-device lane. lint.txt is clean. The benchmark packet is numerical evidence independently of test counts.

Open: matched recompute complete-pair measurement, native residual selection, explicit LSE cotangents, dynamic/composed AD and general batching/transpose closure. These measurements supply no sibling architecture proof and do not promote a performance selector.
