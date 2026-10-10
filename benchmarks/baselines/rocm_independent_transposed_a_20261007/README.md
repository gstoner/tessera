# Independent-prefix transposed-A native integration

Owner: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / ROCM-FP8-BLOCKSCALE-1.
Synchronization: INDEPENDENT-SCALE-BATCH-2026-10-07.

Native Schedule now derives logical M/K from physical A[K,M], seals orientation
in its digest, and carries it through the scaled Tile and ROCm Target operations.
The register recipe reads A through a column-major tile.view and typed fragment
gather. The precomputed address includes the lane row; the architecture-owned
K partition is added by fragment lowering. No host transpose implements execution.
The serialized program records and validates A orientation. Typed plane checks
reject a forged Target orientation or serialized nonboolean flag.

Initial device validation found row 0 correct and later rows wrong because the
new producer address omitted its lane row. The compiler address was repaired,
rebuilt, and all final device cases and benchmark rows rerun. No timing or
numerical claim uses that failed build.

## Validation

- Matching LLVM/MLIR 23.1.1 native compiler rebuild passes.
- 855 host frontend/certificate/registry/timing/audit checks pass.
- Final compiler: 80 native new/existing package and corruption checks pass.
- RTX 5070: 55 sibling NVFP4 orientation/JIT cases pass; two unsupported cases skip.
- RX 9070 XT gfx1201: 180 public primal/JVP cases pass across all fifteen
  nonempty mapped/shared axis masks and both A/B orientations.
- Twelve further gfx1201 cases cover interior 16x16, ragged 17x19 and multiple
  33x35 tiles, K64/K96/K128, both B orientations and FP32/E8M0 scales.
- All 180 benchmark rows pass independent float64 numerics, changed-input
  compiler-free public replay, prepared-owner numerics and stale-generation refusal.

Final compiler SHA256: f27695531af97b9e986a9f210eb389c3c08808754587056db303b67d3fd4bf42.
Maximum benchmark absolute error: 2.7016246173516834e-07.

## Timing boundaries

Public medians include frontend/binding/upload/execution/readback. Prepared
medians exclude preparation and include update/invoke/read. Native HIP launch
windows are already averaged by the runtime over twenty repeats; the recorder
does not divide again. These domains are not speedup comparisons.

Transposed-A median ranges (milliseconds):

| Program | Public warm call | Prepared native event | Prepared update/invoke/read |
| --- | --- | --- | --- |
| primal | 0.862848–1.002192 | 0.010486–0.019948 | 0.386107–0.644012 |
| paired_jvp | 1.477812–1.635834 | 0.051774–0.064383 | 0.577399–0.816730 |

Timing shapes are static M3 N5 K64 with prefix 2x3. Logical operand values are
matched across A orientations. This is a numerical/storage baseline, without
a performance selector promotion. The row-major-A LDS recipe is not transferred
to column-major A; native selection retains the register recipe.

Partial scale groups, nonleading/dynamic maps, arbitrary composition, storage
derivatives, optimized LDS A layout and generic batching/transpose closure
remain open. Apple/x86 execution needs owning-device evidence. Publication and
a fresh green full suite remain unfinished.
