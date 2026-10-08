# Independent-prefix partial scale groups

Owner: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / ROCM-FP8-BLOCKSCALE-1.
Synchronization: INDEPENDENT-SCALE-BATCH-2026-10-07.

Native Graph/Schedule derivation preserves ceiling scale suffixes for partial
trailing K groups. Tile verification, Target projection and both executable
adapters carry the same static independent-plane register contract. Native
WMMA fragment loads are bounded for trailing groups even on interior M/N tiles.
The native loop starts each partial at zero, scales it exactly once, and joins
groups in ascending order. Scale addressing uses ceil(K/scale_k), preserving
row strides. Independent image identity retains static plane and K metadata;
this is not a shape-free partial-group image.

The direct Tile adapter used by paired JVP initially retained a whole-group
gate after the primal Target path passed. That gate was repaired and the final
matching compiler/native and owning-device suites rerun before timing.

## Validation

- Matching LLVM/MLIR 23.1.1 compiler rebuild passes.
- 120 host public projection cases pass.
- 76 native partial-group image, JVP ownership and Target corruption cases pass.
- 308 final diagnostic/pass/audit/event-unit checks pass.
- Final aligned native and exact RTX 5070 sibling lane: 135 pass, two expected skips.
- Exact RX 9070 XT gfx1201: 260 public numerical cases pass, including all
  independent maps, both A/B orientations, FP32/E8M0 scales, K1/K15/K17/K31/
  K33/K37/K63/K65, and interior/multiple/ragged M/N tiles.
- All 180 K37 benchmark rows pass independent float64 numerics, changed-input
  compiler-free public reuse, prepared-owner checks and stale-generation refusal.

Compiler SHA256: 25a4f542b0997ae4192d4e018830fff3aa78f7634a72e20ee1010cd2b9602b31.
Maximum benchmark absolute error: 1.5482812942835267e-07.

## Timing

Public wall time includes frontend/binding/upload/execution/readback. Prepared
wall excludes owner preparation. Native HIP launch windows are already averaged
by the runtime over twenty repeats and are not divided again by either recorder.

| Program | Public warm median range, ms | Native event median range, ms | Prepared update/invoke/read range, ms |
| --- | --- | --- | --- |
| primal | 0.818596–1.091660 | 0.008486–0.019732 | 0.346013–0.761052 |
| paired_jvp | 1.461404–1.900984 | 0.028980–0.063617 | 0.527501–0.825850 |

These are distinct measurement domains and a small static characterization,
not a speedup comparison or selector promotion. Timing shape: M3 N5 K37,
prefix 2x3, scale K32; FP32 scales use N-block 3 and E8M0 uses N-block 1.

Plain scalar JIT projection, other scale-group widths, optimized partial LDS
recipes, dynamic/nonleading maps, arbitrary composition and storage derivatives
still need integration or exact-device proof. Legacy physical contracts remain
aligned. Generic batching/transpose closure, sibling scale execution, full-suite
and publication remain open.
