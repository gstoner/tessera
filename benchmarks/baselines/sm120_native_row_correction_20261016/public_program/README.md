# Public native attention row program

Owner: E2E-REAL-6 / AD-RESIDUAL-EVAL-1 / FRONTEND-IR-MEDIUM-1.
Sync: SM120-NATIVE-ROW-CORRECTION-20261016.

## Executed envelope

RTX 5070 / SM120, CUDA 13.3, driver 610.88, LLVM/MLIR 23.1.1 assertions.
Original Python JIT/native backward packages native MLIR AD, Schedule/Tile,
NVIDIA Target IR and PTX members in strict portable VJP v3. Each member retains
exact image/descriptor/ABI, paired source lineage, geometry, physical storage and
compiler/toolchain identity. Restoration invokes no compiler. The producer is
absent for exactly dV-only activity. VJP v1/v2 and JVP schemas 5 (saved-LSE
bounded) / 6 (saved-LSE static) keep their existing meanings.

71 owning device cases pass, zero skips: restored runtime, original frontend,
changed inputs/seeds, query/key trace-at-one extents, warm compiler/concrete-body
refusal, resident producer-stream ordering, retained output generations and fresh
process compiler-absent replay. Focused compatibility and mandatory registry/audit
gates pass 448 cases. Source hashes and frozen sources accompany the receipts.
Broader required lanes are recorded separately when complete.

The A/B initially caught a full-logical bias represented implicitly by forward
metadata but explicitly by row metadata at Sq=1. Validation now compares the
actual physical bias extent. Dynamic contracts are not widened by this fix.

## Matched public benchmark

Eight rows, 40 samples per arm, alternating AB/BA batches, three warmups,
identical stored operands and bit-identical forward PTX. Independent FP64
comparisons run outside every timed call; arms match bitwise before timing.
Both arms enable native timing output. The public wall interval includes warm
frontend/dispatch, native snapshots, stages, readback and completion; compilation,
oracle checks and the extra device-window query are excluded. Device windows
use the same native event interval from forward start through backward stop.
Individual forward/row/backward samples remain in the raw packet.

| Storage | Sq / Sk | Inline public ms | Prepass public ms | Public ratio | Device-window ratio |
|---|---:|---:|---:|---:|---:|
| float16 | 3 / 5 | 0.9524 | 0.9919 | 0.96x | 0.74x |
| float16 | 1 / 129 | 1.0661 | 1.1009 | 0.97x | 1.04x |
| float16 | 7 / 129 | 1.8991 | 1.0433 | 1.82x | 3.55x |
| float16 | 17 / 129 | 7.5111 | 1.5256 | 4.92x | 10.25x |
| bfloat16 | 3 / 5 | 0.8525 | 0.8868 | 0.96x | 0.97x |
| bfloat16 | 1 / 129 | 1.0752 | 1.0760 | 1.00x | 1.05x |
| bfloat16 | 7 / 129 | 1.6966 | 1.0131 | 1.67x | 3.53x |
| bfloat16 | 17 / 129 | 7.6425 | 1.5370 | 4.97x | 9.88x |

Tiny cases retain regressions: FP16 tiny device ratio is 0.74x and public ratio
0.96x. This packet does not establish universal performance promotion.
Set TESSERA_NVIDIA_ATTENTION_ROW_CORRECTION=prepass to select the static compact
half route explicitly; the default remains inline. Cache identity includes this
policy; inline/prepass/inline switching has a dedicated device test.

## Remaining scope

Dynamic reverse, wider layouts/masks, capture APIs, higher-order composition,
automatic compiler schedule selection and sibling exact-device support remain
separate obligations. This packet does not close the original five-slice goal.
The earlier prepared_owner packet retains its diagnostic owner scope and FP32 dO;
this public measurement uses matching-half dO.

Reproduce in owning WSL with the matching compiler/provider environment:

```sh
python -m pytest -q tests/device/nvidia/test_public_attention_rows_program.py
# Decompress the frozen recorder into the owning scratch directory, then:
python record_public_row_benchmark.py --output public-row-ab.json
```
