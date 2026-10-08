# Scalar typed bounded planes — gfx1201

Owner FRONTEND-IR-MEDIUM-1 / ROCM-FP8-BLOCKSCALE-1.
Sync SCALAR-SCALED-PLANE-2026-10-07.

The shared native MLIR helper derives a bounded independent plane for a
rank-two typed FP8 product with transposed A or a trailing partial scale group.
Schedule and serialized program export use the same derivation. The original
Graph retains rank-two operands and no batching attribute. The physical
program's broadcast profile has an empty prefix and grid Z=1. Typed native
views, fragments and partial accumulation execute through ROCm Target,
LLVM HSACO and checked HIP ownership.

## Validation

32 frontend projection cases and 95 rebuilt native package/JVP/regression
checks pass. The RX 9070 XT gfx1201 owning lane passes 40 primal/JVP cases.
Cases include K1/K32/K37/K65, both A/B orientations, FP32/E8M0 scales and
FP32 scale JVP. Changed inputs/tangents replay with compiler subprocesses
disabled. Native descriptors bind the actual executed image.

48 correctness-gated benchmark rows pass, including changed-input numerical
checks, compiler-free warm replay and stale-generation refusal. Maximum
absolute error is 7.262010137676356e-8. Separate median ranges in milliseconds:

| Domain | Primal | Paired scale JVP |
| --- | --- | --- |
| Public warm call | 0.8076–1.0740 | 1.4124–1.7305 |
| Native HIP event launch window | 0.006682–0.026030 | 0.029150–0.082246 |
| Prepared update/invoke/read wall | 0.3479–0.4522 | 0.5519–0.8267 |

The event API already averages over 20 repetitions; the recorder does not
divide again. These domains are not speedup comparisons. Public wall includes
frontend checks, uploads and readback; prepared wall excludes preparation.

Compiler SHA256:
e4e33848bc4b1b9378789dc2c47539a1f1c405ac0c93ec0e2df9713fdfe071d6.
Source fingerprints bind the native helper, Schedule/export and Python/tests.
The raw packet binds device UUID, native runtime, adapter and recorder hashes.

## Open work

Other scale widths, arbitrary layouts, dynamic/nonleading/composed maps,
storage derivatives, general frontend/AD integration, sibling physical proof,
fresh full-suite closure and PR delivery remain open. This packet does not
promote generic batching or transpose coverage states or a performance selector.

## Sibling regression

The matching compiler passes 83 owning RTX 5070 SM120 NVFP4 host/device
regressions with two unsupported cases skipped. This validates those existing
NVFP4 contracts; it does not establish NVIDIA independent-scale FP8 execution.

## Scalar reverse scale AD

The public scalar native_backward route passes 12 owning gfx1201 tests for
both matrix orientations and individual/reordered SA/SB gradient requests.
All four full-gradient benchmark profiles pass changed-input compiler-free
replay, exact-device attestation and stale-generation refusal. Maximum error
is 3.435916173799569e-8. This reverse profile uses K7 with ragged K4 groups,
separate from the K32 primal/JVP WMMA profile above.

Public warm medians are 0.9009–1.1172 ms; native two-member HIP launch windows
are 0.011824–0.028608 ms; prepared update/invoke/read is 0.5198–0.7408 ms.
These are separate domains, not a speedup. Native structured reduction is
the execution route; the Python float64 implementation is only an oracle.
No generic composed, storage-derivative or sibling scale-AD closure is claimed.
