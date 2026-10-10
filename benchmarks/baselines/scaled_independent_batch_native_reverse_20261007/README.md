# Independent matrix/scale prefixes: native reverse execution

Owning items: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization key: INDEPENDENT-SCALE-BATCH-2026-10-07.

## Implemented contract

Typed E4M3 Graph scaled_matmul with batching="broadcast" retains independent,
right-aligned static batch prefixes for all four operands. Native verification
requires the exact joined result prefix and unchanged logical matrix/scale
suffixes. Singleton, rank-two shared and unequal-rank inputs are distinct;
equal-product but incompatible prefixes are rejected.

The native scale transpose derives gradient coordinates and reduction axes
from each scale's own prefix. Matrix and opposite-scale loads use their own
right-aligned index maps. Ragged K/N group limits and isolated group-local
dots remain in native MLIR. The existing structured reduction carries this
actual body through Schedule, Tile, ROCm Target, LLVM and HSACO; the native
HIP program owns member submission, private output storage and completion.

## Candidate validation

The isolated compiler is linked using the matching aggregate CMake flags and
unchanged library dependencies, replacing only Graph verification and transpose
objects. Candidate/source identities are recorded in source-fingerprints.json.
131 native tests pass: all 15 nonempty mapped/shared combinations, four matrix
orientations, singleton/unequal-rank cases, malformed result/prefix refusals,
reverse export, Schedule/Tile/image packaging and existing transpose regressions.
382 shared gates pass, including exact-device RTX 5070 NVFP4 regressions and
operator/dtype/diagnostic/pass registry checks. No sibling scale-AD execution
claim follows from those NVFP4 regressions.

## Owning gfx1201 execution and measurements

Tajasarus reports AMD Radeon RX 9070 XT, gfx1201. device-results.json records
the HIP UUID bytes, runtime hash, package hash and recorder hash. The packages
were compiled on Super-Bear; Tajasarus replays serialized images with compiler
subprocesses forbidden. All 68 cases pass independent float64 scale-gradient
accumulation before/after timing, changed-value reuse and stale-generation
refusal. Maximum absolute error is 7.896e-8.

The cases cover all 15 nonempty matrix/scale mapping combinations with both
transpose flags, plus singleton/unequal-rank and unbatched cases. These are
tiny ragged M3/N5/K7 groups with block [3,4]. Five native event windows per
case each repeat the two-member program 20 times. Their per-program medians
were previously divided twice; those event values are withdrawn (see the
corrected canonical receipt below). Separate prepared host update/invoke/read medians
range 0.507-1.061 ms. This is an initial baseline, not isolated kernel timing,
a speedup, or a selector decision. Larger regimes still require benchmarking.

## Delivery and remaining integration

Native sources are integrated. The matching aggregate CMake build succeeds,
589 final native/registry/lifecycle/oracle/owning-SM120 tests pass, and all 68
reverse packages pass a fresh canonical gfx1201 execution/timing rerun.
Candidate and integrated program/image manifests are identical for all 68;
compiler, package and recorder identities are verified after collection.
Public static leading-map reverse projection is now proved separately in
benchmarks/baselines/rocm_public_independent_scale_reverse_20261007/README.md.
Primal Schedule batch addressing and public forward/JVP integration, dynamic/nonleading
axes, general composition and generic batching/transpose/full-unit closure
remain open. Reference arithmetic remains an independent oracle.

gfx1201 has the recorded native reverse proof. gfx1151 inherits no RDNA4 FP8
physical route. SM120's existing NVFP4 regression parity is validated; independent
packed-scale batch indexing is follow-up required. Apple Metal and x86 scale
execution require independent physical proof. The aggregate remains unpublished.

## Matching integrated timing receipt

The final canonical owning rerun retains maximum absolute error 7.89566878e-08.
Corrected native two-member per-program launch-window medians span
0.0136979995-0.124925748 ms; prepared host update/invoke/read medians span
0.478699803-1.01585160 ms.
Both domains retain five samples per case; these are baseline measurements.
Final receipts: integrated-gates.log, integrated-device-results.json/log and
integrated-fingerprints.json. Generated-document and Graphify refreshes remain
pending until their terminal receipts are recorded.

## Reproduce on the owning hosts

On the compiler host, with the matching native compiler environment:

    PYTHONPATH=python:. python benchmarks/rocm/record_independent_batch_reverse_packages.py --output /scratch/integrated-packages.json

On verified gfx1201, with TESSERA_ROCM_NATIVE_PROGRAM_LIB pointing to the
matching native HIP program owner and the project Python environment:

    PYTHONPATH=python:. python benchmarks/rocm/record_independent_batch_reverse_device.py --packages /scratch/integrated-packages.json --output /scratch/device-results.json

No benchmark recorder implements production lowering or a Python launch loop.

## Native event-unit correction

The HIP program ABI already divides its event window by the repeat count.
These recorders previously divided again by 20, understating device event
values by 20x. Those event measurements are withdrawn; numerical checks and
host wall timings remain valid. The canonical receipt has been remeasured on
gfx1201 with corrected recorders. It records milliseconds per complete program
invocation over 20 repeats, with no second division and no isolated-kernel claim.
Historical raw receipts remain in owning-host scratch. event-unit-contract.json
binds the runtime source and corrected recorder identities. A host regression
with a known already-averaged value passes.
