# Independent matrix/scale prefixes: native primal and scale JVP

Owning items: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / ROCM-FP8-BLOCKSCALE-1.
Synchronization key: INDEPENDENT-SCALE-BATCH-2026-10-07.

## Implemented native contract

The static gfx1201 Graph broadcasting contract now reaches the existing native
WMMA Schedule/Tile route. All four operand types and the exact joined result
type enter the content-addressed Schedule identity. Tile and ROCm Target retain
these types; native Target admission checks each matrix/scale suffix, storage,
scale policy and batch count. The generated GPU body decomposes block z using
the output prefix, then independently addresses each operand's shared,
singleton or right-aligned plane. Native program export compares these types
against the original member function and retains SSA/lifetime ownership.

Native package binding validates the four independent prefixes and geometry.
Native image projection retains static plane types and suffix strides; the
default primal policy is static_independent_prefix_v1. No runtime projection
erases those static semantics. Existing coupled profiles keep their identity.
No Python tensor arithmetic, Tile construction, replication or production
launch loop implements this route.

The WMMA profile currently requires transposeA=false and complete K scale
groups that are multiples of 16. FP32 block scales and encoded E8M0 K32 scales
are supported; scale JVP uses FP32 derivatives. This implementation is one step
toward the complete independent-prefix contract. It does not close partial
scale groups, transposed A or general batching.

## Validation

The first matching build passes 64 native package tests. After native image
policy integration, 465 focused native/registry/package tests pass. Four
adversarial Target mutations reject forged batch count, result prefix, scale
suffix and storage. The final matching rebuild and 74-case native lane pass, including five
Target corruption cases (count, prefix, suffix, storage and operand policy). Shared operator/dtype/frontend
regressions pass 82 tests. The owning RTX 5070 NVFP4 JIT/map lane passes 14
tests with two expected rank-two/no-batch skips; its GPU/toolchain probe is
recorded in sm120-device.log. That proof covers the existing SM120 profile,
not this gfx1201 independent-scale route.

All 69 final gfx1201 serialized programs pass independent float64-oracle
comparison, changed-input compiler-free replay and stale-generation refusal:
60 primal combinations (15 mapped/shared masks, KN/NK RHS, FP32/E8M0 scales),
three singleton/unequal-rank/unbatched cases and six native paired scale-JVP
cases. Maximum primal error is 2.126443234828912e-7; maximum JVP error is
1.4260600689208758e-7.

Five samples per case separate native event windows from prepared host
update/invoke/read wall time. Native per-program medians are
0.008269950-0.017596599 ms for one-member primal and
0.032247901-0.034086451 ms for four-member JVP. Prepared host medians are
0.337584-0.521992 ms and 0.501775-0.624245 ms respectively.
Shapes are tiny M3/N5/K64. These are baseline windows, not isolated kernel
timing or a speedup comparison; no selector is promoted.

Packages bind source and compiler hashes. Device results bind actual
RX 9070 XT/gfx1201, raw HIP UUID, native library, recorder, adapter and package
hashes. All 69 complete program/image manifests are identical before/after the final
policy guard; the corrected final-compiler rerun passes all 69 cases.
Collected compiler, package, recorder, adapter and current source hashes verify.

## Reproduce

On the compiler host with matching LLVM/MLIR and project Python:

    PYTHONPATH=python:. python -m benchmarks.rocm.record_independent_scaled_primal --output /scratch/packages.json

On the actual gfx1201 host with the full native movement/program HIP bridge:

    PYTHONPATH=python:. python benchmarks/rocm/record_independent_scaled_primal.py --packages /scratch/packages.json --output /scratch/device-results.json

The second command forbids compiler subprocesses after the live architecture
probe and replays the actual serialized images. Numerical checks precede and
follow each measurement domain.

## Open integration and backend assessments

Public independent primal/JVP projection remains open. Transposed A, partial
K scale groups, arbitrary scale-group sizes, dynamic shapes, nonleading maps,
general composition/storage AD and larger regimes require native integration
and physical proof. Generic batching/transpose states remain unchanged;
full-unit green and aggregate PR delivery remain open.

gfx1151 does not inherit RDNA4 FP8 WMMA evidence. Apple Metal and x86 independent
scaled execution are follow-up required. Existing SM120 NVFP4 parity is
validated for its admitted profile; independent packed-scale addressing and
scale AD remain follow-up required.

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
