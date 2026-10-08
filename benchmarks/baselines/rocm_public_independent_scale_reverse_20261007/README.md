# Public independent matrix/scale maps: native reverse execution

Owning items: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization key: INDEPENDENT-SCALE-BATCH-2026-10-07.

## Implemented boundary

Public leading vmap projects independent mapped/shared prefixes for all four
typed scaled-matmul operands into verified Graph MLIR. Native paired scale AD
preserves each scale's own prefix and reduction axes through Schedule, Tile
structured reduction, ROCm Target, LLVM, HSACO and checked native HIP ownership.
Python constructs frontend metadata and binds the package; it does not
implement production tensor arithmetic, Tile construction or a launch loop.

The admitted reverse profile has E4M3 matrices, FP32 scales and exact FP32
per-block accumulation. Both matrix transpose flags are supported. All 15
nonempty mapped/shared combinations execute at leading prefixes (2), (2,3)
and (2,1,3). Source bounds are checked before capture, scalar owners remain
unchanged and AST/tracer certificates agree. Gradient request order is tested.

## Validation and fingerprints

The integrated Super-Bear WSL gate passes 584 tests: public independent maps,
existing typed/nested maps, operator/dtype registries, diagnostics, pass metadata
and audit lifecycle. After the final four-backend document edits, the focused
audit/diagnostic/pass lane passes 307 tests (final-drift-gates.log).
The isolated frontend candidate previously passed 247;
the first candidate harness lacked fixture visibility and is not a passing
receipt. The owning gfx1201 lane passes 261 tests: 189 new cases and 72 existing
public scale-VJP regressions. One pytest unknown timeout-option warning records
a missing runner plugin; no numerical cases skip or fail.

public-timings.json binds the actual RX 9070 XT/gfx1201, raw HIP UUID bytes,
HIP versions, matching compiler/runtime hashes, recorder and adapter identities,
package digests and image hashes. Collection verifies recorder and all recorded
adapter bytes against the authoritative aggregate. Compiler SHA256:
1fcf742130a71ca0d6163aeaee071caba1052f084923cd77856ad87022653174.
Source and test hashes are in source-fingerprints.json. Host/device test logs
are retained in this packet.

## Correctness-gated baseline

All 180 rows compare both scale gradients to independent float64 accumulation
before and after each timing domain. Changed matrix/scale inputs and cotangents
replay with compiler subprocesses forbidden; stale prepared generations reject.
Maximum absolute error is 9.1121950917053e-08.

Five samples per row retain three distinct measurement domains:

| Domain | Per-row median range (ms) | Boundary |
| --- | --- | --- |
| Warm public native_backward | 0.857117–1.376794 | Frontend, package binding, native preparation, uploads, submission, readback and disposal |
| Prepared native launch window | 0.015695950–0.208303258 | Two-member HIP event window already averaged by the native API over 20 repeats |
| Prepared update/invoke/read | 0.462404–0.970170 | Host wall time; preparation excluded |

Shapes are tiny ragged M3/N5/K7 with block [3,4]. Event windows are not isolated
kernel measurements. There is no speedup comparison or selector promotion.

## Reproduce

On the owning gfx1201 host with the matching compiler and full native movement/
program runtime bridge, project Python environment and ROCm environment:

    PYTHONPATH=python:. python -m benchmarks.rocm.record_public_independent_scale_reverse --output /scratch/public-timings.json

The recorder validates the live architecture before packaging. The device gate:

    TESSERA_GFX1201_DEVICE_PROOF=1 python -m pytest -q tests/device/rocm/test_public_independent_scale_vjp.py tests/device/rocm/test_public_scaled_vjp.py

## Remaining engineering and sibling assessments

Independent-map primal and JVP Schedule/Tile addressing remain open. Dynamic
shapes, nonleading/mixed nested axes, general composition, storage gradients,
encoded-scale derivatives, larger regimes and generic batching/transpose
closure remain open. Serial scale reduction remains the default.

gfx1151 has no RDNA4 FP8 WMMA route and inherits no physical proof. SM120's
existing NVFP4 regression evidence does not establish independent packed-scale
maps or scale AD. Apple Metal and x86 need architecture-owned scale execution.
The aggregate is unpublished; this packet does not establish a fresh full-suite
gate or complete the five-slice program.

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
