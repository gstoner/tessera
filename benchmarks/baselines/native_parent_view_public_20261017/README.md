# Native gfx1201 parent-view public integration

Owner: ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1 / LAYOUT-ALG-1 / E2E-REAL-6.
Sync: ROCM-NATIVE-GATE-UP-CONSUMER-20261016.

This supplements the separate native_gate_up_view_20261016 candidate packet.
Static f32 one-parent gate/up Graph programs now execute through public
JIT, native Schedule, typed Tile views, ROCm Target, ROCDL/LLVM, HSACO and the
checked C++ program owner on RX 9070 XT/gfx1201. Thirteen exact-device tests cover
three shapes, literal and captured-literal coordinates, both operand roles, changed inputs, immutable roots, retained
outputs, portable replay and fresh-process replay with compiler paths unavailable
and compiler/eager calls forbidden. See public-parent-view-device-final.txt.

The combined focused compiler, registry, dtype/operation, execution-matrix and audit
gates pass **409 tests** in host WSL. Generated documentation is regenerated;
Ruff and whitespace checks pass. The subsequent current-source manifest,
frontend/cache, native-package and pipeline selection passes **249 tests**.
The exact recorder/plan/foreign-host governance selection passes **1064 tests**;
these selections overlap earlier gates and are not summed. The exploratory full
host-unit run finished with 27547 passed, 7678 skipped and seven failures while
fixes landed during the running process. All seven failure nodes and surrounding
contracts pass together in a fresh current-source selection: **1152 passed**.
This is not an exact-head full-suite pass certificate; required CI must pass on
the published head before merge. See initial-full-unit-summary.txt and
all-failure-replays.txt. A fresh RTX5070/schema-1 source-active JVP regression
passes **12 cases**; nvidia-device-receipt.json records its exact device/scope.

Final binding review passes 55 focused tests and a thirteen-case gfx1201 rerun.
CPU calls avoid parent-view packager discovery. Construction and portable replay
reject relabeling the native image producer away from tessera-rocm-executable;
invalid producer requests fail before compiler discovery. See binding-review-tests.txt.
The shared submitter also retains the existing scaled-program identity guard
for minimal descriptors: 34 runtime/image/package regressions pass after the
guard-order correction, with another thirteen-case gfx1201 device rerun.
See guard-order-tests.txt.

The canonical driver preserves caller Graph ownership and the native ABI frame
across Target and binary stages. Host negative tests reject altered stage bytes,
invalid frame geometry/roles/counts and invalid buffer storage before submission.
No standalone slice route is admitted by the composed slice manifest entry.
Safe captured literal names now compose within list/tuple/dictionary attributes.
SSA tensor shadowing is preserved; executable expressions, attributes/subscripts
and runtime tensor coordinates are not evaluated as static attributes. Dynamic
shape support is not claimed. The frontend/cache regression groups each pass
53 tests; the first group has one existing skip. See capture-tests.txt and
capture-cache-tests.txt. Mypy remains at zero errors.

## Separate timing boundaries

public-parent-view-timings.json records five 8192-invocation HIP-event windows
and five completed warm public calls per shape, with correctness checked before
and after timing. No event window is shorter than 10 ms.

| M / I | Device launch-window median | Completed warm public median |
| --- | --- | --- |
| 3 / 17 | 1.777 us | 1.165 ms |
| 256 / 12288 | 34.464 us | 12.935 ms |

Launch-loop device windows include device dispatch and possible CPU enqueue gaps.
Public wall time includes validation,
owner preparation, transfers, completion and readback. Cold calls additionally
include compiler work; caches were not cleared. Maximum absolute error versus
independent f64 computation rounded to f32 is 3.815e-6. These are characterization
results, not a matched speedup or default-route promotion.

Reproduce on the owning gfx1201 host with matching native compilers and the
checked HIP program runtime:

```sh
python -m pytest tests/device/rocm/test_native_parent_view_jit.py -q
python -m benchmarks.rocm.record_native_parent_view --output parent-view.json
```

## Captured-member attribution follow-up

public-parent-view-captured-timings.json is a separate run of the same package
with five 32768-invocation captured-member windows. Capture and instantiation
are outside the event window, eliminating CPU enqueue gaps; device graph
dispatch remains inside. These are not instruction-counter kernel times.
All captured windows exceed 10 ms and correctness is rechecked afterward.

| M / I | Launch-loop median | Captured-member median | Completed warm public median |
| --- | --- | --- | --- |
| 3 / 17 | 1.811 us | 2.646 us | 1.352 ms |
| 256 / 12288 | 34.423 us | 34.413 us | 16.771 ms |

The model-sized device window is stable across the two launch methods. The tiny
case is sensitive to graph-dispatch overhead; captured timing is not assumed to
be faster. Host wall time varies between these runs, and there is no matched
baseline or regression/speedup conclusion. The initial recorder is frozen as
record_native_parent_view_initial.py; source-identity.json names the recorder
for each packet. Current reproduction uses the recorder under benchmarks/rocm. These measurements
precede the captured-coordinate frontend enhancement. The subsequent image
identity receipt checks both measured literal-profile binaries against the new
frontend; current host wall-time promotion is not inferred from byte identity.

## Remaining obligations

Resident matmul producer -> parent views -> activation chaining, explicit
low-storage semantics, K16 activation quantization/down projection, full NVFP4
MLP/model-quality acceptance and matched performance remain open. The other
four original goal slices retain their own architecture-specific obligations.
The public path currently prepares a native owner per call; host overhead is
substantial and is not hidden by the device timing figure.


## PR964 canonical target-resolution review

The canonical gfx1201 resolution now names tessera-rocm-executable with partial
Level-C coverage. The compiler-spine CSV/Markdown and dependent scoreboard are
regenerated; unknown operations/envelopes and gfx1200/gfx1250 remain unpromoted.
The resolution/package host group passes 90 tests. Owning gfx1201 controls pass
37 cases: 24 ordinary public FP16/BF16/four-FP8-pair matmul/GEMM cases and the
13 parent-view/replay cases (24 foreign-gfx1151 cases explicitly deselected).

resolution-compatibility-replay.json proves that legacy serialized FP16 and
mixed-FP8 artifacts still execute after the metadata change. New packages use
the corrected producer label and retain identical physical machine-code bytes.
The frozen resolution_compatibility_control.py records the tested two-phase
method with original scratch paths. Its capture phase ran at pre-fix head
010b675c9091f139bde2ae0c703c1e656484c3a8; replay ran after the resolution fix.
This is compatibility evidence, not a performance or wider-family promotion.

The target-resolution correction exposed three storage constructors that still
named the old pipeline. Expansion, ingest and folded-storage constructors now
name their actual executable producer. The pre-fix NVFP4 expansion test failed
at the compile-request/image identity guard; after correction, 23 expansion/
ingest tests and 13 folded-storage tests pass on owning gfx1201. The 402-test
registry/conformance/audit lane and all 33 generated documents also pass.
These checks establish route and bitwise/numerical compatibility, not speedup.

Final recorder/audit/reference/package gates: 43 passed, seven device-gated
skips on the NVIDIA build host; gfx1201 storage execution is covered by the
separate owning-host suites above. Final mypy: no issues in 657 source files.
review-source-parity.json binds all 39 source files and three tool/library
identities with zero mismatches.

The preserved failing-test transcript trims trailing whitespace only; the raw
transcript remains in the owning validation scratch directory.
