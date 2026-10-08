# Native resident producer-to-matmul candidate, 2026-10-07

Owner: W1.1 / E2E-REAL-6. Source is isolated from the running aggregate unit
snapshot. The candidate source is integrated into the unpublished aggregate after its
frozen full-unit sweep and Graphify refresh completed. A matching aggregate
runtime rebuild and integrated focused checks are pending. Candidate evidence
retains its own source/runtime hashes; it is not proof of a later rebuilt binary.

The already compiled Graph -> Schedule -> Tile -> NVIDIA Target -> PTX producer
and consumer images are retained by one native owner. A shared C++ submission
helper serves host-staged and resident execution. Resident execution validates
active extents, compact physical pitches, dtype, allocated device capacity,
CUDA allocation/stream context and disjoint output/intermediate storage.
Both kernels execute on one caller stream. Native synchronous completion retires
their uses before caller-owned device buffers can be released or reused.

This does not create a Python numerical backend or a kernel reconstruction path.
Python supplies frontend/ABI bindings and retains the session allocations.
prepare_resident exposes a reusable module owner; execute_resident uses that
same native sequence for its convenience call and returned intermediate/output.

## Validation

- Matching runtime built on Super-Bear with CUDA 13.4.59.
- Initial 209-test lane passed with the candidate native library but original
  Python adapter: tests/unit/conftest.py prepended its own Python tree.
  This is native-runtime regression evidence, not new-adapter evidence.
  The guarded candidate-Python rerun passes all 209 tests, including a
  migrated pointer fixture that forbids Python launch and requires one native
  submission with the returned intermediate/output allocations.
- New physical resident lane: 53 passed. Covers three producer kinds, FP16/BF16,
  fused/unfused, row/column RHS, static/dynamic shapes, changed inputs, closed
  owners, invalid pointers/pitches/bytes/extents, aliasing and recovery.
- Initial lane: 12 failed, 41 passed. Dynamic row fixtures had selected the
  default column-major package. Explicit RHS storage selection repairs the
  fixture; the failed log is retained.
- Existing native host-owner regression: 137 passed.
- Receipt-hoisting/read-only repair: 57 passed, including foreign current
  context, stream and allocation checks, recovery, and independent receipt copies.
- Forty-eight same-image alternating A/B rows pass independent numerics before
  and after each arm, with three separate component event windows per stage.

## Timing

The recorder alternates native submission and the previous Python two-launch
control using identical images and the same resident buffers. It checks an
independent numerical oracle before and after each arm. Component CUDA event
dispatch samples are separate from synchronized resident wall samples.
A concurrent CPU full-unit sweep is recorded as interference; no selector
promotion or isolated-kernel speedup is claimed.

## Profiled overhead repair

The initial native adapter regressed: 0.53-1.19 ms versus 0.35-0.45 ms for
the two-launch control. A 200-call profile attributes over half its time to
artifact receipt serialization/hashing. Immutable component receipts are now
prepared once; invocation retains the complete semantic package snapshot
check and returns independent receipt copies only after native completion.
Read-only inputs are admitted; output/intermediate buffers require write access.

The optimized packet has 48 rows. Paired control/native synchronized
resident wall ratios span 0.977-2.903, median
2.518. These characterize the sampled same-image
call boundary under concurrent full-unit activity; they are not isolated
kernel gains or a selector promotion. Initial timings/profile and initial
fixture/control failures remain available alongside the optimized packet.
The convenience execute_resident call still creates a fresh owner; repeated
calls should use prepare_resident with caller-managed resident buffers.

## Backend assessment

- NVIDIA SM120: candidate physical integration; exact-device lanes above.
- ROCm gfx1201/gfx1151: CUDA ABI does not apply. Their native HIP program owners
  retain separate images, schedules and architecture-specific evidence.
- Apple: CUDA ABI does not apply; no new Metal execution claim.
- x86: CUDA ABI does not apply; no new CPU execution claim.

General producer composition, AD, arbitrary padded resident pitches and
asynchronous result lifetime remain open. Broader existing resident descriptors
remain available through their separate checked launcher. Full-unit green,
aggregate integration, all-plan updates, Graphify refresh and publication are
still required. Source/runtime fingerprints are in candidate-fingerprints.json.

## Aggregate gate status

The frozen aggregate full-unit lane completed with 22,016 passed, 7,529 skipped,
874 deselected and two failures: generic scaled_matmul batching and transpose
closure. The assertions remain unchanged. This suite predates resident-owner
candidate integration and cannot be cited as candidate full-suite evidence.
The queued aggregate Graphify update is now live; it is not yet a completed
candidate integration gate. candidate-source.diff records the isolated change.

## Matching aggregate runtime validation

The aggregate CMake native runtime rebuild succeeds. The integrated focused
lane passes 711 tests across native resident execution, existing host ownership,
diagnostic/pass metadata, audit documents and recorder registration.
The canonical repository recorder passes all 48 numerical A/B rows with the
matching rebuilt runtime. Paired control/native resident wall ratios span
0.974-3.153, median 2.560.
One row remains about 2.63% slower; no universal promotion
is made. Source fingerprints are verified after timing. These measurements
are independent from component CUDA event windows and isolated kernel cost.
The runtime SHA256 is 3b95f5004f676d07568144fd75cb8f56179360fc9451b9f66c6200e0547b2845.
Generated-document validation and fresh Graphify refresh remain in progress.
The red full-unit result predates this integration.
