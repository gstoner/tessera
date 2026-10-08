# gfx1151 current-source compiler revalidation — 2026-10-07

Owner: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / ROCM-E2E-2.
Synchronization key: GFX1151-CURRENT-SOURCE-2026-10-07.

Princess-Luna WSL: AMD Radeon 8060S, live gfx1151, PCI 0000:c5:00.0.
Matching core/ROCm LLVM/MLIR 23.1.1 builds and native HIP movement/image
bindings are used. This source-only scratch checkout is synchronized from the
coordinated Super-Bear checkout. Full file hashes, unchanged compiler driver
hashes, binary/environment hashes and the live GPU probe bind the receipt.
Previous scratch bytes are backed up off home in compiler-sync-backups-20261007.

## Validation

- native-tests.txt: 237 passed, one foreign gfx1201 movement case skipped.
  Ordinary JIT, compiler-free warm reuse, portable replay, explicit f16/bf16
  widening, compiler ancestry, descriptor binding and lifetime checks execute.
- native-tests-initial.txt: 235 passed, three skipped, required-device guard
  failed because the explicit movement flag was absent. The final run sets
  that flag and executes the two owning gfx1151 cases; the guard is retained.
- math.json: 72 rows across f32/f16/bf16 inputs and three shapes, including
  reversed binary roles. Independent NumPy comparisons precede device timing
  and follow each changed-input JIT/portable call. Image identity is reused
  across shapes per operation/storage, with all four IR ancestry digests.
- resident.json: five ordinary-JIT/native C++ resident rows. All 180 warm
  output windows are bit-exact, including NaN payloads, signed zero and
  infinities. Stale generations, bad indices and foreign stream submission
  are rejected; warm compiler subprocesses and image loads/unloads are
  forbidden. Ninety native single-kernel event samples are recorded.
- The full paged read has 32 physical pages and 64 logical entries with
  repeated mappings. Thirty emitted Graph/Schedule/Tile/Target/backend/image
  artifacts are fingerprinted in exported-artifact-sha256.json.
- packet-verification.txt verifies recorder, source, compiler and runtime
  hashes. source-snapshot.json covers 4,526 synchronized files;
  benchmark-source-snapshot.json covers 483 recorder/oracle source files.

## Timings and limits

Resident C++ single-kernel event medians are 10.62–14.58 us for the four
small/large sliced-paged and dispatch rows; the full paged read is 164.01 us.
Synchronized resident-only host walls are 0.106–0.122 ms for those four rows
and 0.467 ms for the full read. Public host calls are 0.526–1.232 ms and
3.847 ms respectively. Resident warm arms exclude input upload and compilation;
download arms include output allocation/readback. Different timing domains
are recorded separately and are not interchangeable.

movement.json retains a separate five-window descriptor/legacy comparison.
Its paged-KV resident-launch event median is 2.887 us against 2.528 us retained,
failing the existing 10% device non-regression gate. Full-call medians are
0.627 versus 0.667 ms; dispatch full-call medians are 0.650 versus 1.931 ms.
These samples include initial native binding startup outliers (14.17 and
8.39 ms for paged read), and launch-window events can include enqueue gaps.
They neither establish isolated-kernel speedup nor promote a selector.

## Scope still open

This closes current-source gfx1151 synchronization for these proved compact
static math/movement envelopes. Arbitrary strides/general layouts, dynamic
composition, broader cache families, W8A8/MXFP4 performance closure and generic
frontend/AD integration remain open. Sibling Apple/x86/gfx1201/SM120 execution
is not inferred. The full unit lane and publication remain separate gates.

## Reproduction

Source .build/validation-env.sh in the owning scratch checkout, then run:

~~~sh
python benchmarks/rocm/benchmark_native_math_package.py --architecture gfx1151 --storage all --samples 5 --iterations 100 --output math.json
python benchmarks/rocm/benchmark_rocm_e2e_movement.py --trials 5 --iterations 100 --output movement.json
python benchmarks/rocm/benchmark_resident_movement.py --architecture gfx1151 --output resident.json
~~~
