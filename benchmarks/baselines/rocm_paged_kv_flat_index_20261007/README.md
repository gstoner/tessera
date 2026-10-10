# Native paged-KV flat-token indexing — 2026-10-07

Owners: ROCM-E2E-2 / E2E-REAL-6.
Synchronization key: ROCM-PAGED-KV-FLAT-INDEX-2026-10-07.
Recorder: benchmarks/rocm/record_paged_kv_index_ab.py.

Native Graph -> Schedule -> Tile -> ROCm Target -> LLVM/HSACO retains compact
f32/i32 storage, the seven-scalar ABI, bounds guards and 256-thread geometry.
Contiguous H*D token indexing eliminates separate head/feature decomposition.
No Python kernel emitter, shape-specialized image or tolerance change is added.

## Owning-device evidence

Each architecture retains compiler hashes, serialized images/descriptors,
five IR stages per arm/case, raw HSACO fingerprints and the recorder source
used for its measurements. One image per arm covers all six shapes.
Coverage includes repeated physical pages, different logical/physical counts,
odd head/feature/page extents and page-size one.

- Princess-Luna, Radeon 8060S / gfx1151: 339 focused tests pass; five foreign
  architecture cases skip. Candidate/reference ratios span 0.935–0.965 for
  five small cases and 0.767 for the large case. Identical-reference controls
  span 0.997–1.009.
- Tajasaurus, RX 9070 XT / gfx1201: 336 focused tests pass; eight foreign
  architecture cases skip. Ratios span 0.892–0.914 for five small cases and
  0.802 for the large case. Identical-reference controls span 0.999–1.002.
  tests-missing-file.txt preserves the initial collection failure. The test
  was restored after its corresponding dependencies matched coordinated source.

These are same-allocation resident HIP-event launch windows: 21 alternating
order samples and 1000 launches each. Python enqueue gaps are included.
They are not isolated kernel-only or end-to-end public-call timings.
Outputs are poisoned before every window and compared bit-exactly with an
independent NumPy indexing oracle, including NaN payloads, signed zero and
infinities. ABI fields and allocation addresses match between arms.

The historical backend_unsigned_divisions field searches a serialized
backend artifact. It is inconclusive as an LLVM/ISA instruction count and
is removed from the canonical recorder. gfx1151 ISA disassemblies are retained.

## Reproduction

From the owning checkout and matching LLVM/MLIR 23.1.1 environment, run the
recorder with --architecture gfx1151 or gfx1201 and --directory /scratch/ab.
Package each arm in a fresh process:
--phase package --arm reference --tool /path/to/reference/tessera-opt
--phase package --arm candidate --tool /path/to/current/tessera-opt
Then run --phase benchmark and separately
--phase benchmark --identical-image-control.

Archived recorder snapshots retain exact measurement hashes. Canonical
recorder cleanup does not rewrite historical results. Reference binaries
remain on owning scratch hosts, excluded from Git; their fingerprints and
reference lowering source are included here.

## Remaining scope

General strided layouts, dynamic/composed consumers, isolated kernel/public
timing closure and wider family performance remain open. No selector promotion
or sibling physical proof follows. The aggregate full unit lane still has
generic scaled_matmul batching and linear-transpose failures.

Canonical recorder replay on gfx1201 passes all six bit-exact cases with three windows of 100 launches; gfx1201/canonical-replay.json matches the current recorder hash. This is recorder validation, not the primary performance sample.

Delivery gates: coordinated compiler build passes; 331 shared compiler/runtime tests pass with 13 device-gated skips; audit/citation/plan-routing gates pass 22 tests; benchmark recorder/coverage gates pass 11 tests. Both owning lowerings match the archived candidate source hash. No full-suite green claim.
