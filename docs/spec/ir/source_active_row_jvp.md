---
last_updated: 2026-10-15
---

# Source-active normalization JVP packages

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-FWD-CORE-1.
Synchronization key: W11-ACTIVE-SOURCE-20261015.

## Named semantic envelope

The first source-active SM120 program is a non-affine, static rank-two
RMSNorm with explicit FP32 accumulation, followed by ReLU and a plain matmul
with FP32 output. Source, RHS and source tangent have matching FP16 or BF16
storage. The source and RHS are distinct original function arguments; their
order may be permuted. The third physical input is the tangent of the original
source argument. Storage policy, when declared, must match the source.

The native forward transform produces the actual primal and tangent SSA.
The producer retains the original normalization policy, working casts,
statistics, reductions, comparison/masking and storage conversions. The
[native normalization precision contract](normalization_ad_accumulator_policy.md)
defines those arithmetic semantics. Unspecified accumulator behavior is not
changed by this package. Fused epilogues, transposes, dynamic dimensions,
resident roots and general reverse products need separate contracts.

The Python frontend supports explicit RMSNorm `numeric_policy` and preserves
original storage before selecting wider eager arithmetic. Its result is rounded
after normalization and optional gamma multiplication. The default eager path
is unchanged. Canonical matmul result inference honors explicit `output_dtype`
in the same way as tracing and eager execution.

## IR and memory contracts

The compiler outlines one two-result producer and two matmul consumers, then
lowers the exact projected members through Graph, Schedule, Tile, NVIDIA Target
and native LLVM/NVVM IR. Python does not reconstruct derivative Graphs or Tile
bodies. Existing payload-free `!tile.tile` and fragment `tile.view` retain their
contracts.

`!tile.buffer_view<element, access, order>` is an additive scalar-memory view.
Its element is F16, BF16 or F32; access is read/write and order is row/column
major. `tile.buffer_view` binds a pointer, logical rows/columns and leading
dimension. `tile.load_element` returns zero outside logical bounds;
`tile.store_element` masks out-of-range writes. Element, access, address space,
constant extents and physical pitch are checked. Dynamic descriptor values
require the owning buffer contract to establish valid dimensions and pitch. Allocation capacity remains
the checked native owner/caller's responsibility.

Bounds checks may be removed only from actual static extents and dominating
SCF induction/unsigned-bound comparisons. An else branch, shifted induction or
unverified metadata cannot assert an access in bounds. The sealed
`tile.row_region_kernel` carries the actual scalar/reduction body. Target
lowering consumes all Tile memory operations and types, verifies original
pointer roots, and preserves the Tile body hash separately when resealing the
native body. ReLU uses an ordered comparison and selects the original storage value
for NaNs and BF16 signed zero; F16 zeros become positive, matching its canonical
activation contract. Row workgroups contain 128 threads.

## Native program and persistence

The additive C API uses seven buffers and three ordered stages. Roots 0/1 retain
original argument order and root 2 is the seed. Private buffers 3/4 are producer
primal/tangent edges; buffers 5/6 are independently owned returned F32 outputs.
A `TesseraNvidiaActiveSourceProgramStep` is 112 bytes on the owning 64-bit ABI;
its version is 1. Metadata binds exact input/output roles, dimensions, storage,
entry symbol and native parameter layout. Existing RHS-only frames are unchanged.

One C++ owner prepares modules, stages three host roots, launches the three
native stages on its private stream, and publishes both outputs only after
completion. It rejects shape, stride, dtype, alias, context and lifetime
mismatches. Profiling requires the same root addresses and a live generation
lease, then refreshes all host root contents and completes one warmup invocation
before opening event/capture windows. In-place mutations are therefore reflected
in the returned profiled outputs. Staging and warmup are excluded from device
timing; each stage is separate from the composed device event. External launch
streams are not accepted.

The native `tessera.native.sm120_active_source_outline.v1` record remains
`artifact_only`: an outline alone is not executable. The compiled portable
schema is `tessera.nvidia.tensor_active_source_jvp.v1`. It carries the native
plan, actual members at every compiler boundary, PTX images, resource records,
compiler/toolchain identity, typed step ABI and cross-bound digests. Existing
RHS-only native v7 and portable schemas are not renumbered.

Restoration validates the complete package and parent Graph/argument ABI before
provider access. Replay requires no compiler or image file reads. Digests detect
corruption and inconsistent witnesses; they do not authenticate arbitrary
third-party native code. A bounded pure certificate cache hashes the entire
payload, including its integrity field. It owns no device allocation. A matching
live JIT owner is borrowed; standalone replay owns and closes its native owner.
Returned receipt dictionaries do not alias saved ABI records.

## Backend assessment and evidence

SM120 has exact RTX 5070 execution proof for the named host-root program and
portable replay. The internal view types are target-neutral; this does not
provide physical producer lowering on Apple, x86 or ROCm. Each sibling needs its
own package contract and device evidence. Apple eager policy handling is shared;
the separate Metal encode API rejects explicit policy until it has an owning
implementation.

Durable tests: `tests/unit/test_native_sm120_active_source_jvp_runtime.py` and
`tests/unit/test_normalization_frontend_policy.py`. A compiler-produced fixture
at `tests/fixtures/native_sm120/active_source_jvp_f16.json` permits provider-free
admission tests. The owning-device tests exercise numerical execution, original
argument permutations, minimal/ragged shapes, fresh-process replay, live-owner
reuse, changed inputs, receipt isolation and malformed artifact rejection.
