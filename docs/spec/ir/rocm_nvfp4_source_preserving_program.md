---
last_updated: 2026-10-16
---

# Native source-preserving NVFP4 program

Owner: ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1.
Synchronization key: NVFP4-NATIVE-MULTI-OUTPUT-20261015.

The native scaled-primal export admits an explicit gfx1201 single-block Graph:
five distinct roots (packed codes, E4M3 scales, FP64 projection globals, E4M3
activation, FP32 activation scales), one tessera.nvfp4_expand and one returned
tessera.scaled_matmul. Weight storage is N-by-K with transposeB=true. Scale
granularity is block [1,16], format FP32, accumulation FP32 and execution
exact_per_block. Product storage is FP32 or BF16. Graph verification checks
the expansion's precise numerical policy. Conflicting consumer policies reject.

Schema tessera.native.nvfp4_program.v3 means this static source-preserving
program specifically. It does not reinterpret v1 (lossy static), v2 (lossy
bounded), or any attention schema. Schema v4 now admits bounded
source-preserving frames through the separate checked capacity/active-row
contract in [NATIVE_NVFP4_BOUNDED_SOURCE_CONTRACT.md](../NATIVE_NVFP4_BOUNDED_SOURCE_CONTRACT.md).
Static v3 retains the original meaning; bounded requests are not relabeled v3.

Native MLIR assigns roots 0..4 in original argument order, expansion outputs
5 and 6, and product output 7. Role indices preserve permuted frontend arguments.
Producer results have first_write=0, last_read=1 and private_scratch ownership.
Product output has first_write=1, last_read=2 and returned_output ownership.
Outlined members clone actual operations with their operands, results, numerical
attributes and module target. Python consumes serialized metadata and checks
projected members against the original export; it does not reconstruct Graph.

The additive TesseraRocmProgramMultiOutputStep and
tessera_rocm_program_prepare_multi_output preserve the single-result ABI.
Inputs precede outputs in flattened kernel memref arguments, then scalars.
Total memrefs per stage are at most seven, inputs at most six, scalars at most
eight. Fresh output slots must follow compiler SSA result order. Native C++
independently derives first writes and last reads and checks all declarations
before allocation. The image-plan cache includes output identities. Private
intermediates cannot be read as returned outputs. Existing ordered submission,
generation-checked transactional readback and teardown rules remain.

The owning-device fixture executes compiler-derived plans through the native
ABI directly. Ordinary composed public JIT and serialized replay remain separate work.
The prepared ownership extension is described below.

## Prepared execution and persistent expansion

The prepared binding consumes this same native v3 manifest and existing native
member descriptors. It retains the static checked Target witness separately
from the native shape-independent image projection. It validates the native
buffer shapes, storage, role permutation, write/read lifetimes and both member
bindings before loading a provider. Input codes/scales/globals receive the same
checked expansion boundary as the isolated public operation.

Preparation copies all five roots into native ownership. import_weights runs
only the producer. invoke_consumer runs only the product after native readiness
checks. update_activations copies only the activation and activation-scale
roots; both expanded weight buffers remain resident. Full update retains its
original meaning and invalidates every intermediate.

The additive update_slots ABI admits distinct original input slots and exact
byte sizes before state changes. It propagates invalidation through actual SSA
edges stored at preparation. Invalid frames preserve the previous generation.
The invoke_range ABI admits a nonempty contiguous stage range only if all
external operands are initialized. Success records initialized results and a
new generation, but readback additionally requires every returned output to be
ready. A producer-only generation cannot publish an uninitialized product.
A weight update invalidates both expansion outputs and downstream products;
consumer-only launch then rejects until the producer executes again.

Cached allocation acquisition resets readiness and uploads the new caller's
roots. No private value from a prior session can satisfy the new session's
dependencies. Failed device completion retains the existing quarantine and
explicit-close behavior. Profiling all pure SSA members restores all outputs'
readiness; it does not change numerical policy.

The prepared API supports FP32 and BF16 stores. The named ordinary public
JIT/replay envelope currently returns FP32; it is described below.

These ranges apply to the admitted pure SSA members. They do not establish
transactional semantics for recurrent SSM checkpoints or other state effects.


## Ordinary JIT and portable replay

A five-host-input Python function containing exactly nvfp4_expand followed by
the exact K16 scaled_matmul reaches this same native v3 partition through the
canonical checkpoint program dispatcher. The current ordinary frontend result
is FP32. Each call consumes all current input contents: warm image/package reuse
never assumes that host checkpoint arrays are immutable. Explicit import-once
reuse belongs to the prepared owner.

Portable schema tessera.rocm.nvfp4_source_frontend_program.v1 retains the native
program, both original member Graph/Schedule/Tile/Target witnesses, static
checked Target, projected image/descriptors and original Python argument names.
It is separate from the older lossy frontend/resident schemas. Restoration
validates the complete frame and digest without compiler tools or Graph
reconstruction. Digests check integrity, not origin authentication; native
artifacts must come from a trusted compiler/storage source.

The runtime checks exact parent target, Graph and argument bindings before
provider access. Compile reports expose the original Graph hash plus ordered
member Schedule/Tile/Target hashes and the portable artifact hash. These are
separate native modules, not one monolithic lowered IR.

Fresh-process tests disable both compiler paths and eager numerical oracles.
Original checkpoint validation additionally checks source identities and
captured model inputs. PR960 adds bounded v4 public/prepared/compiler-free
replay, allocation retention and original gate/up projection evidence; see the
bounded contract above. Full MLP/model-quality acceptance and wider composition
remain open.
