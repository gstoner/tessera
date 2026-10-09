# Mapped-result permutation foundation

Owner: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / LAYOUT-ALG-1.
Synchronization: MAP-RESULT-AXES-FOUNDATION-20261008.
Published as [PR903](https://github.com/gstoner/tessera/pull/903), dependent on
PR902; this is a prerequisite, not GPU map-output closure.

## Architectural change

Graph transpose now checks each result axis against its declared source axis,
rather than accepting any equal multiset of dimensions. The optional canonical
DenseI64 permutation is rank-sized, unique and nonnegative; omitted axes
reverse all dimensions. Raw native axes/perm aliases are rejected so they cannot
silently execute default reversal. Native shape inference uses that same
decoder instead of its retired tessera.perm alias. Frontend keyword/positional axes and negative
aliases normalize to this canonical permutation; conflicting axes/permutation
attributes are rejected. Dot-T and shape-only tracing reverse all dimensions.

Symbolic dimension annotations and SSA propagation now use that same axis order,
including equal-sized axes. Unranked tensors do not acquire a positional proof.
The old symbolic compiler accepts wrong annotations and rejects a valid chained
transpose; six new cases exercise those boundaries. The historical positive
MLIR fixture now declares its intended axis swap explicitly. Three native shape
inference cases fail with the frozen compiler and pass with the updated core.

Native reverse AD now computes the inverse of an explicit permutation. Forward
AD carries its original axes. Reverse AD also swaps input/output symbolic
annotations; a native AD→symbolic-verification case proves that composition. Static general-rank transpose materializes as
linalg.transpose instead of stopping at rank two. Native AD activity and pure
effect facts survive lowering; other policy metadata still needs its owning
consumer. The optimization decoder remains stricter than semantic verification:
an unknown policy cannot be folded away merely because dimensions are equal.

This changes typed Graph semantics and verified MLIR transformation. It does
not add a Python numerical backend or enable nonzero vmap out_axes. The scaled
native program still needs a result-permutation member, Schedule/Tile lowering,
checked ownership/ABI, and exact-device numerical/timing proof before that
production route can be enabled.

## Evidence and limits

- A frozen native compiler accepts all ten original malformed Graph fixtures;
  both explicit-permutation AD/lowering probes also fail. Logs and binary hash
  are retained. These controls are defect reproduction, not current proof.
- 53 current host cases pass: semantic verification, AST/tracer keyword and
  positional/negative axes, general-rank lowering, numerical CPU primal/inverse
  adjoint execution, and ordinary-source changed-input/compiler-free replay.
- 381 focused axis/symbolic/diagnostic/pass cases pass with the final compiler.
- 451 frontend/shape/operation/dtype/diagnostic/pass gates pass; 14 existing
  environment/tool skips. Ruff passes; mypy ratchet remains zero.
- source-tests.txt retains the initial 42-case checkpoint; symbolic-tests.txt
  is the final 381-case run including all 53 axis tests.
- Current-source GPU regression receipts are recorded separately. They prove
  existing routes, not a new GPU transpose/materialization route.
- The core/ROCm compiler was freshly built from this tree with LLVM/MLIR 23.1.1
  assertions. Numerical CPU execution uses the existing recorded native JIT
  engine. A fresh JIT build was unavailable because the assertion install lacks
  libMLIRExecutionEngine.a. No new JIT runtime implementation is claimed.
- cpu.json binds thirteen source hashes, compiler and runtime hashes, typed Graph
  and native Linalg IR, actual CPU model, compile time and completed-call timing.
  identity.json verifies the final source hashes against this tree.

## Diagnostic timing

On AMD Ryzen Threadripper 3970X, five rank-three/rank-four profiles record seven
windows each, all longer than 20ms. Completed native CPU ABI invocation plus
materialization has medians about 19–74 microseconds. Compilation is measured
separately. Independent NumPy output checks precede and follow timing, and warm
invocation does not compile. There is no GPU timing, NumPy speedup comparison,
physical-schedule promotion, or full compiler-route closure claim.

Reproduce in host WSL with the matching TESSERA_OPT and recorded TESSERA_JIT_LIB:
run tests/unit/test_graph_transpose_axis_contract.py and
benchmarks/compiler/benchmark_native_transpose_axes.py --output <scratch JSON>.
The latter is explicitly a diagnostic Graph/Linalg/LLVM gate, not a production
vmap selector or an alternate GPU fast path.

## Remaining

Native Schedule/Tile result permutation and lifetime ownership, public mapped
out_axes integration, dynamic/encoded-layout envelopes, generic scaled_matmul
batching/transpose closure, and owning Apple/NVIDIA/ROCm physical permutation
proof remain open. Primitive coverage states and zero-open CI assertions are
unchanged.
