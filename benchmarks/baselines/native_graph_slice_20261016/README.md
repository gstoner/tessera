# Native Graph slice foundation

Owner: FRONTEND-IR-MEDIUM-1 / LAYOUT-ALG-1 / ROCM-NVFP4-INGEST-1 / E2E-REAL-6.
Sync: ROCM-NATIVE-GATE-UP-VIEWS-20261016.

This is compiler-contract evidence from Super-Bear WSL with matching
LLVM/MLIR 23.1.1. It contains no new GPU execution or performance claim.
The original five-slice objective remains open.

## Verified

- `slice-owning-contract.log` and `slice-junit.xml`: 83 cases, zero skips:
  55 frontend cases and 28 production-compiler cases. These cover both public
  spellings, positional/keyword bindings, shape-only/numerical tracing,
  clipping/empty outputs, integer coercion, signed-i64 overflow boundaries,
  native JVP/VJP structure, guarded dynamic seeds and Tensor IR lowering.
- `slice-final-contract.log`: 155 passed, four unrelated hardware skips in
  the slice/transpose/shape selection. This overlaps the 83-case selection.
- `slice-shared-regression.log`: 541 passed, 32 unrelated skips across
  tracing, operation/dtype/diagnostic/pass registries and typed buffer views.
- `integrated-slice-normalization-oracle.json`: the actual C++
  `SliceContract.h` normalizer agrees with Python `slice.indices` on 101,210
  cases, including signed-64 extrema and logical extents through INT64_MAX.
  This arithmetic diagnostic allocates no tensors.
- `post-slice-boundary.json`: a public positional trace retains one parent,
  two slices and SiLU, performs shape-only frontend capture, and passes the
  native Graph verifier. Schedule still reports the isolated-operand contract.
- `buffer-view-lowering-proof.json`: four padded row/column-major gate/up
  addressing cases execute through the existing shared Tile materializer and
  LLVM on the CPU. They preserve the parent and mask 24 invalid stores.
  This is addressing proof, not SiLU arithmetic or a ROCm package.

`source-identity.json` binds modified candidate source and compiler tool bytes
against merged main c02b662df. The measurement used an uncommitted candidate;
file hashes identify the tested implementation. The normalization JSON binds
the exact header and diagnostic binary. Raw reproduction scripts retained in
this packet use the original owning-host scratch paths.

## Remaining execution gate

The real gate/up Graph must next reach sealed Schedule/Tile view SSA, ROCm
Target/native image and checked program-frame ownership. Parent pitch,
read-only lifetime, separate output storage and device numerics must be proved
on gfx1201, with separate producer/consumer and completed-host timings.
No compact host repacking, sibling hardware evidence or FP4 instruction
support is inferred here. Full activation quantization/downprojection,
original-checkpoint model quality, dynamic-source transpose and the wider
NVIDIA/ROCm obligations remain open.

## Focused reproduction

From the Tessera root in host WSL with matching production compiler paths:

```sh
python -m pytest tests/unit/test_slice_semantic_contract.py -q
python -m pytest tests/unit/test_graph_transpose_axis_contract.py tests/unit/test_shape_rule_registry.py -q
```

Contract: `docs/spec/NATIVE_GRAPH_SLICE_CONTRACT.md`.

Frozen owning-host reproduction: `build_slice_contract_normalize.py` builds
the header diagnostic; `check_integrated_slice_normalization.py` runs the
independent Python oracle. Both retain their original host scratch paths.

Review follow-through: concrete frontend slices retain borrowed NumPy views
without compaction so non-native downstream dtype promotion remains valid.
The actual native SiLU capture still performs no eager element arithmetic.
The reconciliation selection passes 904 tests; the production compiler lane
passes 1,340 with zero skips and its executed-case gate.
