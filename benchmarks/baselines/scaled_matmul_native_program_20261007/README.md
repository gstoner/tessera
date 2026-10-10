# Native scaled-product paired program export — 2026-10-07

Owner: FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Synchronization key: SCALED-MATMUL-NATIVE-PROGRAM-2026-10-07.

The existing native forward AD pass has export-scaled-program=true.
It outlines the actual paired single-block scaled-product/add SSA into private
member functions. The semantic paired root remains as its witness. Admission
and symbol collision checks precede export mutation. No production Python
reconstructs member Graph operations or emits kernel arithmetic.

The native schema-one dictionary tessera.autodiff.scaled_program records:
root; argument_count; ordered steps (member symbol, input buffer IDs, output
buffer ID); returned output IDs; typed buffers with logical_bytes,
argument_attributes, readonly_input/returned_output/private_scratch ownership,
first_write and last_read. Returned buffers escape through call completion;
scratch read lifetimes follow actual SSA uses. Logical byte extents are not
physical padded allocation sizes. Runtime storage projection remains required.
Static byte-addressable integer/floating types without tensor encodings are
admitted; packed, dynamic and encoded-layout physical contracts remain open.

Native forward AD now preserves all primal argument attributes. Tangents
inherit layout, sharding and dimension names without becoming model-parameter
declarations. Outlined member arguments preserve the actual root argument
contracts. The native fixture checks scale-seed operand order, distinct returned
outputs, scratch lifetime through the sum, unchanged block scale layout and
model-parameter/tangent metadata, plus conflicting export-mode rejection.

Both native tools rebuild. Combined core/backend lane: 643 pass, 66 unsupported
(709 discovered). Focused AD/diagnostic/pass tests: 347 pass, 16 skip.
The skipped unit cases do not establish device evidence. This export is not a
launchable AD image. Native program member compilation/projection, sum lowering,
checked physical allocation/stream/completion ownership and owning AMD numerical
and separate kernel/end-to-end timing proof still need implementation.
Generic batching/linear-transpose and the full unit closure gates remain open.

Reproduce from Super-Bear WSL with .build-sm120-w1-1/validation-env.sh:
tessera-opt tests/tessera-ir/phase_f4/autodiff_forward_scaled_matmul_program.mlir
  --tessera-autodiff-forward='export-scaled-program=true'

Fresh RTX 5070 regression proof after argument-metadata preservation passes
159 NVFP4/attention numerical tests, no skips, in 140.82 seconds. These are
existing executable routes; the new scaled AD program still lacks launch proof.
The query records SM12.0 and the actual GPU UUID in the device receipt.
