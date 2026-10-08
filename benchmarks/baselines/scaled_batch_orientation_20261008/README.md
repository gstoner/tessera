# Named scaled-batch operand orientation

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync SCALED-BATCH-ORIENTATION-20261008.

The public eager typed reference now accepts transposed LHS storage under
shared-RHS, shared-LHS and independent batch policies. Logical matrix and scale
indexing already used the oriented operand dimensions; the obsolete admission
check rejected mathematically valid calls before that implementation executed.

24 new numerical cases cover both operand orientations, two leading batch
axes, shared/independent ownership, FP32 and encoded E8M0 scales, and ragged
block boundaries. A scalar-indexed FP64 oracle uses the original logical
operands independently of the reference's matrix products and transposes.

Host WSL validation: 385 passed, zero skipped, including native Graph
orientation verification and operation/dtype/diagnostic/pass drift gates.
No physical backend selector, image, ABI or performance claim changes.
Generic batching, transpose AD and wider derivatives remain open; the two
aggregate PR895 closure checks are not claimed fixed by this correction.
