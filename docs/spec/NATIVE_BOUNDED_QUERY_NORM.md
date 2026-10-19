---
last_updated: 2026-10-19
audit_role: reference
---

# Native bounded-query normalization

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync: NATIVE-BOUNDED-SOURCE-ATTENTION-20261019.

## Native member contract

The named SM120 component is an isolated, compact, unweighted RMSNorm with
homogeneous FP16 or BF16 input/output, explicit FP32 accumulation and immutable
positive finite epsilon. Graph retains tensor<BxHx?xD> with fixed positive B/H/D
and a dynamic query axis. The function carries the positive i64
tessera.native.query_sequence_bound. It is native member intent, not a public
JIT option or an instruction to discard unsupported Graph operations.

The entry has one argument, one result and exactly the normalization and return.
Encoded tensors, argument layout overrides, extra roots/operations, layer norm,
FP32 storage and other targets are outside this named profile. Existing static
normalization admission and hash recipes remain unchanged.

Graph-to-Schedule derives capacity from the actual logical type and marker.
It preserves the dynamic type and records an atomic attribute group:

| Field | Meaning |
| --- | --- |
| shape_policy | bounded_query_v1 |
| sequence_bound | maximum Sq |
| row_capacity | B*H*sequence_bound |
| fixed_columns | D |
| row_multiple | B*H |

The positive products fit signed 32-bit row geometry and checked byte offsets.
Schedule and Tile use the same native verifier. Partial or conflicting policies
are rejected; Schedule-to-Tile rederives the fields from retained Graph intent
before erasing the semantic entry. Capacity changes participate in the native
Schedule digest and generated entry symbol. Static operations carry none of
these fields.

## Physical ABI and guards

The existing four-argument pointer/pointer/i64/i64 kernel ABI remains:
source, destination, actual_rows, columns. Actual rows equal B*H*runtime Sq.
Native lowering checks positive rows, rows<=capacity, divisibility by the fixed
prefix and columns==D before access. The per-row guard additionally excludes
inactive queries. In cooperative_128 mode the entry/row guard is CTA-uniform,
so every participating thread reaches each reduction barrier.

Serial compensation, cooperative reduction order, epsilon rounding and output
storage conversion retain their existing numerical policies. This profile
does not introduce a Python numerical lowering, pointer reinterpretation or
new dtype. Native prepared composition must still validate allocation spans,
stream ordering and distinct private normalized output ownership.

## Backend coordination

The physical contract belongs to exact SM120. ROCm and x86 consumers must not
erase its shape fields and bind another launch ABI. Apple has no consumer for
this SM120 launch carrier. Sibling static profiles retain their own independent
proof; no NVIDIA schedule or performance result transfers.

## Current integration state

The private normalization component passes compiler positive/negative contracts,
eight exact-SM120 serial/cooperative device cases, twelve static artifact
identity profiles and two counterbalanced timing orders. The mixed validation
cohort passes 510 tests with 15 declared skips. Source-bound evidence and timing
domain limits are recorded in
benchmarks/baselines/native_bounded_query_norm_20261019/README.md.
Full CPU/production delivery gates have not been rerun for this increment.

Full original-source bounded attention reverse additionally needs native paired
member projection, complete capacity/lifetime manifests, retained CUDA ownership,
ordinary JIT and compiler-free portable replay. Those are separate gates.
The prior row prerequisite is frozen at7a2aeea11 with its own source-bound
evidence in benchmarks/baselines/native_bounded_query_rows_20261019/.
No aggregate five-slice requirement closes from this component.
