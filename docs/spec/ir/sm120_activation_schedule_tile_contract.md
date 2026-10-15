---
last_updated: 2026-10-15
---

# SM120 storage-preserving activation Schedule/Tile contract

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Synchronization key: W11-RELU-PRODUCER-20261014.

## Named route

The initial integration is RMSNorm -> ReLU -> matmul from the ordinary textual
frontend, including separate f16/bf16 normalization and activation stores.
Native member outlining records original SSA inputs/outputs and the allocation,
first-write and last-read intervals of both private intermediate buffers.
The consumer uses `tile.view` and typed fragments. No Python arithmetic or
Tile constructor supplies a member implementation.

## Graph and Schedule

`tessera.relu` must preserve a positive static unencoded rank-two f16/bf16
Graph tensor. In addition to the compiler artifact hash and verified pure
effect annotation, it may carry explicit schedule="serial" or "flat_128".
Other authored attributes require a separate contract.

`schedule.activation` retains the Graph value and records lowercase SHA-256
identity, `arch="sm_120"`, matching storage, `accum="f32"`, `kind="relu"`,
`axis=-1`, `workgroup_size=128`, `schedule="serial"|"flat_128"`, `nan_mode="propagate"`
and explicit `zero_mode`. The f16 mode is `positive`; bf16 is `preserve`.
This follows the existing eager dtype-specific zero convention. The native
schedule digest includes that convention. Schedule->Tile replays the Graph
policy and rejects altered attributes or artifact identity.

## Tile, lowering and ABI

`tile.activation_kernel` has no SSA result and mutates its destination. Its
operands are `(source ptr, destination ptr, rows i64, columns i64)`; both data
planes use the same row-major half storage. Its attributes carry the checked
Schedule policy and digest. Serial assigns one thread to a row and loops
over columns with 128 threads per block. Explicit flat_128 assigns one
thread per stored element, guards against Rows*Columns, and launches
ceil(Rows*Columns/128) blocks. Positive dimensions must be below 2^31 and
their product no greater than 2^31. Serial remains the default.

Native NVIDIA lowering widens only for comparison, selects positive zero for
negative values and the f16 zero case, and otherwise selects the original
storage value. NaNs propagate through the selected original storage payload.
The result is stored in the original dtype. There is no normalization epsilon,
shared-memory reduction, or new generic dtype promotion.

Distinct checked ABIs are `tessera.nvidia.relu.x_o_rows_columns.f16.v1` and
`tessera.nvidia.relu.x_o_rows_columns.bf16.v1`. Descriptors bind ordered input/
output buffers, shape guards, Rows/Columns scalar roles, mapping-specific geometry,
image identity and the immutable numerical policy. Flat symbols include
_flat_128_ before the digest and use sm120_activation_flat_elements geometry.
Serial symbols and sm120_activation_serial_rows are unchanged. The native
tessera_nvidia_matmul_supports_flat_activation probe prevents older
prefix-only prepared providers from using a serial grid. Negotiation occurs
before owner initialization/allocation. Native provider attachment
requires matching storage and schedule. One shared descriptor checker rejects forged numerical/schedule/storage fields
in composed, prepared, DAG and direct launch paths. Existing stream ordering, scratch
separation, context identity and retained-output checks remain in force.

## Evidence and open scope

`tests/unit/test_sm120_scheduled_activation.py` contains native verifier and
Schedule mutation gates plus exact RTX 5070 public-JIT/bitwise/numerical cases.
The packet is `benchmarks/baselines/sm120_norm_relu_matmul_20261014/guarded.json`.
Only the recorded static named envelope has this slice's exact-device proof.
Symbolic frontend metadata, additional producer/AD envelopes, captured whole-program
measurement and default activation selection remain open. Sibling backends need
their own physical consumers and owning-device proof.

## Execution and coverage registration

The nvidia_sm120 ReLU backend-manifest row distinguishes this standalone
f16/bf16 producer from its existing accumulator-epilogue formats. The actual
public composition uses canonical_nvidia_lhs_program / nvidia_lhs_program;
the execution matrix names its private stores and serial activation launch.
Its numerical fixture is tests/unit/test_sm120_scheduled_activation.py.
The matmul_relu conformance row and tests/device/nvidia/test_matmul_relu.py
continue to prove the separate fused post-matmul envelope.


## Optional flat schedule evidence

W11-RELU-FLAT-20261015 adds tests/unit/test_sm120_coalesced_activation.py.
The owning RTX5070 gate passes 46 new/existing cases: all 65,536 FP16 and
BF16 encodings, ragged/singleton columns, independent composed numerics,
unchanged norm/matmul PTX, Graph-copy isolation, retained outputs, two-sided
bounded replay and stale-provider refusal. Matched captured producer timing
and native program/host timing remain separate. Single-column regressions
and program enqueue noise preclude default-selection claims. See
benchmarks/baselines/sm120_relu_flat_20261015/README.md for final packet and
delivery gates. Wider storage/AD/architectures remain independent obligations.
