---
last_updated: 2026-10-14
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
Graph tensor. The only admitted non-semantic Graph annotation is the verified
`tessera.effect_kind="pure"`, in addition to the compiler's artifact hash.
Other authored attributes require a separate contract.

`schedule.activation` retains the Graph value and records lowercase SHA-256
identity, `arch="sm_120"`, matching storage, `accum="f32"`, `kind="relu"`,
`axis=-1`, `workgroup_size=128`, `schedule="serial"`, `nan_mode="propagate"`
and explicit `zero_mode`. The f16 mode is `positive`; bf16 is `preserve`.
This follows the existing eager dtype-specific zero convention. The native
schedule digest includes that convention. Schedule->Tile replays the Graph
policy and rejects altered attributes or artifact identity.

## Tile, lowering and ABI

`tile.activation_kernel` has no SSA result and mutates its destination. Its
operands are `(source ptr, destination ptr, rows i64, columns i64)`; both data
planes use the same row-major half storage. Its attributes carry the checked
Schedule policy and digest. The initial physical mapping assigns a thread to
a row, 128 threads per block, with an active-row guard and a column loop.

Native NVIDIA lowering widens only for comparison, selects positive zero for
negative values and the f16 zero case, and otherwise selects the original
storage value. NaNs propagate through the selected original storage payload.
The result is stored in the original dtype. There is no normalization epsilon,
shared-memory reduction, or new generic dtype promotion.

Distinct checked ABIs are `tessera.nvidia.relu.x_o_rows_columns.f16.v1` and
`tessera.nvidia.relu.x_o_rows_columns.bf16.v1`. Descriptors bind ordered input/
output buffers, shape guards, Rows/Columns scalar roles, serial row geometry,
image identity and the immutable numerical policy. Native provider attachment
requires matching storage and schedule. One shared descriptor checker rejects forged numerical/schedule/storage fields
in composed, prepared, DAG and direct launch paths. Existing stream ordering, scratch
separation, context identity and retained-output checks remain in force.

## Evidence and open scope

`tests/unit/test_sm120_scheduled_activation.py` contains native verifier and
Schedule mutation gates plus exact RTX 5070 public-JIT/bitwise/numerical cases.
The packet is `benchmarks/baselines/sm120_norm_relu_matmul_20261014/guarded.json`.
Only the recorded static named envelope has this slice's exact-device proof.
Symbolic frontend metadata, additional producer/AD envelopes, captured kernel
measurement and activation schedule tuning remain open. Sibling backends need
their own physical consumers and owning-device proof.

## Execution and coverage registration

The nvidia_sm120 ReLU backend-manifest row distinguishes this standalone
f16/bf16 producer from its existing accumulator-epilogue formats. The actual
public composition uses canonical_nvidia_lhs_program / nvidia_lhs_program;
the execution matrix names its private stores and serial activation launch.
Its numerical fixture is tests/unit/test_sm120_scheduled_activation.py.
The matmul_relu conformance row and tests/device/nvidia/test_matmul_relu.py
continue to prove the separate fused post-matmul envelope.
