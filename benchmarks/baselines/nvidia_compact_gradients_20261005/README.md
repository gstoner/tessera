# Native compact requested attention gradients — 2026-10-06

Owner **AD-RESIDUAL-EVAL-1**; siblings FRONTEND-IR-MEDIUM-1 / W1.1 / E2E-REAL-6.
Synchronization key **NVIDIA-COMPACT-GRADIENTS-2026-10-06**. Implementation is landing; the full five-slice compiler objective remains open.

## Compiler and runtime contract

The isolated public attention VJP keeps its complete logical typed Graph product. Native paired AD derives activity from the requested frontend argument indices and verified SSA mapping. Graph-to-Schedule seals only the requested physical results, launch layout and 64/128-thread geometry into the native contract/hash. Schedule-to-Tile emits the matching LLVM pointer/scalar entry and verified backward kernel. NVIDIA Target lowering retains FP32 accumulation and saved O/LSE, omits inactive stores, and emits LLVM/NVVM/PTX.

The compact ABI is `tessera.nvidia.sm120.attention_backward_lse.compact.f32.v1`. Physical buffer bindings, exact shape guards, scalar ordinals, output order, entry symbol, thread width, launch policy, workspace and completion are checked together. Physical broadcast-bias entries retain eleven scalar slots. C++ owns host/resident submission and precomputed benchmark arguments; no Python GPU body constructor or numerical derivative reconstruction is used. Python remains the public frontend and checked marshalling layer.

Opt in with `compact_gradients=True`, selecting `compact_launch="logical_v1"` or `"packed_v1"` and `compact_threads=64` or `128`. Complete physical outputs remain the public default. No global schedule promotion follows from this bounded experiment.

## Exact-device proof

[packet.json](packet.json) queries and gates NVIDIA GeForce RTX 5070 / SM120, UUID GPU-cba12639-821a-7a10-4cd3-f918f9c0a545, driver 610.88. It records source/compiler/runtime SHA256 identities and per-arm Tile/Target/PTX/descriptors in [artifacts](artifacts/).

Forty cases, each with five arms: complete 128-thread output, compact packed 128/64, and compact preserved-logical ranges 128/64. Twenty-four plain cases cover two shapes, causal/noncausal, six requested-gradient sets and reordered frontend arguments. Twelve cases cover grouped heads and full/broadcast physical bias. Four V-only cases prove linearity with Inf/NaN primal V. Maximum absolute error against the independent FP64 oracle is **7.42713e-8**, using the original `atol=rtol=4e-5`. Nonfinite primal trace warnings are retained; native V-only derivatives pass.

Checked host and resident descriptor launches pass for all finite compact arms. Capture uses private CUDA allocations; actual caller inputs are overwritten after capture, repeated cotangents preserve correctness and requested result order, prior outputs remain valid, and closed frames refuse execution. Actual native allocation bytes and retained gradient counts are recorded. These are synchronous static isolated-attention proofs, not arbitrary dynamic/aliased/async or composed AD admission.

## Balanced measurements

Five alternating trials keep the same forward image and values. CUDA-event Python dispatch, per-call native resident dispatch, and precomputed native launch argument windows are separate from checked allocating/synchronous backward wall time. CUDA-event windows can include host submission gaps; none is claimed as isolated kernel time or application throughput. Resources use the actual 64/128 launch block. History preserves the unsuccessful initial packed schedule and parser/argument attribution iterations.

For shape `B=1,Hq=2,Hkv=1,Sq=8,Sk=129,D=8,Dv=6`:

| Case | Complete prepared event µs | Packed 128 µs | Logical 128 µs | Logical 64 µs | Complete wall µs | Logical 128 wall µs |
|---|---:|---:|---:|---:|---:|---:|
| noncausal k | 15.061 | 17.345 | 15.103 | 14.984 | 175.661 | 163.671 |
| noncausal v | 12.755 | 12.384 | 11.842 | 12.291 | 195.051 | 156.111 |
| causal k | 16.443 | 14.851 | 14.856 | 14.950 | 188.701 | 165.131 |
| causal v | 11.984 | 12.470 | 12.439 | 12.408 | 172.881 | 166.301 |

V-only retained gradient storage drops **7,736 → 3,096 bytes**, three outputs → one; K-only drops to 4,128 bytes. This excludes the cotangent and retained forward state. Preserved-logical launch ranges retain inactive CTA slots without allocating or storing inactive gradients.

Across the 36 finite rows, logical-128 candidate/complete median ratios are 0.9993 for precomputed native events and 0.9374 for checked backward wall time. Their worst ratios are 1.0435 and 1.0368. Packed-128 reaches 1.1816 in the prepared event window; reducing threads alone does not resolve every packed schedule loss. Logical-64 fixes the named long-row K-only gap, but has a 1.0679 worst prepared-event ratio elsewhere. [comparison.json](comparison.json) retains all four candidates and an observed 10% envelope gate. These measurements support explicit preserved-logical candidates, not a universal winner or default change.

## Validation

- Matching full `tessera-opt`, NVIDIA target tool and native CUDA runtime build: [build-threads.txt](build-threads.txt).
- Focused compact Schedule/Tile/ABI/resident tests: **114 passed**, [unit-threads-final.txt](unit-threads-final.txt).
- Owning RTX 5070 compact, saved-LSE and broadcast device regressions: **18 passed**, [device-threads-final.txt](device-threads-final.txt).
- Matching gfx1201 shared compiler build: [.validation-compact-threads/build.txt](.validation-compact-threads/build.txt); **121 shared tests passed**, [shared.txt](.validation-compact-threads/shared.txt).
- Owning RX 9070 XT/gfx1201 existing norm/epilogue regressions: **18 passed, 80 deselected**, [device.txt](.validation-compact-threads/device.txt). This is shared-regression proof; HIP compact gradient execution is not claimed.
- Focused registry/runtime/AD/descriptor contracts: **750 passed**, [contracts-final.txt](contracts-final.txt), with one existing NumPy reshape deprecation warning. Audit documents: **11 passed**, [audit.txt](audit.txt); compiler-plan ownership/log links pass, [plan.txt](plan.txt). All **32 generated documents are in sync**, [generated-check.txt](generated-check.txt); Ruff passes, [ruff.txt](ruff.txt). The source/compiler/runtime fingerprints matched the recorded revision at slice completion, [fingerprints.txt](fingerprints.txt). The subsequent [JVP argument-order integration](../nvidia_jvp_argument_order_20261006/README.md) changes current frontend/AD source and compiler; this packet retains its original measurement hashes and is not relabeled as that newer build.
- Graphify update was attempted in the authoritative WSL checkout, but its CLI is unavailable (exit 127); no fresh graph claim.
- No Apple or x86 exact-device compact-gradient claim. No gfx1151 or quantized-format performance transfer.

## Remaining

General composed/dynamic/higher-order attention AD, bias/dropout JVP, sparse/solver consumers, shared native compact consumers on other architectures, kernel-isolated performance attribution, asynchronous ownership and universal automatic policy remain open. FP8, MXFP8 and MXFP4 retain independent numerical/performance gates. Larger W1.1/E2E/AD programs and the full five-slice objective are not closed.
