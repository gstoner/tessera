# Bounded public saved-LSE JVP on RTX 5070

Owning items: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Synchronization key: SM120-BOUNDED-SAVED-LSE-20261011.

## Contract and proof

The public explicit JIT program accepts sequence_bounds and supplies a native
MLIR request without projecting Python Graph tensor types. Forward and reverse
Graph AD share the same verified native projection. Paired saved-LSE uses Graph
AD schema 5 with bounded sequences, schema 6 with static sequences; schemas 1–4
retain their output-only meanings. Schedule/Tile retain dlse before scratch and
the actual query/key-size scalar arguments.

The generic native host/resident attention forward and backward launchers now
carry all declared physical bias scalars. Dynamic broadcast kernels consume
11 dimension arguments; legacy seven-scalar/full-shape kernels retain their
ABI. Before the repair, the resident broadcast timing path read past its
13-slot argument vector and crashed inside the CUDA driver.

Fresh matching LLVM/MLIR 23.1.1 tools and native CUDA providers are used.
The owning device is NVIDIA GeForce RTX 5070, sm_120, UUID
GPU-cba12639-821a-7a10-4cd3-f918f9c0a545, driver 610.88, CUDA 13.3.

Exact-device tests prove serialized compiler-free replay, original Graph
preservation, inactive value/bias roles, exact-zero value-only dLSE, repeated
direction products, separate old generations and capacity refusal. The named
capacities are (9,11) and (17,129). Direct generic host/resident forward/backward
launches additionally prove key/query/full-shape physical bias indexing.

Tests: tests/device/nvidia/test_bounded_saved_lse_jvp.py and the adjacent
public-native JVP/bounded forward/backward suites. Focused compiler/schema,
portable artifact and diagnostic/pass registry coverage is recorded separately.
74 exact-device tests and 428 focused compiler/registry tests pass.

## Two fresh packets

Each packet contains 30 correctness-gated profiles: five activity/bias variants
across six actual sequence shapes within (17,129). All four results O, LSE,
dO and dLSE are checked against the independent FP64 finite-difference oracle
before and after timing. Each activity/bias variant reuses one sealed package
across shapes. Both packets have identical program/image/binding identities,
all 22 source pins match current source, and all four binary pins match.

| Measurement | run1 | run2 |
| --- | --- | --- |
| Maximum absolute error | 7.1888e-7 | 7.1888e-7 |
| Forward CUDA-event median range | 7.606–55.883 us | 7.774–55.617 us |
| JVP CUDA-event median range | 8.964–10.885 us | 9.042–10.401 us |
| Checked JVP completion median range | 0.295–0.401 ms | 0.302–0.386 ms |
| Resident capture+JVP completion median range | 5.029–6.159 ms | 4.955–6.063 ms |

Forward timing uses the preloaded native descriptor CUDA-event window.
JVP timing uses a preloaded raw CUDA-event dispatch window, with the actual
128-thread launch block and its checked scratch size. Neither window is
presented as isolated kernel-body time. Completed-call timing includes
allocation, snapshot and binding work; uploads, downloads and close are outside
that window. Cold compile time is recorded separately in every profile.
The raw timer borrows the caller's current CUDA context.

These packets characterize this named f32 contract and its remaining host
overhead. They do not establish a speedup, default-route promotion, half-storage
or quantized AD, generic compositions, higher-order transforms or sibling
backend execution. The broader five-slice goal remains active.

The git_base field records the main parent 009a3b226; the changed source is
identified by explicit source_sha256 fields, not by that parent alone.

Recorder:
    python -m benchmarks.nvidia.record_bounded_saved_lse_jvp --samples 3 --repetitions 32 --output benchmarks/baselines/bounded_saved_lse_jvp_20261011/run1.json

Source scripts/_nvidia_env.sh with CUDA_HOME=/usr/local/cuda-13.3 and set
TESSERA_OPT, TESSERA_NVIDIA_OPT, LLVM_BIN, TESSERA_NVIDIA_PTX_LAUNCH_LIB and
TESSERA_NVIDIA_GEMM_LIB to freshly built matching tools/providers.


## Candidate revalidation after the 2026-10-14 WSL reboot

Tajasaurus / RX 9070 XT / gfx1201 rebuilt the candidate with LLVM/MLIR 23.1.1
and passed 77 ingest, package, numerical and shared compiler-admission cases.
See gfx1201-candidate-tests-20261014.txt. This validates the ROCm package
admission change; it does not transfer the NVIDIA attention timings to AMD.
The original 2026-10-11 NVIDIA packets retain their recorded source/binary pins.

Super-Bear / RTX 5070 / sm_120 passed 75 exact-device cases after the reboot,
using the matching LLVM/MLIR 23.1.1 compiler and byte-identical native providers.
See nvidia-candidate-tests-20261014.txt. Both original packets were rechecked:
all 22 source hashes, four binary hashes and corresponding sealed identities
still match. Temporary providers were moved outside the code checkout.


## Scratch reconciliation

The old worktrees were reviewed against merged main before pruning.
Clean merged copies were removed with source commits retained in Git.
Unique scaled-multiply and historical NVFP4 prototype overlays were preserved
as recovery commits before their active checkouts were removed. These
experiments are not part of this validated bounded-attention/admission change;
the scaled-multiply carrier still requires a port onto the current repaired
compiler and fresh owning-device validation before landing.
No recovery commit is treated as a successful validation receipt.

The final broad non-slow unit run on the fully provisioned Super-Bear WSL
host completed: 26,041 passed, 7,568 skipped, 11 warnings; no failures.
See full-unit-tests-20261014.txt. Skipped rows are not physical-backend proof.
The lean Tajasaurus broad sweep is not used as a passing gate: it lacks some
unit dependencies and exposes generic-target/partial-compiler assumptions.
The explicit 77-case candidate gfx1201 lane above is its physical authority.
