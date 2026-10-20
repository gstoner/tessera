# Native original-source tensor package proof

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NATIVE-TENSOR-PROGRAM-PIPELINE-20261019.

This packet proves native admission and complete original-Graph-to-Target
replay plus the explicit frontend compile_native_tensor_program API. Ordinary
JIT dispatch and compiler-free portable deserialization remain open.

## Validation

- source-package-canonical-tests.txt records the initial 80 source/compiler/
  immutable-package cases. The final combined source-package-final-host-gates.txt
  records **618 passed**, including the final **81** source/compiler/package
  cases, caller-owned Graph preservation, shared registries, dtype attributes,
  NVIDIA compiler/fragment paths, pipeline/Target membership/verifier audits,
  document lifecycle and bounded/dynamic/portable legacy contracts. These
  totals overlap and must not be added.
- source-package-device.xml records **8 passed, zero skipped/failed** on
  RTX5070 / SM120: f16/bf16 ReLU, explicit FP32-policy RMSNorm, two-sided
  ReLU/softmax and reversed frontend argument order. Each executes three input
  generations with independent float64 references, half intermediate rounding,
  intact guards/inputs, independent completed results and stable arena capacity.
  External processes are forbidden during native binding and retained calls.
- source-package-legacy-device.txt records **38 passed** with the matching
  compiler and unchanged production native provider. These are adjacent
  native/public/resident/portable DAG regression cases, not generic closure.
- source-package-mypy.txt records no errors in the new transport module;
  focused Ruff and diff-whitespace checks pass.

source-package-device-identity.json records actual GPU UUID/driver, LLVM23.1.1
compiler/layout-library/provider hashes and executed Python source hashes.
host-source-identity.json records native compiler and final host-test sources.
The device/compiler hashes and shared source files match the authoritative
WSL worktree. The production provider is the one already recorded in the
native-frame packet.

programs/ retains exact compressed source and complete Target bytes for all
eight measured programs. Hashes match the timing records. All eight distinct
device images are byte-identical to the earlier pipeline packet's images/;
no shader copies are added here.

## Measurements

Reproducer: benchmarks/nvidia/record_native_tensor_program_package.py, with
--provider, --output and --artifacts arguments. The final timing JSON retains
seven rounds of 32 completed calls and captured device/stage event windows.
Correctness is checked before and after timing. No compiler process may run
during retained calls or profiling.

- Source compilation/package validation: **1996.740146–3191.683347 ms**. This includes frontend construction/tracing, native admission/lowering/images and original-source replay, with process/tool caches described by the recorder.
- Native binding: **3.737130–336.709486 ms**. The first bind includes lazy driver context/module initialization; the packet preserves every case rather than hiding that first cost.
- Completed retained calls: **0.177383–0.317977 ms**. Includes validated roots, H2D, kernels, D2H, completion and result allocation.
- Captured device program: **0.003130–0.005662 ms**. Excludes transfers/readback/setup; stage windows are independent and nonadditive.

These are tiny M17/N11/K32 latency observations, not throughput improvements
or a physical schedule/format promotion. The measured source compile cost is
an open caching/orchestration optimization, separate from runtime execution.

## Remaining integration

Connect ordinary JIT admission and ownership caching without reconstructing
numerical Graph/Tile in Python, preserve existing fused/dynamic routes, and
deliver compiler-free portable restoration with source/member ancestry checks.
The current immutable program constructor requires the matching compiler and
native image replay; there is no unchecked JSON loader. Broader source/layout/
dynamic/AD and the full five-slice program remain open.
