# Current native compiler regression repair — 2026-10-07

Owners: W1.1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Synchronization key: COMPILER-NATIVE-LANES-2026-10-07.

This packet checks the native Graph/AD/Schedule/Tile/Target layers after
the five-slice integration, separately from the earlier frozen Python
non-slow unit sweep. It does not declare the five-slice aggregate complete.

## Reproduced failures and repairs

Initial core.json/core.txt: seven failures, 478 passed, 66 unsupported.
Initial backends.json/backends.txt: one failure, 152 passed.

- Bounded SM120 row-major RHS is now supported; its obsolete negative fixture
  becomes a positive Schedule/typed-view/fragment test.
- MXFP8 LDS stages clamp invalid rows and load valid contiguous vectors;
  the test verifies that current safe loading structure and barrier path.
- The core lit environment dropped ROCM_PATH/HIP_PATH. Preserve the selected
  SDK so native HSACO linking uses the same device libraries as the shell.
  Binary tests are conditional on actual device libraries; Target/LLVM checks
  stay active without them. core-final.json confirms the binary arm executed.
- The scaled carrier fixture incorrectly used fp32 storage for E8M0 exponent
  bytes, a noncanonical block shape, and no exact numeric policy. It now uses
  the existing i8/K32/per-column exact contract. Invalid contract rejection
  remains in the negative suite.
- Negative scale diagnostics follow the declared fp32/E8M0 storage message.
  The formerly unsupported transposed MXFP8 case now tests wrong encoded
  storage; the existing positive LDS fixture covers admitted transposition.
- Rank-two f32 layout materialization tests retain their SM90 tensor scaffold.
  SM120 dynamic row-major storage is separately checked through native views.
- The legacy SM90 emitter attached nvvm.kernel to tensor-valued func.func,
  which violates the registered NVVM verifier. Kernel intent now remains
  tessera.nvidia.kernel until executable LLVM ABI materialization. This is
  high-level artifact correctness, not new SM90 execution proof.
- The SM120 NVFP4 positive Tile fixture now states the required named K16
  physical contract and scale-vector size.

core-after.json preserves the intermediate failure caused by an overly broad
test-fixture replacement changing the f32 result spelling. The final fixture
restores f32 output; no production dtype or numerical policy changes.

## Verified results

- core-final.json: 485 passed, 66 unsupported out of 551 discovered.
  Unsupported feature lanes remain visible; this is not a full fleet union.
- backends-after.json: all 153 NVIDIA/ROCm backend fixtures pass.
- registry-tests.txt: 374 diagnostic/pass/lit inventory tests pass.
- device-tests.txt: 159 owning RTX 5070 tests pass with no skips. Ordinary JIT
  NVFP4 orientation/batching and saved-LSE forward/backward numerical and
  residual-ownership checks use the freshly rebuilt compiler tools.
- device.txt records SM12.0 and the owning GPU UUID. CUDA 13.3 environment,
  LLVM/MLIR 23.1.1. All tests execute in Super-Bear WSL.

No physical gfx1151/gfx1201, Apple or x86 proof transfers from this packet.
ROCm HSACO creation on Super-Bear is native artifact proof only.
Numerical gates do not introduce a speedup claim or replace the separate
event/public-call benchmark packets.

## Reproduction

Source .build-sm120-w1-1/validation-env.sh and activate the existing WSL venv.
Rebuild tessera-opt, tessera-nvidia-opt and tessera-rocm-opt, then run the
LLVM 23.1.1 source lit.py with -j 4 -v -o report.json over tests/tessera-ir
and the two configured backend build test directories. The JSON records
retain the commands actually run. invocations.json records the exact pytest selectors and environment hash;
source-sha256.json binds repaired files and compiler-runtime-source-sha256.json
records the enumerated compiler/runtime source snapshot. Native CUDA runtime library fingerprints are retained in tools-runtime-sha256.json.

## Remaining scope

Generic scaled_matmul batching/linear-transpose AD still fails the global
closure gates; no assertion or partial/planned state is weakened. Dynamic,
nested/composed attention and broader producer/layout/performance envelopes,
matching sibling host updates, full unit and PR delivery remain open.

## Additional owning W1.1 dynamic row-major proof

dynamic-row-device-tests.txt records 84 passed and 11 deliberately deselected.
The selected cases cover all seven nonempty M/N/K dynamic-axis combinations,
RMSNorm/LayerNorm/softmax producers, fp16/BF16 and fused/unfused consumers.
Each compares ordinary JIT, prepared native ownership and portable replay
against the numerical oracle across three runtime shape/pitch frames,
including 1x1x1. Warm tracing/compilation is forbidden and scratch capacity
must stay fixed. This supplies owning execution evidence for the positive
row-major RHS fixture; it does not close arbitrary composed/dynamic AD.

The final compiler rebuild is byte-identical to the tools used for the
numerical and native fixtures. Audit/citation/plan-routing gates pass 22 cases,
the integrated-plan checker passes, all 32 generated documents are in sync
and Graphify refresh completes.
