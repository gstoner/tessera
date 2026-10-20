# PR994 standalone activation integration review

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NATIVE-TENSOR-MLP-20261020.
Review: https://github.com/gstoner/tessera/pull/994#discussion_r4324175607

The review correctly found that standalone FP32 ReLU lowered while the public
registry described only its private MLP use. This follow-up completes the
standalone route instead of narrowing native arithmetic to a wrapper marker.

## Native route and ABI

The ordinary frontend retains its typed ReLU Graph and native
Graph -> Schedule -> Tile -> NVIDIA Target -> PTX. The existing scheduled
consumer packages FP16/BF16/FP32 storage. FP32 has the explicit
tessera.nvidia.relu.x_o_rows_columns.f32.v1 ABI and native CUDA entry/width
dispatch. The two existing half ABIs keep their semantics.

Typed ABI validation checks buffer width, rank, layout, alignment, direction
and ordinal, plus Rows/Columns int64 scalar order. Forging narrower buffers
or deleting invalid rank guards cannot bypass this boundary. An unused ReLU
cannot replace a returned input. Canonical native dispatch produces the same
image and descriptor as the scheduled consumer.

The final design introduces no additional Python Graph-input packager:
the existing canonical dispatcher consumes the existing typed scheduled
consumer. The original no-growth scoreboard gate is preserved. The initial
extra-wrapper failure is retained in the raw full-unit log; 69 focused
pruning/dispatch/source tests pass after removing it.

## Exact-device and compiler proof

On RTX5070 SM120 with immutable provider
03bd30ed513d3362d915eb5cca7f61cb7e850b2cd120b2e89f362a210a88ffde,
50 combined standalone/composed cases pass. A separate final standalone
rerun passes 9 cases, including a fresh process with compiler/subprocess
access disabled. These overlapping runs are not added.
FP16/BF16/FP32 public host calls preserve NaN payloads, zero signs,
infinities/subnormals, changed inputs, snapshots and output canaries.
FP32 resident serial/flat packages have separate bitwise proof.

The compiler route passes 2640 cases with zero skips; final additional
source/dispatch controls have a separate focused lane. Applicable
registry/activation gates pass 511 (8 declared non-SM120 skips), and the
standalone contract lane passes 21 after its final dispatch control.
The repeated full CPU lane passes 30,788 cases with 3,710 declared skips.
All 33 generated documents are in sync, all five surface audits pass, and
Ruff and the zero-error mypy ratchet pass. Graphify refresh completes.
Textual frontend execution separately reaches the checked FP32 native ABI
on RTX5070. Updated-commit CI and review remain required; old-head CI success
does not authorize merging this follow-up.

## Census and sibling assessment

The primitive census now recognizes existing generic native Schedule/Tile
consumers per target instead of publishing them as bootstrap-only. Activation
has explicit NVIDIA-only membership and a typed-consumer declaration.
Removing a target's generic consumer cannot inherit a compiled sibling route.
The fixed census lane passes 19 cases. This is structural route coverage,
not new device evidence or datatype/AD closure on another architecture.

All four backend queues are assessed. No new Apple, x86 or ROCm physical
activation support follows from the CUDA proof. The prior 67 gfx1201 scalar
regressions remain a distinct earlier receipt.

## Reproduction and timing domains

Run from the owning WSL checkout with the matching compiler:

    TESSERA_RELU_RUNTIME_OUTPUT=/tmp/tessera-relu-new.so bash benchmarks/baselines/native_tensor_mlp_20261020/review/build-native-provider.sh
    export TESSERA_NVIDIA_PTX_LAUNCH_LIB=/tmp/tessera-relu-new.so
    export TESSERA_NATIVE_FRAME_TEST_LIB=$TESSERA_NVIDIA_PTX_LAUNCH_LIB
    python -m pytest tests/device/nvidia/test_public_standalone_relu.py -q
    python -m benchmarks.nvidia.record_native_tensor_mlp --standalone-relu --output /tmp/relu.json

CUDA_HOME must identify the installed CUDA SDK. The recipe refuses an existing
output path and leaves earlier validated providers intact. The provider
source/build/device identity is in public-relu-provider.json.

The final three-shape timing packet is public-relu-timing-reviewed.json.
Public serial completion, cold calls and resident serial/flat CUDA-event
windows are separate. Native event windows can contain enqueue gaps.
Mixed tiny outcomes and the wide-row flat improvement do not promote a
default schedule. Intermediate timing attempts remain in scratch and are
not pooled into this packet.

Raw failures and passes are preserved losslessly in gzip; lossless-logs.json
records compressed and original SHA256 identities. The full aggregate
five-slice objective, general producer/layout/dynamic/AD and model-quality
obligations remain open. Generic multi-member source-package restoration
still requires the matching compiler; standalone native-image packages have
their separate compiler-free replay proof.


## Unit CI timeout and census evaluation repair

The fd6f75dba unit job reached 95% but was cancelled at the workflow's
20-minute limit. Lint, audit and the production compiler-route jobs passed.
The raw cancellation log is preserved without changing the fan-in gate.

A read-only route-map probe measured 416 AST parses and 11.269 seconds before
scoped reuse, versus 6 parses and 4.504 seconds for the final scoped source on Tajasaurus.
An intermediate run measured 2.787 seconds; both diagnostic samples are
preserved and neither is a controlled throughput comparison.
The complete derived route output is identical (SHA256
6bac9fbd00f28c8a3a8d5932327087de7122a067e3817ca2d76d679acf893480).
These are diagnostic host measurements under concurrent validation, not
hardware benchmark ratios or guaranteed CI duration.

A context-local snapshot shares parsed sources and derived inventories only
within one evaluation. Each next evaluation rereads live sources. Controls
cover same-size/same-timestamp edits, inventory edits and cleanup after
exceptions; existing target-removal and body-forgery controls remain enforced.
No permanent source cache or weaker route proof is introduced.
The corrected full unit job and exact-head CI still must pass.

Final focused census/adversarial tests pass 59 cases; all 33 generated
documents and 23 ownership/audit/scoreboard cases pass, and Ruff plus the
zero-error mypy ratchet pass. One initial new-test NameError was corrected;
its raw failure is preserved. Full host CPU and updated-head CI remain active.


## Specialized activation selection review

Review https://github.com/gstoner/tessera/pull/994#discussion_r4324678453
correctly found that the operation-name request selected unsupported ReLU
contracts before checking the specialized trace. The JIT now validates that
actual module with supports_activation before canonical native compilation.
Unselected ranks, FP64 and empty extents leave the bounded descriptor path;
errors after native admission still propagate.

The new host-free controls reproduce six failures before the fix, with three
supported-storage controls already passing. Afterward the focused lane passes
86 cases with eight declared non-SM120 skips. The nine standalone RTX5070
host/resident/replay cases pass on unchanged a2cc compiler and 03bd runtime.
The source/binary/GPU identity is in pr994-dispatch-device-identity.json.
Mypy remains zero; all 33 generated documents and the plan gate pass.
The complete post-fix host CPU lane and updated-head CI remain active.
This is bounded dispatch selection proof, not native FP64/rank-one support.
