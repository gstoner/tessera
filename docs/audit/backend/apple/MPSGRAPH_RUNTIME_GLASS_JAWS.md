# Apple MPSGraph Runtime Glass Jaws

Stages 13-15 audit note for Apple GPU value execution. This is a current-truth guardrail, not a control-flow segfault investigation.

## Current Proof Lane

- `tessera_apple_gpu_ppo_policy_loss_f32` is the first RL Apple GPU value executor.
- `tessera_apple_gpu_ppo_policy_loss_ex_f32` extends that executor to optional PPO `mask`, reference-KL, and entropy side tensors.
- Both are MPSGraph-backed: host inputs are copied into Metal buffers, the graph computes the PPO formula on the Metal queue, and execution runs through `runWithMTLCommandQueue`.
- The C ABI returns `1` only when the MPSGraph path runs. The non-Darwin stub returns `0`, so symbol presence alone is not an executable GPU claim.
- Python dispatch checks exact arity from value-call metadata, fp32 coercion at the ABI boundary, matching shapes, `reduction="mean"`, positive `clip_epsilon`, and side-term flags/coefficients.
- A fresh dylib in a fresh Python process is required after adding these symbols. `_apple_gpu_ppo_policy_loss_available()` / `_apple_gpu_ppo_policy_loss_ex_available()` must pass their tiny numerical probes before benchmark/runtime metadata may claim `apple_gpu_value_target_ir`.
- Native Metal/MPSGraph proof tests must run with Metal access in the current process. Codex sandboxed exec can hide `MTLCreateSystemDefaultDevice()` even on Metal 4-capable Apple Silicon, producing false negatives such as `DeviceTensor.is_metal() == False`, `ts_dev_is_metal == 0`, or skipped Metal stress tests. Run decisive Apple GPU validation outside the sandbox / escalated, with `TESSERA_APPLE_GPU_RUNTIME_LIB` or a freshly built/temp `libTesseraAppleRuntime.dylib` loaded in a fresh Python process.
- GRPO/CISPO are compiler-decomposed references only; they have no Apple GPU value executor.

## Glass Jaws To Keep Guarded

- **Executor labeling:** a CPU/reference fallback must never be labeled `apple_gpu_value_target_ir`. New GPU rows need a real Metal/MPSGraph path plus numerical proof.
- **Return-code discipline:** GPU value symbols that can fail should return a status bit. Python must turn non-success into structured non-success, not a fabricated output.
- **Cache lifetime:** cached `MPSGraph`, placeholders, and output tensors must stay alive beyond each autoreleasepool. Cache keys must include shape, dtype, and semantic attrs such as `clip_epsilon`.
- **Autoreleasepool boundaries:** per-call `MPSGraphTensorData` and temporary Metal buffers should live inside an autoreleasepool; cached graph objects should live in the shared cache.
- **Command-queue semantics:** host-pointer value executors may use `runWithMTLCommandQueue`; device-resident/decode-chain work should use `encodeToCommandBuffer` so it composes on one GPU timeline.
- **Tensor-data retention:** result reads must occur before transient `MPSGraphTensorData` leaves scope. Device-resident lanes should write through `resultsDictionary`.
- **Dtype honesty:** Stage 13-14 PPO is fp32 only. f16/bf16 PPO must stay gated until a real dtype path and tolerance contract land.
- **Zero-fill/reference fallback:** zero-fill and host-reference stubs are acceptable for non-executable test scaffolding only. They must not advertise `status="executable"`.

## Stage 13-15 Guards

- Static tests assert the strict and extended PPO symbol bodies contain MPSGraph APIs and no host loop/reference implementation.
- The stub test asserts `tessera_apple_gpu_ppo_policy_loss_f32` and `..._ex_f32` return `0`.
- Runtime tests either compare the MPSGraph PPO result against `tessera.rl.ppo_policy_loss` on an active Darwin GPU runtime or report structured non-success off-platform.
- RL benchmark rows split `python_reference`, `compiler_decomposed_reference`, and `apple_gpu_value_target_ir` so benchmark telemetry cannot conflate compiler visibility with hardware execution.
- GRPO/CISPO benchmark rows may report `compiler_decomposed_reference`; they must not report Apple GPU executor metadata.
