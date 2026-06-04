# Apple MPSGraph Runtime Glass Jaws

Stage 13 audit note for Apple GPU value execution. This is a current-truth guardrail, not a control-flow segfault investigation.

## Current Proof Lane

- `tessera_apple_gpu_ppo_policy_loss_f32` is the first RL Apple GPU value executor.
- It is MPSGraph-backed: host inputs are copied into Metal buffers, the graph computes `mean(-min(exp(logp_new-logp_old)*adv, clip(exp(logp_new-logp_old))*adv))`, and execution runs through `runWithMTLCommandQueue`.
- The C ABI returns `1` only when the MPSGraph path runs. The non-Darwin stub returns `0`, so symbol presence alone is not an executable GPU claim.
- Python dispatch checks exact arity, fp32 coercion at the ABI boundary, matching shapes, `reduction="mean"`, positive `clip_epsilon`, and `kl_coef=0.0`.

## Glass Jaws To Keep Guarded

- **Executor labeling:** a CPU/reference fallback must never be labeled `apple_gpu_value_target_ir`. New GPU rows need a real Metal/MPSGraph path plus numerical proof.
- **Return-code discipline:** GPU value symbols that can fail should return a status bit. Python must turn non-success into structured non-success, not a fabricated output.
- **Cache lifetime:** cached `MPSGraph`, placeholders, and output tensors must stay alive beyond each autoreleasepool. Cache keys must include shape, dtype, and semantic attrs such as `clip_epsilon`.
- **Autoreleasepool boundaries:** per-call `MPSGraphTensorData` and temporary Metal buffers should live inside an autoreleasepool; cached graph objects should live in the shared cache.
- **Command-queue semantics:** host-pointer value executors may use `runWithMTLCommandQueue`; device-resident/decode-chain work should use `encodeToCommandBuffer` so it composes on one GPU timeline.
- **Tensor-data retention:** result reads must occur before transient `MPSGraphTensorData` leaves scope. Device-resident lanes should write through `resultsDictionary`.
- **Dtype honesty:** Stage 13 PPO is fp32 only. f16/bf16 PPO must stay gated until a real dtype path and tolerance contract land.
- **Zero-fill/reference fallback:** zero-fill and host-reference stubs are acceptable for non-executable test scaffolding only. They must not advertise `status="executable"`.

## Stage 13 Guards

- Static tests assert the PPO symbol body contains MPSGraph APIs and no host loop/reference implementation.
- The stub test asserts `tessera_apple_gpu_ppo_policy_loss_f32` returns `0`.
- Runtime tests either compare the MPSGraph PPO result against `tessera.rl.ppo_policy_loss` on an active Darwin GPU runtime or report structured non-success off-platform.
- RL benchmark rows split `python_reference`, `compiler_decomposed_reference`, and `apple_gpu_value_target_ir` so benchmark telemetry cannot conflate compiler visibility with hardware execution.

