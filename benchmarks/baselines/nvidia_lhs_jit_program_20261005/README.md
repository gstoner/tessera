# NVIDIA ordinary LHS producer program

Owner: W1.1; sibling FRONTEND-IR-MEDIUM-1. Sync: NVIDIA-LHS-JIT-PROGRAM-2026-10-05.

## Verified route

Python @jit → complete typed Graph verification → native producer/consumer Schedule IR → Tile IR with views and typed fragments → NVIDIA Target IR → PTX images → checked resident CUDA ABI → host output.

The wrapper sequences existing native packages. It emits no GPU semantic source template. Caller Graph is copied before partitioning; unsupported semantic metadata is rejected. Bias/residual tracer role markers bind to typed SSA operands, while explicit named bindings retain their meaning. Native output conversion follows the epilogue.

Static fp16/BF16 RMSNorm, LayerNorm and last-axis softmax LHS edges execute. Device tests exercise plain fp32 output, bias/ReLU/residual fp16 output, activation-only GELU/SiLU, padded host views, reordered arguments, unchanged caller Graph, cache reuse and portable replay. Fresh processes replay both storages without a compiler.

## Validation

- RTX 5070 / sm_120: 53 LHS/RHS device tests pass.
- Shared Schedule/runtime/ABI/diagnostic/pass gates: 512 pass, 49 target/compiler-dependent skips.
- Matching gfx1201 compiler: 16 shared role-binding tests pass.
- RX 9070 XT / gfx1201: 15 existing native bias/activation epilogue tests pass. gfx1151 exact-device parity is pending.
- Apple/x86 shared guards preserve fused-route refusal; no physical producer parity is claimed.

## Measurements

NVIDIA GeForce RTX 5070, GPU-cba12639-821a-7a10-4cd3-f918f9c0a545, 610.88, 12.0

Twenty-four matched cases validate independently before timing. Cold wall includes trace and compilation. Warm/replay wall includes staging, private allocation, module dispatch, readback and cleanup. Producer/consumer CUDA event windows include device dispatch and are not isolated kernel-only measurements.

| Storage | Producer | Fused | M/K/N | Warm wall ms | Producer event window ms | Consumer event window ms |
| --- | --- | --- | --- | --- | --- | --- |
| fp16 | rmsnorm | False | 128/1024/64 | 4.8871 | 0.207449 | 0.012611 |
| fp16 | rmsnorm | True | 128/1024/64 | 5.5165 | 0.207592 | 0.013244 |
| fp16 | layernorm | False | 128/1024/64 | 5.3891 | 0.254602 | 0.012658 |
| fp16 | layernorm | True | 128/1024/64 | 5.6515 | 0.254902 | 0.013790 |
| fp16 | softmax | False | 128/1024/64 | 4.9552 | 0.250747 | 0.012250 |
| fp16 | softmax | True | 128/1024/64 | 5.6535 | 0.250270 | 0.013176 |
| bf16 | rmsnorm | False | 128/1024/64 | 5.5659 | 0.207446 | 0.012449 |
| bf16 | rmsnorm | True | 128/1024/64 | 5.5121 | 0.207506 | 0.013838 |
| bf16 | layernorm | False | 128/1024/64 | 4.8498 | 0.254643 | 0.012833 |
| bf16 | layernorm | True | 128/1024/64 | 5.7929 | 0.254796 | 0.012968 |
| bf16 | softmax | False | 128/1024/64 | 4.9826 | 0.250600 | 0.012676 |
| bf16 | softmax | True | 128/1024/64 | 5.5314 | 0.250175 | 0.013263 |

## Remaining work

General producer graphs, dynamic ordinary frontend edges, composed AD, sibling physical routes and the broader five-slice objective remain open. Producer windows dominate this named device edge; further native scheduling needs matched numerical and timing proof. FP8, MXFP8 and MXFP4 remain separate arithmetic, quality and performance gates before strategy/default decisions.

timings.json retains image/contract/compiler/source fingerprints. Device, shared-role and gfx1201 epilogue logs retain execution evidence. Audit and generated-document gate logs are stored beside them.
