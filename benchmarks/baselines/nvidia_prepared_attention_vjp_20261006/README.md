# Native prepared saved-LSE reverse ownership

Owner **AD-RESIDUAL-EVAL-1**; siblings E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Sync **NVIDIA-PREPARED-ATTENTION-VJP-2026-10-06**.

Four C ABI exports move repeated static reverse module loading, buffer binding,
private output/LSE storage, compact gradients, stream, events and completion
into the existing C++ attention owner. JVP shares its module/context/retirement
machinery; no second device-math implementation is introduced.

The canonical reverse family compiles a certified traced Graph through native
AD/Schedule/Tile/Target once. Its bounded planning cache retains immutable
serialized products and provenance. A bounded per-thread native registration
imports the pin once; warm calls check input dtype/shape and identity, then
invoke C++. A private saved generation is consumed and completed before the
requested host gradients return. External CUDA streams/resident inputs and
asynchronous generations are not admitted by this host-input API.

## Owning-device evidence

RTX5070 / SM120, driver 610.88, LLVM/MLIR 23.1.1. The runtime rebuild used
CUDA SDK 13.4.59; this is recorded in toolchain.json rather than inferred from
the older validation environment's CUDA_HOME setting.

- 88 public reverse cases pass independent FP64 gradients, changed cotangents,
  argument/request permutations, grouped and causal attention, full/broadcast
  bias and retained prior host outputs.
- 88 complete native program digests match the previous public reverse packet.
  GPU images, Schedule/Tile contracts and numerical policy are unchanged.
- 88 matched common-runtime A/B cases pass before nine alternating trials.
  Both arms use the same pinned artifact and include uploads/downloads.
- Median prepared/unprepared wall ratio **0.0700637** across cases.
  Median case wall medians: prepared **0.443518 ms**, unprepared **6.288189 ms**.
  These are small static envelopes and host-runtime gains, not general GPU
  kernel speedup or scheduling-policy promotion.
- Forward and backward CUDA event windows are retained separately from host
  timing in matched.json. Transfers, image import and host validation are
  outside those event windows.
- Public warm host median **1.43974 ms**, including frontend certification and
  dispatch. The matched runtime comparison above is the performance evidence.
- Three externally pinned compiler-free fresh-process replays pass.
- Native extent rejection preserves the next generation; closed handles,
  wrong CUDA context, inherited handles/PIDs and Python locks are checked.
  The device/JVP regression and host guard batch passes 34 tests, including
  separately compiled 64-thread logical-range Q-only and bias-gradient products,
  and a tight V-gradient proof for K=65,537;
  456 shared host tests pass with one Apple hardware skip requiring a Darwin host; intentionally
  forking with a held lock emits Python's expected fork deprecation warnings.

Public data: public.json and artifacts/. Matched samples: matched.json.
Native/SDK identities: toolchain.json. Fresh replays: replay-*.json.
Device gates: device-tests.txt. Focused shared contracts: contracts.txt.

## Reproduce on owning WSL

~~~sh
source .build-sm120-w1-1/validation-env.sh
cmake --build .build-sm120-w1-1 --target tessera_nvidia_ptx_launch -j4
.venv/bin/python -m benchmarks.nvidia.benchmark_public_attention_vjp --output benchmarks/baselines/nvidia_prepared_attention_vjp_20261006/public.json
.venv/bin/python -m benchmarks.nvidia.benchmark_prepared_attention_vjp --artifacts benchmarks/baselines/nvidia_public_attention_vjp_20261006/artifacts --output benchmarks/baselines/nvidia_prepared_attention_vjp_20261006/matched.json
TESSERA_NVIDIA_DEVICE_PROOF=1 .venv/bin/python -m pytest -q tests/device/nvidia/test_prepared_attention_vjp.py tests/device/nvidia/test_prepared_attention_jvp.py
~~~

Run timing recorders sequentially on the GPU. The public and matched recorders ran sequentially after the final runtime rebuild.

## Remaining scope

Static f32 canonical flash_attn public reverse ownership is proved. General
Graph composition/bufferization, layouts/dynamic shapes, dropout/higher AD,
resident/borrowed tensors and async checkpoint lifetimes remain open.
Apple/ROCm/x86 assess shared registry and ABI changes but receive no SM120
physical proof. FP8/MXFP8/MXFP4 evaluation and full five-slice closure remain
independent. The graph refresh is blocked by the missing WSL Graphify CLI.
