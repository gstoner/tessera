# Public JIT attention bias VJP on SM120

Owner E2E-REAL-6 / AD-HIGHER-1; sync NVIDIA-ATTENTION-BIAS-GRADIENT-2026-10-02.

Exact device: NVIDIA GeForce RTX 5070, GPU-cba12639-821a-7a10-4cd3-f918f9c0a545, 610.88, 12.0.
Public JIT tracing preserves Q/K/V and the exact-shaped f32 additive bias
as real typed Graph operands. The native paired AD pass emits saved O/LSE
and all four cotangents; checkpoint export retains requested selection/order
in paired lineage and carries the complete physical product through Schedule,
Tile and NVIDIA Target/NVVM/PTX. The resident tape validates the checked
eleven-buffer backward ABI before CUDA use and owns private Q/K/V/bias/O/LSE.

18 recorded rows cover full/causal, Sq<Sk and Sq>Sk, grouped query, batch two,
independent D/Dv, and wrt=(bias), (bias,k), (v,bias,q,k).
After capture, caller Q/K/V/bias are overwritten. Repeated backward and a
second cotangent still match the independent float64 oracle; closed frames
reject backward. Maximum absolute gradient error: 1.62360434e-07.
A separate exact-device unit also proves Q-only activity on the biased graph.

441 focused tests passed, including traced bias lineage, requested result
order, saved O/LSE identity, gradient launch coverage, and rollback after an
injected launch failure. Both compiler tools were rebuilt on Super-Bear WSL;
compiler-tools.json records their actual hashes.

Three wall samples per row: capture medians 2.071390–2.618315 ms,
backward medians 0.177032–0.338623 ms.
Capture includes private copies, module loading, forward and synchronization.
Backward includes allocation and synchronization. Selected wrt still computes
all four physical gradients. Separate resident CUDA-event windows are in
the sibling nvidia_checkpoint_bias_gradient_20261002 packet. No speedup,
default-dispatch promotion, or sibling-device parity claim follows.

Remaining: broadcast bias reduction, bias JVP/higher derivatives, wider
dtype/layout and general composed AD, plus Apple/ROCm/x86 native checkpoint
consumers. This proves the explicit compile_native_attention_vjp API for one
dense static f32 Graph family; it does not claim arbitrary JIT execution.

Reproduce from the owning repository checkout:
    source .build-sm120-w1-1/validation-env.sh
    .venv/bin/python benchmarks/nvidia/benchmark_jit_attention_bias_vjp.py --output packet.json
