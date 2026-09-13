# GFX1201 automatic public attention AD

Owner: E2E-REAL-6 / ROCM-2. Sync: `GFX1201-PUBLIC-AD-2026-09-13`.

Tajasarus RX 9070 XT gfx1201; ROCm 10.0; LLVM/MLIR 23.1.1 with assertions;
Ubuntu 26.04 WSL2. This is correctness evidence, not selector-grade timing.

The scheduled backward package now retains gfx1201 through the direct Tile
adapter, native compilation, image, descriptors and exact-device admission.
The public `jit(target="rocm", autodiff="reverse").native_backward` path selects
the live owning architecture, lowers the traced reverse carrier once, and
executes that package. Certificates derive architecture from the artifact and
compare it with runtime physical attestation; gfx1151 is not presumed.

Tests cover f16/bf16 public GQA Q/K/V gradients and four explicit backward
programs with causal/windowed attention, f32 bias, softcap and deterministic
dropout, against float64 host oracles. Outputs are fp32. Explicit saved-LSE
requests and unknown architectures fail closed. Existing matrix/forward package
checks run alongside the new AD tests.

The gfx1201 policy is recompute, including shapes above gfx1151's threshold.
This synchronous five-stage program owns workspace for one call. It is not a
reusable tape and cannot claim asynchronous ownership or persistent saved LSE.
Sparse SWMMAC remains ISA/catalog-only: compressed A values, selection-index
packing, a verified native producer and device numeric tests are still required.
General matrix dtype/shape/epilogue envelopes remain open. The earlier profiler
packet has zero kernel dispatch/code-object rows; no kernel attribution or
performance promotion is claimed here. No sibling device proof transfers.
