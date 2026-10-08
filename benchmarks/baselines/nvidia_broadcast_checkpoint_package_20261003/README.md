# SM120 broadcast checkpoint checked package and public reverse execution

Owner: E2E-REAL-6 / AD-HIGHER-1. Sync: NVIDIA-BROADCAST-CHECKPOINT-PACKAGE-2026-10-03.

The native arithmetic core now connects to checked host/resident packages and public JIT reverse AD. Separate broadcast ABI identifiers carry BiasB/BiasH/BiasQ/BiasK in the descriptor. The native kernel retains seven logical dimensions; its verified Schedule/Tile attributes bake physical extents. The host bridge validates broadcast extents before allocation/copy and uses physical sizes for bias/dBias and backward geometry. Physical guards and saved-state identity bind the compiled shape. The tape validates the descriptor before CUDA, privately copies physical bias and Q/K/V/O/LSE, and allocates physical gradients.

## Numerical and ownership protocol

The owning RTX 5070 recorder rejects other architecture/device configurations and records GPU UUID, driver, compiler and bridge hashes plus source hashes. Fourteen paired checked/public cases cover every axis, combined axes, GQA, causal/full masks and both sequence orderings. Independent float64 attention supplies O, natural-log LSE and Q/K/V gradients; dense oracle dBias is reduced over broadcast axes independently of native physical-owner reduction. Two additional explicit Q/K/V-only consumers exercise the broadcast ABI without a dBias result. Public Q/V-only selection is also checked: paired native AD currently computes all operand cotangents, then returns only requested roles.

Public capture is tested after caller buffer mutation, across repeated and changed cotangents, and after close. Descriptor mutations must fail before loading CUDA. Existing full-shape/native attention and matmul routes remain regression gates. Correctness is checked before and after timing.

## Timing and limits

The packet separates checked host-copy wall time, resident backward CUDA-event dispatch windows, and public capture/backward wall time. Device windows include driver gaps. No speedup, fully optimized reduction, universal AD completion or low-precision format promotion is claimed. FP8, MXFP8 and MXFP4 remain separate required evaluation points before schedule/format decisions.

Broadcast higher-order/JVP, lower-rank/dynamic bias, pruning unrequested cotangents and sibling native consumers remain open. This increment completes the physical rank-four broadcast package/tape connection; the full five-slice goal remains active.

Evidence: rtx5070.json, run.txt, runtime-build.txt, host-contracts.txt, device-regressions.txt and the final drift/document checks.

## Recorded result

Fourteen paired cases, two explicit Q/K/V-only consumers and two public Q/V selections passed. Maximum checked gradient error was 1.3e-07; maximum public gradient error was 2.09e-07. The regression lane passed 101 exact-device tests; 537 focused host/registry/audit tests passed. Physical-scalar mismatch checks reject otherwise broadcast-valid scalars that differ from the compiled storage before any native copy. All 32 generated documents were regenerated; compiler-plan, Ruff and diff gates passed. Graphify query/update could not run because the CLI is unavailable in WSL.
