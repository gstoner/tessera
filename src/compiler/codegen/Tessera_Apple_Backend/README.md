# Tessera Apple Backend

Apple Silicon is Tessera's **most mature native-execution lane** — both the CPU
(Accelerate) and GPU (Metal) paths execute on real hardware today. This backend
defines the hardware-free Target IR contracts and the lowering + runtime that
drive them:

```text
textual DSL / @jit -> Graph IR -> Schedule IR -> Tile IR -> Apple Target IR -> native execution
```

- `tessera_apple.cpu.*` models Accelerate/vecLib/BNNS CPU calls.
- `tessera_apple.gpu.*` models Metal/MPS-style GPU kernels and dispatch.

The hardware-free Target IR object model lives in
`python/tessera/compiler/target_ir.py` and is covered by
`tests/unit/test_target_ir.py` plus target-contract tests. It verifies required
Apple CPU attrs (`framework`, `abi`, `dtype`) and Apple GPU attrs (`kernel`,
`framework`, `status`, dispatch queue, artifact kind) — the layer that keeps the
backend lit-testable independent of a Darwin host.

## Native execution (Darwin)

- **Apple CPU** — `@jit(target="apple_cpu")` lowers matmul-style ops through
  `MatmulToAppleCPU` to an Accelerate shim (`cblas_sgemm` rank-2/rank-3 + BNNS
  f16/bf16) and executes.
- **Apple GPU** — `@jit(target="apple_gpu")` runs a 17-pass Tile→Apple lowering
  into an Objective-C++ runtime (`apple_gpu_runtime.mm`) with **MPS**, **MSL**
  (custom `simdgroup_matrix` kernels), and **MPSGraph** lanes, plus additive
  **Metal 4** paths (bf16/f16 `matmul2d`, fused epilogues, resident MLP
  sessions, pipeline archives, opt-in conv2d) and packaged `.mtlpackage`
  loading. The runtime also carries the fused MoE-SwiGLU expert-FFN kernel and
  the GA/EBM fused kernels.

Both lanes also run through the backend-agnostic **C-ABI launch bridge**
(`tsrLaunchKernel` → `tsrRegisterGpuLauncher`) — the mechanism later reused for
the ROCm and NVIDIA hardware lanes.

> **Honest scope (Decision #27 grounding):** Apple runtime claims require a
> capable Darwin host. Non-Darwin builds fall back to deterministic stubs for CI
> — those are **not** hardware proof; promote a new Apple claim only from a
> fresh-process runtime check or the backend-specific benchmark/test lane. Count
> truth for the C ABI surface and kernel families is in
> [`docs/audit/generated/runtime_abi.md`](../../../../docs/audit/generated/runtime_abi.md)
> (never copy the numbers into prose).

## Canonical references

- [`docs/backends/apple/`](../../../../docs/backends/apple/) — the canonical
  Apple CPU+GPU reference.
- [`docs/audit/backend/apple/archive/apple_gpu_metal4_adoption.md`](../../../../docs/audit/backend/apple/archive/apple_gpu_metal4_adoption.md)
  — the forward-looking Metal 4 adoption ladder.
- [`docs/audit/backend/apple/`](../../../../docs/audit/backend/apple/) — the Apple
  theme audit (what's proven vs. open).
