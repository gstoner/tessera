# Backend Audit

This document consolidates shared backend/runtime audit work. Platform-specific
Apple, NVIDIA, and ROCm details live in sibling platform folders.

## Finished

- **Runtime execution matrix:** `../generated/runtime_execution_matrix.md`
  owns what `runtime.launch()` can execute.
- **Runtime ABI dashboard:** `../generated/runtime_abi.md` owns C ABI symbol
  and header counts.
- **Executable rows:** Apple CPU, Apple GPU, native CPU, and JIT CPU numpy are
  explicit executable rows.
- **Non-executable targets are honest:** NVIDIA, ROCm, and Metalium are
  recognized but return unsupported/unimplemented behavior rather than fake
  success.
- **Toolchain pins:** CUDA, NCCL, and ROCm pins agree in generated dashboards.
- **Hardware frontier framing:** archived Phase G/H/I material establishes that
  backend-kernel completion requires real device proof.

## Still Open

- **NVIDIA runtime execution:** no execution-matrix rows yet.
- **ROCm runtime execution:** no execution-matrix rows yet.
- **Metalium runtime execution:** still shared-backend frontier material, not a
  platform-specific execution lane.
- **Universal backend-kernel axis:** generated S-series status still reports all
  432 primitive entries open on backend-kernel proof.
- **Numerical proof discipline:** backend rows need explicit compare fixtures or
  hardware/package validation before promotion.

## Dtype Policy Snapshot

The current backend dtype contract is capability-aware, not a blanket lowering
claim. Canonical Tessera dtype names that must remain synchronized with the
language/IR spec include:

- Floating point: `fp64`, `f64`, `fp32`, `f32`, `tf32`, `fp16`, `f16`, `bf16`,
  `fp8_e4m3`, `fp8_e5m2`, `fp6_e2m3`, `fp6_e3m2`, `fp4_e2m1`, `nvfp4`.
- Integer/storage: `int64`, `i64`, `int32`, `i32`, `int16`, `i16`, `int8`,
  `i8`, `uint8`, `u8`, `uint16`, `uint32`, `uint64`, `int4`, `i4`, `bool`.
- Planned policy families: `mxfp8`, `mxfp6`, `mxfp4`, `bfp8_b`, `bfp4_b`,
  `blockfp*`.

Policy rules:

- `tf32` is a math mode on `fp32` tensors, not a storage dtype.
- Direct `int4` requires an explicit packed-storage policy before target
  support can be advertised as native execution.
- AMD MX formats, NVIDIA `nvfp4`, and Tenstorrent BFP/BLOCKFP formats must stay
  separate policy families.
- Apple CPU/GPU claims should remain conservative: `fp32`, `fp16`, and integer
  storage are primary; BF16 depends on OS/Metal exposure; FP8/FP4 are not native
  Apple M1 Max acceleration claims.

## Next Work

1. Keep all non-Apple hardware claims at artifact/toolchain status until real
   execute-and-compare hardware proof exists.
2. Add runtime execution rows only after target launch paths are implemented.
3. Promote manifest rows only with runtime ABI, hardware smoke, and numerical
   proof.
4. Split Metalium into its own folder only when there is enough target-specific
   audit material to justify it.

## Source Material Consolidated

- `archive/phase_ghi_hardware_frontier.md`
- `archive/hardware_dtype_support_matrix.md`
- `archive/nvidia_rocm_execute_and_compare_plan.md`
