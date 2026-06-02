# Apple Backend Audit

Apple backend audit tracking lives here. This theme covers Apple CPU/GPU,
Metal 4, MPS/MPSGraph/custom MSL, packaged kernels, command-buffer discipline,
and Apple performance proof.

## Current Truth

- Start with [APPLE_AUDIT.md](APPLE_AUDIT.md) for the consolidated Apple audit:
  what shipped, what remains open, and which archived docs were absorbed.
- Apple is a platform-specialized backend, not just a shared-backend footnote.
- The Apple path spans compiler lowering, Python runtime dispatch, C ABI
  symbols, Metal 4 lanes, encode-session batching, and packaged `.mtlpackage`
  validation.
- Generated Apple status should be read from:
  - [../../generated/apple_target_map.md](../../generated/apple_target_map.md)
  - [../../generated/runtime_abi.md](../../generated/runtime_abi.md)
  - [../../generated/runtime_execution_matrix.md](../../generated/runtime_execution_matrix.md)

## Open Items

- Make Apple tensor/kernel binding specs common to packaged and non-packaged
  kernels.
- Wire Apple feature limits into Schedule/Tile/Target decisions.
- Finish canonical `tessera.ops` / `@jit` integration for one-command-buffer
  decode chains.
- Populate production packaged kernels; fixture-backed package validation is
  not the same as production packaged backend coverage.

## Archived Source Material

- [2026_06_01_apple_gpu_chain_audit.md](archive/2026_06_01_apple_gpu_chain_audit.md)
- [apple_ga_ebm_native_execution_gap.md](archive/apple_ga_ebm_native_execution_gap.md)
- [single_command_buffer_decode_plan.md](archive/single_command_buffer_decode_plan.md)
