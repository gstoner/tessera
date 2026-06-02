# Apple Backend Audit

This document consolidates Apple backend audit material across Metal runtime,
Metal 4, packaged kernels, command-buffer work, and Apple-specific performance.

## Finished

- **Apple CPU runtime:** Apple CPU artifacts execute through Accelerate.
- **Apple GPU runtime:** Apple GPU artifacts execute through MPS, MPSGraph, and
  custom MSL runtime paths.
- **Metal 4 surface:** Metal 4 probes and lanes exist for selected kernels.
- **Runtime ABI breadth:** generated ABI inventory reports a large Apple symbol
  surface and 84 Apple GPU families.
- **One-command-buffer substrate:** encode-session APIs and chain planning exist
  for resident decode-style chains.
- **Auto-batch substrate:** Apple GPU chain planner and `apple_gpu_ops` tracing
  pieces exist.
- **Conv2d encode-session lanes:** f32/f16/bf16 wrapper lanes exist.
- **Packaged-kernel lifecycle:** PK1-PK7 are proven against a real Apple sample
  package, including reflection, validation, argument layout, and dispatch.
- **GA/EBM Apple specialization:** fused Apple kernels and benchmarks exist for
  important GA/EBM paths.

## Still Open

- **Binding specs are not universal.** Packaged kernels have binding validation;
  ordinary Apple kernels still rely too much on convention and symbol shape.
- **Feature limits are underused.** Apple feature/limit data exists but does not
  dominate Schedule/Tile/Target choices yet.
- **One-CB path — canonical route landed (2026-06-02); residual polish.**
  `@jit(target="apple_gpu", auto_batch=True)` now runs a decode body written
  with the canonical `tessera.ops.*` surface on one command buffer per encode
  segment (the global interception shim forwards to the encode session under
  an active trace), and `max_ops_per_cb` threads the chunking budget through
  the decorator. Locked by `tests/unit/test_apple_gpu_jit_auto_batch_canonical.py`.
  Residual: it is **opt-in** (`auto_batch=True`), not auto-detected; and an
  auto_batch body still pays Graph-IR-emission overhead it never uses, and
  must keep shape kwargs as literals/args (a general `@jit` AST-lowering
  constraint, not auto_batch-specific).
- **Production packaged kernels are empty.** Fixture proof is not the same as
  production packaged backend coverage.
- **Target IR does too much.** Apple source strings, fusion recognition, and
  runtime dispatch decisions need a descriptor-backed backend registry.
- **Performance gates are uneven.** Benchmarks exist, but manifest-attached
  benchmark metadata and ratchets are not systematic.

## Next Work

1. Promote `AppleTensorBindingSpec` / `AppleKernelBindingSpec` to all Apple
   kernel families.
2. Add Apple kernel descriptors for MPSGraph, MSL, Metal 4, packaged, and
   encode-session paths.
3. Wire Apple feature limits into schedule/tile/kernel selection.
4. ~~Finish `@jit(target="apple_gpu", auto_batch=True)` canonical
   one-command-buffer route.~~ **Landed 2026-06-02** — canonical
   `tessera.ops.*` decode through `@jit` runs on one cb, with
   `max_ops_per_cb` chunking threaded through the decorator. Follow-on
   (optional): make auto_batch bypass unused Graph-IR emission; consider
   auto-detection so the route is on by default for recognized decode loops.
5. Add production packaged kernels with reflection validation and numerical
   compare fixtures.
6. Attach benchmark metadata for Apple hot paths such as matmul, matmul
   epilogues, conv2d, decode chain, and packaged kernels.

## Source Material Consolidated

- `archive/2026_06_01_apple_gpu_chain_audit.md`
- `archive/single_command_buffer_decode_plan.md`
- `archive/apple_ga_ebm_native_execution_gap.md`
- `../../compiler/archive/compiler_apple_backend_end_to_end_audit_2026_06_02.md`

