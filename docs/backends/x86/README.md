---
classification: Backend architecture reference
authority: x86 / AVX-512 / AMX reader entry point
last_updated: 2026-07-13
---

# x86 Backend

This page is the reader-facing entry point for Tessera's x86 CPU target,
including the AVX-512/AMX lowering and the MLIR-to-LLVM JIT execution lanes.

## Current evidence

- [Compiler reference](../../spec/COMPILER_REFERENCE.md) defines the x86
  lowering and JIT surface.
- [Runtime execution matrix](../../audit/generated/runtime_execution_matrix.md)
  is the current placement/proof dashboard.
- [Runtime ABI dashboard](../../audit/generated/runtime_abi.md) records the
  exported ABI surface.

## Design and implementation

The x86 route includes the generic CPU JIT plus target-specific AVX-512 and AMX
lowering where the target contract permits it. Read the compiler reference for
the supported compiler form; do not infer native execution from an artifact or
reference fallback alone.

## Decisions and open work

[Shared backend audit](../../audit/backend/BACKEND_AUDIT.md) owns cross-target
ABI and proof rules. [Compiler audit](../../audit/compiler/COMPILER_AUDIT.md)
owns compiler integration and roadmap decisions.
