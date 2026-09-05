---
last_updated: 2026-09-05
audit_role: index
---

# Backend audit map

The [integrated compiler plan](../compiler/INTEGRATED_COMPILER_PLAN.md) owns
cross-domain order and dependencies for the MLIR/LLVM native-code foundation.
This directory owns backend findings, execution details, and device evidence.
It does not define a second compiler roadmap.

| Document | Responsibility |
|---|---|
| [Shared backend audit](BACKEND_AUDIT.md) | Shared findings and historical outcomes. |
| [E2E compilation plan](E2E_COMPILATION_AUDIT.md) | Level A/B/C definitions and detailed seam closure gates, subordinate to integrated sequencing. |
| [Apple queue](apple/todo.md) / [audit map](apple/README.md) | Apple compiler, Metal/AIR packaging, runtime and owning-device proof. |
| [NVIDIA queue](nvidia/todo.md) / [audit](nvidia/NVIDIA_AUDIT.md) | NVVM/PTX packaging, CUDA execution and architecture-specific schedules. |
| [ROCm queue](rocm/todo.md) / [audit](rocm/ROCM_AUDIT.md) | ROCDL/AMDGPU packaging, HIP execution and architecture-specific schedules. |
| [x86 queue](x86/todo.md) | LLVM native images, CPU ABI and ISA-specific execution proof. |
| [Execution matrix](../generated/runtime_execution_matrix.md) | Generated runtime route status; does not establish performance superiority. |

## Current integrated-plan handoff

Use synchronization key `IR-NATIVE-FOUNDATION-1` (E2E-REAL-5 / W2.4 / W2.4a).
The five-action native ownership loop records bounded paired saved-LSE, signed
INT4 and paged-read migration, verifier recovery, and the queue ownership spike.
Allocation-specific release and control-flow lifetime remain open. NVFP4/MX,
fused paged attention, recompute backward, replay-SSM and MoE remain separate
work; existing native constructors are not evidence that migration is complete.

Each queue's dated native checkpoint/packed-state entry records its own proof.
CUDA results do not establish ROCm parity; compiler-only measurements do not
establish device latency; Apple and x86 require their own follow-ups. Implementation
in the current working tree is not a claim of merged or production-selected work.

## Retention and cleanup

Architecture queues retain dated evidence and stable work-item anchors. Read
the current handoff before older synchronization entries. Scoped Blackwell,
Strix Halo and E2E plans remain `landing`: age alone does not close their residual
work. Their original bring-up snapshots are provenance, not current support.

The superseded [Apple codegen design](apple/archive/APPLE_GPU_CODEGEN_PLAN.md)
is archived; its rationale remains available without competing with the active
queue. Close a plan only after its residuals are discharged or explicitly handed
to an owning active item, then archive it and summarize it in its owning audit.
