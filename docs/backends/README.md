---
classification: Documentation index
authority: Reader-facing backend navigation
last_updated: 2026-07-13
---

# Backend Documentation

Start here when the question is about a hardware target. Every target has the
same split of responsibility:

| Need | Location |
|---|---|
| Architecture, compiler route, runtime model, and implementation guidance | The target directories listed below |
| Current op/target placement and proof evidence | [`docs/audit/generated/`](../audit/generated/) |
| Decisions, open work, audits, and historical plans | [`docs/audit/backend/`](../audit/backend/) |

Generated dashboards are the status authority. A backend guide explains the
design and links to evidence; it must not duplicate mutable coverage counts.

| Target | Start here | Implementation detail | Decision and history |
|---|---|---|---|
| Apple CPU + GPU | [apple/](apple/) | [Apple kernel guide](apple/kernel-guide.md) | [Apple audit](../audit/backend/apple/APPLE_AUDIT.md) |
| x86 / AVX-512 / AMX | [x86/](x86/) | compiler and ABI references linked there | [shared backend audit](../audit/backend/BACKEND_AUDIT.md) |
| ROCm / AMD | [rocm/](rocm/) | [ROCm kernel inventory](rocm/kernel-inventory.md) | [ROCm audit](../audit/backend/rocm/ROCM_AUDIT.md) |
| NVIDIA / CUDA | [nvidia/](nvidia/) | [NVIDIA kernel inventory](nvidia/kernel-inventory.md) | [NVIDIA audit](../audit/backend/nvidia/NVIDIA_AUDIT.md) |

## Taxonomy

Use these terms consistently: **target** (Apple GPU, x86, ROCm, NVIDIA),
**execution unit** (op, fused region, or package subgraph), **compiler form**
(generic dispatch, strict Target IR, or artifact), **runtime executor**,
**placement**, and **proof**. Compiler support, artifact emission, native
execution, and hardware proof are distinct claims.
