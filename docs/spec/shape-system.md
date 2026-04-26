---
status: Normative
classification: Normative
last_updated: 2026-04-26
---

# shape-system.md
# Shape System (Normative core + Informative appendix)

## Normative
- Shape variables with constraints (e.g., `N % block == 0`).
- Compile‑time checking; error codes and message schema.

## Diagnostics (Informative)
```text
E1302: Shape mismatch: expected Tensor[B, N, D], found Tensor[B, M, D]
  at attention(q, k): parameter `k`
  note: N and M must be equal or broadcastable