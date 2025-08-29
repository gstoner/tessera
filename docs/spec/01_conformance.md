
### 2.2 `docs/spec/01_conformance.md`
```markdown
# Tessera Conformance (Normative)

## 1. Scope
This document defines requirements for a conformant Tessera implementation, including language subset, runtime ABI surface, and TileIR semantics.

## 2. Definition of Conformance
A **Conformant Implementation** shall:
1. Implement the required **Language Core** (Spec §02) and **ABI Core** (Spec §03).
2. Accept and correctly compile the **Conformance Test Suite** programs.
3. Emit TileIR meeting the invariants in Spec §04.

## 3. Profiles
- **Profile T0 (Kernel‑only)** — minimal language + host ABI.
- **Profile T1 (Single‑node)** — adds device memory mgmt; streams/events.
- **Profile T2 (Cluster)** — adds P2P/NVLink fabrics; distributed tiles.

## 4. Compliance Testing
- Structure and expected outputs TBD; see Appendix A1 for examples (informative).