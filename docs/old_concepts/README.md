# old_concepts — Archived Research Material

This folder contains design explorations and early-phase documents that no longer reflect the current Tessera architecture. They are preserved for historical reference only.

**Nothing in this folder should be used as a reference for the current implementation.**

---

## Contents

### `Rust_Frontend_Research/`
Early architectural proposal to implement the Tessera frontend parser and type system in Rust, with Python as the user-facing API layer.

**Status:** Not adopted. The Tessera frontend is pure Python, permanently. The MLIR C++ pass pipeline handles the performance-critical compilation stages. A Rust intermediate layer adds FFI complexity without user-visible benefit.

### `Tracing_JIT_Research/`
Research exploration of a multi-tier meta-tracing JIT compiler for Tessera — a Tier 1 interpreter, Tier 2 standard compilation, and Tier 3 adaptive specialization with hotspot detection.

**Status:** Not on the roadmap. The implemented Tessera compiler is a static AOT pipeline (Python → Graph IR → Schedule IR → Tile IR → Target IR). Constraint checking at decoration time and deterministic effect inference both depend on static analysis. Phases 1–6 do not include a JIT tier.

### `Target_IR_CPP_Artifacts/`
C++ source code that was placed in the docs folder by mistake. These are implementation artifacts from early Target IR work, not documentation.

**Status:** The actual documentation is `docs/architecture/tessera_target_ir_usage_guide.md`. The files here are C++ and should be referenced from `src/` if needed.

### `Tessera_Programming_Model_V1.md`
The original programming model document written before Phase 1. Uses the pre-canonical API (`@tessera.function`, pre-`Region` syntax).

**Status:** Superseded. The canonical API is defined in `CLAUDE.md` and documented in `docs/spec/PYTHON_API_SPEC.md` (once written). Do not use this as a reference.

---

*Archived: April 26, 2026*
