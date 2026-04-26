# Tessera Linear Algebra Solvers (Mixed Precision)

This package provides:
- A spec (`docs/Tessera_Linalg_Solvers_Spec.md`)
- An ODS skeleton for the Solver dialect
- Pass stubs for mixed precision and iterative refinement
- A test sketch and minimal CMake

This package now lives under `src/solvers/linalg` so linear algebra solver work stays with the rest of the solver stack. Wire registration in the parent solver build when the scaffold graduates.
