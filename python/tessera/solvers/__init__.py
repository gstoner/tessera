"""Solver-side Python frontmatter for Tessera.

This package surfaces the Python entry points to the solver dialects
shipped under ``src/solvers/``.  The dialects themselves live in C++
MLIR (see ``src/solvers/{core,linalg,scaling_resilience,spectral,tpp}``);
the modules here expose the per-dialect *names*, *pass-pipeline aliases*,
and *status* so Python tooling and audit infrastructure can reason
about which solver surfaces are available without linking against
``tessera-opt``.

Each submodule is dependency-free (no MLIR Python bindings required).
"""

from __future__ import annotations

from . import tpp

__all__ = ["tpp"]
