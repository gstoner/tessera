# Compiler integration drift repair — 2026-10-07

Super-Bear WSL full non-slow unit lane: 21,920 passed, 7,503 skipped,
874 deselected, five failures. Three inventory/evidence failures are repaired:
native scaled-matmul adjoint recognition; exact ODS inventory 673 with
real structured-reduction consumers; architecture-specific scale-transpose
evidence registration. Focused repair lane: 780 passed.

The gfx1201 scale-transpose packet does not certify the generic ROCm alias.
Two generic scaled-matmul batching/transpose closure failures remain unchanged.
No full-suite green result or sibling device execution is claimed.
See full-unit-before-repair.log and focused.log.
