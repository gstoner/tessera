# INT4 sparse execution — 2026-09-14

Tajasarus / gfx1201 / WSL2 / ROCm 10.0 / LLVM 23.1.1 with assertions.

The compiled logical producer accepts byte-addressable signed or unsigned inputs
with a declared 4-bit range, performs packing in native lowering, and issues
K=32 SWMMAC. Four signedness combinations pass on 32x48x64 matrices, exercising
all six sparse index pairs. Tests check disassembly, exact integer results and
range refusal in both host preflight and GPU validity output. This does not
certify K=64 instructions, compressed host storage, or performance promotion.

`focused_tests.txt`: 453 passing tests including owning-device sparse, capture,
runtime, registry and audit checks. Ruff and focused mypy passed.

Automatic sparse selection/native Graph lowering and arbitrary AD remain open.
The AD plan requires differentiation before physical sparse packing, with tests
preserving derivatives at numerical zeros and composed result ownership.
