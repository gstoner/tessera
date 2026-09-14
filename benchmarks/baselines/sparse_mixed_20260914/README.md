# Mixed sparse operand validation — 2026-09-14

Owning host: Tajasarus, gfx1201, ROCm 10.0, WSL2, LLVM 23.1.1 with assertions.
Compiler target: `build-assertions/tools/tessera-opt/tessera-opt`.

16 byte-format cases cover signed/signed, unsigned/unsigned, both mixed-sign orders,
both same-format FP8 forms and both mixed FP8 forms at 16x16x32 and 32x48x64.
Each checks the disassembled SWMMAC instruction, exact numerical agreement on
representable inputs, all six sparse index pairs, and invalid-pattern refusal.
Unsigned values exceed 127. These are correctness checks, not precision/error-budget
or performance promotion evidence. No sibling execution is inferred.

`focused_tests.txt`: 407 passing tests, including byte/half device checks and registries.
`contract_tests.txt`: 33 passing sparse-contract and audit tests.

Native Graph lowering/automatic selection, INT4, arbitrary AD, actual driver-hang
recovery and remaining cohort/breadth routes remain open. Process termination
and workload-scoped replacement checks do not establish GPU reset recovery.
