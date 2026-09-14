# Native sparse Graph and logical AD — 2026-09-14

Owning host: Tajasarus / gfx1201 / ROCm 10.0 / WSL2.
Compiler: LLVM 23.1.1 assertions-enabled tessera-opt.

460 focused tests passed, covering native sparse Graph capture, signed/FP8/INT4
runtime regressions, dtype/diagnostic/pass registries, audit gates and ROCm AD.
Ruff and focused mypy passed. `focused_tests.txt` records the run.

Native Graph-to-Schedule lowering owns packing/index/status generation for one
static checked-2:4 f16/bf16 matrix product. Tests disable the Python recipe
producer during JIT capture. Metadata is projected from the emitted descriptor.
The explicit capture API supports AD-configured JIT parents; their backward
continues to consume logical source rather than packing/index IR.

Device tests verify gradients at numerical zeros, operand reversal and repeated
operands. Repeated edges accumulate both contributions. Existing composed HIP
GEMM executes backward; this is not a new general region AD package. Device
identity follows the selected adapter.

Still open: automatic default/arbiter sparse-versus-dense selection, nonisolated
Graph envelopes, arbitrary public AD composition/control flow/higher order, and
measured performance promotion. No evidence transfers to sibling backends.
