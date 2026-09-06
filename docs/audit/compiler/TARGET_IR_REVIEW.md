---
last_updated: 2026-09-06
audit_role: reference
scope: Target dialects, native package ownership and verifier evidence
---

# Target IR: current architecture and remaining work

This refresh supersedes the [August inventory](archive/TARGET_IR_REVIEW_2026-08-02.md).
It preserves X1–X7 finding IDs for traceability and removes stale dialect/pass
counts and effort estimates. The [integrated plan](INTEGRATED_COMPILER_PLAN.md)
and [native foundation survey](MLIR_NATIVE_FOUNDATION_SURVEY.md) own migration
order and route inventories; the [execution matrix](../generated/runtime_execution_matrix.md)
owns recorded execution evidence.

## X1 — x86 Target IR exists

`TesseraX86Dialect.td` and `TesseraX86Ops.td` declare a real dialect, including
kernel/ABI, vector/packing and legacy AMX operations. Positive and negative
fixtures exist in `tests/tessera-ir/phase2/x86_target_ir*.mlir`.
The old “no dialect” finding and proposed build-or-exempt decision are obsolete.

A declaration does not establish current production support: the
[x86 queue](../backend/x86/todo.md) explicitly retires AMX as project direction.
Do not revive that route because historical AMX ODS or fixtures remain.

## X2 — matrix contracts remain a per-operation audit

The standalone ROCm `ROCM_MFMAOp` still declares `AnyType` operands/results.
Typed Tile fragments and admitted native package checks do not automatically
harden every standalone Target operation. Keep tightening reachable contracts
by `(architecture, instruction, shape, operand/accumulator type, role)` and
negative-test unsupported combinations. Never force one WMMA vector shape onto
all RDNA/CDNA families or copy CUDA fragment layouts to AMD.

NVIDIA now also has real native verifier bodies for block coordinates, macro-CTA
matmul and delegate/kernel contracts. The old description “NVIDIA codegen is
Python” is not an adequate description of current native scheduled packaging.
The remaining task is a route-by-route ownership and verification census, not
blanket replacement of either dialect.

## X3 — semantic strings are partially constrained

`TesseraROCMDialect.td::ROCM_EnumStrAttr` validates admitted semantic values while
preserving textual string syntax. Therefore “zero EnumAttr” no longer implies
“no value validation.” Kernel symbol names are intentionally open strings;
turning every `$name` into an enum would be wrong.

Audit genuine selector fields per operation (dtype, mode, route, counter and
policy), including required versus optional meaning. Test absence, unknown
values and incompatible combinations. Typed C++ accessors may improve the API,
but a syntax migration is distinct from fixing a fail-open verifier.

## X4 — distinguish spelling checks from actual verification

`tests/unit/test_target_ir_contract.py` still contains compatibility-text smoke
assertions. `tests/tessera-ir/phase8/target_ir_contracts.mlir` runs FileCheck on
its own text; that fixture is not a parser/verifier test. Keep these checks
labelled as smoke evidence.

Actual native evidence comes from target-loaded positive and negative fixtures
and the native package pipeline. For example, x86's wrong-tile-operand fixture
invokes `tessera-opt` and requires rejection. For every promoted family, require
parse/verify/lower checks and a numerical owning-host result. An unregistered-op
parse alone does not validate the Target contract.

## X5 / X6 — native ownership is mixed, but the destination is settled

| Surface | Role to preserve or migrate |
|---|---|
| Native Graph→Schedule→Tile producers | Semantic/schedule identity and native SSA; new shared compiler work belongs here. |
| `nvidia_native.py`, `rocm_native.py`, `x86_native.py` and Apple package routes | Orchestration and binding. Some entries still accept `GraphIRModule`; classify each and migrate historical lower-IR regeneration to consumption of verified native artifacts. |
| C++ native generators / NVVM / ROCDL / LLVM | Backend-owned instruction and image lowering; retain existing native producers while changing their callers. |
| `target_ir.py` | Compatibility/value artifact construction; emitted text is not interchangeable with a verified native package. |
| `emit/` source generators and specialized kernels | Bounded candidate/reference/compatibility roles. Do not make them the canonical replacement for MLIR/LLVM lowering. |

New proof includes scheduled unary/attention/stateful boundaries, allocation
lifetimes and native split AD products; it is not a statement that every package
family has migrated. Inspect the actual selected producer and preserve package,
schedule, layout, numeric-policy, ABI and residual identity through the boundary.

## X7 — hardware-independent inspection, architecture-specific semantics

Target IR is testable without the physical device; it is not a portable
hardware-neutral instruction set. ROCm/NVIDIA ops can name matrix instructions,
Apple ops can name library or MSL boundaries, and x86 can name CPU ABI/vector
operations. Shared semantics live above the physical lowering boundary.

Host-only tests establish contracts. Exact-device numerical/performance proof
belongs to the required backend host. x86 host companions do not prove GPU
execution; WSL pruning timings do not acquire selector authority by appearing
in a native package. ROCm ISA admission follows the in-tree RDNA archive.

## Carried-forward acceptance

| Original item | Disposition |
|---|---|
| X-U1 / X-U2 | W1.1 and the current typing census: finish reachable matrix/selector contracts, preserving per-architecture variants. |
| X-U3 | Retain smoke checks; expand actual native positive/negative coverage for each migrated family. |
| X-U4 | Dialect-existence question resolved. Execution breadth follows the x86 queue, including AMX retirement. |
| X-U5 / X-U6 | W3.2 and IR-NATIVE-FOUNDATION-1: record one producer per family, remove Graph re-entry, verify artifact identity, compare numerically on the owning host before retiring a producer. |

Source anchors: [x86 ODS](../../../src/compiler/codegen/tessera_x86_backend/include/TesseraX86/IR/TesseraX86Ops.td),
[ROCm ODS](../../../src/compiler/codegen/Tessera_ROCM_Backend/include/TesseraROCM/IR/TesseraROCMOps.td),
[ROCm attribute constraints](../../../src/compiler/codegen/Tessera_ROCM_Backend/include/TesseraROCM/IR/TesseraROCMDialect.td),
and [NVIDIA verifiers](../../../src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/IR/TesseraNVIDIADialect.cpp).
