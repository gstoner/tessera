---
status: Normative
classification: Normative
authority: High-level DSL and multi-level IR specification; defers public Python symbols to docs/spec/PYTHON_API_SPEC.md and backend ABI details to docs/spec/RUNTIME_ABI_SPEC.md
last_updated: 2026-04-28
---

# Tessera Language And IR Specification

This document defines the Tessera high-level DSL, type system, static and
dynamic semantics, and the multi-level IR stack: Graph IR, Schedule IR, Tile IR,
and Target IR. It is modeled after LLVM LangRef and CUDA documentation.

Conformance keywords `must`, `must not`, `shall`, `shall not`, `should`, and
`may` follow RFC 2119 meanings. BNF terminals are written in backticks.
Optional elements are `[ ... ]`; zero or more is `{ ... }`.

## 1. Scope

The Tessera DSL is a textual front-end for constructing Graph IR modules. It
provides:

- module, function, and kernel declarations
- tensor, fragment, memref, mesh, and function types
- standard operator calls through `op.*`
- distributed calls through `dist.*`
- scheduling and numerics annotations
- structured control flow with SSA-like region semantics

The DSL is normative as a language model. The current Python implementation may
construct the same IR through decorators and API calls rather than this textual
syntax.

## 2. Lexical Structure

| Token | Definition |
|-------|------------|
| `IDENT` | `[A-Za-z_][A-Za-z0-9_]*` |
| `INT` | `[0-9]+` |
| `FLOAT` | Decimal with `.` or exponent. |
| `STRING` | Double-quoted string with escapes for `\"`, `\\`, and `\n`. |
| `COMMENT` | `//` line comment or `/* ... */` block comment. |

Reserved keywords:

```text
module func kernel let return if else for while in schedule dist op mesh type
dtype layout precision numerics pipeline barrier shared align asm import from as
```

## 3. Type Terminals

Scalar dtype terminals:

```text
fp64 fp32 tf32 bf16 fp16
fp8_e4m3 fp8_e5m2
fp6_e2m3 fp6_e3m2
fp4_e2m1 nvfp4
int64 int32 int16 int8 int4
uint64 uint32 uint16 uint8
bool
```

Rubin target capability note: NVIDIA public Rubin material identifies NVFP4,
FP8/FP6, INT8, FP16, BF16, TF32, and FP64 as Tensor Core precision families,
with FP64, FP32, INT32, FP16, and BF16 listed for CUDA-core execution. Tessera
therefore treats `nvfp4`, `fp4_e2m1`, `fp6_e2m3`, and `fp6_e3m2` as standard
low-precision dtype names, while exact bit encodings remain target-specific.

Compound types:

```text
tensor< Shape x DType [; layout=Layout] >
fragment< m, n, k, DType [; layout=Layout] >
memref< Shape x DType > [addrspace=local|shared|global|distributed|host|tmem]
mesh< axes=[id {, id}], shape=[int {, int}] >
fn( ParamTypes ) -> RetType
```

## 4. Core BNF

```text
program        ::= { module_decl }
module_decl    ::= "module" ident "{" { decl } "}"
decl           ::= func_decl | kernel_decl | type_decl | const_decl | mesh_decl

type_decl      ::= "type" ident "=" type_expr ";"
const_decl     ::= "let" ident ":" type_expr "=" expr ";"
mesh_decl      ::= "mesh" ident "=" "mesh" "<" mesh_params ">" ";"

func_decl      ::= "func" ident "(" [param_list] ")" [ret_ann] [attr_block] block
kernel_decl    ::= "kernel" ident "(" [param_list] ")" [ret_ann] [attr_block] block
param_list     ::= param { "," param }
param          ::= ident ":" type_expr [ "=" expr ]
ret_ann        ::= "->" type_expr
attr_block     ::= "@" "{" { attr [","] } "}"
attr           ::= ident "=" attr_val
attr_val       ::= integer | float | string | ident | "[" { attr_val [","] } "]"

block          ::= "{" { stmt } "}"
stmt           ::= var_decl ";" | assign ";" | op_stmt ";" | ctrl_stmt | return_stmt ";"
var_decl       ::= "let" ident ":" type_expr [ "=" expr ]
assign         ::= lvalue "=" expr
lvalue         ::= ident | ident "[" slice_list "]"
op_stmt        ::= call_expr | schedule_stmt | dist_stmt | barrier_stmt | assert_stmt
ctrl_stmt      ::= if_stmt | for_stmt | while_stmt
if_stmt        ::= "if" "(" expr ")" block [ "else" block ]
for_stmt       ::= "for" "(" iter_decl "in" range_expr ")" block
while_stmt     ::= "while" "(" expr ")" block
iter_decl      ::= ident [ "," ident ]
range_expr     ::= expr ":" expr [ ":" expr ]
return_stmt    ::= "return" [ expr_list ]

expr_list      ::= expr { "," expr }
expr           ::= primary { binop primary }
primary        ::= literal | ident | call_expr | tensor_expr | "(" expr ")"
call_expr      ::= qual_ident "(" [ arg_list ] ")" [ attr_block]
qual_ident     ::= ident { "." ident }
arg_list       ::= expr { "," expr }

tensor_expr    ::= "tensor" "<" shape "x" dtype [ ";" "layout" "=" layout_spec ] ">"
shape          ::= dim { "x" dim }
dim            ::= integer | "?"
dtype          ::= ident
layout_spec    ::= ident [ "(" param_list ")" ]

schedule_stmt  ::= "schedule" "." ident "(" [ arg_list ] ")" [ attr_block ]
dist_stmt      ::= "dist" "." ident "(" [ arg_list ] ")" [ attr_block ]
barrier_stmt   ::= "barrier" "(" [string] ")"
assert_stmt    ::= "assert" "(" expr [ "," string ] ")"
```

## 5. Static Semantics

Every expression must have a statically known type. Tensor ranks must be known.
Unknown dimensions are allowed at Graph IR and Schedule IR, but must be resolved,
guarded by a runtime shape witness, or explicitly padded before lowering to
Tile IR.

Shape semantics are defined by `docs/spec/SHAPE_SYSTEM.md`. Tensor types carry
logical shape separately from physical layout and shard map. Reusing a symbolic
dimension name asserts equality within the current specialization. Derived
dimensions such as `D = H * Dh` are legal in Graph IR, but products must be
resolved before Tile IR lowering.

Implicit casts shall not occur except for precision promotions declared by the
operator numeric policy, such as BF16 or FP8 storage with FP32 accumulation.
Broadcasting is forbidden unless an operator explicitly declares NumPy-style
broadcast semantics.

Layout is first-class. Operators must declare legal input/output layouts or
require an explicit `layout_cast` at Schedule IR. Fragment and Tile layouts must
match target constraints such as `ldmatrix`, WGMMA, TMA, or MFMA alignment.

## 6. Dynamic Semantics

The DSL denotes a pure functional graph except for declared effects:

```text
pure < random < movement < state < collective < memory < io < top
```

> **Implementation gap (Phases 1–3):** The compiler currently implements a **5-level** subset: `pure < random < memory < io < top`. The intermediate levels `movement`, `state`, and `collective` are specified here as the full aspirational contract but are not yet emitted or inferred by `EffectAnnotationPass`. They will be activated in Phases 4–5. See `docs/CANONICAL_API.md §Effect System` for the currently implemented table.

Execution order is defined by SSA data dependencies, explicit synchronization,
state effects, movement effects, and collective dependencies. Determinism is
governed by the active numeric profile: `fast`, `deterministic`, or `strict`.

## 7. Graph IR

Graph IR represents algebraic operators, autodiff, state objects, and
distributed intent.

Core type families:

```text
TensorType(shape, dtype, layout?)
MeshType(axes, shape)
NumericPolicy(storage, accum, rounding, scale, quant_axis, deterministic)
KVCacheType(shape, dtype_policy, eviction, page_size)
```

Selected operations:

```mlir
%y = tessera.graph.matmul %a, %b
%p = tessera.graph.softmax %x {axis = -1}
%f = tessera.graph.fft %x
%c = tessera.graph.kv_cache.create {eviction = "rolling_window"}
%o = tessera.graph.flash_attn %q, %k, %v, %c
%r = tessera.graph.all_reduce %x {axis = "dp", op = "sum"}
%a = tessera.graph.arch.parameter {size = 4, dtype = "fp32"}
%g = tessera.graph.arch.gumbel_softmax %a {temperature = 4.0}
%m = tessera.graph.arch.mixed %x, %g {candidates = ["flash", "performer", "gmlp"]}
```

Verification:

- matmul requires `lhs.shape[-1] == rhs.shape[-2]`
- batch matmul and elementwise ops use broadcasting only when the operator
  declares broadcast semantics
- reshape preserves element count, including derived-dimension products
- shard maps require logical dimensions to be divisible by their mesh axes when
  concrete, or guarded by runtime witnesses when dynamic
- softmax axis must be in range and stable by construction
- cache append must match cache head dimension and dtype policy
- all-reduce requires sharding compatibility with the axis communicator
- operators must carry legal numeric policies for requested dtype/target pairs
- architecture parameters must remain FP32 and candidate gates must match
  candidate counts

## 8. Schedule IR

Schedule IR legally transforms Graph IR into tiled, fused, pipelined kernels
with explicit movement and memory staging.

Selected operations:

```mlir
%t = tessera.schedule.tile %x {m = 128, n = 128, k = 64}
%f = tessera.schedule.fuse %a, %b
%p = tessera.schedule.pipeline %t {double_buffer = true, depth = 3}
%m = tessera.schedule.prefetch %t {scope = "shared", align = 32}
%l = tessera.schedule.layout_cast %t {to = "row_major"}
%a = tessera.schedule.artifact %p {hash = "..."}
%k = tessera.schedule.knob %p {name = "tile_m", choices = [64, 128]}
```

Constraints:

- tiling shall preserve dependences
- prefetch scopes are `shared`, `global`, `distributed`, or `host`
- pipeline stages must be acyclic and have depth at least one
- movement plans must be explicit before Tile IR
- schedule feasibility shall prune tile/layout candidates that violate shape
  divisibility or shard constraints before autotuning
- schedule artifacts must include shape, layout, target, numeric policy,
  movement plan, tile knobs, and hash

## 9. Tile IR

Tile IR binds scheduled computation to accelerator execution primitives:
blocks, warps, fragments, shared memory, transaction barriers, and MMA.

Selected operations:

```mlir
%sa = tile.alloc_shared %desc : memref<128x64xbf16, shared>
tile.async_copy %global, %shared {stage = 0, vector = 16}
%bar = tile.mbarrier.alloc {count = 1, scope = "block"}
%tok = tile.mbarrier.arrive_expect_tx %bar {bytes = 16384, semantics = "release", scope = "block"}
%ok = tile.mbarrier.try_wait %bar, %tok
%mm = tile.mma %a, %b, %c {accum = "fp32"}
tile.barrier
```

Hopper forward rule: mbarrier transaction barriers are available for NVIDIA
targets with `isa >= SM_90`. They are required for TMA-style asynchronous copy
completion tracking and may lower to PTX `mbarrier.*` primitives.

Tile verification:

- `ldmatrix` and MMA operands must satisfy alignment and layout constraints
- MMA tile shapes must be supported for the dtype on the target architecture
- shared allocations must fit per-block shared-memory limits
- mbarriers must be initialized before arrival or wait
- transaction byte counts must match associated asynchronous movement
- barriers must not appear in divergent control paths for their scope

## 10. Target IR And ABI

Target IR lowers Tile IR to vendor-specific intrinsics and the runtime ABI.

NVIDIA lowering includes:

```text
mma.sync, wgmma.mma_async, tcgen05, ldmatrix, cp.async,
cp.async.bulk.tensor, mbarrier.*, bar.sync, shfl.sync
```

AMD lowering includes:

```text
MFMA/WMMA intrinsics, LDS operations, DS swizzles, cache controls
```

Kernel entry ABI:

```text
kernel @k(%arg0: !llvm.ptr<global>, %arg1: i64, %arg2: f32, ...)
  attributes {grid=(gx,gy,gz), block=(bx,by,bz), smem_bytes=N, stream=S}
```

Pointers are passed as address-space-qualified global pointers. Scalar uniforms
are 32-bit or 64-bit values. Grid, block, and dynamic shared-memory size must be
explicit.

## 11. Distributed Semantics

`mesh<axes=[...], shape=[...]>` defines a process/device grid. `dist.*` ops must
declare axis and operation. Collective operand shapes and layouts shall match
across participating ranks after sharding rules are applied.

Deterministic profiles require fixed collective ordering and reduction trees.

## 12. Verification Checklist

- all ops are well-typed
- ranks are known before Tile IR
- shape contracts are satisfied
- layouts are compatible or materialized casts exist
- schedule transforms preserve dependences
- pipeline and movement plans are acyclic
- tile alignment and intrinsic constraints hold for the target
- mbarriers are legal for target and scope
- distributed axes and shapes are consistent
- numeric policies are legal for dtype and target

## 13. Worked Example

```text
module demo {
  func mm(A: tensor<1024x1024xbf16>, B: tensor<1024x1024xbf16>)
      -> tensor<1024x1024xbf16> {
    let C: tensor<1024x1024xbf16> = op.matmul(A, B);
    return C;
  }
}
```

Graph IR:

```mlir
%C = tessera.graph.matmul %A, %B
  : (tensor<1024x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16>
```

Schedule IR:

```mlir
%Ct = tessera.schedule.tile %C {m = 128, n = 128, k = 64}
%Cp = tessera.schedule.pipeline %Ct {double_buffer = true, depth = 3}
```

Tile IR:

```mlir
%bar = tile.mbarrier.alloc {count = 1, scope = "block"}
tile.async_copy %A_global, %A_shared {stage = 0, vector = 16}
%tok = tile.mbarrier.arrive_expect_tx %bar {bytes = 16384, semantics = "release", scope = "block"}
%ready = tile.mbarrier.try_wait %bar, %tok
%fragC = tile.mma %fragA, %fragB, %fragC {m = 16, n = 16, k = 16, accum = "fp32"}
```

## 14. References

- `docs/spec/MEMORY_MODEL_SPEC.md`
- `docs/guides/Tessera_Tensor_Layout_And_Data_Movement_Guide.md`
- `docs/spec/GRAPH_IR_SPEC.md`
- `docs/spec/TARGET_IR_SPEC.md`
- `docs/spec/PYTHON_API_SPEC.md`
