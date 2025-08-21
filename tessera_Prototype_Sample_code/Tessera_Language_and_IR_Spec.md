# Tessera Language and IR Specification
*(Normative — modeled after LLVM LangRef & CUDA docs; includes a BNF-like grammar for the Tessera DSL and formal IR layer definitions.)*

**Status:** Draft v1.0 (Normative unless explicitly marked *Informative*)  
**Scope:** Defines the Tessera high-level DSL, type system, semantics, and the multi-level IRs (Graph IR, Schedule IR, Tile IR, Target IR), including verification rules and lowering constraints.

---

## 1. Conformance, Terms, and Notation

**Conformance keywords**: “**must**”, “**must not**”, “**shall**”, “**shall not**”, “**should**”, “**may**” follow RFC 2119 meanings.  
**BNF Notation**: This document uses an LL(1)-friendly BNF. Terminals are in `monospace`. Optional elements are `[ ... ]`. Zero or more is `{ ... }`.

**IR Semantics style**: SSA-based, MLIR-like region semantics with explicit attributes. Types and operations include verification rules. Lowering rules are normative unless stated otherwise.

---

## 2. Language Overview (DSL)
(*Normative*)

The Tessera DSL is a textual front-end for constructing Graph IR modules. It provides:
- **Module** and **function** declarations (`module`, `func`, `kernel`).
- **Tensor** types and values.
- **Operator** calls (`op.*`) and **distributed** calls (`dist.*`).
- **Scheduling annotations** (`@schedule(...)`) and **numerics/precision** annotations.
- **Control constructs** (`if`, `for`, `while`) with SSA-like region semantics.

### 2.1 Lexical Structure
- **Identifiers**: `[A-Za-z_][A-Za-z0-9_]*`
- **Integer literals**: decimal; **Float literals**: decimal with `.` or exponent.
- **String literals**: double-quoted; escape `\"` `\\n`.
- **Comments**: `//` to end-of-line; `/* ... */` block comments.
- **Keywords (reserved)**:  
  `module func kernel let return if else for while in schedule dist op mesh type dtype layout precision numerics pipeline barrier shared align asm import from as`

### 2.2 Types (Terminals)
Scalar dtypes terminal set (non-exhaustive):  
`fp64 | fp32 | tf32 | bf16 | fp16 | fp8_e4m3 | fp8_e5m2 | int64 | int32 | int16 | int8 | bool`

Compound types (normative forms):
- **Tensor**: `tensor< Shape x DType [; layout=Layout] >`
- **Fragment** (Tensor Core MMA tile): `fragment< m, n, k, DType, layout=... >`
- **MemRef** for shared/local memory tiles: `memref< Shape x DType > [addrspace=shared|global]`
- **Mesh**: `mesh< axes=[id {, id}], shape=[int {, int}] >`
- **Fn type**: `fn( ParamTypes ) -> RetType`

### 2.3 BNF Grammar (DSL)

```
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
attr_block     ::= "@" "{" { attr ("," attr) } "}"
attr           ::= ident "=" attr_val
attr_val       ::= integer | float | string | ident | "[" { attr_val ("," attr_val) } "]"

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
range_expr     ::= expr ":" expr [ ":" expr ]   // start:stop[:step]
return_stmt    ::= "return" [ expr_list ]

expr_list      ::= expr { "," expr }
expr           ::= primary { binop primary }
primary        ::= literal | ident | call_expr | tensor_expr | "(" expr ")"
call_expr      ::= qual_ident "(" [ arg_list ] ")" [ attr_block ]
qual_ident     ::= ident { "." ident }          // e.g., op.softmax
arg_list       ::= expr { "," expr }

tensor_expr    ::= "tensor" "<" shape "x" dtype [ ";" "layout" "=" layout_spec ] ">"
shape          ::= dim { "x" dim }              // e.g., 128x64x?
dim            ::= integer | "?"                // unknown-dimension
dtype          ::= ident
layout_spec    ::= ident [ "(" param_list ")" ] // e.g., row_major, col_major, blocked(bm=128,bn=128)

binop          ::= "+" | "-" | "*" | "/" | "@" | "==" | "!=" | "<" | ">" | "<=" | ">="
literal        ::= integer | float | string | "true" | "false"

schedule_stmt  ::= "schedule" "." ident "(" [ arg_list ] ")" [ attr_block ]
dist_stmt      ::= "dist" "." ident "(" [ arg_list ] ")" [ attr_block ]
barrier_stmt   ::= "barrier" "(" [string] ")"
assert_stmt    ::= "assert" "(" expr [ "," string ] ")"
type_expr      ::= tensor_type | scalar_type | memref_type | fragment_type | mesh_type | fn_type

tensor_type    ::= "tensor" "<" shape "x" dtype [ ";" "layout" "=" layout_spec ] ">"
memref_type    ::= "memref" "<" shape "x" dtype ">" [ "[" "addrspace" "=" ident "]" ]
fragment_type  ::= "fragment" "<" integer "," integer "," integer "," dtype [ ";" "layout" "=" layout_spec ] ">"
mesh_type      ::= "mesh" "<" "axes" "=" "[" ident { "," ident } "]" "," "shape" "=" "[" integer { "," integer } "]" ">"
fn_type        ::= "fn" "(" [ type_expr { "," type_expr } ] ")" "->" type_expr
```

**Notes**:
- `kernel` functions denote entry points intended for device execution.  
- `schedule.*` and `dist.*` calls **do not** produce tensors; they mutate IR scheduling and distribution metadata.  
- `op.*` calls are pure by default unless operator semantics declare effects (e.g., collectives).

---

## 3. Static Semantics (Type & Shape System)
(*Normative*)

### 3.1 Type rules
- Every expression **must** have a statically known type.
- Tensor ranks **must** be known; unknown dimensions (`?`) are allowed but must be resolved prior to lowering to Tile IR.
- Implicit casts **shall not** occur except for precision promotions explicitly listed (e.g., BF16→FP32 accumulation).

### 3.2 Shape rules
- Algebraic ops (e.g., `matmul`) **must** satisfy dimension contracts (K matches).
- Broadcasting **may** follow NumPy semantics when an operator declares it; otherwise, it is forbidden.

### 3.3 Layout rules
- Layout is a first-class attribute; operators **must** define legal input/output layouts or request an explicit `layout_cast` at Schedule IR.
- Fragment/Tile layouts **must** match hardware constraints at Tile IR (e.g., `ldmatrix` alignment).

---

## 4. Dynamic Semantics (Evaluation)
(*Normative*)

- The DSL denotes a **pure functional graph** except for: collectives, random, stateful norms/optimizers, and explicit `schedule.*` mutations.
- Execution order is defined by data dependencies and explicit synchronization (`barrier`, `pipeline` stages).
- Determinism is governed by the **Numerics profile** (see §10).

---

## 5. Graph IR (tessera.graph)
(*Normative*)

**Purpose**: Algebraic operator graph, autodiff, distributed intent.  
**Representation**: SSA values, MLIR-like region ops, module-scoped symbols.

### 5.1 Types
- `TensorType(shape, dtype, layout?)`
- `MeshType(axes, shape)`

### 5.2 Core Operations (Selection)
```
%t  = tessera.graph.const_tensor  : tensor<...>
%y  = tessera.graph.matmul %a, %b : (tensor<MxKxT>, tensor<KxNxT>) -> tensor<MxNxT>
%p  = tessera.graph.softmax %x    : tensor<...> -> tensor<...>
%f  = tessera.graph.fft %x        : tensor<...> -> tensor<...>
%ar = tessera.graph.all_reduce %x {axis="dp", op="sum"} : tensor<...> -> tensor<...>
%g  = tessera.graph.grad %y, %wrt(%x) : tensor<...>
```

### 5.3 Attributes
- `shape`, `dtype`, `layout`, `mesh_axes`, `precision`, `numerics_profile`

### 5.4 Verification
- **Matmul**: `%a.shape[-1] == %b.shape[-2]`
- **Softmax**: `axis` in range; stable by construction.
- **AllReduce**: tensor sharding matches `axis` communicator.

### 5.5 Autodiff Semantics
- Reverse mode defined by operator adjoints; operators **must** declare their VJP/JVP or inherit default rule.

---

## 6. Schedule IR (tessera.schedule)
(*Normative*)

**Purpose**: Legally transforms Graph IR into tiled, fused, pipelined kernels with explicit memory staging.

### 6.1 Operations
```
%t1 = tessera.schedule.tile %t {m=128, n=128, k=64}
%t2 = tessera.schedule.fuse %tA, %tB
%p  = tessera.schedule.pipeline %t {double_buffer=true, depth=3}
%ps = tessera.schedule.prefetch %t {scope="shared", align=32}
%lc = tessera.schedule.layout_cast %t {to="row_major"}
```

### 6.2 Constraints
- Tiling **shall** preserve dependences.
- Prefetch scopes: `"shared" | "global"`; shared requires per-block legality.
- Pipeline stages must be acyclic; `depth ≥ 1`.

### 6.3 Verification
- Tile sizes must divide or cover shapes with explicit padding.
- Layout casts must be supported by producer/consumer ops or be materialized copies.

---

## 7. Tile IR (tessera.tile)
(*Normative*)

**Purpose**: Binding of scheduled computation to GPU execution primitives (blocks, warps, Tensor Cores, shared/reg memory).

### 7.1 Types
- `fragment<m,n,k,dtype,layout>`
- `memref<shape x dtype>[addrspace=shared|global]`

### 7.2 Operations (Selection)
```
%sa = tessera.tile.shared.alloc : memref<128x128xbf16>[addrspace=shared]
%la = tessera.tile.ldmatrix %sa {transpose=false, stride=...} : fragment<16,16,16,bf16>
%mm = tessera.tile.mma.sync %a, %b, %c {m=16, n=16, k=16, accum="fp32"} : fragment<...>
%cp = tessera.tile.cp.async %global, %shared {bytes=...}
     tessera.tile.barrier
%sh = tessera.tile.shfl.sync %v {mask=0xffffffff, delta=...}
```

### 7.3 Constraints
- `ldmatrix` alignment **must** satisfy hardware requirements (e.g., 128-bit).
- `mma.sync` tile shapes **must** be supported for the `dtype` on target architecture.
- Shared allocations **shall** fit per-block smem limits; verifier computes static upper bound.

---

## 8. Target IR & ABI
(*Normative*)

**Purpose**: Lower to LLVM dialect and emit PTX (NVIDIA) or ROCm LLVM (AMD). Define kernel entry ABI and calling conventions.

### 8.1 Kernel Entry ABI
```
kernel @k(%arg0: !llvm.ptr<global>, %arg1: i64, %arg2: f32, ...)
    attributes { grid=(gx,gy,gx), block=(bx,by,bz), smem_bytes=N, stream=S }
```
- All device pointers passed as global pointers; scalar uniforms as 32/64-bit.  
- Grid/block dimensions **must** be explicit.  
- Shared memory size **must** be declared; dynamic segment allowed.

### 8.2 Lowering Rules (Graph→Schedule→Tile→Target)
- **Graph→Schedule**: introduce tiling/fusion/prefetch; **shall** preserve semantics.  
- **Schedule→Tile**: map loop nests to blocks/warps; introduce explicit smem ops.  
- **Tile→Target**: emit vendor intrinsics:  
  - NVIDIA: `mma.sync`, `ldmatrix`, `cp.async`, `bar.sync`, `shfl.sync`  
  - AMD: corresponding MFMA/WMMA intrinsics, LDS ops, DS swizzles.

### 8.3 Memory Model
- Address spaces: `global`, `shared`, `private`.  
- Synchronization: `tessera.tile.barrier` is block-scoped; distributed sync via collectives is mesh-scoped.  
- Atomics follow target’s sequentially consistent model unless annotated otherwise.

---

## 9. Distributed Semantics
(*Normative*)

- `mesh<axes=[...], shape=[...]>` defines a process/device grid.  
- `dist.*` ops **must** declare the axis and operation; shapes **shall** match across ranks.  
- Collectives are deterministic under `numerics.profile("deterministic"|"strict")` with fixed reduction trees.

Example (Graph IR):
```
%y = tessera.graph.all_reduce %x {axis="dp", op="sum"} : tensor<...> -> tensor<...>
```

---

## 10. Numerics, Precision, and Determinism
(*Normative*)

- Operators **must** specify accumulation precision (`accum="fp32"` default for mixed-precision GEMM).  
- Stable softmax semantics (max-subtraction) are mandatory.  
- Profiles: `fast | deterministic | strict`; stricter profiles **must not** reorder reductions or use differing math approximations.

---

## 11. Verification Summary
(*Normative checklist*)

- **Types**: all ops well-typed; ranks known (except allowed `?` before Tile IR).  
- **Shapes**: dimension contracts satisfied; broadcasts legal where declared.  
- **Layouts**: producer/consumer layout compatibility or materialized casts present.  
- **Schedule legality**: no dependence violations; pipeline acyclic; smem/reg usage within bounds.  
- **Tile legality**: alignment, tile sizes, and intrinsics supported on target.  
- **Distributed**: collective axis & shapes consistent across ranks.  
- **Numerics**: accumulation precision and determinism profile consistent.

---

## 12. Diagnostics (Informative)
- Verifiers **should** produce actionable messages with op location, operand shapes, and suggested remedies.  
- IR dumps at each level **may** be emitted for debugging.

---

## 13. Worked Examples (Informative)

### 13.1 DSL → IR (MatMul)
```
module demo {
  func mm(A: tensor<1024x1024xbf16>, B: tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16> {
    let C: tensor<1024x1024xbf16> = op.matmul(A, B);
    return C;
  }
}
```
**Graph IR**
```
%C = "tessera.graph.matmul"(%A, %B)
     : (tensor<1024x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16>
```
**Schedule IR**
```
%Ct = "tessera.schedule.tile"(%C) { m=128, n=128, k=64 }
%Cp = "tessera.schedule.pipeline"(%Ct) { double_buffer=true, depth=3 }
```
**Tile IR**
```
%smA = "tessera.tile.shared.alloc" : memref<128x64xbf16>[addrspace=shared]
%smB = "tessera.tile.shared.alloc" : memref<64x128xbf16>[addrspace=shared]
%fragC = "tessera.tile.mma.sync"(%fragA, %fragB, %fragC)
         {m=16, n=16, k=16, accum="fp32"} : fragment<16,16,16,bf16>
```
**Target IR** (PTX via LLVM)
```
call @llvm.nvvm.wmma.m16n16k16.mma.sync(...)
```

### 13.2 DSL with Distribution
```
mesh g = mesh<axes=[dp,tp], shape=[4,2]>;
func step(X: tensor<?,Dxbf16; layout=row_major>) {
  let Y: tensor<?,Dxbf16> = dist.all_reduce(op.matmul(X, X), axis=dp);
  return Y;
}
```

---

## 14. Appendix A — Complete Token & Grammar Reference (Normative)

### 14.1 Tokens
```
IDENT  ::= [A-Za-z_][A-Za-z0-9_]*
INT    ::= [0-9]+
FLOAT  ::= [0-9]+ "." [0-9]* ( [eE] [+-]? [0-9]+ )?
STRING ::= " ( [^"\\] | \\["\\nrt] )* "
WS     ::= [ \t\r\n]+
COMMENT::= "//" .*? "\n" | "/*" .*? "*/"
```

### 14.2 Precedence (from high to low)
```
()  (call, indexing)
unary + -
* /
+ -
@      (matrix multiply)
comparisons (== != < > <= >=)
```

### 14.3 Extended EBNF
(See §2.3 for the core grammar.)

---

## 15. Appendix B — Operator Semantics (Normative excerpts)

**MatMul**  
Type: `(tensor<MxKxT>, tensor<KxNxT>) -> tensor<MxNxT>`  
Preconditions: `K` equal, layouts compatible or castable.  
Postconditions: Output layout inherits left operand unless overridden.  
Numerics: Accumulation in `fp32` if `T ∈ {fp16, bf16, fp8_*}`.

**Softmax**  
Stable by definition: subtract max along `axis`; all finite inputs → finite outputs in (0,1).

**AllReduce**  
Deterministic tree order under `deterministic|strict` profiles. Shapes and dtypes identical across ranks. Side effect: collective comm.

---

## 16. Appendix C — ABI Summary (Normative)

- Kernel symbol naming: `tss_kernel_<mangled>`  
- Parameters order: buffers first, then scalars, then launch config metadata (hidden).  
- Alignment: 16-byte alignment for TensorCore fragments and `ldmatrix` tiles.  
- Calling convention: C-compatible for host, target-specific for device (NVVM/ROCm).

---

## 17. Change Log
- v1.0 — Initial publication
