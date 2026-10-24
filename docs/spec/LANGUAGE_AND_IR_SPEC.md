---
status: Normative
classification: Normative
authority: High-level DSL and multi-level IR specification; defers public Python symbols to docs/spec/PYTHON_API_SPEC.md and backend ABI details to docs/spec/RUNTIME_ABI_SPEC.md
last_updated: 2026-10-24
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

The DSL is normative as a language model. The current Python implementation can
construct Graph IR through decorators/API calls or through the textual DSL
parser. The active Python compiler carries verifier-backed object models for
Graph IR, Schedule IR, Tile IR, and Apple/ROCm Target IR artifact lowering.

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

> **Note 2026-10-24.** The scalar terminal list above diverges from the
> canonical dtype set in `python/tessera/dtype.py` /
> `docs/reference/tessera_tensor_attributes.md` (Decision #15a, which wins):
> `tf32` is rejected as a storage dtype there (model it as
> `numeric_policy.math_mode="tf32"` on `fp32`). `uint64`/`uint32`/`uint16`/
> `uint8` are reserved names whose status is `planned_gated`, not
> first-class. The terminal list is left as specified pending a spec
> decision on `tf32`.

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

The Python effect lattice and C++ `EffectAnnotationPass` both implement this
8-level order. Current lowering paths may still support only a subset of
effectful operations, but emitted `tessera.effect` attributes use these names.

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

Selected operations (the Graph IR dialect is `tessera`; op names are
`tessera.<op>` as declared in `src/compiler/ir/TesseraOps.td`):

```mlir
%y = tessera.matmul %a, %b
%p = tessera.softmax %x {axis = -1}
%f = tessera.fft %x
%c = tessera.kv_cache.create {eviction = "rolling_window"}
%o = tessera.flash_attn %q, %k, %v, %c
%r = tessera.all_reduce %x {axis = "dp", op = "sum"}
%a = tessera.arch.parameter {size = 4, dtype = "fp32"}
%g = tessera.arch.gumbel_softmax %a {temperature = 4.0}
%m = tessera.arch.mixed %x, %g {candidates = ["flash", "performer", "gmlp"]}
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

Selected operations and Python object-model coverage:

```mlir
schedule.mesh.define {dims = [2, 4], axis_names = ["dp", "tp"]}
schedule.mesh.region {mesh = @M, axis = "dp"} {
  schedule.pipeline.region {schedule = "gpipe", micro_batches = 2} {
    schedule.stage {devices = ["gpu:0"]} {
      schedule.yield
    }
    schedule.yield
  }
  schedule.yield
}
%t = schedule.tile %x {m = 128, n = 128, k = 64}
%m = schedule.prefetch %t {into = "shared", overlap = "compute"}
%a = schedule.artifact {hash = "..."}
%k = schedule.knob %p {name = "tile_m", choices = [64, 128]}
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

Selected operations and active artifact surface:

```mlir
%sa = tile.alloc_shared %desc : memref<128x64xbf16, shared>
tile.async_copy %global, %shared {stage = 0, vector = 16}
tile.wait_async {stage = 0}
%bar = tile.mbarrier.alloc {count = 1, scope = "block"}
%tok = tile.mbarrier.arrive_expect_tx %bar {bytes = 16384, semantics = "release", scope = "block"}
%ok = tile.mbarrier.try_wait %bar, %tok
%mm = tile.mma %a, %b, %c {accum = "fp32"}
tessera_attn.online_softmax %scores {policy = "safe"}
tile.barrier
```

> **Note (2026-08-10, updated 2026-10-24).** The `tessera.queue.*` forms that
> used to appear in this listing are gone from both surfaces. The `tessera.queue`
> MLIR dialect was deleted 2026-08-10 (Decisions #29/#31), and the Python Tile
> IR verifier (`python/tessera/compiler/tile_ir.py`) now rejects any
> `tessera.queue.*` op as a forbidden legacy compatibility op
> (`TILE_IR_LEGACY_QUEUE`). Async ordering is expressed by SSA token lineage:
> on the MLIR side `WarpSpecializationPass` synchronizes warp roles through
> `!tile.pipeline_state` + `!tile.async_token` SSA chains; see
> `LOWERING_PIPELINE_SPEC.md` §3.7.

> **Note 2026-10-24.** The MLIR Tile dialect
> (`src/compiler/ir/include/Tessera/Dialect/Tile/TileOps.td`) spells the
> transaction-barrier ops `tile.mbarrier.init` (with `slots` / `phase_bits`
> attributes, not `tile.mbarrier.alloc` with `count` / `scope`),
> `tile.mbarrier.arrive_expect_tx` (`slot`, `bytes`; no `semantics` / `scope`
> attribute), `tile.mbarrier.try_wait`, and `tile.mbarrier.wait`. It has no
> `tile.barrier`; execution barriers are `tile.cta_sync` (CTA-wide) and
> `tile.s_barrier` (workgroup/wave). `tile.alloc_shared` is recognized by
> name in the arena/reuse passes (`TileBufferArenaPass`,
> `TileBufferReusePass`) rather than declared in that ODS file. The listing
> above is kept as the specified vocabulary; the release/scope semantics it
> states are not carried as attributes by the current MLIR ops.

Hopper forward rule: mbarrier transaction barriers are available for NVIDIA
targets whose target profile carries the `mbarrier_arrive_tx` feature
(sm_90, sm_100, and sm_120 in `python/tessera/compiler/capabilities.py`).
Gate on that feature flag, not on an `isa >= SM_90` ordering, which reads as
excluding sm_120 (CLAUDE.md Decision #1, corrected 2026-08-30). They are required for TMA-style asynchronous copy
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

Apple lowering includes toolchain-free Target IR artifacts for CPU and GPU
(Decision #19: parseable and verifiable without the vendor toolchain or
device; ops may name the machine):

```text
tessera_apple.cpu.accelerate_gemm, tessera_apple.cpu.vector_reduce,
tessera_apple.cpu.vector_op, tessera_apple.gpu.metal_kernel,
tessera_apple.gpu.dispatch, tessera_apple.diagnostic
```

ROCm lowering includes toolchain-free Target IR artifacts (MFMA is CDNA-only;
the fleet's RDNA parts use `tessera_rocm.wmma`):

```text
tessera_rocm.mfma, tessera_rocm.wmma, tessera_rocm.async_copy, tessera_rocm.wait,
tessera_rocm.elementwise, tessera.target.diagnostic
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
%C = tessera.matmul %A, %B
  : (tensor<1024x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16>
```

Schedule IR:

```mlir
schedule.mesh.define {dims = [1], axis_names = ["local"]}
%Ct = schedule.tile %C {m = 128, n = 128, k = 64}
schedule.pipeline.region {schedule = "gpipe", micro_batches = 1} {
  schedule.stage {devices = ["cpu:0"]} {
    schedule.yield
  }
  schedule.yield
}
```

Tile IR:

```mlir
tile.async_copy %A_global, %A_shared {stage = 0, vector = 16}
%fragC = tile.mma %fragA, %fragB, %fragC {m = 16, n = 16, k = 16, accum = "fp32"}
tile.wait_async {stage = 0}
```

## 14. References

- `docs/spec/MEMORY_MODEL_SPEC.md`
- `docs/guides/Tessera_Tensor_Layout_And_Data_Movement_Guide.md`
- `docs/spec/GRAPH_IR_SPEC.md`
- `docs/spec/TARGET_IR_SPEC.md`
- `docs/spec/PYTHON_API_SPEC.md`
