//===- TesseraAppleOps.td - Apple Silicon Target IR -------*- tablegen -*-===//

#ifndef TESSERA_TARGET_APPLE_OPS
#define TESSERA_TARGET_APPLE_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TesseraApple_Dialect : Dialect {
  let name = "tessera_apple";
  let cppNamespace = "::tessera::apple";
  let summary = "Tessera Target IR for Apple Silicon CPU and GPU artifacts";
  let description = [{
    The Apple target dialect is a hardware-free artifact contract. CPU ops map
    to Accelerate/vecLib/BNNS-style calls. GPU ops map to Metal/MPS-style kernel
    and dispatch artifacts.
  }];

  // Use plain attributes (not op-properties storage). Keeps the dialect
  // header-only and round-trippable without the Op<> getProperties() machinery.
  let useDefaultTypePrinterParser = 1;
}

// A `simdgroup_matrix` register value.
//
// Metal's simdgroup matrix is an opaque value distributed across the 32 lanes
// of a SIMD-group, not a per-thread array -- modelling it as an SSA type is
// what lets the verifier reject a multiply whose operands never came from a
// `simdgroup_load`. Mirrors `!tessera_x86.tile` for AMX and
// `!tessera_rocm.token` for async copies.
//
// The element type is carried; the 8x8 shape is not, because Apple7 supports
// exactly one simdgroup-matrix shape and encoding a single legal value in the
// type would be ceremony. The MMA op verifies it instead, so a future shape
// fails closed at the operation rather than silently type-checking.
def Apple_SimdgroupMatrixType : TypeDef<TesseraApple_Dialect, "SimdgroupMatrix"> {
  let mnemonic = "simdgroup_matrix";
  let summary = "An 8x8 Metal simdgroup_matrix register value";
  let parameters = (ins "::mlir::Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
  // Metal's simdgroup_matrix exists for half, bfloat and float. An
  // unconstrained parameter accepted `simdgroup_matrix<i32>`, which has no
  // Metal spelling and no MMA -- it verified and round-tripped, then had
  // nothing to lower to. Constraining the type is what makes the load/store
  // element-match check meaningful rather than a comparison between two
  // arbitrary types.
  let genVerifyDecl = 1;
}

class TesseraApple_Op<string mnemonic, list<Trait> traits = []> :
    Op<TesseraApple_Dialect, mnemonic, traits>;

class Apple_EnumStrAttr<list<string> cases, string summaryText>
    : StringBasedAttr<
        CPred<"::llvm::is_contained(::llvm::ArrayRef<::llvm::StringRef>({"
              # !interleave(!foreach(c, cases, "\"" # c # "\""), ", ") #
              "}), ::llvm::cast<::mlir::StringAttr>($_self).getValue())">,
        summaryText>;
def Apple_KVCacheKindAttr
    : Apple_EnumStrAttr<["tessera.kv_cache.create", "tessera.kv_cache.append",
                         "tessera.kv_cache.prune", "tessera.kv_cache.read"],
                        "canonical KV-cache operation kind">;

//===----------------------------------------------------------------------===//
// Apple GPU machine primitives.
//
// Everything else this dialect declares is a *dispatch container* -- msl_kernel,
// mps_matmul, dispatch, kernel_call -- which is why the MLIR pipeline could
// only ever say "call this kernel" and never express what a kernel does. The
// machine vocabulary lived entirely in `emit/apple_msl.py`, which is the
// two-disconnected-compilers seam CLAUDE.md names for Apple.
//
// These are the peers of `nvvm.mma.sync` and `rocdl.mfma`: they name the
// machine. Per Decision #19 as amended, every contract attribute below is
// REQUIRED, not an optional bag -- an `OptionalAttr` contract verifies with the
// field unset and so fails open, which is what the membership audit found
// across most of the existing Target IR (38 of 138 ops require anything).
//===----------------------------------------------------------------------===//

def Apple_MemoryScopeAttr
    : Apple_EnumStrAttr<["none", "threadgroup", "device", "threadgroup_device"],
        "Metal mem_flags scope the barrier orders against">;

def Apple_AddressSpaceAttr
    : Apple_EnumStrAttr<["threadgroup", "device"],
        "Metal address space the matrix is loaded from or stored to">;

def Apple_MmaStorageAttr
    : Apple_EnumStrAttr<["f16", "bf16", "f32"],
        "simdgroup_matrix input element type">;

def AppleGPU_ThreadgroupAllocOp : TesseraApple_Op<"gpu.threadgroup_alloc"> {
  let summary = "Allocate a threadgroup-memory tile against a declared budget";
  let description = [{
    Threadgroup memory is the staging area a tiled matmul needs and the reason
    ragged shapes work at all: Metal's `simdgroup_load` has **no bounds
    predicate**, so out-of-range elements cannot be masked at the load. They
    are masked when the tile is *copied in* — `As[e] = (gr < M && gk < K) ?
    A[...] : 0` — and the MMA then reads a staged tile that is always fully
    in range. Zero padding is exact: a zero operand contributes nothing to the
    dot product, so every valid output element is unaffected.

    `budget_bytes` is the target's per-threadgroup limit, carried **in the IR**
    rather than compiled into the verifier. The limit is a device property
    (`[MTLDevice maxThreadgroupMemoryLength]` — 32768 on Apple7, queried, not
    assumed), and baking it into C++ would make the verifier's answer depend on
    the host it runs on. That would break the property Decision #19 actually
    buys: this layer must be checkable on any machine, including one with no
    Metal device. Carrying it as an attribute keeps the check exact and the
    compiler host-free, and it makes the budget a fact a reader can see.

    Exceeding the budget is a pipeline-creation failure at runtime, far from
    the pass that caused it, which is why it fails here instead.
  }];
  let arguments = (ins I64Attr:$elements, I64Attr:$budget_bytes);
  let results = (outs AnyMemRef:$tile);
  let assemblyFormat = "attr-dict `:` type($tile)";
  let hasVerifier = 1;
}

def AppleGPU_SimdgroupFillOp : TesseraApple_Op<"gpu.simdgroup_fill", [Pure]> {
  let summary = "A constant-filled simdgroup_matrix (make_filled_simdgroup_matrix)";
  let description = [{
    `make_filled_simdgroup_matrix<T, 8, 8>(v)`.

    The K-loop accumulator has to start somewhere, and the alternative --
    loading it from a zeroed buffer -- is both an extra memory round-trip and
    a different program: it makes the accumulator's initial value depend on
    memory the compiler would then have to prove was zeroed. Metal provides
    the fill directly and `emit/apple_msl.py` uses it, so the IR says the same
    thing.

    `Pure` because the result depends only on the attributes: it reads no
    memory and has no ordering requirement, which lets CSE share one zero
    accumulator across tiles.
  }];
  let arguments = (ins F32Attr:$value);
  let results = (outs Apple_SimdgroupMatrixType:$matrix);
  let assemblyFormat = "attr-dict `:` type($matrix)";
}

def AppleGPU_SimdgroupLoadOp : TesseraApple_Op<"gpu.simdgroup_load"> {
  let summary = "Load an 8x8 simdgroup_matrix from strided memory";
  let description = [{
    `simdgroup_load(dst, src, elements_per_row)`.

    `leading_dim` is the source's **row stride in elements**, and it is the
    single most dangerous number in this op: Metal reads row *r* at
    `src + r * leading_dim`, so a stride smaller than the matrix width makes
    consecutive rows overlap and the MMA silently consumes the wrong elements.
    There is no fault -- the kernel runs and the numbers are wrong. The
    verifier therefore requires `leading_dim >= 8`.

    `space` is required because `threadgroup` and `device` loads have different
    visibility rules; a `device` load that should have been `threadgroup` reads
    memory the staging barrier never ordered.
  }];
  let arguments = (ins AnyMemRef:$source, Index:$offset,
                       I64Attr:$leading_dim,
                       Apple_AddressSpaceAttr:$space);
  let results = (outs Apple_SimdgroupMatrixType:$matrix);
  let assemblyFormat = [{
    $source `,` $offset attr-dict `:` type($source) `,` type($offset)
    `->` type($matrix)
  }];
  let hasVerifier = 1;
}

def AppleGPU_SimdgroupStoreOp : TesseraApple_Op<"gpu.simdgroup_store"> {
  let summary = "Store an 8x8 simdgroup_matrix to strided memory";
  let description = [{
    `simdgroup_store(m, dst, elements_per_row)`. `leading_dim` carries the same
    row-stride contract as the load, and the same silent-corruption failure
    when it is too small -- here it overwrites neighbouring rows instead of
    reading them.
  }];
  let arguments = (ins Apple_SimdgroupMatrixType:$matrix,
                       AnyMemRef:$destination, Index:$offset,
                       I64Attr:$leading_dim,
                       Apple_AddressSpaceAttr:$space);
  let assemblyFormat = [{
    $matrix `,` $destination `,` $offset attr-dict `:` type($matrix) `,`
    type($destination) `,` type($offset)
  }];
  let hasVerifier = 1;
}

def AppleGPU_SimdgroupMatmulOp : TesseraApple_Op<"gpu.simdgroup_matmul"> {
  let summary = "8x8x8 simdgroup matrix multiply-accumulate: d = a*b + c";
  let description = [{
    `simdgroup_multiply_accumulate(d, a, b, c)` computes **d = a·b + c** over
    8x8 operands, so this is a full 8x8x8 MMA with K = 8.

    Two numerical facts are contract, not convention, and both are verified:

    * **The accumulator is always fp32**, whatever the inputs are. Metal's
      accumulate form takes `simdgroup_float8x8` and `emit/apple_msl.py`
      relies on it -- "the accumulator is always fp32, so the epilogue sees
      full-precision matmul results". An f16 accumulator would change the
      numerics of every kernel built on this op, so `c` and the result must be
      f32 and an f16 accumulator is rejected rather than accepted and
      quietly re-rounded. This is Decision #15a's storage-versus-accumulator
      split expressed at the machine level.
    * **Both inputs carry the same storage type**, matching `storage`. Mixing
      an f16 and an f32 operand is not a Metal simdgroup MMA; it would have to
      be a convert plus an MMA, and accepting it here would hide that
      conversion from the numerics the epilogue reasons about.

    The 8x8x8 shape is verified rather than assumed. Apple7 supports exactly
    one simdgroup-matrix shape; stating it means a future part fails closed
    here instead of lowering to a `simdgroup_matrix<T, R, C>` the hardware
    does not have.
  }];
  let arguments = (ins Apple_SimdgroupMatrixType:$a,
                       Apple_SimdgroupMatrixType:$b,
                       Apple_SimdgroupMatrixType:$c,
                       Apple_MmaStorageAttr:$storage,
                       I64Attr:$m, I64Attr:$n, I64Attr:$k);
  let results = (outs Apple_SimdgroupMatrixType:$d);
  let assemblyFormat = [{
    $a `,` $b `,` $c attr-dict `:` type($a) `,` type($b) `,` type($c)
    `->` type($d)
  }];
  let hasVerifier = 1;
}

def AppleGPU_ThreadgroupBarrierOp : TesseraApple_Op<"gpu.threadgroup_barrier"> {
  let summary = "threadgroup_barrier with an explicit Metal memory scope";
  let description = [{
    `threadgroup_barrier(mem_flags::...)`.

    The scope is required and never defaults. A barrier is only a barrier for
    the memory it names: staging a tile into threadgroup memory and then
    ordering with `mem_none` compiles, runs, and races. Decision #21a -- the
    scope selects semantics, so absence must fail closed rather than pick the
    common case.
  }];
  let arguments = (ins Apple_MemoryScopeAttr:$memory_scope);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def AppleCPU_AccelerateGemmOp : TesseraApple_Op<"cpu.accelerate_gemm", [Pure]> {
  let summary = "Accelerate/CBLAS GEMM artifact for Apple Silicon CPU";
  let arguments = (ins StrAttr:$source,
                       StrAttr:$result,
                       I64Attr:$ordinal,
                       OptionalAttr<StrAttr>:$framework,
                       OptionalAttr<StrAttr>:$abi,
                       OptionalAttr<StrAttr>:$dtype);
  let assemblyFormat = "attr-dict";
}

def AppleCPU_VectorOp : TesseraApple_Op<"cpu.vector_op", [Pure]> {
  let summary = "Accelerate/vecLib vector operation artifact";
  let arguments = (ins StrAttr:$source,
                       StrAttr:$result,
                       I64Attr:$ordinal,
                       OptionalAttr<StrAttr>:$framework,
                       OptionalAttr<StrAttr>:$abi,
                       OptionalAttr<StrAttr>:$pattern,
                       OptionalAttr<StrAttr>:$dtype);
  let assemblyFormat = "attr-dict";
}

def AppleCPU_VectorReduceOp : TesseraApple_Op<"cpu.vector_reduce", [Pure]> {
  let summary = "Accelerate/vDSP reduction artifact";
  let arguments = (ins StrAttr:$source,
                       StrAttr:$result,
                       I64Attr:$ordinal,
                       OptionalAttr<StrAttr>:$framework,
                       OptionalAttr<StrAttr>:$abi,
                       OptionalAttr<StrAttr>:$dtype);
  let assemblyFormat = "attr-dict";
}

def AppleGPU_MetalKernelOp : TesseraApple_Op<"gpu.metal_kernel"> {
  let summary = "Metal compute kernel artifact";
  let arguments = (ins StrAttr:$source,
                       StrAttr:$result,
                       I64Attr:$ordinal,
                       OptionalAttr<StrAttr>:$kernel,
                       OptionalAttr<StrAttr>:$framework,
                       OptionalAttr<StrAttr>:$dtype,
                       OptionalAttr<StrAttr>:$grid,
                       OptionalAttr<StrAttr>:$threadgroup,
                       OptionalAttr<StrAttr>:$threadgroup_memory,
                       OptionalAttr<StrAttr>:$temporary_memory,
                       OptionalAttr<StrAttr>:$status);
  let assemblyFormat = "attr-dict";
}

def AppleGPU_DispatchOp : TesseraApple_Op<"gpu.dispatch"> {
  let summary = "Metal command queue dispatch artifact";
  let arguments = (ins I64Attr:$ordinal,
                       OptionalAttr<StrAttr>:$queue,
                       OptionalAttr<StrAttr>:$artifact,
                       OptionalAttr<StrAttr>:$execution_mode);
  let assemblyFormat = "attr-dict";
}

// Phase 8.3 — MPS-shaped Target IR ops. These describe the GPU-side analogue of
// the Apple CPU Accelerate ops: hardware-shaped but not hardware-bound. The
// AppleGPUToMPS lowering pass + tessera_apple_gpu_runtime.mm runtime turn the
// matmul op into an executable MTLDevice + MPSMatrixMultiplication call. The
// softmax and dispatch ops are artifact-only in 8.3 — Phase 8.4 broadens the
// runtime envelope.

def AppleGPU_MPSMatmulOp : TesseraApple_Op<"gpu.mps_matmul", [Pure]> {
  let summary = "Metal Performance Shaders matrix multiplication artifact";
  let arguments = (ins StrAttr:$source,
                       StrAttr:$result,
                       I64Attr:$ordinal,
                       OptionalAttr<StrAttr>:$framework,
                       OptionalAttr<StrAttr>:$abi,
                       OptionalAttr<StrAttr>:$dtype,
                       OptionalAttr<I64Attr>:$m,
                       OptionalAttr<I64Attr>:$n,
                       OptionalAttr<I64Attr>:$k);
  let assemblyFormat = "attr-dict";
}

def AppleGPU_MPSSoftmaxOp : TesseraApple_Op<"gpu.mps_softmax", [Pure]> {
  let summary = "Metal Performance Shaders softmax artifact";
  let arguments = (ins StrAttr:$source,
                       StrAttr:$result,
                       I64Attr:$ordinal,
                       OptionalAttr<StrAttr>:$framework,
                       OptionalAttr<StrAttr>:$abi,
                       OptionalAttr<StrAttr>:$dtype,
                       OptionalAttr<I64Attr>:$rows,
                       OptionalAttr<I64Attr>:$cols,
                       OptionalAttr<I64Attr>:$axis);
  let assemblyFormat = "attr-dict";
}

def AppleGPU_MPSDispatchOp : TesseraApple_Op<"gpu.mps_dispatch"> {
  let summary = "Metal command queue dispatch for MPS-encoded artifacts";
  let arguments = (ins I64Attr:$ordinal,
                       OptionalAttr<StrAttr>:$queue,
                       OptionalAttr<StrAttr>:$framework,
                       OptionalAttr<StrAttr>:$execution_mode);
  let assemblyFormat = "attr-dict";
}

// Phase 8.4 — Custom MSL kernel artifact. Carries the kernel's MSL source
// inline as a StringAttr so the Target IR module is a self-contained,
// replayable record. The runtime compiles via
// `[device newLibraryWithSource:options:error:]` and caches the resulting
// MTLLibrary by `cache_key` (sha256 of msl_source).
//
// Used for ops that MPS doesn't ship a kernel for, or where Tessera wants a
// purpose-built custom kernel even when MPS has coverage (flash-attention,
// rope, fused softmax variants, etc.).

def AppleGPU_MSLKernelOp : TesseraApple_Op<"gpu.msl_kernel", [Pure]> {
  let summary = "Custom Metal Shading Language kernel artifact";
  let arguments = (ins StrAttr:$source,
                       StrAttr:$result,
                       I64Attr:$ordinal,
                       StrAttr:$entry_point,
                       StrAttr:$msl_source,
                       OptionalAttr<StrAttr>:$framework,
                       OptionalAttr<StrAttr>:$dtype,
                       OptionalAttr<StrAttr>:$grid,
                       OptionalAttr<StrAttr>:$threadgroup,
                       OptionalAttr<StrAttr>:$cache_key);
  let assemblyFormat = "attr-dict";
}

def Apple_TargetDiagnosticOp : TesseraApple_Op<"diagnostic"> {
  let summary = "Stable Apple target diagnostic artifact";
  let arguments = (ins StrAttr:$source,
                       StrAttr:$result,
                       I64Attr:$ordinal,
                       StrAttr:$severity,
                       StrAttr:$reason);
  let assemblyFormat = "attr-dict";
}

// kv_cache_coverage_matrix.md (2026-05-10) — KV-cache lowering target
// for both CPU and GPU sides of the Apple backend. The TileToApple
// pass emits these in place of the historical
// `tessera_apple.diagnostic("unsupported")`. The Python runtime
// dispatches on (target, kind) to the existing `KVCacheHandle`
// reference path; a future native Metal kernel for the in-cache
// score-matrix decode is gated on Phase G.
// Distinct mnemonics (cpu.kv_cache_cpu / gpu.kv_cache_gpu) so the
// auto-generated C++ class names don't collide on `KVCacheOp`. Op
// names in the IR remain user-friendly via the dialect prefix.
def AppleCPU_KVCacheCPUOp : TesseraApple_Op<"cpu.kv_cache_cpu", [Pure]> {
  let summary = "Apple CPU KV-cache op artifact (handle-shaped)";
  let arguments = (ins StrAttr:$source,
                       StrAttr:$result,
                       I64Attr:$ordinal,
                       OptionalAttr<StrAttr>:$framework,
                       OptionalAttr<StrAttr>:$abi,
                       Apple_KVCacheKindAttr:$kind);
  let assemblyFormat = "attr-dict";
}

def AppleGPU_KVCacheGPUOp : TesseraApple_Op<"gpu.kv_cache_gpu"> {
  let summary = "Apple GPU KV-cache op artifact (handle-shaped)";
  let arguments = (ins StrAttr:$source,
                       StrAttr:$result,
                       I64Attr:$ordinal,
                       OptionalAttr<StrAttr>:$framework,
                       OptionalAttr<StrAttr>:$abi,
                       Apple_KVCacheKindAttr:$kind);
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Value-producing Target IR ops (Apple Value Target IR sprint, 2026-06-03)
//
// Unlike the attribute-only artifact ops above (which are inspection/dashboard
// metadata), these carry the real SSA tensor operands and results, so the
// value-preserving `-full` pipelines can `replaceOp` the Tile op with a target
// op that actually produces its values.  Each names the C ABI `symbol` the
// runtime dispatches to (the seam-closure contract).  `status` is "executable"
// when a runtime dispatcher exists, else "artifact" (a value op may still be
// emitted for ops whose values must flow even if execution is pending).
//===----------------------------------------------------------------------===//

// Apple CPU value call — dispatches to an Accelerate/LAPACK/BNNS C ABI symbol.
// W1.1 step 6 — the value domain of an Apple runtime call.
//
// `Variadic<AnyType>` on a call into Accelerate / a compiled MSL kernel is
// unexplained: those entry points take BUFFERS OF DATA. It admitted things that
// are meaningless at that boundary -- a `!tile.fragment`, an async token, a
// symbol ref -- and the ODS said nothing about it.
//
// Scalar parameters are NOT operands on these ops; they ride as attributes
// (`lower`, `trans`, `unit_diag`, `full_matrices`, `dtype`, ...), which is why
// this can be buffers-only without losing expressiveness.
//
// Deliberately NOT applied to the control-flow ops: `gpu.control_{if,loop,while}`
// carry `iter_args`, and loop-carried values are legitimately polymorphic.
// `AnyType` there is explained, not unexplained -- see the note on those ops.
def AppleTarget_Buffer : AnyTypeOf<[AnyRankedTensor, AnyMemRef],
                                   "a ranked tensor or memref data buffer">;

def AppleCPU_CallOp : TesseraApple_Op<"cpu.call", [Pure]> {
  let summary = "Value-producing Apple CPU runtime call (Accelerate/LAPACK)";
  let arguments = (ins Variadic<AppleTarget_Buffer>:$operands,
                       StrAttr:$op_kind,
                       StrAttr:$symbol,
                       OptionalAttr<StrAttr>:$abi,
                       OptionalAttr<StrAttr>:$status,
                       OptionalAttr<StrAttr>:$dtype,
                       OptionalAttr<StrAttr>:$framework,
                       OptionalAttr<StrAttr>:$argument_layout,
                       // Linalg semantic attrs (Value Target IR sprint 3) —
                       // copied from the Graph/Tile source so runtime dispatch
                       // never silently assumes a default.
                       OptionalAttr<BoolAttr>:$lower,
                       OptionalAttr<BoolAttr>:$trans,
                       OptionalAttr<BoolAttr>:$unit_diag,
                       OptionalAttr<BoolAttr>:$full_matrices);
  let results = (outs Variadic<AppleTarget_Buffer>:$results);
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

// Apple GPU value call — dispatches to a custom MSL kernel C ABI symbol.
def AppleGPU_KernelCallOp : TesseraApple_Op<"gpu.kernel_call", [Pure]> {
  let summary = "Value-producing Apple GPU MSL kernel call";
  let arguments = (ins Variadic<AppleTarget_Buffer>:$operands,
                       StrAttr:$op_kind,
                       StrAttr:$symbol,
                       OptionalAttr<StrAttr>:$abi,
                       OptionalAttr<StrAttr>:$status,
                       OptionalAttr<StrAttr>:$dtype,
                       OptionalAttr<StrAttr>:$framework,
                       OptionalAttr<StrAttr>:$argument_layout,
                       // Linalg semantic attrs (Value Target IR sprint 3) —
                       // copied from the Graph/Tile source so runtime dispatch
                       // never silently assumes a default.
                       OptionalAttr<BoolAttr>:$lower,
                       OptionalAttr<BoolAttr>:$trans,
                       OptionalAttr<BoolAttr>:$unit_diag,
                       OptionalAttr<BoolAttr>:$full_matrices);
  let results = (outs Variadic<AppleTarget_Buffer>:$results);
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

// Apple GPU value call — dispatches through an authored .mtlpackage (PK8 lane).
def AppleGPU_PackageCallOp : TesseraApple_Op<"gpu.package_call", [Pure]> {
  let summary = "Value-producing Apple GPU packaged (.mtlpackage) call";
  let arguments = (ins Variadic<AppleTarget_Buffer>:$operands,
                       StrAttr:$op_kind,
                       StrAttr:$symbol,
                       OptionalAttr<StrAttr>:$abi,
                       OptionalAttr<StrAttr>:$status,
                       OptionalAttr<StrAttr>:$dtype,
                       OptionalAttr<StrAttr>:$framework,
                       OptionalAttr<StrAttr>:$argument_layout,
                       // Linalg semantic attrs (Value Target IR sprint 3) —
                       // copied from the Graph/Tile source so runtime dispatch
                       // never silently assumes a default.
                       OptionalAttr<BoolAttr>:$lower,
                       OptionalAttr<BoolAttr>:$trans,
                       OptionalAttr<BoolAttr>:$unit_diag,
                       OptionalAttr<BoolAttr>:$full_matrices);
  let results = (outs Variadic<AppleTarget_Buffer>:$results);
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

// Phase-G G-B — bounded control-flow loop Target-IR op. Value-preserving
// (iter_args in, final carry out) so the lowering is a plain replaceOp. Carries
// the body symbol + static range + the runtime `symbol`
// (`tessera_apple_gpu_run_graph_loop_f32`). `status="artifact"`: the IR path is
// lit-verified; MLIR-driven execution off this op is a follow-on (today the
// GraphFn lane calls the runtime symbol directly).
def AppleGPU_ControlLoopOp : TesseraApple_Op<"gpu.control_loop", [Pure]> {
  let summary = "Apple GPU bounded control-flow loop (Phase-G G-B)";
  let arguments = (ins Variadic<AnyType>:$iter_args,
                       FlatSymbolRefAttr:$body,
                       I64Attr:$start, I64Attr:$stop, I64Attr:$step,
                       I64Attr:$ordinal,
                       OptionalAttr<StrAttr>:$status,
                       OptionalAttr<StrAttr>:$symbol,
                       OptionalAttr<StrAttr>:$framework,
                       // G-B.2 executable payload (carried through from
                       // tessera.control_for) — the body serialized to the
                       // run_graph_loop op-list ABI, so the lowered op dispatches
                       // via `symbol` directly off its attributes.
                       OptionalAttr<I64Attr>:$carry_arg_index,
                       OptionalAttr<DenseI32ArrayAttr>:$body_opcodes,
                       OptionalAttr<DenseI32ArrayAttr>:$body_in0,
                       OptionalAttr<DenseI32ArrayAttr>:$body_in1,
                       OptionalAttr<DenseI32ArrayAttr>:$body_iattr,
                       OptionalAttr<DenseF32ArrayAttr>:$body_fattr,
                       OptionalAttr<I64Attr>:$body_out_id);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

// W1.1 step 6 — `AnyType` on the control-flow ops is EXPLAINED, not residual.
//
// `gpu.control_{if,loop,while}` carry `iter_args`: values threaded across
// iterations of a bounded loop or across the arms of a conditional. Those are
// legitimately polymorphic -- an accumulator tensor, a scalar induction value, a
// predicate -- exactly as `scf.for`'s iter_args are. Constraining them to the
// buffer domain used by the runtime CALL ops would reject valid programs.
//
// Step 6 asks for the removal of UNEXPLAINED `AnyType`. This is the explanation,
// recorded so the next audit does not read these three as leftovers and tighten
// them. If they are ever narrowed, it must be to the set `scf` itself permits,
// not to the call-operand set.

def AppleGPU_ControlIfOp : TesseraApple_Op<"gpu.control_if", [Pure]> {
  let summary = "Apple GPU divergent control-flow if/else (Phase-G close-out C)";
  let arguments = (ins Variadic<AnyType>:$iter_args,
                       FlatSymbolRefAttr:$then_branch,
                       FlatSymbolRefAttr:$else_branch,
                       I64Attr:$flag_arg_index,
                       I64Attr:$ordinal,
                       OptionalAttr<StrAttr>:$status,
                       OptionalAttr<StrAttr>:$symbol,
                       OptionalAttr<StrAttr>:$framework,
                       // executable payload (carried through from
                       // tessera.control_if) — then/else branches serialized to
                       // the run_graph_cond op-list ABI, so the lowered op
                       // dispatches via `symbol` directly off its attributes.
                       OptionalAttr<DenseI32ArrayAttr>:$then_opcodes,
                       OptionalAttr<DenseI32ArrayAttr>:$then_in0,
                       OptionalAttr<DenseI32ArrayAttr>:$then_in1,
                       OptionalAttr<DenseI32ArrayAttr>:$then_iattr,
                       OptionalAttr<DenseF32ArrayAttr>:$then_fattr,
                       OptionalAttr<I64Attr>:$then_out_id,
                       OptionalAttr<DenseI32ArrayAttr>:$else_opcodes,
                       OptionalAttr<DenseI32ArrayAttr>:$else_in0,
                       OptionalAttr<DenseI32ArrayAttr>:$else_in1,
                       OptionalAttr<DenseI32ArrayAttr>:$else_iattr,
                       OptionalAttr<DenseF32ArrayAttr>:$else_fattr,
                       OptionalAttr<I64Attr>:$else_out_id,
                       OptionalAttr<DenseI64ArrayAttr>:$out_shape);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

def AppleGPU_ControlWhileOp : TesseraApple_Op<"gpu.control_while", [Pure]> {
  let summary = "Apple GPU bounded while-loop (Phase-G close-out D)";
  let arguments = (ins Variadic<AnyType>:$iter_args,
                       FlatSymbolRefAttr:$body,
                       FlatSymbolRefAttr:$cond,
                       I64Attr:$carry_arg_index,
                       I64Attr:$max_iters,
                       I64Attr:$ordinal,
                       OptionalAttr<StrAttr>:$status,
                       OptionalAttr<StrAttr>:$symbol,
                       OptionalAttr<StrAttr>:$framework,
                       OptionalAttr<DenseI32ArrayAttr>:$body_opcodes,
                       OptionalAttr<DenseI32ArrayAttr>:$body_in0,
                       OptionalAttr<DenseI32ArrayAttr>:$body_in1,
                       OptionalAttr<DenseI32ArrayAttr>:$body_iattr,
                       OptionalAttr<DenseF32ArrayAttr>:$body_fattr,
                       OptionalAttr<I64Attr>:$body_out_id,
                       OptionalAttr<DenseI32ArrayAttr>:$cond_opcodes,
                       OptionalAttr<DenseI32ArrayAttr>:$cond_in0,
                       OptionalAttr<DenseI32ArrayAttr>:$cond_in1,
                       OptionalAttr<DenseI32ArrayAttr>:$cond_iattr,
                       OptionalAttr<DenseF32ArrayAttr>:$cond_fattr,
                       OptionalAttr<I64Attr>:$cond_out_id);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

#endif // TESSERA_TARGET_APPLE_OPS

//===----------------------------------------------------------------------===//
// Structural Target-IR nodes (W0.9) -- emitted by the Python lowering before
// being declared here, so the artifact named the dialect without being it.
// Directive-style: structural markers, not value-carrying computation.
//===----------------------------------------------------------------------===//

def Apple_CPUKVCacheReadOp : TesseraApple_Op<"cpu.kv_cache_read"> {
  let summary = "Hardware-free marker for a KV-cache read lowered to Apple CPU";
  let assemblyFormat = "attr-dict";
}

def Apple_CPUMoeSolverOp : TesseraApple_Op<"cpu.moe_solver"> {
  let summary = "Hardware-free marker for the MoE solver lowered to Apple CPU";
  let assemblyFormat = "attr-dict";
}

// Profiler probe (W0.9 follow-up). `annotate_target_ir_with_probes()` emits
// this op into every target lane; it was emitted without being declared, so a
// probe-annotated module did not parse. Same defect class as the five op names
// W0.9 declared -- found via the `_copy_target_op` fix in review of PR #490.
def Apple_CPUProfilerProbeOp : TesseraApple_Op<"cpu.profiler_probe"> {
  let summary = "Intra-kernel profiler probe on the Apple CPU lane";
  let assemblyFormat = "attr-dict";
}

def Apple_GPUProfilerProbeOp : TesseraApple_Op<"gpu.profiler_probe"> {
  let summary = "Intra-kernel profiler probe on the Apple GPU lane";
  let assemblyFormat = "attr-dict";
}
