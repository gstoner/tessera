//===- TesseraAppleOps.td - Apple Silicon Target IR -------*- tablegen -*-===//

#ifndef TESSERA_TARGET_APPLE_OPS
#define TESSERA_TARGET_APPLE_OPS

include "mlir/IR/OpBase.td"
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
  let usePropertiesForAttributes = 0;
}

class TesseraApple_Op<string mnemonic, list<Trait> traits = []> :
    Op<TesseraApple_Dialect, mnemonic, traits>;

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
                       StrAttr:$kind);
  let assemblyFormat = "attr-dict";
}

def AppleGPU_KVCacheGPUOp : TesseraApple_Op<"gpu.kv_cache_gpu"> {
  let summary = "Apple GPU KV-cache op artifact (handle-shaped)";
  let arguments = (ins StrAttr:$source,
                       StrAttr:$result,
                       I64Attr:$ordinal,
                       OptionalAttr<StrAttr>:$framework,
                       OptionalAttr<StrAttr>:$abi,
                       StrAttr:$kind);
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
def AppleCPU_CallOp : TesseraApple_Op<"cpu.call", [Pure]> {
  let summary = "Value-producing Apple CPU runtime call (Accelerate/LAPACK)";
  let arguments = (ins Variadic<AnyType>:$operands,
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
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

// Apple GPU value call — dispatches to a custom MSL kernel C ABI symbol.
def AppleGPU_KernelCallOp : TesseraApple_Op<"gpu.kernel_call", [Pure]> {
  let summary = "Value-producing Apple GPU MSL kernel call";
  let arguments = (ins Variadic<AnyType>:$operands,
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
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

// Apple GPU value call — dispatches through an authored .mtlpackage (PK8 lane).
def AppleGPU_PackageCallOp : TesseraApple_Op<"gpu.package_call", [Pure]> {
  let summary = "Value-producing Apple GPU packaged (.mtlpackage) call";
  let arguments = (ins Variadic<AnyType>:$operands,
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
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
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

#endif // TESSERA_TARGET_APPLE_OPS
