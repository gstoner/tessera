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

#endif // TESSERA_TARGET_APPLE_OPS
