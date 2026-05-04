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
}

class TesseraApple_Op<string mnemonic, list<Trait> traits = []> :
    Op<TesseraApple_Dialect, mnemonic, traits>;

def AppleCPU_AccelerateGemmOp : TesseraApple_Op<"cpu.accelerate_gemm", [Pure]> {
  let summary = "Accelerate/CBLAS GEMM artifact for Apple Silicon CPU";
  let arguments = (ins StrAttr:$source,
                       StrAttr:$result,
                       I64Attr:$ordinal,
                       OptionalAttr<StrAttr>:$abi,
                       OptionalAttr<StrAttr>:$dtype);
  let assemblyFormat = "attr-dict";
}

def AppleCPU_VectorOp : TesseraApple_Op<"cpu.vector_op", [Pure]> {
  let summary = "Accelerate/vecLib vector operation artifact";
  let arguments = (ins StrAttr:$source,
                       StrAttr:$result,
                       I64Attr:$ordinal,
                       OptionalAttr<StrAttr>:$abi);
  let assemblyFormat = "attr-dict";
}

def AppleCPU_VectorReduceOp : TesseraApple_Op<"cpu.vector_reduce", [Pure]> {
  let summary = "Accelerate/vDSP reduction artifact";
  let arguments = (ins StrAttr:$source,
                       StrAttr:$result,
                       I64Attr:$ordinal,
                       OptionalAttr<StrAttr>:$abi);
  let assemblyFormat = "attr-dict";
}

def AppleGPU_MetalKernelOp : TesseraApple_Op<"gpu.metal_kernel"> {
  let summary = "Metal compute kernel artifact";
  let arguments = (ins StrAttr:$source,
                       StrAttr:$result,
                       I64Attr:$ordinal,
                       OptionalAttr<StrAttr>:$kernel,
                       OptionalAttr<StrAttr>:$framework,
                       OptionalAttr<StrAttr>:$threadgroup_memory,
                       OptionalAttr<StrAttr>:$status);
  let assemblyFormat = "attr-dict";
}

def AppleGPU_DispatchOp : TesseraApple_Op<"gpu.dispatch"> {
  let summary = "Metal command queue dispatch artifact";
  let arguments = (ins I64Attr:$ordinal,
                       OptionalAttr<StrAttr>:$queue,
                       OptionalAttr<StrAttr>:$artifact);
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
