//===- TesseraNVIDIADialect.td - Tessera NVIDIA Target IR ----*- tablegen -*-===//
//
// Hardware-free NVIDIA Target IR contract dialect (Decision #19).
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_NVIDIA_DIALECT
#define TESSERA_NVIDIA_DIALECT

include "mlir/IR/OpBase.td"

def TesseraNVIDIADialect : Dialect {
  let name = "tessera_nvidia";
  let cppNamespace = "::tessera::nvidia";
  let summary = "Tessera NVIDIA Target IR contract dialect";
  let description = [{
    Hardware-free NVIDIA Target IR contracts spanning Hopper (`wgmma`),
    datacenter Blackwell (`tcgen05_mma` + TMEM), and consumer Blackwell
    (warp-level `mma_sync`) matmul lowering, plus the `tma_async_copy` /
    `mbarrier` async companions.

    Decision #19: a backend exposes a *typed*, hardware-free Target IR dialect
    between Tile IR and hardware-specific (NVVM) lowering. Until now this
    dialect was `isExtensible` with zero registered ops, so `LowerTileToNVIDIA`
    built generic operations that no verifier constrained. These op definitions
    give it a real typed surface: `LowerTileToNVIDIA` still populates them via
    the generic `OperationState` builders, but they now round-trip, verify, and
    parse without `--allow-unregistered-dialect`. `LowerNVIDIAToNVVM` rewrites
    each to its NVVM artifact marker.
  }];

  // Plain attribute storage (not op-properties) so the ops round-trip and the
  // generic `OperationState` builders in NVIDIALowering.cpp populate the
  // attributes without the `getProperties()` machinery — mirrors the Apple
  // reference dialect (TesseraAppleOps.td).
  let usePropertiesForAttributes = 0;
}

// v1 typed contract. Each op carries the region's values as variadic operands
// and yields variadic results; the specialization metadata (arch / shape /
// dtype / warpgroup / …) rides as optional attributes. The attribute set is the
// union the lowering can emit — an op ignores those it does not use — which
// types the previously-extensible dialect while tolerating the per-SM-branch
// attribute combinations `LowerTileToNVIDIA` produces. Ops carry no side-effect
// trait so generic DCE never drops a marker whose result is unused.
class TesseraNVIDIA_Op<string mnemonic, list<Trait> traits = []> :
    Op<TesseraNVIDIADialect, mnemonic, traits> {
  let arguments = (ins
      Variadic<AnyType>:$inputs,
      OptionalAttr<StrAttr>:$arch,
      OptionalAttr<StrAttr>:$source,
      OptionalAttr<StrAttr>:$shape,
      OptionalAttr<StrAttr>:$dtype_ab,
      OptionalAttr<StrAttr>:$dtype_c,
      OptionalAttr<StrAttr>:$accum,
      OptionalAttr<StrAttr>:$src_space,
      OptionalAttr<StrAttr>:$dst_space,
      OptionalAttr<BoolAttr>:$block_scaled,
      OptionalAttr<I64Attr>:$warpgroup,
      OptionalAttr<I64Attr>:$cta_group,
      OptionalAttr<I64Attr>:$columns,
      OptionalAttr<I64Attr>:$bytes);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = [{
    $inputs attr-dict `:` functional-type($inputs, $outputs)
  }];
}

def NVIDIA_MmaSyncOp : TesseraNVIDIA_Op<"mma_sync"> {
  let summary = "Consumer Blackwell (sm_120) warp-level mma.sync contract";
}
def NVIDIA_WgmmaOp : TesseraNVIDIA_Op<"wgmma"> {
  let summary = "Hopper (sm_90a) warpgroup wgmma contract";
}
def NVIDIA_Tcgen05MmaOp : TesseraNVIDIA_Op<"tcgen05_mma"> {
  let summary = "Datacenter Blackwell (sm_100a) tcgen05 MMA contract";
}
def NVIDIA_WmmaOp : TesseraNVIDIA_Op<"wmma"> {
  let summary = "Pre-Hopper warp-level wmma contract";
}
def NVIDIA_TmaAsyncCopyOp : TesseraNVIDIA_Op<"tma_async_copy"> {
  let summary = "TMA cp.async.bulk.tensor async-copy contract (sm_90+)";
}
def NVIDIA_MbarrierOp : TesseraNVIDIA_Op<"mbarrier"> {
  let summary = "mbarrier arrive/wait contract";
}
def NVIDIA_TmemAllocOp : TesseraNVIDIA_Op<"tmem_alloc"> {
  let summary = "Blackwell TMEM allocation contract (sm_100a)";
}
def NVIDIA_TmemLoadOp : TesseraNVIDIA_Op<"tmem_load"> {
  let summary = "Blackwell TMEM load contract (sm_100a)";
}
def NVIDIA_TmemStoreOp : TesseraNVIDIA_Op<"tmem_store"> {
  let summary = "Blackwell TMEM store contract (sm_100a)";
}
def NVIDIA_CudaKernelOp : TesseraNVIDIA_Op<"cuda_kernel"> {
  let summary = "Generic CUDA kernel artifact contract";
}

#endif // TESSERA_NVIDIA_DIALECT
