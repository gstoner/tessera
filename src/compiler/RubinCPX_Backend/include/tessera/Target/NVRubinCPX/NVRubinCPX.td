
//===- NVRubinCPX.td --------------------------------------------*- tablegen -*-===//
// Tessera Target-IR: NVIDIA Rubin CPX extensions (device kinds, caps, memspaces)
// This file depends on core Tessera Target IR (not included here).
//-------------------------------------------------------------------------------
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def Tessera_NVRubinCPX_Dialect : Dialect {
  let name = "tessera.target.cpx";
  let cppNamespace = "::tessera::target";
  let summary = "Tessera Target-IR: NVIDIA Rubin CPX extensions";
  let description = [{
    Device kinds, NVFP4/NVFP6 scalar types, capability bits, and ops for
    KV cache transport and context-phase primitives.
  }];
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

def NVFP4Type : TypeDef<Tessera_NVRubinCPX_Dialect, "NVFP4"> {
  let mnemonic = "nvfp4";
  let summary = "NVIDIA 4-bit floating-point (nvfp4)";
  let description = "Storage/computation type used by CPX tensor units.";
}

def NVFP6Type : TypeDef<Tessera_NVRubinCPX_Dialect, "NVFP6"> {
  let mnemonic = "nvfp6";
  let summary = "NVIDIA 6-bit floating-point (nvfp6)";
  let description = "Optional 6-bit float if exposed; lowers to packed lanes.";
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

def DeviceKindAttr : StrEnumAttr<"DeviceKind", "cpx_device_kind",
  [
    StrEnumAttrCase<"nvidia.rubin_cpx">,
    StrEnumAttrCase<"nvidia.rubin_gpu">
  ]> {
  let summary = "Device kind enumeration (Rubin CPX or Rubin GPU)";
}

def CapabilitiesAttr : AttrDef<Tessera_NVRubinCPX_Dialect, "Capabilities"> {
  let parameters = (ins
    I1Attr:$has_context_accel,
    I1Attr:$has_hw_venc,
    I1Attr:$has_hw_vdec,
    I1Attr:$mem_gddr7,
    I1Attr:$p2p_pcie_gen6,
    I1Attr:$link_cx9
  );
  let assemblyFormat = "`<` $has_context_accel `,` $has_hw_venc `,` $has_hw_vdec `,`"
                       "`" " $mem_gddr7 `,` $p2p_pcie_gen6 `,` $link_cx9 `>`";
}

// Memory space enum for address space tagging
def MemSpaceAttr : StrEnumAttr<"MemSpace", "cpx_memspace",
  [
    StrEnumAttrCase<"gddr7.global">,
    StrEnumAttrCase<"hbm.global">,
    StrEnumAttrCase<"nvlink.scratch">,
    StrEnumAttrCase<"host.pcie">
  ]> { let summary = "Named memory spaces relevant to CPX"; }

// KV role and layout
def KVRoleAttr : StrEnumAttr<"KVRole", "kv_role",
  [ StrEnumAttrCase<"keys">, StrEnumAttrCase<"values"> ]>;

def KVLayoutAttr : StrEnumAttr<"KVLayout", "kv_layout",
  [ StrEnumAttrCase<"row_major">, StrEnumAttrCase<"paged"> ]>;

def KVPolicyAttr : StrEnumAttr<"KVPolicy", "kv_policy",
  [ StrEnumAttrCase<"pcie+cx9">, StrEnumAttrCase<"nvlink"> ]>;

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

class Tessera_Op<string mnemonic, list<Trait> traits = []>
    : Op<Tessera_NVRubinCPX_Dialect, mnemonic, traits>;

def KVCacheOp : Tessera_Op<"kv.cache", [NoSideEffect]> {
  let summary = "Declare/mark a KV cache tensor with role/layout hints";
  let arguments = (ins AnyMemRef:$buffer, KVRoleAttr:$role, KVLayoutAttr:$layout);
  let results = (outs AnyMemRef:$out);
  let assemblyFormat = "$buffer attr-dict `:` type($buffer) `->` type($out)";
}

def KVExportOp : Tessera_Op<"kv.export", []> {
  let summary = "Export a KV page/chunk from CPX to a transport channel";
  let arguments = (ins AnyMemRef:$src, KVPolicyAttr:$policy, I64Attr:$chunk_bytes);
  let results = (outs AnyType:$token);
  let assemblyFormat = "$src `,` $policy `,` $chunk_bytes attr-dict `:` type($src)";
}

def KVImportOp : Tessera_Op<"kv.import", []> {
  let summary = "Import a KV page/chunk on the Rubin GPU side";
  let arguments = (ins AnyType:$token, AnyMemRef:$dst);
  let results = (outs UnitAttr:$done);
  let assemblyFormat = "$token `->` $dst attr-dict `:` type($dst)";
}

def KVPrefetchOp : Tessera_Op<"kv.prefetch", [NoSideEffect]> {
  let summary = "Prefetch a logical range of KV pages to overlap compute";
  let arguments = (ins AnyMemRef:$kv, I64Attr:$start_page, I64Attr:$num_pages);
  let assemblyFormat = "$kv `[` $start_page `:` $num_pages `]` attr-dict `:` type($kv)";
}

// Context-phase fused attention (rope+attn+kv-pack)
def AttnPrefillFusedOp : Tessera_Op<"attn.prefill_fused", []> {
  let summary = "Fused prefill attention with RoPE + KV packing for long context";
  let arguments = (ins AnyType:$q, AnyType:$k, AnyType:$v,
                        AnyMemRef:$kv_cache, I64Attr:$seq_len);
  let results = (outs AnyType:$o);
  let assemblyFormat = "$q `,` $k `,` $v `,` $kv_cache `,` $seq_len attr-dict "
                       "`:` type($q) `,` type($k) `,` type($v) `,` type($kv_cache) "
                       "`->` type($o)";
}

// Video decode/encode stubs (device-accelerated on CPX)
def VideoDecodeOp : Tessera_Op<"video.decode", []> {
  let summary = "Hardware-accelerated video decode on CPX";
  let arguments = (ins AnyType:$bitstream, StrAttr:$codec);
  let results = (outs AnyType:$frames);
  let assemblyFormat = "$bitstream `,` $codec attr-dict `:` type($bitstream) `->` type($frames)";
}

def VideoEncodeOp : Tessera_Op<"video.encode", []> {
  let summary = "Hardware-accelerated video encode on CPX";
  let arguments = (ins AnyType:$frames, StrAttr:$codec);
  let results = (outs AnyType:$bitstream);
  let assemblyFormat = "$frames `,` $codec attr-dict `:` type($frames) `->` type($bitstream)";
}
