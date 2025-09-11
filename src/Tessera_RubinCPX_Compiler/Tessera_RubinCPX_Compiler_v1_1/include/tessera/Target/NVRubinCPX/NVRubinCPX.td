
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def Tessera_NVRubinCPX_Dialect : Dialect {
  let name = "tessera.target.cpx";
  let cppNamespace = "::tessera::target";
  let summary = "Rubin CPX Target-IR extensions (v1.1)";
}

def NVFP4Type : TypeDef<Tessera_NVRubinCPX_Dialect, "NVFP4"> { let mnemonic = "nvfp4"; }
def NVFP6Type : TypeDef<Tessera_NVRubinCPX_Dialect, "NVFP6"> { let mnemonic = "nvfp6"; }

def DeviceKindAttr : StrEnumAttr<"DeviceKind","cpx_device_kind",
  [StrEnumAttrCase<"nvidia.rubin_cpx">, StrEnumAttrCase<"nvidia.rubin_gpu">]>;

def CapabilitiesAttr : AttrDef<Tessera_NVRubinCPX_Dialect, "Capabilities"> {
  let parameters = (ins I1Attr:$has_context_accel, I1Attr:$has_hw_venc, I1Attr:$has_hw_vdec,
                        I1Attr:$mem_gddr7, I1Attr:$p2p_pcie_gen6, I1Attr:$link_cx9);
  let assemblyFormat = "`<` $has_context_accel `,` $has_hw_venc `,` $has_hw_vdec `,` "
                       "$mem_gddr7 `,` $p2p_pcie_gen6 `,` $link_cx9 `>`";
}

def MemSpaceAttr : StrEnumAttr<"MemSpace","cpx_memspace",
  [StrEnumAttrCase<"gddr7.global">, StrEnumAttrCase<"hbm.global">,
   StrEnumAttrCase<"nvlink.scratch">, StrEnumAttrCase<"host.pcie">]>;

def KVRoleAttr    : StrEnumAttr<"KVRole","kv_role", [StrEnumAttrCase<"keys">, StrEnumAttrCase<"values">]>;
def KVLayoutAttr  : StrEnumAttr<"KVLayout","kv_layout", [StrEnumAttrCase<"row_major">, StrEnumAttrCase<"paged">]>;
def KVPolicyAttr  : StrEnumAttr<"KVPolicy","kv_policy", [StrEnumAttrCase<"pcie+cx9">, StrEnumAttrCase<"nvlink">]>;

class CPX_Op<string mnemonic, list<Trait> traits = []>
    : Op<Tessera_NVRubinCPX_Dialect, mnemonic, traits> { let hasVerifier = 1; }

// Async token type reference
def Async_Token : Type<CPred<"$_self.isa<::mlir::Type>()">, "async.token">;

def KVCacheOp : CPX_Op<"kv.cache", [NoSideEffect]> {
  let arguments = (ins AnyMemRef:$buffer, KVRoleAttr:$role, KVLayoutAttr:$layout);
  let results   = (outs AnyMemRef:$out);
  let assemblyFormat = "$buffer attr-dict `:` type($buffer) `->` type($out)";
}

def KVExportOp : CPX_Op<"kv.export"> {
  let arguments = (ins AnyMemRef:$src, KVPolicyAttr:$policy, I64Attr:$chunk_bytes);
  let results   = (outs AnyType:$token); // Will be !async.token in lowering
  let assemblyFormat = "$src `,` $policy `,` $chunk_bytes attr-dict `:` type($src)";
}

def KVImportOp : CPX_Op<"kv.import"> {
  let arguments = (ins AnyType:$token, AnyMemRef:$dst);
  let results   = (outs UnitAttr:$done);
  let assemblyFormat = "$token `->` $dst attr-dict `:` type($dst)";
}

def KVPrefetchOp : CPX_Op<"kv.prefetch", [NoSideEffect]> {
  let arguments = (ins AnyMemRef:$kv, I64Attr:$start_page, I64Attr:$num_pages);
  let assemblyFormat = "$kv `[` $start_page `:` $num_pages `]` attr-dict `:` type($kv)";
}

def AttnPrefillFusedOp : CPX_Op<"attn.prefill_fused"> {
  let arguments = (ins AnyType:$q, AnyType:$k, AnyType:$v,
                        AnyMemRef:$kv_cache, I64Attr:$seq_len);
  let results   = (outs AnyType:$o);
  let assemblyFormat = "$q `,` $k `,` $v `,` $kv_cache `,` $seq_len attr-dict "
                       "`:` type($q) `,` type($k) `,` type($v) `,` type($kv_cache) `->` type($o)";
}

// New fused ingest region wrapper (holds decoded frames + transforms on CPX)
def VideoIngestFusedOp : CPX_Op<"video.ingest_fused", [NoSideEffect, SingleBlockImplicitTerminator<"func::ReturnOp">]> {
  let summary = "Fused video decode -> patchify -> tokenizer -> prefill on CPX";
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "attr-dict with region($body)";
}

// Leaf decode/encode used by the fuser
def VideoDecodeOp : CPX_Op<"video.decode"> {
  let arguments = (ins AnyType:$bitstream, StrAttr:$codec);
  let results   = (outs AnyType:$frames);
  let assemblyFormat = "$bitstream `,` $codec attr-dict `:` type($bitstream) `->` type($frames)";
}
def VideoEncodeOp : CPX_Op<"video.encode"> {
  let arguments = (ins AnyType:$frames, StrAttr:$codec);
  let results   = (outs AnyType:$bitstream);
  let assemblyFormat = "$frames `,` $codec attr-dict `:` type($frames) `->` type($bitstream)";
}
