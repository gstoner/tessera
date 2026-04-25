//===- TileMem.td (v1.1) ---------------------------------------------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"
include "mlir/IR/DialectSpecification.td"

def Tessera_Tile_Dialect : Dialect {
  let name = "tessera.tile";
  let cppNamespace = "::tessera::tile";
}

def TMEMType : TypeDef<"TMEMType"> {
  let summary = "Blackwell TMEM memory space with capacity and columns";
  let parameters = (ins I64Parameter:$capacity, I32Parameter:$columns);
  let assemblyFormat = "`<` $capacity `,` $columns `>`";
}

def AllocTMEMOp : Op<"alloc_tmem", [NoSideEffect]> {
  let summary = "Allocate a TMEM tile buffer";
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$type);
  let results = (outs AnyMemRef:$buffer);
  let assemblyFormat = "$type attr-dict";
  let hasVerifier = 1;
}

def TmemLoadOp : Op<"ld.tmem", [NoSideEffect]> {
  let summary = "Load a scalar/vector from TMEM to registers";
  let arguments = (ins AnyMemRef:$src, Variadic<Index>:$indices);
  let results = (outs AnyType:$res);
  let assemblyFormat = "$src `[` $indices `]` attr-dict";
  let hasVerifier = 1;
}

def TmemStoreOp : Op<"st.tmem", []> {
  let summary = "Store from registers into TMEM";
  let arguments = (ins AnyType:$val, AnyMemRef:$dst, Variadic<Index>:$indices);
  let assemblyFormat = "$val `,` $dst `[` $indices `]` attr-dict";
  let hasVerifier = 1;
}

def Tcgen05MMAOp : Op<"mma.tcgen05", []> {
  let summary = "Blackwell tensor core MMA with CTA-group";
  let arguments = (ins AnyType:$a, AnyType:$b, AnyType:$acc, I32Attr:$cta_group);
  let results = (outs AnyType:$out);
  let assemblyFormat = "$a `,` $b `->` $acc `,` $cta_group attr-dict";
  let hasVerifier = 1;
}
