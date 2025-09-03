//===- Cerebras.td - Cerebras target IR (sketch) ---------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"

def Cerebras_Dialect : Dialect {
  let name = "cerebras";
  let cppNamespace = "::tessera::cerebras";
}

def Cerebras_RegionOp : Op<Cerebras_Dialect, "region", [IsolatedFromAbove]> {
  let summary = "Physical rectangular PE region";
  let arguments = (ins I32Attr:$x0, I32Attr:$y0, I32Attr:$x1, I32Attr:$y1, I32Attr:$color);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "`{` $x0 `,` $y0 `,` $x1 `,` $y1 `,` $color `}` $body attr-dict";
}

def Cerebras_RouteOp : Op<Cerebras_Dialect, "route", []> {
  let summary = "Route color/class between regions";
  let arguments = (ins I32Attr:$from, I32Attr:$to, I32Attr:$color);
  let assemblyFormat = "`(` $from `->` $to `)` attr-dict";
}

def Cerebras_MemcpyOp : Op<Cerebras_Dialect, "memcpy", []> {
  let summary = "Memcpy between address spaces";
  let arguments = (ins AnyType:$dst, AnyType:$src, StrAttr:$dst_space, StrAttr:$src_space);
  let assemblyFormat = "$dst `,` $src attr-dict";
}

def Cerebras_LoadSRAMOp : Op<Cerebras_Dialect, "load_sram", []> {
  let summary = "Load from global into SRAM tile buffer";
  let arguments = (ins AnyType:$dst_sram, AnyType:$src_global);
  let assemblyFormat = "$dst_sram `,` $src_global attr-dict";
}

def Cerebras_StoreSRAMOp : Op<Cerebras_Dialect, "store_sram", []> {
  let summary = "Store from SRAM tile buffer to global";
  let arguments = (ins AnyType:$dst_global, AnyType:$src_sram);
  let assemblyFormat = "$dst_global `,` $src_sram attr-dict";
}

def Cerebras_MatmulOp : Op<Cerebras_Dialect, "matmul", []> {
  let summary = "Matmul stub op";
  let arguments = (ins AnyType:$A, AnyType:$B, AnyType:$C,
                       I32Attr:$M, I32Attr:$N, I32Attr:$K);
  let assemblyFormat = "$A `,` $B `->` $C `(` $M `,` $N `,` $K `)` attr-dict";
}

def Cerebras_EmitOp : Op<Cerebras_Dialect, "emit", []> {
  let summary = "Serialize module to CSL + layout.json";
  let arguments = (ins StrAttr:$csl_out, StrAttr:$layout_out);
  let assemblyFormat = "attr-dict";
}
