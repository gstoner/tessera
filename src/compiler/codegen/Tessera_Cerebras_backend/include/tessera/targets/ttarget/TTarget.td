//===- TTarget.td - Minimal Tessera Target IR (sketch) ---------*- tablegen -*-===//
include "mlir/IR/OpBase.td"

def TTarget_Dialect : Dialect {
  let name = "ttarget";
  let cppNamespace = "::tessera::ttarget";
}

def TTarget_RegionOp : Op<TTarget_Dialect, "region", [IsolatedFromAbove]> {
  let summary = "Logical region for placement";
  let description = [{ Holds a body representing a placed region. Attributes: x0,y0,x1,y1 (i32) }];
  let arguments = (ins I32Attr:$x0, I32Attr:$y0, I32Attr:$x1, I32Attr:$y1);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "`{` $x0 `,` $y0 `,` $x1 `,` $y1 `}` $body attr-dict";
}

def TTarget_CopyOp : Op<TTarget_Dialect, "copy", []> {
  let summary = "Copy tensor between mem classes";
  let description = [{ src -> dst. Attributes: src_space, dst_space (str). }];
  let arguments = (ins AnyType:$src, AnyType:$dst,
                       StrAttr:$src_space, StrAttr:$dst_space);
  let assemblyFormat = "$src `->` $dst attr-dict";
}

def TTarget_RouteOp : Op<TTarget_Dialect, "route", []> {
  let summary = "Declare a routing color or link between regions";
  let arguments = (ins I32Attr:$from, I32Attr:$to, I32Attr:$color);
  let assemblyFormat = "`(` $from `->` $to `)` attr-dict";
}

def TTarget_MatmulOp : Op<TTarget_Dialect, "matmul", []> {
  let summary = "Trivial matmul stub";
  let arguments = (ins AnyType:$A, AnyType:$B, AnyType:$C,
                       I32Attr:$M, I32Attr:$N, I32Attr:$K);
  let assemblyFormat = "$A `,` $B `->` $C `(` $M `,` $N `,` $K `)` attr-dict";
}
