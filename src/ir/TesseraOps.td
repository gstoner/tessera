
include "mlir/IR/OpBase.td"
include "mlir/IR/OpImplementation.h.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"

def Tessera_EpilogueKind : I32EnumAttr<
  "EpilogueKind",
  "Which post-op is fused",
  [
    I32EnumAttrCase<"None", 0>,
    I32EnumAttrCase<"Relu", 1>,
    I32EnumAttrCase<"Gelu", 2>,
    I32EnumAttrCase<"Silu", 3>
  ]
> {
  let convertFromStorage = [{(tessera::EpilogueKind)$_self;}];
  let stringify = [{tessera::stringifyEpilogueKind($_self);}];
  let cppNamespace = "::tessera";
}
def Tessera_EpilogueKindAttr : EnumAttr<Tessera_Dialect, Tessera_EpilogueKind, "epilogue">;

def TensorType : AnyTypeOf<[AnyRankedTensor, AnyUnrankedTensor]>;

def Tessera_MatmulOp : Op<
  Tessera_Dialect, "matmul",
  [Pure, NoSideEffect, DeclareOpInterfaceMethods<TilingInterface>]
> {
  let summary = "Matrix multiply (A @ B)";
  let arguments = (ins TensorType:$lhs, TensorType:$rhs,
                        OptionalAttr<I64Attr>:$tile_k,
                        DefaultValuedAttr<BoolAttr, "false">:$transposeA,
                        DefaultValuedAttr<BoolAttr, "false">:$transposeB);
  let results = (outs TensorType:$result);

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_Conv2DNHWCOp : Op<
  Tessera_Dialect, "conv2d_nhwc",
  [Pure, NoSideEffect, DeclareOpInterfaceMethods<TilingInterface>]
> {
  let summary = "2D convolution NHWC x HWCF -> NHWC";
  let arguments = (ins TensorType:$input, TensorType:$filter,
                        ArrayAttr:$strides, ArrayAttr:$dilations,
                        OptionalAttr<Tessera_EpilogueKindAttr>:$epilogue);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_FlashAttnOp : Op<
  Tessera_Dialect, "flash_attn",
  [Pure, NoSideEffect]
> {
  let summary = "FlashAttention fused op";
  let arguments = (ins TensorType:$q, TensorType:$k, TensorType:$v,
                        I64Attr:$head_dim,
                        OptionalAttr<F64Attr>:$dropout_p,
                        DefaultValuedAttr<BoolAttr, "false">:$causal);
  let results = (outs TensorType:$o);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_FusedEpilogueOp : Op<
  Tessera_Dialect, "fused_epilogue",
  [Pure, NoSideEffect]
> {
  let summary = "Fused epilogue for matmul/conv with optional bias";
  let arguments = (ins TensorType:$lhs, TensorType:$rhs, TensorType:$bias,
                        Tessera_EpilogueKindAttr:$epilogue,
                        DefaultValuedAttr<BoolAttr, "false">:$has_bias);
  let results = (outs TensorType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tessera_TransposeOp : Op<
  Tessera_Dialect, "transpose",
  [Pure, NoSideEffect]
> {
  let arguments = (ins TensorType:$x);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tessera_CastOp : Op<
  Tessera_Dialect, "cast",
  [Pure, NoSideEffect]
> {
  let arguments = (ins TensorType:$x);
  let results = (outs TensorType:$y);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}
