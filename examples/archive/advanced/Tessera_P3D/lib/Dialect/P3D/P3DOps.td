//===- P3DOps.td -----------------------------------------*- tablegen -*-===//
include "P3DBase.td"

def P3D_HaloAttr : StrAttr<CPred<"true">> {
  let summary = "Boundary condition policy (e.g. periodic, dirichlet, neumann)";
}

class P3D_Op<string mnemonic, list<Trait> traits = []> :
    Op<P3D_Dialect, mnemonic, traits>;

def P3D_Conv3DOp : P3D_Op<"conv3d",
  [AttrSizedOperandSegments, Pure, InferTypeOpInterface]> {
  let summary = "NCDHW/NCDHW conv3d with explicit kernel/stride/dilation/pad";
  let arguments = (ins
    AnyMemRef:$input,    // [N,C,D,H,W]
    AnyMemRef:$weight,   // [K,C,KT,KH,KW]
    Optional<AnyMemRef>:$bias,
    I64ArrayAttr:$stride,      // [sT,sH,sW]
    I64ArrayAttr:$dilation,    // [dT,dH,dW]
    I64ArrayAttr:$padding,     // [pT,pH,pW]
    OptionalAttr<P3D_HaloAttr>:$bc
  );
  let results = (outs AnyMemRef:$output);
  let assemblyFormat = "$input `,` $weight (`,` $bias^)? "
                       "attr-dict `:` type($input) `,` type($weight) `->` type($output)";
}

def P3D_PyramidBuildOp : P3D_Op<"pyramid.build", [Pure]> {
  let summary = "Build multi-res pyramid (levels, factors)";
  let arguments = (ins AnyMemRef:$input, I64ArrayAttr:$factors);
  let results = (outs Variadic<AnyMemRef>:$levels);
  let assemblyFormat = "$input `,` $factors attr-dict `:` type($input) `->` type($levels)";
}

def P3D_GlobalContextOp : P3D_Op<"global_context",
  [Pure, InferTypeOpInterface]> {
  let summary = "Global context mixing over 3D volume (attention-like)";
  let arguments = (ins AnyMemRef:$input, Optional<AnyMemRef>:$mask);
  let results = (outs AnyMemRef:$output);
  let assemblyFormat = "$input (`,` $mask^)? attr-dict `:` type($input) `->` type($output)";
}

def P3D_Upsample3DOp : P3D_Op<"upsample3d", [Pure]> {
  let summary = "3D upsampling with mode and target size";
  let arguments = (ins AnyMemRef:$input, I64ArrayAttr:$size, StrAttr:$mode);
  let results   = (outs AnyMemRef:$output);
  let assemblyFormat = "$input `,` $size `,` $mode attr-dict `:` type($input) `->` type($output)";
}

def P3D_Downsample3DOp : P3D_Op<"downsample3d", [Pure]> {
  let summary = "3D downsampling with window/stride and mode";
  let arguments = (ins AnyMemRef:$input, I64ArrayAttr:$window, I64ArrayAttr:$stride, StrAttr:$mode);
  let results = (outs AnyMemRef:$output);
  let assemblyFormat = "$input `,` $window `,` $stride `,` $mode attr-dict "
                       "`:` type($input) `->` type($output)";
}
