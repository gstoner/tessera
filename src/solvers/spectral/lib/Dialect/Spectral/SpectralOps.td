
//===- SpectralOps.td ------------------------------------------*- tablegen -*-===//
// Tessera Spectral Dialect (fft, ifft, plan, twiddle, conv_fft).
//===----------------------------------------------------------------------===//
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"

def Spectral_Dialect : Dialect {
  let name = "tessera_spectral";
  let cppNamespace = "::tessera::spectral";
  let summary = "Tessera Spectral ops (FFT/iFFT/Plan/Twiddle/Conv)";
}

def TS_PrecisionPolicy : StrAttr;
def TS_AccPolicy : StrAttr;
def TS_ScalingPolicy : StrAttr;

def TS_FFTPlanOp : Op<Spectral_Dialect, "plan",
      [NoSideEffect, AttrSizedOperandSegments]> {
  let summary = "Create an FFT plan with numerics policies";
  let arguments = (ins
    I64ArrayAttr:$axes,
    OptionalAttr<I64ArrayAttr>:$radix_seq,
    TS_PrecisionPolicy:$elem_precision,
    TS_AccPolicy:$acc_precision,
    TS_ScalingPolicy:$scaling,
    BoolAttr:$inplace,
    BoolAttr:$is_real_input,
    StrAttr:$norm_policy
  );
  let results = (outs AnyType:$plan);
  let assemblyFormat = [{ 
    $axes (`radix` $radix_seq^)? `elem` $elem_precision `acc` $acc_precision
    `scale` $scaling `inplace` $inplace `real` $is_real_input `norm` $norm_policy
  }];
}

def TS_TwiddleTableOp : Op<Spectral_Dialect, "twiddle_table", [NoSideEffect]> {
  let summary = "Materialize (possibly quantized) twiddle table for a stage";
  let arguments = (ins AnyType:$plan, I64Attr:$stage);
  let results = (outs AnyMemRef:$table);
}

def TS_FFTOp : Op<Spectral_Dialect, "fft", []> {
  let summary = "Execute FFT using a plan";
  let arguments = (ins AnyType:$plan, AnyMemRef:$src, AnyMemRef:$dst);
  let assemblyFormat = [{ $plan `,` $src `->` $dst }];
}

def TS_IFFTOp : Op<Spectral_Dialect, "ifft", []> {
  let summary = "Execute inverse FFT using a plan";
  let arguments = (ins AnyType:$plan, AnyMemRef:$src, AnyMemRef:$dst);
  let assemblyFormat = [{ $plan `,` $src `->` $dst }];
}

def TS_SpectralConvOp : Op<Spectral_Dialect, "conv_fft", []> {
  let summary = "Pad→FFT→pointwise cmul→iFFT→crop composite op";
  let arguments = (ins AnyType:$plan, AnyMemRef:$src, AnyMemRef:$kernel, AnyMemRef:$dst);
}
