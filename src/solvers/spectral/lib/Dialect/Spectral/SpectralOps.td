
//===- SpectralOps.td ------------------------------------------*- tablegen -*-===//
// Tessera Spectral Dialect (fft, ifft, plan, twiddle, conv_fft).
//===----------------------------------------------------------------------===//
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Spectral_Dialect : Dialect {
  let name = "tessera_spectral";
  let cppNamespace = "::tessera::spectral";
  let summary = "Tessera Spectral ops (FFT/iFFT/Plan/Twiddle/Conv)";
  let useDefaultTypePrinterParser = 1;
}

// Opaque handle produced by `tessera_spectral.plan` and consumed by the
// exec ops.  It carries no parameters — the numerics policy lives in the
// plan op's attributes — so `!tessera_spectral.plan` parses/prints with the
// dialect's default type printer/parser.
def Spectral_PlanType : TypeDef<Spectral_Dialect, "Plan"> {
  let mnemonic = "plan";
  let summary = "Opaque FFT plan handle";
}

// Numerics policies are free-form string attributes (validated at the pass
// level, not in ODS).  `StrAttr` is a concrete attribute constraint (a def,
// not a class), so it is used directly on the operands below rather than via
// `def X : StrAttr;` aliases, which is not valid TableGen.

def TS_FFTPlanOp : Op<Spectral_Dialect, "plan", [Pure]> {
  let summary = "Create an FFT plan with numerics policies";
  let arguments = (ins
    I64ArrayAttr:$axes,
    OptionalAttr<I64ArrayAttr>:$radix_seq,
    StrAttr:$elem_precision,
    StrAttr:$acc_precision,
    StrAttr:$scaling,
    BoolAttr:$inplace,
    BoolAttr:$is_real_input,
    StrAttr:$norm_policy
  );
  let results = (outs Spectral_PlanType:$plan);
  // Uses the generic assembly form.  A custom assemblyFormat would need
  // explicit `attr-dict` + result/operand type directives under modern ODS;
  // the dialect is only consumed programmatically and in tests (generic
  // form), so the pretty syntax is not carried.
}

def TS_TwiddleTableOp : Op<Spectral_Dialect, "twiddle_table", [Pure]> {
  let summary = "Materialize (possibly quantized) twiddle table for a stage";
  let arguments = (ins AnyType:$plan, I64Attr:$stage);
  let results = (outs AnyMemRef:$table);
}

def TS_FFTOp : Op<Spectral_Dialect, "fft", []> {
  let summary = "Execute FFT using a plan";
  let arguments = (ins AnyType:$plan, AnyMemRef:$src, AnyMemRef:$dst);
}

def TS_IFFTOp : Op<Spectral_Dialect, "ifft", []> {
  let summary = "Execute inverse FFT using a plan";
  let arguments = (ins AnyType:$plan, AnyMemRef:$src, AnyMemRef:$dst);
}

def TS_SpectralConvOp : Op<Spectral_Dialect, "conv_fft", []> {
  let summary = "Pad→FFT→pointwise cmul→iFFT→crop composite op";
  let arguments = (ins AnyType:$plan, AnyMemRef:$src, AnyMemRef:$kernel, AnyMemRef:$dst);
}
