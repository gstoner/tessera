#pragma once
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringSwitch.h"
namespace tessera::numeric_policy {
using namespace mlir;
inline unsigned accumSignificandBits(FloatType type) {
  return llvm::APFloat::semanticsPrecision(type.getFloatSemantics());
}

//: Outcome of reading `numeric_policy.accum` against the storage element type.
enum class AccumRealizability {
  //: No policy, no accum, or an accumulator no more precise than storage —
  //: nothing to widen, and the lowering is unchanged.
  NoWidening,
  //: Strictly wider in BITS: a plain extf/truncf pair expresses it.
  Widen,
  //: More precise but not wider in bits (bf16 storage, fp16 accum). No cast
  //: expresses it and substituting fp32 would over-deliver precision the
  //: program did not ask for, so this lane cannot provide it.
  Unrealizable,
};

inline AccumRealizability
readAccumulator(Operation *op, Type storageElem, FloatType &widened) {
  widened = nullptr;
  auto policy = op->getAttrOfType<DictionaryAttr>("numeric_policy");
  if (!policy) return AccumRealizability::NoWidening;
  auto accum = policy.getAs<StringAttr>("accum");
  if (!accum) return AccumRealizability::NoWidening;
  auto storageFloat = dyn_cast<FloatType>(storageElem);
  if (!storageFloat) return AccumRealizability::NoWidening;
  MLIRContext *ctx = op->getContext();
  FloatType accumType =
      llvm::StringSwitch<FloatType>(accum.getValue())
          .Case("fp64", Float64Type::get(ctx))
          .Case("fp32", Float32Type::get(ctx))
          .Case("fp16", Float16Type::get(ctx))
          .Case("bf16", BFloat16Type::get(ctx))
          .Default(nullptr);
  if (!accumType) return AccumRealizability::NoWidening;
  if (accumSignificandBits(accumType) <= accumSignificandBits(storageFloat))
    return AccumRealizability::NoWidening;
  if (accumType.getWidth() <= storageFloat.getWidth())
    return AccumRealizability::Unrealizable;
  widened = accumType;
  return AccumRealizability::Widen;
}


}
