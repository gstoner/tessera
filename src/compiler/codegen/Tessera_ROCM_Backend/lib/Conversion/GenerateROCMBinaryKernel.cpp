//===- GenerateROCMBinaryKernel.cpp - elementwise binary math kernel -----===//
//
// Expands a `tessera_rocm.binary` directive into a flat 2-operand elementwise
// gpu kernel applying a pointwise binary arithmetic function over N elements
// (one thread per element, strided grid) — the standalone S2 binary-arithmetic
// family, the binary sibling of the unary-math lane:
//
//   sub  = a - b      div = a / b      pow = a ** b
//   maximum = max(a, b)   minimum = min(a, b)   (IEEE NaN-propagating)
//
// Both operands and the output share the same storage dtype; the body computes
// in f32 regardless (transcendental `pow` lowers through convert-math-to-rocdl).
// N is a runtime index arg. Validated vs a numpy reference on gfx1151.
//===----------------------------------------------------------------------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"

#include <limits>

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;

enum class Bin { Sub, Div, Pow, Maximum, Minimum, Add, Mul, Mod, FloorDiv };

void emitBinaryBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f, Type storeTy,
                    Bin bin) {
  Type f32 = b.getF32Type();
  bool isF32 = storeTy.isF32();
  auto slt = arith::CmpIPredicate::slt;

  b.setInsertionPointToStart(&f.getBody().front());
  Value A = f.getArgument(0), B = f.getArgument(1), O = f.getArgument(2),
        N = f.getArgument(3);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, bid, cBD),
                                      tid);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, N);
  auto ifo = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(ifo.thenBlock());

  Value ra = b.create<memref::LoadOp>(loc, A, ValueRange{gid});
  Value rb = b.create<memref::LoadOp>(loc, B, ValueRange{gid});
  Value a = isF32 ? ra : b.create<arith::ExtFOp>(loc, f32, ra);
  Value bb = isF32 ? rb : b.create<arith::ExtFOp>(loc, f32, rb);
  Value y;
  switch (bin) {
  case Bin::Sub:
    y = b.create<arith::SubFOp>(loc, a, bb);
    break;
  case Bin::Div:
    y = b.create<arith::DivFOp>(loc, a, bb);
    break;
  case Bin::Pow:
    y = b.create<math::PowFOp>(loc, a, bb);
    break;
  case Bin::Maximum:
    // IEEE-754-2019 contract, fleet-wide decision 2026-08-23: NaN
    // propagates and signed zeros are ordered (max tie -> +0.0, min tie
    // -> -0.0) — exactly arith.maximumf/minimumf's own semantics, so emit
    // them bare. The numpy-emulating `select(a==b, b, ...)` wrapper that
    // used to sit here returned the SECOND operand on a ±0 tie; numpy's
    // tie sign is an ISA accident (SSE second operand, NEON IEEE), not a
    // portable contract. The x86 CPU JIT lane pins the same semantics in
    // test_jit_tensor_elementwise_totality.py.
    y = b.create<arith::MaximumFOp>(loc, a, bb);
    break;
  case Bin::Minimum:
    y = b.create<arith::MinimumFOp>(loc, a, bb);
    break;
  case Bin::Add:
    y = b.create<arith::AddFOp>(loc, a, bb);
    break;
  case Bin::Mul:
    y = b.create<arith::MulFOp>(loc, a, bb);
    break;
  case Bin::FloorDiv:
  case Bin::Mod: {
    Value zero = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
    Value oneNeg = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(-1.0f));
    Value inf = b.create<arith::ConstantOp>(
        loc, f32, b.getF32FloatAttr(std::numeric_limits<float>::infinity()));
    Value nan = b.create<arith::ConstantOp>(
        loc, f32, b.getF32FloatAttr(std::numeric_limits<float>::quiet_NaN()));
    Value isInfA = b.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, b.create<math::AbsFOp>(loc, a), inf);
    Value isInfB = b.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, b.create<math::AbsFOp>(loc, bb), inf);
    Type i32 = b.getI32Type();
    Value ai = b.create<arith::BitcastOp>(loc, i32, a);
    Value bi = b.create<arith::BitcastOp>(loc, i32, bb);
    Value zi = b.create<arith::ConstantIntOp>(loc, i32, 0);
    Value signA = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, ai, zi);
    Value signB = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, bi, zi);
    Value signDiff = b.create<arith::XOrIOp>(loc, signA, signB);
    Value aZero = b.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, a, zero);
    Value quotient = b.create<arith::DivFOp>(loc, a, bb);
    Value floored = b.create<math::FloorOp>(loc, quotient);
    Value finiteA = b.create<arith::XOrIOp>(
        loc, isInfA, b.create<arith::ConstantIntOp>(loc, 1, 1));
    Value finiteB = b.create<arith::XOrIOp>(
        loc, isInfB, b.create<arith::ConstantIntOp>(loc, 1, 1));
    if (bin == Bin::FloorDiv) {
      Value negNonzero = b.create<arith::AndIOp>(
          loc, b.create<arith::AndIOp>(loc, isInfB, finiteA),
          b.create<arith::AndIOp>(
              loc, signDiff,
              b.create<arith::XOrIOp>(
                  loc, aZero, b.create<arith::ConstantIntOp>(loc, 1, 1))));
      Value bInfResult = b.create<arith::SelectOp>(loc, negNonzero, oneNeg,
                                                   quotient);
      Value aInfOnly = b.create<arith::AndIOp>(loc, isInfA, finiteB);
      y = b.create<arith::SelectOp>(
          loc, aInfOnly, nan,
          b.create<arith::SelectOp>(loc, isInfB, bInfResult, floored));
    } else {
      Value regular = b.create<arith::SubFOp>(
          loc, a, b.create<arith::MulFOp>(loc, floored, bb));
      Value signedZero = b.create<math::CopySignOp>(loc, zero, bb);
      Value nonzeroInfResult = b.create<arith::SelectOp>(loc, signDiff, bb, a);
      Value bInfResult = b.create<arith::SelectOp>(loc, aZero, signedZero,
                                                   nonzeroInfResult);
      Value bInfFiniteA = b.create<arith::AndIOp>(loc, isInfB, finiteA);
      Value bZero = b.create<arith::CmpFOp>(
          loc, arith::CmpFPredicate::OEQ, bb, zero);
      Value zeroModuloNonzero = b.create<arith::AndIOp>(
          loc, aZero,
          b.create<arith::XOrIOp>(
              loc, bZero, b.create<arith::ConstantIntOp>(loc, 1, 1)));
      y = b.create<arith::SelectOp>(
          loc, zeroModuloNonzero, signedZero,
          b.create<arith::SelectOp>(loc, bInfFiniteA, bInfResult, regular));
    }
    break;
  }
  }
  Value sv = isF32 ? y : b.create<arith::TruncFOp>(loc, storeTy, y);
  b.create<memref::StoreOp>(loc, sv, O, ValueRange{gid});

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMBinaryKernelPass
    : PassWrapper<GenerateROCMBinaryKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMBinaryKernelPass)

  StringRef getArgument() const final { return "generate-rocm-binary-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.binary directive into a flat 2-operand "
           "elementwise binary-arithmetic gpu kernel (compiler-generated)";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    math::MathDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.binary")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      if (!nameAttr) {
        op->emitError("tessera_rocm.binary missing name");
        return signalPassFailure();
      }
      StringRef kindStr = "sub";
      if (auto a = op->getAttrOfType<StringAttr>("kind"))
        kindStr = a.getValue();
      Bin bin = llvm::StringSwitch<Bin>(kindStr)
                    .Case("sub", Bin::Sub)
                    .Case("div", Bin::Div)
                    .Case("pow", Bin::Pow)
                    .Case("maximum", Bin::Maximum)
                    .Case("minimum", Bin::Minimum)
                    .Case("add", Bin::Add)
                    .Case("mul", Bin::Mul)
                    .Case("mod", Bin::Mod)
                    .Case("floor_div", Bin::FloorDiv)
                    .Default(Bin::Sub);
      static const llvm::StringSet<> kValid = {
          "sub", "div", "pow", "maximum", "minimum",
          "add", "mul", "mod", "floor_div"};
      if (!kValid.contains(kindStr)) {
        op->emitError("generate-rocm-binary-kernel: unknown kind '")
            << kindStr << "' (sub/div/pow/maximum/minimum/add/mul/mod/floor_div)";
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();

      Type storeTy = b.getF32Type();
      if (auto a = op->getAttrOfType<StringAttr>("dtype")) {
        StringRef dt = a.getValue();
        if (dt == "f16" || dt == "float16")
          storeTy = b.getF16Type();
        else if (dt == "bf16" || dt == "bfloat16")
          storeTy = b.getBF16Type();
        else if (dt != "f32" && dt != "float32") {
          op->emitError("generate-rocm-binary-kernel: dtype must be f32, f16, "
                        "or bf16 (got '")
              << dt << "')";
          return signalPassFailure();
        }
      }

      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      Type idxTy = b.getIndexType();
      auto memTy = MemRefType::get({ShapedType::kDynamic}, storeTy);
      // (A, B, O : memref<?xstore>, N : index)
      auto fnTy = b.getFunctionType({memTy, memTy, memTy, idxTy}, {});
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitBinaryBody(body, loc, gpuFunc, storeTy, bin);
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMBinaryKernelPass() {
  return std::make_unique<GenerateROCMBinaryKernelPass>();
}
