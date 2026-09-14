#include "TesseraROCM/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

// Stage J — lower a `tessera_rocm.wmma` that carries REAL RDNA WMMA fragment
// vectors to the real `rocdl.wmma.*.16x16x16.*` intrinsic op (which translates
// to `llvm.amdgcn.wmma.*`, the same instruction the hand-written builtins emit
// and llc proves). The gfx11 / RDNA 3.5 WMMA ABI (all 16x16x16, wave32):
//   * f16  in, f32 acc : A/B `vector<16xf16>`, acc/res `vector<8xf32>`.
//   * bf16 in, f32 acc : A/B `vector<16xbf16>` (bitcast to `vector<16xi16>` —
//     the intrinsic takes the bit-pattern), acc/res `vector<8xf32>`.
//   * int8 in, i32 acc : A/B `vector<4xi32>` (16 int8 packed/lane), acc/res
//     `vector<8xi32>`; signed (signA=signB=1), no saturating clamp.
//   * int4 in, i32 acc : A/B `vector<2xi32>` (16 int4 packed/lane), acc/res
//     `vector<8xi32>`; signed.
// All confirmed supported on gfx1151 by the device compiler (hipcc
// --offload-arch=gfx1151). FP8/F32/TF32 WMMA do not exist on RDNA 3.5.
//
// Returns true if it emitted the real op (and replaced/erased `op`); false when
// the operands are not executable fragments. Target-only inspection stops
// before this pass; reaching binary lowering with an abstract contract is an
// error, never a request to synthesize a placeholder call.
bool lowerRealWMMA(Operation *op, PatternRewriter &rewriter) {
  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return false;
  Value a = op->getOperand(0), b = op->getOperand(1), c = op->getOperand(2);
  Type resTy = op->getResult(0).getType();
  auto isVec = [](Type t, int64_t n, Type elt) {
    auto v = dyn_cast<VectorType>(t);
    return v && v.getRank() == 1 && v.getNumElements() == n &&
           v.getElementType() == elt;
  };
  Location loc = op->getLoc();
  Type f32 = rewriter.getF32Type();
  Type i32 = rewriter.getIntegerType(32);
  StringRef inputDType;
  if (auto attr = op->getAttrOfType<StringAttr>("input_dtype"))
    inputDType = attr.getValue();
  StringRef inputBType = inputDType;
  if (auto attr = op->getAttrOfType<StringAttr>("input_b_dtype"))
    inputBType = attr.getValue();
  StringRef fragmentFamily;
  if (auto attr = op->getAttrOfType<StringAttr>("fragment_family"))
    fragmentFamily = attr.getValue();

  // CDNA5 WMMA doubles K for f16/bf16 and has a distinct intrinsic ABI.
  // The five modifiers are instruction attributes, not extra fragment values:
  // no sign inversion, no C modification, and no explicit operand reuse is the
  // exact dense A*B+C operation represented by Tile's logical mma contract.
  if (fragmentFamily == "cdna5_wmma" &&
      isVec(c.getType(), 8, f32) && isVec(resTy, 8, f32)) {
    Operation *real = nullptr;
    if (isVec(a.getType(), 16, rewriter.getF16Type()) &&
        isVec(b.getType(), 16, rewriter.getF16Type()))
      real = rewriter.create<ROCDL::wmma_f32_16x16x32_f16>(
          loc, resTy, a, b, ROCDL::WMMACModifier::none, c,
          /*reuseA=*/false, /*reuseB=*/false);
    else if (isVec(a.getType(), 16, rewriter.getBF16Type()) &&
             isVec(b.getType(), 16, rewriter.getBF16Type()))
      real = rewriter.create<ROCDL::wmma_f32_16x16x32_bf16>(
          loc, resTy, a, b, ROCDL::WMMACModifier::none, c,
          /*reuseA=*/false, /*reuseB=*/false);
    if (real) {
      rewriter.replaceOp(op, real->getResults());
      return true;
    }
    return false;
  }

  // --- f32-accumulate family (f16 / bf16 inputs) ---
  if (isVec(c.getType(), 8, f32) && isVec(resTy, 8, f32)) {
    Type f16 = rewriter.getF16Type();
    Type bf16 = rewriter.getBF16Type();
    bool f16Fragments =
        (isVec(a.getType(), 16, f16) && isVec(b.getType(), 16, f16)) ||
        (isVec(a.getType(), 8, f16) && isVec(b.getType(), 8, f16));
    if (f16Fragments) {
      Operation *real = rewriter.create<ROCDL::wmma_f32_16x16x16_f16>(
          loc, resTy, ValueRange{a, b, c});
      rewriter.replaceOp(op, real->getResults());
      return true;
    }
    int64_t bf16Elements = isVec(a.getType(), 16, bf16) ? 16 : 8;
    if ((isVec(a.getType(), 16, bf16) || isVec(a.getType(), 8, bf16)) &&
        isVec(b.getType(), bf16Elements, bf16)) {
      // RDNA bf16 WMMA takes the bf16 bit-pattern as <16 x i16>.
      Type i16Vec =
          VectorType::get({bf16Elements}, rewriter.getIntegerType(16));
      Value ai = rewriter.create<LLVM::BitcastOp>(loc, i16Vec, a);
      Value bi = rewriter.create<LLVM::BitcastOp>(loc, i16Vec, b);
      Operation *real = rewriter.create<ROCDL::wmma_f32_16x16x16_bf16>(
          loc, resTy, ValueRange{ai, bi, c});
      rewriter.replaceOp(op, real->getResults());
      return true;
    }
    if (fragmentFamily == "rdna4_wmma" &&
        isVec(a.getType(), 2, i32) && isVec(b.getType(), 2, i32)) {
      Operation *real = nullptr;
      bool aFp8 = inputDType == "e4m3" || inputDType == "fp8";
      bool bFp8 = inputBType == "e4m3" || inputBType == "fp8";
      bool aBf8 = inputDType == "e5m2" || inputDType == "bf8";
      bool bBf8 = inputBType == "e5m2" || inputBType == "bf8";
      if (aFp8 && bFp8)
        real = rewriter.create<ROCDL::wmma_f32_16x16x16_fp8_fp8>(loc, resTy, ValueRange{a,b,c});
      else if (aFp8 && bBf8)
        real = rewriter.create<ROCDL::wmma_f32_16x16x16_fp8_bf8>(loc, resTy, ValueRange{a,b,c});
      else if (aBf8 && bFp8)
        real = rewriter.create<ROCDL::wmma_f32_16x16x16_bf8_fp8>(loc, resTy, ValueRange{a,b,c});
      else if (aBf8 && bBf8)
        real = rewriter.create<ROCDL::wmma_f32_16x16x16_bf8_bf8>(loc, resTy, ValueRange{a,b,c});
      if (real) {
        rewriter.replaceOp(op, real->getResults());
        return true;
      }
    }
    return false;
  }

  // RDNA4 has compact eight-element low-precision accumulators; opsel=0.
  if (fragmentFamily == "rdna4_wmma") {
    Type half = rewriter.getF16Type(), bf = rewriter.getBF16Type();
    if (isVec(a.getType(),8,half) && isVec(b.getType(),8,half) &&
        isVec(c.getType(),8,half) && isVec(resTy,8,half)) {
      auto real = rewriter.create<ROCDL::wmma_f16_16x16x16_f16>(loc,resTy,a,b,c,false);
      rewriter.replaceOp(op,real->getResults()); return true;
    }
    if (isVec(a.getType(),8,bf) && isVec(b.getType(),8,bf) &&
        isVec(c.getType(),8,bf) && isVec(resTy,8,bf)) {
      Type bits = VectorType::get({8}, rewriter.getI16Type());
      Value ai = rewriter.create<LLVM::BitcastOp>(loc,bits,a);
      Value bi = rewriter.create<LLVM::BitcastOp>(loc,bits,b);
      Value ci = rewriter.create<LLVM::BitcastOp>(loc,bits,c);
      auto real = rewriter.create<ROCDL::wmma_bf16_16x16x16_bf16>(loc,bits,ai,bi,ci,false);
      Value result = rewriter.create<LLVM::BitcastOp>(loc,resTy,real.getResult());
      rewriter.replaceOp(op,result); return true;
    }
  }

  // --- explicitly admitted f16-accumulate family (gfx11) ---
  // The intrinsic reads and writes only the 16-bit register halves selected
  // by opsel. The generator uses opsel=false and stores indices 0,2,...,14;
  // the unselected halves remain outside the mathematical result.
  Type f16 = rewriter.getF16Type();
  if (isVec(a.getType(), 16, f16) && isVec(b.getType(), 16, f16) &&
      isVec(c.getType(), 16, f16) && isVec(resTy, 16, f16)) {
    Operation *real = rewriter.create<ROCDL::wmma_f16_16x16x16_f16>(
        loc, resTy, a, b, c, /*opsel=*/false);
    rewriter.replaceOp(op, real->getResults());
    return true;
  }

  // --- i32-accumulate family (int8 / int4 inputs), signed, non-saturating ---
  if (isVec(c.getType(), 8, i32) && isVec(resTy, 8, i32)) {
    // signA/signB/clamp are immarg attributes (the IU intrinsic class). Signed
    // inputs (signA=signB=1); clamp=0 = no i32 saturation (wrap), matching a
    // plain integer matmul against numpy's int32 accumulate.
    bool signedA = true, signedB = true;
    for (auto [name, value] : {std::pair<StringRef, bool *>("signed_a", &signedA),
                               std::pair<StringRef, bool *>("signed_b", &signedB)}) {
      if (Attribute raw = op->getAttr(name)) {
        auto attr = dyn_cast<BoolAttr>(raw);
        if (!attr) return false;
        *value = attr.getValue();
      }
    }
    SmallVector<NamedAttribute> attrs = {
        rewriter.getNamedAttr("signA", rewriter.getBoolAttr(signedA)),
        rewriter.getNamedAttr("signB", rewriter.getBoolAttr(signedB)),
        rewriter.getNamedAttr("clamp", rewriter.getBoolAttr(false)),
    };
    if ((isVec(a.getType(), 4, i32) && isVec(b.getType(), 4, i32)) ||
        (fragmentFamily == "rdna4_wmma" && inputDType == "int8" &&
         isVec(a.getType(), 2, i32) && isVec(b.getType(), 2, i32))) {
      Operation *real = rewriter.create<ROCDL::wmma_i32_16x16x16_iu8>(
          loc, TypeRange{resTy}, ValueRange{a, b, c}, attrs);
      rewriter.replaceOp(op, real->getResults());
      return true;
    }
    if (fragmentFamily == "rdna4_wmma" && inputDType == "int4" &&
        isVec(a.getType(), 1, i32) && isVec(b.getType(), 1, i32)) {
      Value ai = rewriter.create<LLVM::ExtractElementOp>(loc, a,
          rewriter.create<LLVM::ConstantOp>(loc, i32, rewriter.getI32IntegerAttr(0)));
      Value bi = rewriter.create<LLVM::ExtractElementOp>(loc, b,
          rewriter.create<LLVM::ConstantOp>(loc, i32, rewriter.getI32IntegerAttr(0)));
      auto real = rewriter.create<ROCDL::wmma_i32_16x16x16_iu4>(
          loc, TypeRange{resTy}, ValueRange{ai, bi, c}, attrs);
      rewriter.replaceOp(op, real->getResults());
      return true;
    }
    if (isVec(a.getType(), 2, i32) && isVec(b.getType(), 2, i32)) {
      Operation *real;
      if (fragmentFamily == "rdna4_wmma" && inputDType == "int4")
        real = rewriter.create<ROCDL::wmma_i32_16x16x32_iu4>(
            loc, TypeRange{resTy}, ValueRange{a, b, c}, attrs);
      else
        real = rewriter.create<ROCDL::wmma_i32_16x16x16_iu4>(
            loc, TypeRange{resTy}, ValueRange{a, b, c}, attrs);
      rewriter.replaceOp(op, real->getResults());
      return true;
    }
    return false;
  }
  return false;
}

// CDNA's canonical 16x16x16 f16/bf16 MFMA ABI is wave64: four input
// elements and four f32 accumulator elements per lane. MLIR 23 models the
// unmodified dense-product form with only the A, B, and accumulator operands.
bool lowerRealMFMA(Operation *op, PatternRewriter &rewriter) {
  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return false;
  Value a = op->getOperand(0), b = op->getOperand(1), c = op->getOperand(2);
  Type resTy = op->getResult(0).getType();
  auto isVec = [](Type t, int64_t n, Type elt) {
    auto v = dyn_cast<VectorType>(t);
    return v && v.getRank() == 1 && v.getNumElements() == n &&
           v.getElementType() == elt;
  };
  Type f32 = rewriter.getF32Type();
  if (!isVec(c.getType(), 4, f32) || !isVec(resTy, 4, f32))
    return false;

  Location loc = op->getLoc();
  if (isVec(a.getType(), 4, rewriter.getF16Type()) &&
      isVec(b.getType(), 4, rewriter.getF16Type())) {
    Operation *real = rewriter.create<ROCDL::mfma_f32_16x16x16f16>(
        loc, resTy, a, b, c, /*cbsz=*/0, /*abid=*/0,
        ROCDL::MFMAPermB::none);
    rewriter.replaceOp(op, real->getResults());
    return true;
  }
  Type bf16 = rewriter.getBF16Type();
  if (isVec(a.getType(), 4, bf16) && isVec(b.getType(), 4, bf16)) {
    Type i16Vec = VectorType::get({4}, rewriter.getIntegerType(16));
    Value ai = rewriter.create<LLVM::BitcastOp>(loc, i16Vec, a);
    Value bi = rewriter.create<LLVM::BitcastOp>(loc, i16Vec, b);
    Operation *real = rewriter.create<ROCDL::mfma_f32_16x16x16bf16_1k>(
        loc, resTy, ai, bi, c, /*cbsz=*/0, /*abid=*/0,
        ROCDL::MFMAPermB::none);
    rewriter.replaceOp(op, real->getResults());
    return true;
  }
  return false;
}

struct LoweringPass : PassWrapper<LoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, ROCDL::ROCDLDialect>();
  }
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoweringPass)

  StringRef getArgument() const final { return "lower-tessera-target-to-rocdl"; }

  StringRef getDescription() const final {
    return "Lower executable Tessera ROCm target ops to real ROCDL and reject "
           "target-only or abstract contracts";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> waitOps;
    SmallVector<Operation *> rocmOps;

    module.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tessera_rocm.wait")
        waitOps.push_back(op);
      else if (name == "tessera_rocm.mfma" ||
               name == "tessera_rocm.wmma" ||
               name == "tessera_rocm.swmmac" ||
               name == "tessera_rocm.async_copy" ||
               name == "tessera_rocm.buffer_load" ||
               name == "tessera_rocm.ds_read_tr")
        rocmOps.push_back(op);
    });
    waitOps.append(rocmOps.begin(), rocmOps.end());
    rocmOps = std::move(waitOps);

    for (Operation *op : rocmOps) {
      PatternRewriter rewriter(op->getContext());
      rewriter.setInsertionPoint(op);

      StringRef opName = op->getName().getStringRef();

      if (opName == "tessera_rocm.swmmac") {
        auto arch = op->getAttrOfType<StringAttr>("arch");
        auto aTy = dyn_cast<VectorType>(op->getOperand(0).getType());
        if (!arch || arch.getValue() != "gfx1201" || !aTy) {
          op->emitError("sparse WMMA requires its verified gfx1201 contract");
          signalPassFailure();
          return;
        }
        SmallVector<Value> args(op->getOperands());
        auto width = op->getAttrOfType<IntegerAttr>("integer_bits");
        bool int4 = width && width.getInt() == 4;
        bool integer = aTy.getElementType().isInteger(8) || int4;
        bool fp8 = isa<Float8E4M3FNType>(aTy.getElementType());
        bool bf8 = isa<Float8E5M2Type>(aTy.getElementType());
        if (integer || fp8 || bf8) {
          if (!integer) {
            args[0] = rewriter.create<arith::BitcastOp>(op->getLoc(), VectorType::get({8}, rewriter.getI8Type()), args[0]);
            args[1] = rewriter.create<arith::BitcastOp>(op->getLoc(), VectorType::get({16}, rewriter.getI8Type()), args[1]);
          }
          if (int4) {
            // Pack explicitly in i32: avoid sub-byte vector truncation in the
            // target legalizer and retain the byte-addressable logical ABI.
            auto i32 = rewriter.getI32Type();
            auto constant = [&](int value) -> Value {
              return rewriter.create<LLVM::ConstantOp>(op->getLoc(), i32, rewriter.getI32IntegerAttr(value));
            };
            for (unsigned side = 0; side != 2; ++side) {
              unsigned words = side == 0 ? 1 : 2;
              auto packedTy = VectorType::get({static_cast<int64_t>(words)}, i32);
              Value packed = rewriter.create<LLVM::UndefOp>(op->getLoc(), packedTy);
              for (unsigned word = 0; word != words; ++word) {
                Value bits = constant(0);
                for (unsigned nibble = 0; nibble != 8; ++nibble) {
                  Value byte = rewriter.create<LLVM::ExtractElementOp>(op->getLoc(), args[side], constant(word * 8 + nibble));
                  Value wide = rewriter.create<LLVM::ZExtOp>(op->getLoc(), i32, byte);
                  Value low = rewriter.create<LLVM::AndOp>(op->getLoc(), wide, constant(15));
                  Value shifted = rewriter.create<LLVM::ShlOp>(op->getLoc(), low, constant(nibble * 4));
                  bits = rewriter.create<LLVM::OrOp>(op->getLoc(), bits, shifted);
                }
                packed = rewriter.create<LLVM::InsertElementOp>(op->getLoc(), packedTy, packed, bits, constant(word));
              }
              if (side == 0)
                args[side] = rewriter.create<LLVM::ExtractElementOp>(op->getLoc(), packed, constant(0));
              else
                args[side] = packed;
            }
          } else {
            args[0] = rewriter.create<LLVM::BitcastOp>(op->getLoc(),
                VectorType::get({2}, rewriter.getI32Type()), args[0]);
            args[1] = rewriter.create<LLVM::BitcastOp>(op->getLoc(),
                VectorType::get({4}, rewriter.getI32Type()), args[1]);
          }
          if (integer) {
            auto sign = [&](StringRef name) {
              auto attr = op->getAttrOfType<BoolAttr>(name);
              return rewriter.create<LLVM::ConstantOp>(op->getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(!attr || attr.getValue()));
            };
            auto no = rewriter.create<LLVM::ConstantOp>(op->getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false));
            args = {sign("a_signed"), args[0], sign("b_signed"), args[1], args[2], args[3], no};
          }
          bool rhsFP8 = isa<Float8E4M3FNType>(cast<VectorType>(op->getOperand(1).getType()).getElementType());
          StringRef intrinsicName = integer ? (int4 ? "llvm.amdgcn.swmmac.i32.16x16x32.iu4" : "llvm.amdgcn.swmmac.i32.16x16x32.iu8") :
              (fp8 ? (rhsFP8 ? "llvm.amdgcn.swmmac.f32.16x16x32.fp8.fp8" : "llvm.amdgcn.swmmac.f32.16x16x32.fp8.bf8")
                   : (rhsFP8 ? "llvm.amdgcn.swmmac.f32.16x16x32.bf8.fp8" : "llvm.amdgcn.swmmac.f32.16x16x32.bf8.bf8"));
          auto call = rewriter.create<LLVM::CallIntrinsicOp>(op->getLoc(),
              op->getResult(0).getType(), rewriter.getStringAttr(intrinsicName), args);
          rewriter.replaceOp(op, call.getResult(0));
          continue;
        }
        bool bf16 = aTy.getElementType().isBF16();
        if (bf16) {
          args[0] = rewriter.create<LLVM::BitcastOp>(op->getLoc(),
              VectorType::get({8}, rewriter.getI16Type()), args[0]);
          args[1] = rewriter.create<LLVM::BitcastOp>(op->getLoc(),
              VectorType::get({16}, rewriter.getI16Type()), args[1]);
        }
        auto resultTy = cast<VectorType>(op->getResult(0).getType());
        bool lowAcc = !resultTy.getElementType().isF32();
        Type intrinsicTy = resultTy;
        if (bf16 && lowAcc) {
          intrinsicTy = VectorType::get({8}, rewriter.getI16Type());
          args[2] = rewriter.create<LLVM::BitcastOp>(op->getLoc(), intrinsicTy, args[2]);
        }
        StringRef name = bf16
            ? (lowAcc ? "llvm.amdgcn.swmmac.bf16.16x16x32.bf16" : "llvm.amdgcn.swmmac.f32.16x16x32.bf16")
            : (lowAcc ? "llvm.amdgcn.swmmac.f16.16x16x32.f16" : "llvm.amdgcn.swmmac.f32.16x16x32.f16");
        auto intrinsic = rewriter.create<LLVM::CallIntrinsicOp>(op->getLoc(),
            intrinsicTy, rewriter.getStringAttr(name), args);
        Value result = intrinsic.getResult(0);
        if (bf16 && lowAcc)
          result = rewriter.create<LLVM::BitcastOp>(op->getLoc(), resultTy, result);
        rewriter.replaceOp(op, result);
        continue;
      }

      // A matrix op carrying real fragment vectors lowers to the real ROCDL
      // intrinsic. Abstract/scalar contracts fail closed below.
      if (opName == "tessera_rocm.wmma" && lowerRealWMMA(op, rewriter))
        continue;
      if (opName == "tessera_rocm.mfma" && lowerRealMFMA(op, rewriter))
        continue;

      if (opName == "tessera_rocm.wmma" || opName == "tessera_rocm.mfma")
        op->emitError("executable ROCm matrix lowering requires typed hardware "
                      "fragment vectors; scalar/abstract contracts are target-only");
      else if (opName == "tessera_rocm.async_copy" ||
               opName == "tessera_rocm.wait")
        op->emitError("executable ROCm async operations must pass through "
                      "lower-rocm-async-copy with the memref/token contract");
      else
        op->emitError("ROCm target operation has no executable ROCDL lowering; "
                      "retain it with output=target until a physical consumer lands");
      signalPassFailure();
      return;
    }

    bool leakedROCMOp = false;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef().starts_with("tessera_rocm.")) {
        op->emitError("unsupported ROCm target op after ROCDL lowering");
        leakedROCMOp = true;
      }
    });
    if (leakedROCMOp)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::tessera_rocm::createLowerTesseraToROCDLImpl() {
  return std::make_unique<LoweringPass>();
}
