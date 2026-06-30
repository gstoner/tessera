//===- GenerateROCMControlForWmmaKernel.cpp — WMMA matmul recurrence -------===//
//
// CF4d-3 of docs/audit/roadmap/CF_CROSS_ELEMENT_PLAN.md (ROCm-led, gfx1151,
// RDNA3.5). A single-TILE WMMA matmul recurrence: a control_for whose body is
// `carry = carry @ W`, carry and W both 16x16 f16 — one RDNA `16x16x16` WMMA
// tile, one wave, no cross-workgroup sync → carry @ W^max_iters on device.
//
// The new piece vs the CF4d-1/2 LDS kernels is the accumulator→input FRAGMENT
// shuffle through LDS between iterations: the WMMA result is a vector<8xf32>
// accumulator fragment, but the next iteration needs `carry` as the vector<16xf16>
// A-fragment. We round-trip through LDS in LOGICAL [row][col] order (the WMMA
// store/load lane layouts both index the matrix logically), so the LDS is just a
// plain 16x16 f16 matrix and the handoff is layout-correct.
//
//   wave32 layout (matching GenerateWMMAGemmKernel / RDNA WMMA):
//     lane = tid & 15 ; lhi = tid >> 4
//     A-frag a[i] = carry[lane][i]        (row `lane`)              i in 0..15
//     B-frag b[i] = W[i][lane]            (col `lane`)              i in 0..15
//     acc   c[e] -> D[2*e + lhi][lane]    (8 outputs per lane)      e in 0..7
//
//   gpu.func @ctrl_for_wmma(%CARRY,%W,%OUT: memref<?xf16>) kernel {  // 1 wave
//     lds = workgroup memref<256xf16>
//     cooperatively load CARRY -> lds ; barrier
//     B = build b-frag from W (loop-invariant, once)
//     scf.for %i = 0 to max_iters {
//       A = build a-frag from lds
//       D = rocdl.wmma.f32.16x16x16.f16(A, B, 0)        // carry @ W
//       store (f16)D[e] -> lds[(2e+lhi)*16 + lane]      // new carry, logical
//       barrier
//     }
//     cooperatively store lds -> OUT
//   }
//
// Fixed 16x16x16 (the only RDNA WMMA tile). Larger carries (multi-tile) need a
// grid-wide barrier — CF4d-4. Bodies/shapes outside this exact form are left for
// the guard / SCF.

#include "TesseraROCM/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

constexpr int64_t TILE = 16;  // RDNA WMMA 16x16x16
constexpr int64_t WAVE = 32;  // one wave (wave32)

// Body must be exactly one tessera.matmul(%carry, %W) no-transpose, carry/W/
// result all 16x16 f16. control_for operands = [carry (idx 0), W]. Returns true.
static bool validateWmmaBody(Operation *op, SymbolTable &symTab) {
  auto bodySym = op->getAttrOfType<FlatSymbolRefAttr>("body");
  auto carryA = op->getAttrOfType<IntegerAttr>("carry_arg_index");
  if (!bodySym || !carryA || carryA.getInt() != 0 ||
      op->getNumOperands() != 2 || op->getNumResults() != 1)
    return false;
  auto fn = dyn_cast_or_null<func::FuncOp>(
      symTab.lookupNearestSymbolFrom(op, bodySym.getAttr()));
  if (!fn || fn.isExternal())
    return false;
  FunctionType ft = fn.getFunctionType();
  if (ft.getNumInputs() != 2 || ft.getNumResults() != 1)
    return false;
  auto is16x16f16 = [](Type t) {
    auto r = dyn_cast<RankedTensorType>(t);
    return r && r.getRank() == 2 && r.getDimSize(0) == TILE &&
           r.getDimSize(1) == TILE && r.getElementType().isF16();
  };
  auto carryT = ft.getInput(0);
  if (!is16x16f16(carryT) || !is16x16f16(ft.getInput(1)) ||
      ft.getResult(0) != carryT || op->getOperand(0).getType() != carryT ||
      op->getResult(0).getType() != carryT ||
      op->getOperand(1).getType() != ft.getInput(1))
    return false;
  Block &blk = fn.getBody().front();
  auto it = blk.begin();
  if (it == blk.end() || it->getName().getStringRef() != "tessera.matmul" ||
      it->getNumOperands() != 2 || it->getOperand(0) != blk.getArgument(0) ||
      it->getOperand(1) != blk.getArgument(1) || it->getNumResults() != 1)
    return false;
  auto ta = it->getAttrOfType<BoolAttr>("transposeA");
  auto tb = it->getAttrOfType<BoolAttr>("transposeB");
  if ((ta && ta.getValue()) || (tb && tb.getValue()))
    return false;
  auto ret = dyn_cast<func::ReturnOp>(std::next(it));
  return ret && ret.getNumOperands() == 1 &&
         ret.getOperand(0) == it->getResult(0);
}

struct GenerateROCMControlForWmmaKernelPass
    : public PassWrapper<GenerateROCMControlForWmmaKernelPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMControlForWmmaKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-control-for-wmma-kernel";
  }
  StringRef getDescription() const final {
    return "CF4d-3: lower a single-tile WMMA matmul recurrence control_for "
           "(carry @ W, both 16x16 f16) to one wave gpu.func using "
           "rocdl.wmma.f32.16x16x16.f16 + an accumulator->input LDS fragment "
           "shuffle per iteration, for gfx1151.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect, vector::VectorDialect,
                    ROCDL::ROCDLDialect, func::FuncDialect>();
  }

  void emitKernel(Operation *op, ModuleOp module, unsigned idx) {
    auto startA = op->getAttrOfType<IntegerAttr>("start");
    auto stopA = op->getAttrOfType<IntegerAttr>("stop");
    auto stepA = op->getAttrOfType<IntegerAttr>("step");
    if (!startA || !stopA || !stepA)
      return;

    MLIRContext *ctx = module.getContext();
    OpBuilder b(module.getBodyRegion());
    b.setInsertionPointToEnd(module.getBody());
    Location loc = op->getLoc();
    std::string kname = ("tessera_control_for_wmma_" + Twine(idx)).str();

    Type f16 = b.getF16Type(), f32 = b.getF32Type(), idxTy = b.getIndexType();
    auto fragTy = VectorType::get({16}, f16);   // A/B fragment (per lane)
    auto accTy = VectorType::get({8}, f32);      // accumulator (per lane)
    auto memTy = MemRefType::get({ShapedType::kDynamic}, f16);

    auto gpuMod = gpu::GPUModuleOp::create(b, loc, kname + "_mod");
    b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
    auto fnTy = b.getFunctionType({memTy, memTy, memTy}, {});  // CARRY, W, OUT
    auto f = gpu::GPUFuncOp::create(b, loc, kname, fnTy);
    f->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());

    auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
    auto ldsT = MemRefType::get({TILE * TILE}, f16, MemRefLayoutAttrInterface(),
                                ws);
    Value lds = f.addWorkgroupAttribution(ldsT, loc);

    OpBuilder kb(f.getContext());
    kb.setInsertionPointToStart(&f.getBody().front());
    Value CARRY = f.getArgument(0), W = f.getArgument(1), OUT = f.getArgument(2);
    auto ci = [&](int64_t v) { return arith::ConstantIndexOp::create(kb, loc, v); };
    Value c0 = ci(0), c1 = ci(1), c16 = ci(16);
    Value tid = gpu::ThreadIdOp::create(kb, loc, gpu::Dimension::x);
    Value lane = arith::RemUIOp::create(kb, loc, tid, c16);   // tid & 15
    Value lhi = arith::DivUIOp::create(kb, loc, tid, c16);    // tid >> 4

    // Cooperatively load CARRY -> LDS: 256 elems, WAVE threads, 8 each.
    auto coopCopy = [&](Value src, Value dst) {
      auto lp = scf::ForOp::create(kb, loc, c0, ci(TILE * TILE / WAVE), c1);
      OpBuilder::InsertionGuard ig(kb);
      kb.setInsertionPointToStart(lp.getBody());
      Value j = lp.getInductionVar();
      Value off = arith::AddIOp::create(
          kb, loc, tid, arith::MulIOp::create(kb, loc, j, ci(WAVE)));
      Value v = memref::LoadOp::create(kb, loc, src, ValueRange{off});
      memref::StoreOp::create(kb, loc, v, dst, ValueRange{off});
    };
    coopCopy(CARRY, lds);
    gpu::BarrierOp::create(kb, loc);

    // Build a fragment vector<16xf16> from a memref — UNROLLED (16 is the WMMA
    // tile, a compile-time constant), so the per-element index/insert use static
    // positions. idxOf(i) maps element i (compile-time) to its (dynamic) memref
    // offset.
    auto buildFrag = [&](Value mem,
                         llvm::function_ref<Value(int64_t)> idxOf) -> Value {
      Value frag = arith::ConstantOp::create(
          kb, loc, fragTy, DenseElementsAttr::get(fragTy, kb.getF16FloatAttr(0)));
      for (int64_t i = 0; i < TILE; ++i) {
        Value v = memref::LoadOp::create(kb, loc, mem, ValueRange{idxOf(i)});
        frag = vector::InsertOp::create(kb, loc, v, frag, ArrayRef<int64_t>{i});
      }
      return frag;
    };

    // B-frag = column `lane` of W (loop-invariant): b[i] = W[i*16 + lane].
    Value Bf = buildFrag(W, [&](int64_t i) {
      return arith::AddIOp::create(kb, loc, ci(i * TILE), lane);
    });
    Value accZero = arith::ConstantOp::create(
        kb, loc, accTy, DenseElementsAttr::get(accTy, kb.getF32FloatAttr(0)));

    Value lb = ci(startA.getInt()), ub = ci(stopA.getInt()),
          st = ci(stepA.getInt());
    auto loop = scf::ForOp::create(kb, loc, lb, ub, st);
    {
      OpBuilder::InsertionGuard ig(kb);
      kb.setInsertionPointToStart(loop.getBody());
      // A-frag = row `lane` of the LDS carry: a[i] = lds[lane*16 + i].
      Value laneRow = arith::MulIOp::create(kb, loc, lane, c16);
      Value Af = buildFrag(lds, [&](int64_t i) {
        return arith::AddIOp::create(kb, loc, laneRow, ci(i));
      });
      Value D = ROCDL::wmma_f32_16x16x16_f16::create(kb, loc, accTy,
                                                     ValueRange{Af, Bf, accZero});
      gpu::BarrierOp::create(kb, loc);  // all A-frag reads of lds done
      // Store the accumulator back as the new carry (logical [2e+lhi][lane]),
      // UNROLLED over the 8 acc elements (static extract positions).
      for (int64_t e = 0; e < 8; ++e) {
        Value de = vector::ExtractOp::create(kb, loc, D, ArrayRef<int64_t>{e});
        Value h = arith::TruncFOp::create(kb, loc, f16, de);
        // row = 2*e + lhi ; col = lane ; off = row*16 + col
        Value row = arith::AddIOp::create(kb, loc, ci(2 * e), lhi);
        Value off = arith::AddIOp::create(
            kb, loc, arith::MulIOp::create(kb, loc, row, c16), lane);
        memref::StoreOp::create(kb, loc, h, lds, ValueRange{off});
      }
      gpu::BarrierOp::create(kb, loc);  // new carry visible before next A-frag
    }

    coopCopy(lds, OUT);
    kb.setInsertionPointToEnd(&f.getBody().front());
    gpu::ReturnOp::create(kb, loc);
    op->setAttr("tessera.rocm_kernel", b.getStringAttr(kname));
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symTab(module);
    SmallVector<Operation *> fors;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera.control_for")
        fors.push_back(op);
    });
    unsigned idx = 0;
    for (Operation *op : fors)
      if (validateWmmaBody(op, symTab))
        emitKernel(op, module, idx++);
  }
};

}  // namespace

std::unique_ptr<Pass>
mlir::tessera_rocm::createGenerateROCMControlForWmmaKernelPass() {
  return std::make_unique<GenerateROCMControlForWmmaKernelPass>();
}
