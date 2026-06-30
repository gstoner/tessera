//===- GenerateROCMControlForWmmaTileKernel.cpp — multi-tile WMMA recurrence =//
//
// CF4d-4 of docs/audit/roadmap/CF_CROSS_ELEMENT_PLAN.md (ROCm-led, gfx1151,
// RDNA3.5). A MULTI-tile WMMA matmul recurrence: a control_for whose body is
// `carry = carry @ W`, carry an M×K f16 tile-grid and W a K×K f16 tile-grid
// (M, K multiples of 16) → carry @ W^max_iters on device, ONE dispatch.
//
// The CF4d-3 insight that makes this cheap: a carry of MT×KT WMMA tiles still
// fits in ONE workgroup's LDS (e.g. 64×64 f16 = 8 KB ≪ 64 KB), so we do NOT need
// a grid-wide barrier / cooperative launch. We launch ONE workgroup of MT*KT
// waves; each wave owns exactly one output tile (ti, tj) and accumulates it over
// the shared-K dimension:
//
//     D[ti][tj] = Σ_tk  carry[ti][tk] @ W[tk][tj]        (tk in 0..KT)
//
// The carry lives in LDS as a plain M×K f16 matrix (logical [row][col], off =
// row*K + col). Per control-loop iteration every wave (a) reads its whole carry
// row-block from LDS and accumulates its output tile, (b) a WORKGROUP barrier
// ensures all reads of the old carry finish, (c) writes its output tile back to
// LDS as the new carry, (d) a second barrier publishes it. Two barriers, no
// cross-workgroup sync. Larger carries (exceeding one workgroup's LDS) are the
// only case that would need grid.sync — out of scope here.
//
//   wave32 layout per wave (matching CF4d-3 / GenerateWMMAGemmKernel):
//     laneInWave = tid & 31 ; lane = laneInWave & 15 ; lhi = laneInWave >> 4
//     waveId = tid >> 5 ; ti = waveId / KT ; tj = waveId % KT
//     A-frag a[i] = carry[ti*16 + lane][tk*16 + i]      (LDS, per tk)
//     B-frag b[i] = W[tk*16 + i][tj*16 + lane]          (global, per tk, invariant)
//     acc   c[e] -> D[ti*16 + 2e+lhi][tj*16 + lane]     (8 outputs per lane)
//
// Fixed RDNA 16×16×16 WMMA tiles. Bodies/shapes outside this exact form are left
// for the CF4d-3 single-tile pass / the guard / SCF.

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

constexpr int64_t TILE = 16;     // RDNA WMMA 16x16x16
constexpr int64_t WAVE = 32;     // one wave (wave32)
// One workgroup must be co-resident on a single WGP, so the tile grid is bounded
// by how many of THIS kernel's waves fit there, not by the 1024-thread/workgroup
// hardware max. The WMMA accumulator + the A/B fragments push VGPR/wave high
// enough that gfx1151 holds 8 waves (256 threads) of this kernel per WGP;
// measured on-device, 9+ waves fail to launch (hipErrorLaunchFailure). We cap at
// 8 so the pass only ever emits a kernel that actually launches on the target —
// never a silently-unlaunchable one. Carries needing >8 tiles are the multi-
// workgroup / grid.sync frontier (CF4d-5, see CF_CROSS_ELEMENT_PLAN.md).
constexpr int64_t MAX_WAVES = 8;

// A multi-tile carry: carry M×K, W K×K, both f16, M/K multiples of 16, and the
// tile count MT*KT in [1, MAX_WAVES]. Body is exactly tessera.matmul(%carry, %W)
// no-transpose with result == carry shape. On success fills M, K and returns true.
static bool validateWmmaTileBody(Operation *op, SymbolTable &symTab, int64_t &M,
                                 int64_t &K) {
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
  auto carryT = dyn_cast<RankedTensorType>(ft.getInput(0));
  auto wT = dyn_cast<RankedTensorType>(ft.getInput(1));
  if (!carryT || !wT || carryT.getRank() != 2 || wT.getRank() != 2 ||
      !carryT.getElementType().isF16() || !wT.getElementType().isF16())
    return false;
  M = carryT.getDimSize(0);
  K = carryT.getDimSize(1);
  if (M <= 0 || K <= 0 || M % TILE != 0 || K % TILE != 0 ||
      wT.getDimSize(0) != K || wT.getDimSize(1) != K)
    return false;
  int64_t numWaves = (M / TILE) * (K / TILE);
  if (numWaves < 1 || numWaves > MAX_WAVES)
    return false;
  // result/operand types must all be the carry's M×K f16.
  if (ft.getResult(0) != Type(carryT) ||
      op->getOperand(0).getType() != Type(carryT) ||
      op->getResult(0).getType() != Type(carryT) ||
      op->getOperand(1).getType() != Type(wT))
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

struct GenerateROCMControlForWmmaTileKernelPass
    : public PassWrapper<GenerateROCMControlForWmmaTileKernelPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GenerateROCMControlForWmmaTileKernelPass)

  StringRef getArgument() const final {
    return "generate-rocm-control-for-wmma-tile-kernel";
  }
  StringRef getDescription() const final {
    return "CF4d-4: lower a multi-tile WMMA matmul recurrence control_for "
           "(carry M×K @ W K×K, f16) to ONE workgroup of MT*KT waves using "
           "rocdl.wmma.f32.16x16x16.f16 + a per-iteration LDS carry handoff "
           "(workgroup barrier, no grid.sync), for gfx1151.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect, vector::VectorDialect,
                    ROCDL::ROCDLDialect, func::FuncDialect>();
  }

  void emitKernel(Operation *op, ModuleOp module, unsigned idx, int64_t M,
                  int64_t K) {
    auto startA = op->getAttrOfType<IntegerAttr>("start");
    auto stopA = op->getAttrOfType<IntegerAttr>("stop");
    auto stepA = op->getAttrOfType<IntegerAttr>("step");
    if (!startA || !stopA || !stepA)
      return;

    const int64_t MT = M / TILE, KT = K / TILE;
    const int64_t numWaves = MT * KT;
    const int64_t blockDim = numWaves * WAVE;
    const int64_t elems = M * K;                  // LDS / CARRY / OUT element count
    const int64_t perThread = elems / blockDim;   // = 256/32 = 8, exact

    MLIRContext *ctx = module.getContext();
    OpBuilder b(module.getBodyRegion());
    b.setInsertionPointToEnd(module.getBody());
    Location loc = op->getLoc();
    std::string kname = ("tessera_control_for_wmma_tile_" + Twine(idx)).str();

    Type f16 = b.getF16Type(), f32 = b.getF32Type();
    auto fragTy = VectorType::get({16}, f16);   // A/B fragment (per lane)
    auto accTy = VectorType::get({8}, f32);      // accumulator (per lane)
    auto memTy = MemRefType::get({ShapedType::kDynamic}, f16);

    auto gpuMod = gpu::GPUModuleOp::create(b, loc, kname + "_mod");
    b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
    auto fnTy = b.getFunctionType({memTy, memTy, memTy}, {});  // CARRY, W, OUT
    auto f = gpu::GPUFuncOp::create(b, loc, kname, fnTy);
    f->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());

    auto ws = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
    auto ldsT = MemRefType::get({elems}, f16, MemRefLayoutAttrInterface(), ws);
    Value lds = f.addWorkgroupAttribution(ldsT, loc);

    OpBuilder kb(f.getContext());
    kb.setInsertionPointToStart(&f.getBody().front());
    Value CARRY = f.getArgument(0), W = f.getArgument(1), OUT = f.getArgument(2);
    auto ci = [&](int64_t v) { return arith::ConstantIndexOp::create(kb, loc, v); };
    Value c0 = ci(0), c1 = ci(1), c16 = ci(16), c32 = ci(WAVE), cK = ci(K);
    Value tid = gpu::ThreadIdOp::create(kb, loc, gpu::Dimension::x);
    Value laneInWave = arith::RemUIOp::create(kb, loc, tid, c32); // tid & 31
    Value lane = arith::RemUIOp::create(kb, loc, laneInWave, c16); // & 15
    Value lhi = arith::DivUIOp::create(kb, loc, laneInWave, c16);  // >> 4
    Value waveId = arith::DivUIOp::create(kb, loc, tid, c32);      // tid >> 5
    // ti = waveId / KT ; tj = waveId % KT  → this wave's output tile.
    Value ti = arith::DivUIOp::create(kb, loc, waveId, ci(KT));
    Value tj = arith::RemUIOp::create(kb, loc, waveId, ci(KT));
    Value ti16 = arith::MulIOp::create(kb, loc, ti, c16);
    Value tj16 = arith::MulIOp::create(kb, loc, tj, c16);
    // colB = tj*16 + lane is shared by the B-fragment and the acc store-back.
    Value colB = arith::AddIOp::create(kb, loc, tj16, lane);
    // rowA = ti*16 + lane is the A-fragment row.
    Value rowA = arith::AddIOp::create(kb, loc, ti16, lane);

    // Cooperatively copy `elems` f16 between two flat memrefs: each thread moves
    // `perThread` (= 8) strided elements.
    auto coopCopy = [&](Value src, Value dst) {
      auto lp = scf::ForOp::create(kb, loc, c0, ci(perThread), c1);
      OpBuilder::InsertionGuard ig(kb);
      kb.setInsertionPointToStart(lp.getBody());
      Value j = lp.getInductionVar();
      Value off = arith::AddIOp::create(
          kb, loc, tid, arith::MulIOp::create(kb, loc, j, ci(blockDim)));
      Value v = memref::LoadOp::create(kb, loc, src, ValueRange{off});
      memref::StoreOp::create(kb, loc, v, dst, ValueRange{off});
    };
    coopCopy(CARRY, lds);
    gpu::BarrierOp::create(kb, loc);

    // Build a fragment vector<16xf16> from a memref — UNROLLED (16 is the WMMA
    // tile, compile-time), so per-element insert positions are static.
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

    // B-fragments are loop-invariant across control-loop iterations (W is fixed)
    // and depend only on (tk, this thread's tj) — build all KT once, up front.
    // b_tk[i] = W[(tk*16 + i)*K + colB].
    SmallVector<Value> Bf(KT);
    for (int64_t tk = 0; tk < KT; ++tk) {
      Value tkBase = arith::AddIOp::create(
          kb, loc, arith::MulIOp::create(kb, loc, ci(tk * TILE), cK), colB);
      Bf[tk] = buildFrag(W, [&](int64_t i) {
        return arith::AddIOp::create(kb, loc, tkBase,
                                     arith::MulIOp::create(kb, loc, ci(i), cK));
      });
    }
    Value accZero = arith::ConstantOp::create(
        kb, loc, accTy, DenseElementsAttr::get(accTy, kb.getF32FloatAttr(0)));

    Value lb = ci(startA.getInt()), ub = ci(stopA.getInt()),
          st = ci(stepA.getInt());
    auto loop = scf::ForOp::create(kb, loc, lb, ub, st);
    {
      OpBuilder::InsertionGuard ig(kb);
      kb.setInsertionPointToStart(loop.getBody());
      // Accumulate this wave's output tile over the shared-K dimension:
      // acc = Σ_tk wmma(A[ti][tk], B[tk][tj]).  A-frag a[i] = lds[rowA*K + tk*16+i].
      Value acc = accZero;
      for (int64_t tk = 0; tk < KT; ++tk) {
        Value rowBase = arith::AddIOp::create(
            kb, loc, arith::MulIOp::create(kb, loc, rowA, cK), ci(tk * TILE));
        Value Af = buildFrag(lds, [&](int64_t i) {
          return arith::AddIOp::create(kb, loc, rowBase, ci(i));
        });
        acc = ROCDL::wmma_f32_16x16x16_f16::create(kb, loc, accTy,
                                                   ValueRange{Af, Bf[tk], acc});
      }
      gpu::BarrierOp::create(kb, loc);  // all A-frag reads of the old carry done
      // Store the accumulator as this wave's output tile of the new carry, in
      // LOGICAL [row][col] order: D[e] -> lds[(ti*16 + 2e+lhi)*K + colB].
      for (int64_t e = 0; e < 8; ++e) {
        Value de = vector::ExtractOp::create(kb, loc, acc, ArrayRef<int64_t>{e});
        Value h = arith::TruncFOp::create(kb, loc, f16, de);
        Value row = arith::AddIOp::create(
            kb, loc, ti16,
            arith::AddIOp::create(kb, loc, ci(2 * e), lhi));  // ti*16 + 2e + lhi
        Value off = arith::AddIOp::create(
            kb, loc, arith::MulIOp::create(kb, loc, row, cK), colB);
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
    for (Operation *op : fors) {
      int64_t M = 0, K = 0;
      if (validateWmmaTileBody(op, symTab, M, K))
        emitKernel(op, module, idx++, M, K);
    }
  }
};

}  // namespace

std::unique_ptr<Pass>
mlir::tessera_rocm::createGenerateROCMControlForWmmaTileKernelPass() {
  return std::make_unique<GenerateROCMControlForWmmaTileKernelPass>();
}
