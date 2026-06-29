//===- GenerateROCMCliffordKernel.cpp - geometric-algebra bilinear -------===//
//
// Expands a `tessera_rocm.clifford` directive into a table-driven
// geometric-algebra bilinear-product gpu kernel — the GA lane (P12 of
// S_SERIES_GAP_CLOSURE_PLAN) backing geometric_product / wedge /
// left_contraction on Cl(3,0) (8 blades).
//
// A multivector product is a STRUCTURED BILINEAR FORM driven by a compile-time
// (sign, i, j -> out-blade) Cayley table: out[k] = Σ sign·a[i]·b[j]. One thread
// per batch element loads the 8 blade coefficients of `a` and `b` (batch-major
// [n, 8]) and accumulates the 8 output blades by the triples for `kind`
// (0 = geometric_product, 1 = wedge, 2 = left_contraction). The triples are
// UNROLLED at generation time, so the Cayley table is a true compile-time
// constant — the GPU shape the gap plan calls out. The tables were lifted from
// tessera.ga.signature.Cl(3,0).product_table(). CPU analog: avx512_clifford_f32.
// Args: (A : memref<?xf32>, B : memref<?xf32>, n : index, out : memref<?xf32>).
//===----------------------------------------------------------------------===//

#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include <array>

using namespace mlir;

namespace {

static constexpr int64_t BD = 256;

struct Triple { int k, i, j, s; };  // out[k] += s * a[i] * b[j]

// Cl(3,0) geometric product — all 64 nonzero terms.
static const Triple GP[] = {
    {0,0,0,1},{1,0,1,1},{2,0,2,1},{3,0,3,1},{4,0,4,1},{5,0,5,1},{6,0,6,1},
    {7,0,7,1},{1,1,0,1},{0,1,1,1},{3,1,2,1},{2,1,3,1},{5,1,4,1},{4,1,5,1},
    {7,1,6,1},{6,1,7,1},{2,2,0,1},{3,2,1,-1},{0,2,2,1},{1,2,3,-1},{6,2,4,1},
    {7,2,5,-1},{4,2,6,1},{5,2,7,-1},{3,3,0,1},{2,3,1,-1},{1,3,2,1},{0,3,3,-1},
    {7,3,4,1},{6,3,5,-1},{5,3,6,1},{4,3,7,-1},{4,4,0,1},{5,4,1,-1},{6,4,2,-1},
    {7,4,3,1},{0,4,4,1},{1,4,5,-1},{2,4,6,-1},{3,4,7,1},{5,5,0,1},{4,5,1,-1},
    {7,5,2,-1},{6,5,3,1},{1,5,4,1},{0,5,5,-1},{3,5,6,-1},{2,5,7,1},{6,6,0,1},
    {7,6,1,1},{4,6,2,-1},{5,6,3,-1},{2,6,4,1},{3,6,5,1},{0,6,6,-1},{1,6,7,-1},
    {7,7,0,1},{6,7,1,1},{5,7,2,-1},{4,7,3,-1},{3,7,4,1},{2,7,5,1},{1,7,6,-1},
    {0,7,7,-1}};

// wedge — geometric-product terms with i & j == 0.
static const Triple WEDGE[] = {
    {0,0,0,1},{1,0,1,1},{2,0,2,1},{3,0,3,1},{4,0,4,1},{5,0,5,1},{6,0,6,1},
    {7,0,7,1},{1,1,0,1},{3,1,2,1},{5,1,4,1},{7,1,6,1},{2,2,0,1},{3,2,1,-1},
    {6,2,4,1},{7,2,5,-1},{3,3,0,1},{7,3,4,1},{4,4,0,1},{5,4,1,-1},{6,4,2,-1},
    {7,4,3,1},{5,5,0,1},{7,5,2,-1},{6,6,0,1},{7,6,1,1},{7,7,0,1}};

// left contraction — grade(out) == grade(j) - grade(i) >= 0.
static const Triple LC[] = {
    {0,0,0,1},{1,0,1,1},{2,0,2,1},{3,0,3,1},{4,0,4,1},{5,0,5,1},{6,0,6,1},
    {7,0,7,1},{0,1,1,1},{2,1,3,1},{4,1,5,1},{6,1,7,1},{0,2,2,1},{1,2,3,-1},
    {4,2,6,1},{5,2,7,-1},{0,3,3,-1},{4,3,7,-1},{0,4,4,1},{1,4,5,-1},{2,4,6,-1},
    {3,4,7,1},{0,5,5,-1},{2,5,7,1},{0,6,6,-1},{1,6,7,-1},{0,7,7,-1}};

void tableFor(StringRef kind, const Triple** t, int* nt) {
  if (kind == "wedge") { *t = WEDGE; *nt = (int)(sizeof(WEDGE)/sizeof(Triple)); }
  else if (kind == "left_contraction") {
    *t = LC; *nt = (int)(sizeof(LC)/sizeof(Triple));
  } else { *t = GP; *nt = (int)(sizeof(GP)/sizeof(Triple)); }
}

void emitCliffordBody(OpBuilder &b, Location loc, gpu::GPUFuncOp f,
                      StringRef kind) {
  Type f32 = b.getF32Type();
  Type idxTy = b.getIndexType();
  auto slt = arith::CmpIPredicate::slt;

  b.setInsertionPointToStart(&f.getBody().front());
  Value A = f.getArgument(0), B = f.getArgument(1);
  Value N = f.getArgument(2), OUT = f.getArgument(3);

  Value bid = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cBD = b.create<arith::ConstantIndexOp>(loc, BD);
  Value gid = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, bid, cBD), tid);
  Value inb = b.create<arith::CmpIOp>(loc, slt, gid, N);
  auto guard = b.create<scf::IfOp>(loc, inb, /*withElse=*/false);
  b.setInsertionPointToStart(guard.thenBlock());

  Value c8 = b.create<arith::ConstantIndexOp>(loc, 8);
  Value base = b.create<arith::MulIOp>(loc, gid, c8);

  // Load the 8 blade coefficients of a and b.
  std::array<Value, 8> a, bb;
  for (int t = 0; t < 8; ++t) {
    Value off = b.create<arith::AddIOp>(
        loc, base, b.create<arith::ConstantIndexOp>(loc, t));
    a[t] = b.create<memref::LoadOp>(loc, A, ValueRange{off});
    bb[t] = b.create<memref::LoadOp>(loc, B, ValueRange{off});
  }

  // Accumulate the 8 output blades from the (unrolled) Cayley triples.
  Value zero = b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(0.0f));
  std::array<Value, 8> acc;
  acc.fill(zero);
  const Triple *tbl;
  int ntri;
  tableFor(kind, &tbl, &ntri);
  for (int t = 0; t < ntri; ++t) {
    const Triple &tr = tbl[t];
    Value prod = b.create<arith::MulFOp>(loc, a[tr.i], bb[tr.j]);
    if (tr.s < 0)
      acc[tr.k] = b.create<arith::SubFOp>(loc, acc[tr.k], prod);
    else
      acc[tr.k] = b.create<arith::AddFOp>(loc, acc[tr.k], prod);
  }

  for (int t = 0; t < 8; ++t) {
    Value off = b.create<arith::AddIOp>(
        loc, base, b.create<arith::ConstantIndexOp>(loc, t));
    b.create<memref::StoreOp>(loc, acc[t], OUT, ValueRange{off});
  }

  b.setInsertionPointToEnd(&f.getBody().front());
  b.create<gpu::ReturnOp>(loc);
}

struct GenerateROCMCliffordKernelPass
    : PassWrapper<GenerateROCMCliffordKernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateROCMCliffordKernelPass)

  StringRef getArgument() const final { return "generate-rocm-clifford-kernel"; }
  StringRef getDescription() const final {
    return "Expand a tessera_rocm.clifford directive into a table-driven "
           "geometric-algebra bilinear-product gpu kernel (Cl(3,0))";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> directives;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "tessera_rocm.clifford")
        directives.push_back(op);
    });
    for (Operation *op : directives) {
      auto nameAttr = op->getAttrOfType<StringAttr>("name");
      auto kindAttr = op->getAttrOfType<StringAttr>("kind");
      if (!nameAttr || !kindAttr) {
        op->emitError("tessera_rocm.clifford missing name/kind");
        return signalPassFailure();
      }
      OpBuilder b(module.getBodyRegion());
      b.setInsertionPointToEnd(module.getBody());
      Location loc = op->getLoc();
      std::string kname = nameAttr.getValue().str();
      Type f32 = b.getF32Type();
      Type idxTy = b.getIndexType();
      auto memF32 = MemRefType::get({ShapedType::kDynamic}, f32);
      auto fnTy = b.getFunctionType({memF32, memF32, idxTy, memF32}, {});
      auto gpuMod = b.create<gpu::GPUModuleOp>(loc, kname + "_mod");
      b.setInsertionPointToStart(&gpuMod.getBodyRegion().front());
      auto gpuFunc = b.create<gpu::GPUFuncOp>(loc, kname, fnTy);
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
      OpBuilder body(gpuFunc.getContext());
      emitCliffordBody(body, loc, gpuFunc, kindAttr.getValue());
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createGenerateROCMCliffordKernelPass() {
  return std::make_unique<GenerateROCMCliffordKernelPass>();
}
