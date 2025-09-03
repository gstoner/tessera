#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/IR/MemRefOps.h"
#include "mlir/Dialect/MemRef/Utils/Utils.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

// Forward declarations for Tessera Tile IR ops (adjust include paths in real project)
namespace mlir {
namespace tessera {
namespace tile {
class CopyOp;
class GemmOp;
} // namespace tile
} // namespace tessera
} // namespace mlir

// Generated Metalium ops (expected when tblgen is wired)
#ifdef GEN_TESSERA_METALIUM_OPS_DECL
  #include "Tessera/Target/Metalium/TesseraMetaliumOps.h.inc"
#endif

using namespace mlir;

namespace {

// -------- Helpers: memref shape/strides & element size --------

static LogicalResult extract2DShapeAndStrides(MemRefType memTy,
                                              SmallVectorImpl<int64_t> &shape2D,
                                              SmallVectorImpl<int64_t> &strides2D) {
  if (!memTy || memTy.getRank() == 0)
    return failure();

  int64_t offset = 0;
  SmallVector<int64_t, 4> fullStrides;
  if (failed(getStridesAndOffset(memTy, fullStrides, offset))) {
    // Assume contiguous layout: row-major
    fullStrides.resize(memTy.getRank());
    int64_t running = 1;
    for (int i = memTy.getRank() - 1; i >= 0; --i) {
      fullStrides[i] = running;
      int64_t dim = memTy.getDimSize(i);
      running *= (dim == ShapedType::kDynamic ? 1 : dim);
    }
  }

  if (memTy.getRank() == 1) {
    int64_t n = memTy.getDimSize(0);
    shape2D.assign({n, 1});
    strides2D.assign({fullStrides[0], 1});
    return success();
  }

  int64_t r = memTy.getDimSize(memTy.getRank() - 2);
  int64_t c = memTy.getDimSize(memTy.getRank() - 1);
  int64_t rs = fullStrides[memTy.getRank() - 2];
  int64_t cs = fullStrides[memTy.getRank() - 1];

  shape2D.assign({r, c});
  strides2D.assign({rs, cs});
  return success();
}

static int64_t getElementByteWidth(Type t) {
  if (auto vt = dyn_cast<VectorType>(t))
    t = vt.getElementType();

  if (auto it = dyn_cast<IntegerType>(t)) return it.getWidth() / 8;
  if (auto ft = dyn_cast<FloatType>(t))   return ft.getWidth() / 8;
  return 0; // unknown
}

static StringRef getMemSpaceName(Type ty) {
  if (auto mr = dyn_cast<MemRefType>(ty)) {
    Attribute ms = mr.getMemorySpace();
    if (auto str = dyn_cast_or_null<StringAttr>(ms))
      return str.getValue();
    // If you wrap memorySpace in a custom attr, unwrap here.
  }
  return StringRef();
}

static StringAttr chooseDmaDirection(RewriterBase &rewriter, Type dstTy, Type srcTy) {
  auto srcSpace = getMemSpaceName(srcTy);
  auto dstSpace = getMemSpaceName(dstTy);
  if (srcSpace == "dram" && dstSpace == "sram")
    return rewriter.getStringAttr("dram_to_sram");
  if (srcSpace == "sram" && dstSpace == "dram")
    return rewriter.getStringAttr("sram_to_dram");
  if (srcSpace == "sram" && dstSpace == "sram")
    return rewriter.getStringAttr("sram_to_sram");
  if (srcSpace == "dram" && dstSpace == "dram")
    return rewriter.getStringAttr("dram_to_dram");
  // Fallback
  return rewriter.getStringAttr("dram_to_sram");
}

// -------- Conversion patterns --------

/// tessera.tile.copy → tessera_metalium.dma
struct TileCopyToMetaliumDMA : public OpConversionPattern<mlir::tessera::tile::CopyOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(mlir::tessera::tile::CopyOp op,
                                typename mlir::tessera::tile::CopyOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value src = adaptor.getOperands()[0];
    Value dst = adaptor.getOperands()[1];

    auto srcTy = dyn_cast<MemRefType>(getElementTypeOrSelf(src.getType()));
    auto dstTy = dyn_cast<MemRefType>(getElementTypeOrSelf(dst.getType()));
    if (!srcTy || !dstTy) {
      rewriter.eraseOp(op);
      return success();
    }

    SmallVector<int64_t, 2> srcShape2D, dstShape2D, srcStrides2D, dstStrides2D;
    (void)extract2DShapeAndStrides(srcTy, srcShape2D, srcStrides2D);
    (void)extract2DShapeAndStrides(dstTy, dstShape2D, dstStrides2D);

    int64_t rows = 1, cols = 1;
    if (srcShape2D.size() == 2 && dstShape2D.size() == 2) {
      rows = std::min(srcShape2D[0], dstShape2D[0]);
      cols = std::min(srcShape2D[1], dstShape2D[1]);
    }

    auto cRows = rewriter.create<arith::ConstantIndexOp>(loc, rows);
    auto cCols = rewriter.create<arith::ConstantIndexOp>(loc, cols);
    SmallVector<Value, 2> shapeVals{cRows, cCols};

    Value srcRowStride = rewriter.create<arith::ConstantIndexOp>(loc,
      srcStrides2D.size() == 2 ? srcStrides2D[0] : 1);
    Value srcColStride = rewriter.create<arith::ConstantIndexOp>(loc,
      srcStrides2D.size() == 2 ? srcStrides2D[1] : 1);
    Value dstRowStride = rewriter.create<arith::ConstantIndexOp>(loc,
      dstStrides2D.size() == 2 ? dstStrides2D[0] : 1);
    Value dstColStride = rewriter.create<arith::ConstantIndexOp>(loc,
      dstStrides2D.size() == 2 ? dstStrides2D[1] : 1);

    SmallVector<Value, 2> srcStrVals{srcRowStride, srcColStride};
    SmallVector<Value, 2> dstStrVals{dstRowStride, dstColStride};

    int64_t elemBytes = getElementByteWidth(srcTy.getElementType());
    if (elemBytes == 0) elemBytes = getElementByteWidth(dstTy.getElementType());

#ifdef GEN_TESSERA_METALIUM_OPS_DECL
    auto dirAttr = chooseDmaDirection(rewriter, dstTy, srcTy);
    auto elemAttr = rewriter.getI64IntegerAttr(elemBytes);
    auto burstAttr = rewriter.getI64IntegerAttr(256);

    (void)rewriter.create<mlir::tessera::metalium::Metalium_DmaOp>(
      loc, /*result types*/ TypeRange{},
      /*dst*/ dst, /*src*/ src,
      /*shape*/ ValueRange{shapeVals},
      /*dst_strides*/ ValueRange{dstStrVals},
      /*src_strides*/ ValueRange{srcStrVals},
      /*direction*/ dirAttr,
      /*element_size_bytes*/ elemAttr,
      /*burst*/ burstAttr,
      /*async*/ nullptr);
#endif

    rewriter.eraseOp(op);
    return success();
  }
};

/// tessera.tile.gemm → tessera_metalium.matmul with [M,N,K] from attrs if present.
struct TileGemmToMetaliumMatmul : public OpConversionPattern<mlir::tessera::tile::GemmOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(mlir::tessera::tile::GemmOp op,
                                typename mlir::tessera::tile::GemmOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

#ifdef GEN_TESSERA_METALIUM_OPS_DECL
    // Safe defaults
    int64_t M = 64, N = 64, K = 32;

    // If the GemmOp carries integer attrs "m","n","k" (or "tile_m", etc.), read them.
    if (auto dict = op->getAttrDictionary()) {
      auto getI64 = [&](StringRef key, int64_t &dst) {
        if (auto a = dict.get(key)) {
          if (auto ia = dyn_cast<IntegerAttr>(a)) dst = ia.getInt();
        }
      };
      getI64("m", M); getI64("n", N); getI64("k", K);
      getI64("tile_m", M); getI64("tile_n", N); getI64("tile_k", K);
    }

    auto tileAttr = rewriter.getI64ArrayAttr({M, N, K});
    // TODO: plumb real operands/types; we erase the op in this starter.
    // rewriter.replaceOpWithNewOp<mlir::tessera::metalium::Metalium_MatmulOp>(
    //   op, /*result type*/ op.getResult().getType(),
    //   A, B, Cinit, tileAttr, /*layout*/ nullptr, /*accum*/ nullptr);
#endif
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateTileToMetaliumPatterns(RewritePatternSet &patterns, TypeConverter &converter) {
  auto *ctx = patterns.getContext();
  patterns.add<TileCopyToMetaliumDMA, TileGemmToMetaliumMatmul>(ctx);
}

struct LowerTileToMetaliumPass : public PassWrapper<LowerTileToMetaliumPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTileToMetaliumPass)

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();

    RewritePatternSet patterns(ctx);
    TypeConverter converter;
    populateTileToMetaliumPatterns(patterns, converter);

    ConversionTarget target(*ctx);
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createLowerTileToMetaliumPass() {
  return std::make_unique<LowerTileToMetaliumPass>();
}
