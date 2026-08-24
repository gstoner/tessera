//===- TileDialect.h - Tessera Tile IR dialect ----------------*- C++ -*-===//
//
// Sprint 9: registered Tile IR dialect (value-lane lowering spine). See
// TileOps.td.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_DIALECT_TILE_DIALECT_H
#define TESSERA_DIALECT_TILE_DIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Generated dialect declarations (cppNamespace = ::tessera::tile).
#include "Tessera/Dialect/Tile/TileOpsDialect.h.inc"

// C1 (2026-06-23) — generated attribute declarations (#tile.layout/#tile.swizzle).
// Must precede the op classes; some ops carry these attributes.
#define GET_ATTRDEF_CLASSES
#include "Tessera/Dialect/Tile/TileAttrs.h.inc"

// Phase B (2026-06-23) — generated type declarations (!tile.async_token). Must
// precede the op classes; ops may carry the token as an operand/result type.
#define GET_TYPEDEF_CLASSES
#include "Tessera/Dialect/Tile/TileTypes.h.inc"

#define GET_OP_CLASSES
#include "Tessera/Dialect/Tile/TileOps.h.inc"

namespace tessera {
namespace tile {

/// Insert the Tile IR dialect into a DialectRegistry. Call from tessera-opt and
/// from any pass (TilingPass) whose getDependentDialects creates tile.* ops.
void registerTileDialect(::mlir::DialectRegistry &registry);

/// Whether `type` is a Tile *control* type — an ordering/ownership handle
/// rather than a data operand.
///
/// Tile ops routinely carry both: `tile.mma` takes its A/B/accumulator data
/// operands **and** the `!tile.async_token` edge that `WarpSpecLegalityPass`
/// requires to gate the matrix op on copy completion
/// (`WARPSPEC_MMA_NOT_TOKEN_SYNCED`). Anything reasoning about arity has to
/// separate the two, or it will count an ordering edge as an operand.
///
/// This is deliberately Tile-level only. A backend that adds its own control
/// types composes rather than forks:
///
///     isTileControlType(t) || isa<mlir::tessera_rocm::TokenType>(t)
///
/// Tile IR must not depend on a backend dialect.
bool isTileControlType(::mlir::Type type);

/// The data operands of `op` — every operand that is not a Tile control type.
///
/// Previously a file-local `static` in one backend lowering
/// (`TileToROCM.cpp`), which is why the ODS verifier never learned the rule and
/// counted raw operands instead. That mismatch made the typed `tile.mma` form
/// and warp-spec token sync mutually exclusive: no producer could satisfy both.
::llvm::SmallVector<::mlir::Value> dataOperands(::mlir::Operation *op);

/// Revalidate a structured composed-layout carrier through the native L1
/// layout-algebra ABI.  Backend boundaries call this rather than trusting that
/// an earlier verifier happened to run.
bool isNativeMaterializableComposedLayout(TileComposedLayoutAttr layout);

/// One runtime leaf in the canonical preorder used by
/// `tile.materialize_composed_layout`: outer shape, outer stride, then each
/// basis shape/stride pair.  `mode` is the flattened logical mode after nested
/// outer tuples are expanded.
struct ComposedLayoutDynamicLeaf {
  enum class Kind { OuterShape, OuterStride, BasisShape, BasisStride } kind;
  unsigned mode;
  unsigned leaf = 0;
};

struct ComposedLayoutMaterializationMode {
  int64_t outerShape;
  int64_t outerStride;
  int64_t offset;
  ::llvm::SmallVector<int64_t, 4> basisShapes;
  ::llvm::SmallVector<int64_t, 4> basisStrides;
};

/// Extract every scalar-output layout accepted by the native authority. A
/// tuple-valued basis maps one coordinate through mixed-radix div/rem before
/// the outer stride is applied; this remains one exact i64 address, not a
/// flattened approximation.
bool getMaterializableComposedLayout(
    TileComposedLayoutAttr layout,
    ::llvm::SmallVectorImpl<ComposedLayoutMaterializationMode> &modes,
    ::llvm::SmallVectorImpl<ComposedLayoutDynamicLeaf> &dynamicLeaves);

/// Extract the affine composed-layout subset, preserving `-1` for a runtime
/// leaf and returning the exact operand order needed to resolve those leaves.
/// Compatibility extractor for the one-leaf affine subset.
bool getAffineComposedLayout(
    TileComposedLayoutAttr layout,
    ::llvm::SmallVectorImpl<int64_t> &outerStrides,
    ::llvm::SmallVectorImpl<int64_t> &basisStrides,
    ::llvm::SmallVectorImpl<int64_t> &offsets,
    ::llvm::SmallVectorImpl<ComposedLayoutDynamicLeaf> &dynamicLeaves);

/// Extract the static affine subset of a composed layout.  The arrays encode
/// `sum(outerStride[i] * (offset[i] + basisStride[i] * coordinate[i]))`.
/// Any runtime leaf or tuple-valued basis map deliberately returns false.
bool getStaticAffineComposedLayout(
    TileComposedLayoutAttr layout,
    ::llvm::SmallVectorImpl<int64_t> &outerStrides,
    ::llvm::SmallVectorImpl<int64_t> &basisStrides,
    ::llvm::SmallVectorImpl<int64_t> &offsets);

/// Materialize the canonical rank-2 `crd2idx` expression used by physical
/// emitters.  This is deliberately target-neutral: CUDA, ROCm, x86, and Apple
/// may choose different schedules, but a declared row/column-major physical
/// layout has exactly one index expression.
::mlir::FailureOr<::mlir::Value> materializeLinearIndex(
    ::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::Value row,
    ::mlir::Value column, ::mlir::Value leadingDim,
    ::llvm::StringRef order);

} // namespace tile
} // namespace tessera

#endif // TESSERA_DIALECT_TILE_DIALECT_H
