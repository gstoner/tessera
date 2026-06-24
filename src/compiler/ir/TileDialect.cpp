//===- TileDialect.cpp - Tessera Tile IR dialect --------------*- C++ -*-===//

#include "Tessera/Dialect/Tile/TileDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

// C1 (2026-06-23) — generated attribute storage/printer/parser. Included BEFORE
// the dialect defs so the dialect's parse/printAttribute hooks resolve the
// generatedAttribute{Parser,Printer} helpers defined here.
#define GET_ATTRDEF_CLASSES
#include "Tessera/Dialect/Tile/TileAttrs.cpp.inc"

#include "Tessera/Dialect/Tile/TileOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Tessera/Dialect/Tile/TileOps.cpp.inc"

using namespace mlir;

namespace tessera {
namespace tile {

//===----------------------------------------------------------------------===//
// C1 — TileLayoutAttr verifier
//===----------------------------------------------------------------------===//

// The named hardware axes a layout stride may reference. Membership is the
// contract; an unrecognized axis name is a hard error (Decision: no silent
// no-ops — surface a clear diagnostic).
static const llvm::StringSet<> &knownHardwareAxes() {
  static const llvm::StringSet<> kSet = {
      // memory / TMEM storage axes (these are what alias — see C2)
      "m", "tlane", "tcol",
      // thread placement
      "laneid", "warpid", "reg", "tid_in_wg", "wid_in_wg",
      // grid / cluster placement
      "bx", "by", "bz", "cbx", "cby", "cbz",
      // device placement (same algebra as ShardSpec, Decision #3)
      "gpuid_x", "gpuid_y",
  };
  return kSet;
}

LogicalResult TileLayoutAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<int64_t> shardExtents, ArrayRef<int64_t> shardStrides,
    ArrayRef<StringAttr> shardAxes, ArrayRef<int64_t> replicaCounts,
    ArrayRef<int64_t> replicaStrides, ArrayRef<StringAttr> replicaAxes,
    int64_t offset, TileSwizzleAttr swizzle) {
  auto checkTriple = [&](StringRef which, ArrayRef<int64_t> extents,
                         ArrayRef<int64_t> strides,
                         ArrayRef<StringAttr> axes) -> LogicalResult {
    if (extents.size() != strides.size() || extents.size() != axes.size())
      return emitError()
             << "TILE_LAYOUT_RANK_MISMATCH: " << which
             << " extents/strides/axes must have equal length (got "
             << extents.size() << "/" << strides.size() << "/" << axes.size()
             << ")";
    for (int64_t e : extents)
      if (e <= 0)
        return emitError() << "TILE_LAYOUT_NONPOSITIVE_EXTENT: " << which
                           << " extent must be > 0 (got " << e << ")";
    for (StringAttr ax : axes)
      if (!knownHardwareAxes().contains(ax.getValue()))
        return emitError()
               << "TILE_LAYOUT_UNKNOWN_AXIS: " << which << " axis \""
               << ax.getValue()
               << "\" is not a known hardware axis {m, tlane, tcol, laneid, "
                  "warpid, reg, tid_in_wg, wid_in_wg, bx, by, bz, cbx, cby, "
                  "cbz, gpuid_x, gpuid_y}";
    return success();
  };
  if (failed(checkTriple("shard", shardExtents, shardStrides, shardAxes)))
    return failure();
  if (failed(
          checkTriple("replica", replicaCounts, replicaStrides, replicaAxes)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// C1 — TileLayoutAttr custom assembly (empty `[]` arrays must parse)
//===----------------------------------------------------------------------===//

static ParseResult parseIntArray(AsmParser &p, SmallVectorImpl<int64_t> &out) {
  return p.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() {
    int64_t v;
    if (p.parseInteger(v))
      return failure();
    out.push_back(v);
    return success();
  });
}

static ParseResult parseStrArray(AsmParser &p,
                                 SmallVectorImpl<StringAttr> &out) {
  MLIRContext *ctx = p.getContext();
  return p.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() {
    std::string s;
    if (p.parseString(&s))
      return failure();
    out.push_back(StringAttr::get(ctx, s));
    return success();
  });
}

static void printStrArray(AsmPrinter &p, ArrayRef<StringAttr> axes) {
  p << "[";
  llvm::interleaveComma(axes, p,
                        [&](StringAttr a) { p << "\"" << a.getValue() << "\""; });
  p << "]";
}

Attribute TileLayoutAttr::parse(AsmParser &p, Type) {
  MLIRContext *ctx = p.getContext();
  SmallVector<int64_t> se, ss, rc, rs;
  SmallVector<StringAttr> sa, ra;
  int64_t offset = 0;
  TileSwizzleAttr swizzle;

  if (p.parseLess() || p.parseKeyword("shard") || p.parseEqual() ||
      parseIntArray(p, se) || p.parseColon() || parseIntArray(p, ss) ||
      p.parseKeyword("on") || parseStrArray(p, sa) || p.parseComma() ||
      p.parseKeyword("replica") || p.parseEqual() || parseIntArray(p, rc) ||
      p.parseColon() || parseIntArray(p, rs) || p.parseKeyword("on") ||
      parseStrArray(p, ra) || p.parseComma() || p.parseKeyword("offset") ||
      p.parseEqual() || p.parseInteger(offset))
    return {};

  if (succeeded(p.parseOptionalComma())) {
    Attribute sw;
    if (p.parseKeyword("swizzle") || p.parseEqual() || p.parseAttribute(sw))
      return {};
    swizzle = dyn_cast<TileSwizzleAttr>(sw);
    if (!swizzle) {
      p.emitError(p.getCurrentLocation(),
                  "TILE_LAYOUT_BAD_SWIZZLE: expected a #tile.swizzle attribute");
      return {};
    }
  }
  if (p.parseGreater())
    return {};

  return TileLayoutAttr::getChecked(
      [&]() { return p.emitError(p.getNameLoc()); }, ctx, se, ss, sa, rc, rs, ra,
      offset, swizzle);
}

void TileLayoutAttr::print(AsmPrinter &p) const {
  p << "<shard = [";
  llvm::interleaveComma(getShardExtents(), p);
  p << "] : [";
  llvm::interleaveComma(getShardStrides(), p);
  p << "] on ";
  printStrArray(p, getShardAxes());
  p << ", replica = [";
  llvm::interleaveComma(getReplicaCounts(), p);
  p << "] : [";
  llvm::interleaveComma(getReplicaStrides(), p);
  p << "] on ";
  printStrArray(p, getReplicaAxes());
  p << ", offset = " << getOffset();
  if (TileSwizzleAttr sw = getSwizzle())
    p << ", swizzle = " << sw;
  p << ">";
}

//===----------------------------------------------------------------------===//
// C3 — typed-barrier + pipeline-state verifiers
//===----------------------------------------------------------------------===//

LogicalResult
TileBarrierAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                        StringRef kind, int64_t expect) {
  static const llvm::StringSet<> kKinds = {"tma", "tcgen05", "mbarrier"};
  if (!kKinds.contains(kind))
    return emitError() << "TILE_BARRIER_UNKNOWN_KIND: kind \"" << kind
                       << "\" is not one of {tma, tcgen05, mbarrier}";
  if (expect < 0)
    return emitError() << "TILE_BARRIER_NEGATIVE_EXPECT: expect must be >= 0 "
                          "(got "
                       << expect << ")";
  return success();
}

LogicalResult TilePipelineStateAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, int64_t depth,
    int64_t stage, int64_t phase, StringRef role) {
  if (depth < 1)
    return emitError() << "TILE_PIPELINE_BAD_DEPTH: depth must be >= 1 (got "
                       << depth << ")";
  if (stage < 0 || stage >= depth)
    return emitError() << "TILE_PIPELINE_STAGE_OOB: stage " << stage
                       << " is not in [0, " << depth << ")";
  if (phase != 0 && phase != 1)
    return emitError() << "TILE_PIPELINE_BAD_PHASE: phase must be 0 or 1 (got "
                       << phase << ")";
  static const llvm::StringSet<> kRoles = {"producer", "consumer"};
  if (!kRoles.contains(role))
    return emitError() << "TILE_PIPELINE_BAD_ROLE: role \"" << role
                       << "\" is not one of {producer, consumer}";
  return success();
}

LogicalResult TilePipelineDepthsAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, int64_t q, int64_t kv,
    int64_t tmem) {
  if (q < 1 || kv < 1 || tmem < 1)
    return emitError()
           << "TILE_PIPELINE_DEPTHS_NONPOSITIVE: each ring depth (q/kv/tmem) "
              "must be >= 1 (got q="
           << q << ", kv=" << kv << ", tmem=" << tmem << ")";
  return success();
}

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

void TesseraTileDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Tessera/Dialect/Tile/TileOps.cpp.inc"
      >();
  // C1 (2026-06-23): first-class layout attributes.
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Tessera/Dialect/Tile/TileAttrs.cpp.inc"
      >();
  // Sprint 9: the value-lane contraction + linalg ops above are registered and
  // verified. The artifact lane still emits other transient tile.* ops
  // (tile.mma / tile.async_copy / tile.kv_cache / debug husks) that are not yet
  // ODS-registered — allow them as opaque so registering this dialect does not
  // break the artifact pipeline. The Apple *value* lane produces only the
  // registered ops, so it runs with NO --allow-unregistered-dialect (the win);
  // registering the remaining tile ops is a follow-on.
  allowUnknownOperations(true);
}

void registerTileDialect(::mlir::DialectRegistry &registry) {
  registry.insert<TesseraTileDialect>();
}

} // namespace tile
} // namespace tessera
