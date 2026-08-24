//===- TileDialect.cpp - Tessera Tile IR dialect --------------*- C++ -*-===//

#include "Tessera/Dialect/Tile/TileDialect.h"
#include "Tessera/Dialect/Tile/TileEpilogue.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

// C1 (2026-06-23) — generated attribute storage/printer/parser. Included BEFORE
// the dialect defs so the dialect's parse/printAttribute hooks resolve the
// generatedAttribute{Parser,Printer} helpers defined here.
#define GET_ATTRDEF_CLASSES
#include "Tessera/Dialect/Tile/TileAttrs.cpp.inc"

// Phase B (2026-06-23) — generated type storage/printer/parser (!tile.async_token).
#define GET_TYPEDEF_CLASSES
#include "Tessera/Dialect/Tile/TileTypes.cpp.inc"

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
      // memory storage axes (these are what alias — see C2). `m` is generic
      // linear memory; `lds` is AMD shared memory (LDS); `tlane`/`tcol` are
      // NVIDIA Blackwell TMEM.
      "m", "lds", "tlane", "tcol",
      // thread placement. `warpid` (NVIDIA warp) and `waveid` (AMD wave) are
      // the same axis named for each ISA — neither backend is privileged.
      "laneid", "warpid", "waveid", "reg", "tid_in_wg", "wid_in_wg",
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

namespace {

static LogicalResult verifyComposedTree(
    llvm::function_ref<InFlightDiagnostic()> emitError, Attribute value,
    bool shape, StringRef which, unsigned &leafCount) {
  if (auto integer = dyn_cast<IntegerAttr>(value)) {
    const int64_t leaf = integer.getInt();
    if (shape ? (leaf <= 0 && leaf != -1) : (leaf < 0 && leaf != -1))
      return emitError() << "TILE_COMPOSED_LAYOUT_BAD_LEAF: " << which
                         << (shape ? " shape" : " stride")
                         << " leaf must be " << (shape ? "> 0 or -1" : ">= 0 or -1")
                         << " (got " << leaf << ")";
    ++leafCount;
    return success();
  }
  auto group = dyn_cast<ArrayAttr>(value);
  if (!group || group.empty())
    return emitError() << "TILE_COMPOSED_LAYOUT_BAD_TREE: " << which
                       << " must be a non-empty nested integer array";
  for (Attribute child : group)
    if (failed(verifyComposedTree(emitError, child, shape, which, leafCount)))
      return failure();
  return success();
}

static bool sameComposedProfile(Attribute lhs, Attribute rhs) {
  const bool lhsLeaf = isa<IntegerAttr>(lhs);
  const bool rhsLeaf = isa<IntegerAttr>(rhs);
  if (lhsLeaf || rhsLeaf)
    return lhsLeaf == rhsLeaf;
  auto lhsGroup = dyn_cast<ArrayAttr>(lhs);
  auto rhsGroup = dyn_cast<ArrayAttr>(rhs);
  if (!lhsGroup || !rhsGroup || lhsGroup.size() != rhsGroup.size())
    return false;
  for (auto [left, right] : llvm::zip(lhsGroup, rhsGroup))
    if (!sameComposedProfile(left, right))
      return false;
  return true;
}

} // namespace

LogicalResult TileComposedLayoutAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, ArrayAttr outerShape,
    ArrayAttr outerStride, ArrayAttr basis, ArrayRef<int64_t> offsets) {
  unsigned outerLeaves = 0, outerStrideLeaves = 0;
  if (failed(verifyComposedTree(emitError, outerShape, /*shape=*/true,
                                "outer", outerLeaves)) ||
      failed(verifyComposedTree(emitError, outerStride, /*shape=*/false,
                                "outer", outerStrideLeaves)))
    return failure();
  if (!sameComposedProfile(outerShape, outerStride) ||
      outerLeaves != outerStrideLeaves)
    return emitError() << "TILE_COMPOSED_LAYOUT_PROFILE_MISMATCH: outer shape "
                       << "and stride trees must have identical nesting";
  if (basis.size() != outerLeaves || offsets.size() != outerLeaves)
    return emitError() << "TILE_COMPOSED_LAYOUT_BASIS_RANK: outer coordinate "
                       << "count, basis count, and offset count must match (got "
                       << outerLeaves << "/" << basis.size() << "/"
                       << offsets.size() << ")";
  for (Attribute entry : basis) {
    auto pair = dyn_cast<ArrayAttr>(entry);
    if (!pair || pair.size() != 2)
      return emitError() << "TILE_COMPOSED_LAYOUT_BAD_BASIS: each basis entry "
                         << "must be [shape_tree, stride_tree]";
    unsigned shapeLeaves = 0, strideLeaves = 0;
    if (failed(verifyComposedTree(emitError, pair[0], /*shape=*/true,
                                  "basis", shapeLeaves)) ||
        failed(verifyComposedTree(emitError, pair[1], /*shape=*/false,
                                  "basis", strideLeaves)))
      return failure();
    if (!sameComposedProfile(pair[0], pair[1]) || shapeLeaves != strideLeaves)
      return emitError() << "TILE_COMPOSED_LAYOUT_PROFILE_MISMATCH: basis "
                         << "shape and stride trees must have identical nesting";
  }
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
  // Completion-semantics domains, named per ISA — NVIDIA: tma (byte-count/
  // engine), tcgen05 (MMA-complete), mbarrier (thread-arrival); AMD: s_barrier
  // (workgroup/wave arrival), waitcnt (async vmcnt/lgkmcnt counter). Neither
  // backend is privileged in the contract.
  static const llvm::StringSet<> kKinds = {"tma",       "tcgen05",   "mbarrier",
                                           "s_barrier", "waitcnt"};
  if (!kKinds.contains(kind))
    return emitError() << "TILE_BARRIER_UNKNOWN_KIND: kind \"" << kind
                       << "\" is not one of {tma, tcgen05, mbarrier, s_barrier, "
                          "waitcnt}";
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

LogicalResult TileBufferRefAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, StringRef name,
    StringRef space, StringRef access) {
  if (name.empty())
    return emitError() << "TILE_BUFFER_REF_EMPTY_NAME: a buffer reference must "
                          "name a buffer";
  // Memory spaces, named per ISA. `smem` (NVIDIA shared) and `lds` (AMD Local
  // Data Share) are the same on-chip scratchpad named for each backend; `tmem`
  // is NVIDIA Blackwell tensor memory; `gmem` global; `reg` registers.
  static const llvm::StringSet<> kSpaces = {"smem", "lds",  "tmem",
                                            "gmem", "reg"};
  if (!kSpaces.contains(space))
    return emitError() << "TILE_BUFFER_REF_BAD_SPACE: space \"" << space
                       << "\" is not one of {smem, lds, tmem, gmem, reg}";
  static const llvm::StringSet<> kAccess = {"read", "write", "free"};
  if (!kAccess.contains(access))
    return emitError() << "TILE_BUFFER_REF_BAD_ACCESS: access \"" << access
                       << "\" is not one of {read, write, free}";
  return success();
}

LogicalResult TileScaleLayoutAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, StringRef dtype,
    int64_t blockSize, int64_t axis, StringRef layout, int64_t stride,
    int64_t alignment, int64_t offset) {
  bool absent = dtype == "none";
  if (absent) {
    if (blockSize != 0 || axis != -1 || layout != "none" || stride != 0 ||
        alignment != 1 || offset != 0)
      return emitError()
             << "TILE_SCALE_LAYOUT_INVALID: dtype=none requires "
                "block_size=0, axis=-1, layout=none, stride=0, alignment=1, "
                "and offset=0";
    return success();
  }
  static const llvm::StringSet<> kDtypes = {"ue4m3", "ue8m0", "f16", "bf16",
                                            "f32"};
  if (!kDtypes.contains(dtype))
    return emitError() << "TILE_SCALE_LAYOUT_INVALID: unsupported scale dtype "
                       << dtype;
  static const llvm::StringSet<> kLayouts = {"linear", "row_major",
                                             "col_major"};
  if (!kLayouts.contains(layout))
    return emitError() << "TILE_SCALE_LAYOUT_INVALID: layout must be "
                          "linear, row_major, or col_major";
  if (blockSize < 1 || axis < 0 || stride < 1)
    return emitError() << "TILE_SCALE_LAYOUT_INVALID: scaled storage "
                          "requires block_size>=1, axis>=0, and stride>=1";
  if (alignment < 1 || (alignment & (alignment - 1)) != 0)
    return emitError() << "TILE_SCALE_LAYOUT_INVALID: alignment must be "
                          "a positive power of two";
  if (offset < 0)
    return emitError()
           << "TILE_SCALE_LAYOUT_INVALID: offset must be >= 0";
  return success();
}

LogicalResult TilePackedFormatAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, StringRef logicalType,
    StringRef containerType, int64_t logicalBits,
    int64_t elementsPerContainer, StringRef signedness, StringRef encoding,
    StringRef laneOrder) {
  static const llvm::StringMap<int64_t> kContainerBits = {
      {"int8", 8}, {"uint8", 8}, {"int16", 16}, {"uint16", 16},
      {"int32", 32}, {"uint32", 32}};
  auto container = kContainerBits.find(containerType);
  if (container == kContainerBits.end())
    return emitError()
           << "TILE_PACKED_FORMAT_INVALID: container must be an explicitly "
              "sized integer storage type";
  if (logicalBits < 1 || logicalBits > container->second ||
      elementsPerContainer < 1 ||
      logicalBits * elementsPerContainer > container->second)
    return emitError()
           << "TILE_PACKED_FORMAT_INVALID: logical_bits and "
              "elements_per_container do not fit the physical container";

  static const llvm::StringSet<> kLogical = {
      "int4", "uint4", "fp4_e2m1", "nvfp4", "fp6_e2m3", "fp6_e3m2"};
  if (!kLogical.contains(logicalType))
    return emitError()
           << "TILE_PACKED_FORMAT_INVALID: unsupported packed logical dtype "
           << logicalType;
  static const llvm::StringSet<> kSignedness = {
      "signed_twos_complement", "unsigned", "format_defined"};
  if (!kSignedness.contains(signedness))
    return emitError()
           << "TILE_PACKED_FORMAT_INVALID: unsupported signedness "
           << signedness;
  static const llvm::StringSet<> kEncoding = {
      "twos_complement", "unsigned_integer", "e2m1", "nv_e2m1", "e2m3",
      "e3m2"};
  if (!kEncoding.contains(encoding))
    return emitError()
           << "TILE_PACKED_FORMAT_INVALID: unsupported packed encoding "
           << encoding;
  static const llvm::StringSet<> kLaneOrder = {
      "low_to_high", "high_to_low", "scalar_lsb"};
  if (!kLaneOrder.contains(laneOrder))
    return emitError()
           << "TILE_PACKED_FORMAT_INVALID: unsupported lane order "
           << laneOrder;

  if (logicalBits != (logicalType.starts_with("fp6_") ? 6 : 4))
    return emitError()
           << "TILE_PACKED_FORMAT_INVALID: logical_bits disagrees "
              "with the logical dtype";
  if (logicalType.starts_with("fp6_") && elementsPerContainer != 1)
    return emitError()
           << "TILE_PACKED_FORMAT_INVALID: FP6 byte-container "
              "storage requires elements_per_container=1 while retaining "
              "logical_bits=6";
  if (!logicalType.starts_with("fp6_") && containerType == "int8" &&
      elementsPerContainer != 2)
    return emitError()
           << "TILE_PACKED_FORMAT_INVALID: four-bit storage requires exactly "
              "two logical values per int8 container";
  if (logicalType == "int4" &&
      (signedness != "signed_twos_complement" ||
       encoding != "twos_complement"))
    return emitError()
           << "TILE_PACKED_FORMAT_INVALID: int4 requires signed "
              "two's-complement encoding";
  if (logicalType == "uint4" &&
      (signedness != "unsigned" || encoding != "unsigned_integer"))
    return emitError()
           << "TILE_PACKED_FORMAT_INVALID: uint4 requires unsigned "
              "integer encoding";
  return success();
}

LogicalResult TilePackedPhysicalViewAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    TilePackedFormatAttr format, int64_t packingAxis,
    ArrayRef<int64_t> strides, int64_t alignment, int64_t offset,
    TileScaleLayoutAttr scale) {
  if (!format)
    return emitError()
           << "TILE_PACKED_VIEW_INVALID: a packed-format descriptor is "
              "required";
  if (packingAxis < 0)
    return emitError()
           << "TILE_PACKED_VIEW_INVALID: packing_axis must be >= 0";
  if (strides.empty() || llvm::any_of(strides, [](int64_t value) {
        return value < 1;
      }))
    return emitError()
           << "TILE_PACKED_VIEW_INVALID: container strides must be "
              "non-empty and positive";
  if (packingAxis >= static_cast<int64_t>(strides.size()))
    return emitError()
           << "TILE_PACKED_VIEW_INVALID: packing_axis must index the stride "
              "rank";
  if (alignment < 1 || (alignment & (alignment - 1)) != 0)
    return emitError()
           << "TILE_PACKED_VIEW_INVALID: alignment must be a positive "
              "power of two";
  if (offset < 0)
    return emitError()
           << "TILE_PACKED_VIEW_INVALID: offset must be >= 0";
  if (!scale)
    return emitError()
           << "TILE_PACKED_VIEW_INVALID: a scale-layout "
              "descriptor is required, using dtype=none for unscaled storage";

  bool isInt = format.getLogicalType() == "int4" ||
               format.getLogicalType() == "uint4";
  if (isInt && scale.getDtype() != "none")
    return emitError()
           << "TILE_PACKED_VIEW_INVALID: integer packing does not accept "
              "a block-scale operand";
  if (!isInt && scale.getDtype() == "none")
    return emitError()
           << "TILE_PACKED_VIEW_INVALID: FP4/FP6 physical views "
              "require explicit scale metadata";
  if (!isInt && scale.getAxis() != packingAxis)
    return emitError()
           << "TILE_PACKED_VIEW_INVALID: block scales must index "
              "the packed logical axis";
  return success();
}

LogicalResult TileMmaDescAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, StringRef family,
    int64_t m, int64_t n, int64_t k, StringRef aType, StringRef bType,
    StringRef accType, StringRef aLayout, StringRef bLayout,
    int64_t kBlocks) {
  static const llvm::StringSet<> kFamilies = {
      "auto", "mma_sync", "wgmma", "tcgen05", "wmma", "mfma"};
  if (!kFamilies.contains(family))
    return emitError() << "TILE_MMA_DESC_BAD_FAMILY: family \"" << family
                       << "\" is not one of {auto, mma_sync, wgmma, tcgen05, "
                          "wmma, mfma}";
  if (m <= 0 || n <= 0 || k <= 0)
    return emitError() << "TILE_MMA_DESC_NONPOSITIVE_SHAPE: m/n/k must be > 0";
  if (kBlocks < 1)
    return emitError() << "TILE_MMA_DESC_BAD_K_BLOCKS: k_blocks must be >= 1";
  if (aType.empty() || bType.empty() || accType.empty())
    return emitError() << "TILE_MMA_DESC_EMPTY_DTYPE: a/b/acc must be named";
  static const llvm::StringSet<> kLayouts = {"row_major", "col_major"};
  if (!kLayouts.contains(aLayout) || !kLayouts.contains(bLayout))
    return emitError() << "TILE_MMA_DESC_BAD_LAYOUT: a_layout and b_layout "
                          "must be row_major or col_major";
  return success();
}

// W1.1 step 1 (PR #502 review) — domain validation for !tile.fragment.
//
// Deliberately mirrors TileMmaDescAttr::verify above, value for value: the two
// describe the same instruction contract from opposite sides, and letting them
// drift would mean a fragment could state a family or layout the descriptor
// rejects. Decision #31's one-rule principle -- if these ever need to differ,
// that is a finding, not a convenience.
//
// The bare `!tile.fragment` (all-unknown) is legal and is the legacy spelling.
// A PARTIALLY populated tuple is not: it is the dangerous middle state the
// review identified, because `isUnknown()` returns false for it, so step 2's
// type-based verification would read it as a stated contract when the producer
// only filled in half.
LogicalResult FragmentType::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, int64_t m, int64_t n,
    int64_t k, StringRef elem, StringRef acc, StringRef role, StringRef layout,
    StringRef family) {
  const bool shapeUnset = m == 0 && n == 0 && k == 0;
  const bool stringsUnset = elem.empty() && acc.empty() && role.empty() &&
                            layout.empty() && family.empty();
  if (shapeUnset && stringsUnset)
    return success();  // the legacy bare form
  if (shapeUnset || stringsUnset ||
      elem.empty() || acc.empty() || role.empty() || layout.empty() ||
      family.empty())
    return emitError()
           << "TILE_FRAGMENT_PARTIAL_CONTRACT: a fragment states its whole "
              "instruction contract or none of it; write `!tile.fragment` for "
              "the unknown form";

  static const llvm::StringSet<> kFamilies = {
      "auto", "mma_sync", "wgmma", "tcgen05", "wmma", "mfma"};
  if (!kFamilies.contains(family))
    return emitError() << "TILE_FRAGMENT_BAD_FAMILY: family \"" << family
                       << "\" is not one of {auto, mma_sync, wgmma, tcgen05, "
                          "wmma, mfma}";
  if (m <= 0 || n <= 0 || k <= 0)
    return emitError() << "TILE_FRAGMENT_NONPOSITIVE_SHAPE: m/n/k must be > 0";
  // `role` is a heavily overloaded name in this tree -- warp roles
  // (producer/consumer/manager), buffer roles (input/scratch), pass roles
  // (backward) all use it on OTHER ops. These five are the fragment roles, and
  // they are exactly the ones MMAOp::verify and the two backend lowerings read.
  static const llvm::StringSet<> kRoles = {"a", "b", "acc", "scale_a",
                                           "scale_b"};
  if (!kRoles.contains(role))
    return emitError() << "TILE_FRAGMENT_BAD_ROLE: role \"" << role
                       << "\" is not one of {a, b, acc, scale_a, scale_b}";
  static const llvm::StringSet<> kLayouts = {"row_major", "col_major"};
  if (!kLayouts.contains(layout))
    return emitError()
           << "TILE_FRAGMENT_BAD_LAYOUT: layout must be row_major or col_major";
  return success();
}

LogicalResult TileMemoryLayoutAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, StringRef space,
    StringRef order, int64_t leadingDim) {
  static const llvm::StringSet<> kSpaces = {"gmem", "smem", "lds"};
  static const llvm::StringSet<> kOrders = {"row_major", "col_major"};
  if (!kSpaces.contains(space))
    return emitError() << "TILE_MEMORY_LAYOUT_BAD_SPACE: space must be gmem, smem, or lds";
  if (!kOrders.contains(order))
    return emitError() << "TILE_MEMORY_LAYOUT_BAD_ORDER: order must be row_major or col_major";
  if (leadingDim < 0)
    return emitError() << "TILE_MEMORY_LAYOUT_BAD_LEADING_DIM: leading_dim must "
                          "be >= 0 (zero denotes an SSA-supplied dynamic value)";
  return success();
}

LogicalResult TileEpilogueAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, bool bias,
    StringRef activation, StringRef outputType) {
  (void)bias;
  if (!isSupportedActivation(activation))
    return emitError() << "TILE_EPILOGUE_BAD_ACTIVATION: activation must be "
                          "none, relu, gelu, or silu";
  if (!isSupportedOutputType(outputType))
    return emitError() << "TILE_EPILOGUE_BAD_OUTPUT: output must be f64, f32, f16, or i32";
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
  // Phase B (2026-06-23): first-class !tile.async_token type.
  addTypes<
#define GET_TYPEDEF_LIST
#include "Tessera/Dialect/Tile/TileTypes.cpp.inc"
      >();
  // The contraction/linalg value-lane ops and the four sync ops (async_copy /
  // wait_async / mma / s_barrier, registered in Phase A0) are all first-class
  // now. The artifact lane still emits other transient tile.* ops (tile.tma.* /
  // tile.tmem.* / tile.kv_cache / debug husks) that are not yet ODS-registered —
  // allow them as opaque so registering this dialect does not break the artifact
  // pipeline. The Apple *value* lane produces only registered ops, so it runs
  // with NO --allow-unregistered-dialect (the win); registering the remaining
  // tile ops is a follow-on.
  allowUnknownOperations(true);
}

void registerTileDialect(::mlir::DialectRegistry &registry) {
  registry.insert<TesseraTileDialect>();
}

bool isTileControlType(::mlir::Type type) {
  // Ordering / ownership handles, not data. `BufferType` is included to match
  // the set the ROCm lowering has used since it introduced this rule; changing
  // membership changes arity for every caller, so it is deliberately one list.
  return ::mlir::isa<AsyncTokenType, BufferType, PipelineStateType, RoleType>(
      type);
}

::llvm::SmallVector<::mlir::Value> dataOperands(::mlir::Operation *op) {
  ::llvm::SmallVector<::mlir::Value> values;
  for (::mlir::Value operand : op->getOperands())
    if (!isTileControlType(operand.getType()))
      values.push_back(operand);
  return values;
}

} // namespace tile
} // namespace tessera
