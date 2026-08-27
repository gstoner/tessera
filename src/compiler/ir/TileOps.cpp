//===- TileOps.cpp - Tessera Tile IR op verifiers -------------*- C++ -*-===//
//
// Sprint 9: rank/shape verifiers for the contraction tile ops. These mirror the
// Graph IR MatmulOp / BatchedGemmOp verifiers — the value lane only ever
// produces in-envelope tile ops, and this re-checks at the Tile layer so a
// malformed tile op can never reach the Target lowering.
//
//===----------------------------------------------------------------------===//

#include "Tessera/Dialect/Tile/TileDialect.h"
#include "tessera/LayoutAlgebra.h"
#include "tessera/Rank2Index.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace tessera {
namespace tile {

namespace {

static bool encodeLayoutTree(Attribute attr,
                             SmallVectorImpl<TesseraLayoutNodeV1> &out) {
  if (auto integer = dyn_cast<IntegerAttr>(attr)) {
    out.push_back({integer.getInt(), 0});
    return true;
  }
  auto group = dyn_cast<ArrayAttr>(attr);
  if (!group || group.empty() || group.size() > UINT32_MAX)
    return false;
  out.push_back({0, static_cast<uint32_t>(group.size())});
  for (Attribute child : group)
    if (!encodeLayoutTree(child, out))
      return false;
  return true;
}

static bool nativeCoalescible(Attribute shape, Attribute stride) {
  SmallVector<TesseraLayoutNodeV1> shapeNodes, strideNodes;
  if (!encodeLayoutTree(shape, shapeNodes) || !encodeLayoutTree(stride, strideNodes))
    return false;
  const size_t capacity = 2 * (shapeNodes.size() + strideNodes.size() + 1);
  SmallVector<TesseraLayoutNodeV1> outputShape(capacity), outputStride(capacity);
  size_t outputShapeCount = 0, outputStrideCount = 0;
  return tessera_layout_coalesce_v1(
             shapeNodes.data(), shapeNodes.size(), strideNodes.data(),
             strideNodes.size(), outputShape.data(), outputShape.size(),
             outputStride.data(), outputStride.size(), &outputShapeCount,
             &outputStrideCount) == TESSERA_LAYOUT_OK;
}

static bool flattenAffineLeaves(Attribute attr,
                                SmallVectorImpl<int64_t> &out) {
  if (auto integer = dyn_cast<IntegerAttr>(attr)) {
    if (integer.getInt() < -1)
      return false;
    out.push_back(integer.getInt());
    return true;
  }
  auto group = dyn_cast<ArrayAttr>(attr);
  if (!group || group.empty())
    return false;
  for (Attribute child : group)
    if (!flattenAffineLeaves(child, out))
      return false;
  return true;
}

} // namespace

bool isNativeMaterializableComposedLayout(TileComposedLayoutAttr layout) {
  if (!nativeCoalescible(layout.getOuterShape(), layout.getOuterStride()))
    return false;
  for (Attribute entry : layout.getBasis()) {
    auto pair = dyn_cast<ArrayAttr>(entry);
    if (!pair || pair.size() != 2 || !nativeCoalescible(pair[0], pair[1]))
      return false;
  }
  return true;
}

bool getStaticAffineComposedLayout(
    TileComposedLayoutAttr layout, SmallVectorImpl<int64_t> &outerStrides,
    SmallVectorImpl<int64_t> &basisStrides,
    SmallVectorImpl<int64_t> &offsets) {
  SmallVector<ComposedLayoutDynamicLeaf> dynamicLeaves;
  if (!getAffineComposedLayout(layout, outerStrides, basisStrides, offsets,
                               dynamicLeaves) ||
      !dynamicLeaves.empty())
    return false;
  return true;
}

bool getMaterializableComposedLayout(
    TileComposedLayoutAttr layout,
    SmallVectorImpl<ComposedLayoutMaterializationMode> &modes,
    SmallVectorImpl<ComposedLayoutDynamicLeaf> &dynamicLeaves) {
  SmallVector<int64_t> outerShape, outerStrides;
  if (!flattenAffineLeaves(layout.getOuterShape(), outerShape) ||
      !flattenAffineLeaves(layout.getOuterStride(), outerStrides) ||
      outerShape.size() != outerStrides.size() ||
      outerShape.size() != layout.getBasis().size() ||
      outerShape.size() != layout.getOffsets().size() ||
      !isNativeMaterializableComposedLayout(layout))
    return false;
  for (auto [mode, value] : llvm::enumerate(outerShape))
    if (value == -1)
      dynamicLeaves.push_back({ComposedLayoutDynamicLeaf::Kind::OuterShape,
                               static_cast<unsigned>(mode), 0});
  for (auto [mode, value] : llvm::enumerate(outerStrides))
    if (value == -1)
      dynamicLeaves.push_back({ComposedLayoutDynamicLeaf::Kind::OuterStride,
                               static_cast<unsigned>(mode), 0});
  for (auto [mode, entry] : llvm::enumerate(layout.getBasis())) {
    auto pair = dyn_cast<ArrayAttr>(entry);
    if (!pair || pair.size() != 2)
      return false;
    ComposedLayoutMaterializationMode result;
    result.outerShape = outerShape[mode];
    result.outerStride = outerStrides[mode];
    result.offset = layout.getOffsets()[mode];
    if (!flattenAffineLeaves(pair[0], result.basisShapes) ||
        !flattenAffineLeaves(pair[1], result.basisStrides) ||
        result.basisShapes.size() != result.basisStrides.size() ||
        result.basisShapes.empty())
      return false;
    for (auto [leaf, value] : llvm::enumerate(result.basisShapes))
      if (value == -1)
        dynamicLeaves.push_back({ComposedLayoutDynamicLeaf::Kind::BasisShape,
                                 static_cast<unsigned>(mode),
                                 static_cast<unsigned>(leaf)});
    for (auto [leaf, value] : llvm::enumerate(result.basisStrides))
      if (value == -1)
        dynamicLeaves.push_back({ComposedLayoutDynamicLeaf::Kind::BasisStride,
                                 static_cast<unsigned>(mode),
                                 static_cast<unsigned>(leaf)});
    modes.push_back(std::move(result));
  }
  return true;
}

bool getAffineComposedLayout(
    TileComposedLayoutAttr layout, SmallVectorImpl<int64_t> &outerStrides,
    SmallVectorImpl<int64_t> &basisStrides,
    SmallVectorImpl<int64_t> &offsets,
    SmallVectorImpl<ComposedLayoutDynamicLeaf> &dynamicLeaves) {
  SmallVector<ComposedLayoutMaterializationMode> modes;
  if (!getMaterializableComposedLayout(layout, modes, dynamicLeaves))
    return false;
  for (const auto &mode : modes) {
    if (mode.basisShapes.size() != 1)
      return false;
    outerStrides.push_back(mode.outerStride);
    basisStrides.push_back(mode.basisStrides.front());
    offsets.push_back(mode.offset);
  }
  return true;
}

FailureOr<Value> materializeLinearIndex(OpBuilder &builder, Location loc,
                                        Value row, Value column,
                                        Value leadingDim, StringRef order) {
  if (!row.getType().isIntOrIndex() || !column.getType().isIntOrIndex() ||
      row.getType() != column.getType() || row.getType() != leadingDim.getType())
    return failure();
  layout::Rank2Order rank2Order;
  if (order == "row_major")
    rank2Order = layout::Rank2Order::RowMajor;
  else if (order == "col_major")
    rank2Order = layout::Rank2Order::ColumnMajor;
  else
    return failure();

  Value coordinates[2] = {row, column};
  const layout::Rank2IndexPlan plan = layout::rank2IndexPlan(rank2Order);
  Value scaled = arith::MulIOp::create(
      builder, loc, coordinates[plan.majorCoordinate], leadingDim);
  return Value(arith::AddIOp::create(
      builder, loc, scaled, coordinates[plan.minorCoordinate]));
}

namespace {
// Read an optional discardable bool attr (transposeA/transposeB), default false.
bool boolAttr(Operation *op, llvm::StringRef name) {
  if (auto a = op->getAttrOfType<BoolAttr>(name))
    return a.getValue();
  return false;
}

bool sameStaticShape(RankedTensorType a, RankedTensorType b) {
  if (a.getRank() != b.getRank())
    return false;
  if (!a.hasStaticShape() || !b.hasStaticShape())
    return false;
  for (int64_t i = 0, e = a.getRank(); i < e; ++i)
    if (a.getDimSize(i) != b.getDimSize(i))
      return false;
  return true;
}

LogicalResult verifyCollective(Operation *op, Value input, Value output,
                               bool scalesAxis, bool multiply,
                               bool requiresReduction) {
  auto meshAxis = op->getAttrOfType<StringAttr>("mesh_axis");
  auto tensorAxis = op->getAttrOfType<IntegerAttr>("tensor_axis");
  auto reduction = op->getAttrOfType<StringAttr>("reduction");
  auto worldSize = op->getAttrOfType<IntegerAttr>("world_size");
  if (!meshAxis || meshAxis.getValue().empty())
    return op->emitOpError("requires a non-empty mesh_axis");
  if (!tensorAxis)
    return op->emitOpError("requires tensor_axis");
  if (worldSize && worldSize.getInt() <= 0)
    return op->emitOpError("world_size must be positive when present");
  if (requiresReduction &&
      (!reduction ||
       !llvm::is_contained({"sum", "mean", "max", "min", "prod"},
                           reduction.getValue())))
    return op->emitOpError(
        "reduction must be one of sum, mean, max, min, prod");

  auto inTy = dyn_cast<RankedTensorType>(input.getType());
  auto outTy = dyn_cast<RankedTensorType>(output.getType());
  if (!inTy || !outTy)
    return success();
  if (inTy.getRank() != outTy.getRank() ||
      inTy.getElementType() != outTy.getElementType())
    return op->emitOpError("must preserve rank and element type");
  int64_t axis = tensorAxis.getInt();
  if (axis < 0)
    axis += inTy.getRank();
  if (axis < 0 || axis >= inTy.getRank())
    return op->emitOpError("tensor_axis is out of range");
  for (int64_t dim = 0; dim < inTy.getRank(); ++dim) {
    int64_t in = inTy.getDimSize(dim);
    int64_t out = outTy.getDimSize(dim);
    if (ShapedType::isDynamic(in) || ShapedType::isDynamic(out))
      continue;
    if (dim != axis || !scalesAxis) {
      if (in != out)
        return op->emitOpError("must preserve every non-collective dimension");
      continue;
    }
    if (!worldSize)
      continue;
    int64_t size = worldSize.getInt();
    bool valid = multiply ? out == in * size : in == out * size;
    if (!valid)
      return op->emitOpError(
          "collective-axis extent disagrees with world_size");
  }
  return success();
}

LogicalResult requireBoolAttr(Operation *op, llvm::StringRef name,
                              bool expected) {
  if (boolAttr(op, name) != expected)
    return op->emitOpError()
           << name << " must be " << (expected ? "true" : "false");
  return success();
}

TileMmaDescAttr mmaDescAttr(Operation *op) {
  return op->getAttrOfType<TileMmaDescAttr>("mma");
}

// W1.1 step 2 — the parameterized fragment, if this value carries one.
// Null for a bare `!tile.fragment` (legacy) or a non-fragment.
FragmentType typedFragment(Value v) {
  auto f = dyn_cast<FragmentType>(v.getType());
  return (f && !f.isUnknown()) ? f : FragmentType();
}

// Does a `#tile.mma_desc` agree with ONE fragment type, given that fragment's
// role? (PR #503 review.)
//
// The first version compared only m/n/k/family/accType, which let a descriptor
// contradict the type on exactly the fields codegen reads: a bf16 `elem` paired
// with `a = "f16"`, or a `row_major` fragment paired with
// `a_layout = "col_major"`. Both `NVIDIALowering.cpp` and `TileToROCM.cpp`
// select the instruction variant and the physical register layout from the
// DESCRIPTOR, so that IR verifies while codegen follows a different contract
// than the type states -- silently wrong, in the one place the two sources of
// truth still overlap.
//
// Role-dependent because the descriptor has always carried A and B separately
// (`aType`/`bType`, `a_layout`/`b_layout`); that asymmetry is why the shared
// checks in `verifyMMAFromTypes` deliberately exclude `elem` and `layout`.
// One implementation, used by both the consumer and the producer side.
LogicalResult descriptorAgreesWithFragment(Operation *op, TileMmaDescAttr desc,
                                           FragmentType f) {
  auto err = [&]() -> InFlightDiagnostic {
    return op->emitOpError() << "TILE_MMA_DESC_DISAGREES: ";
  };
  if (desc.getM() != f.getM() || desc.getN() != f.getN() ||
      desc.getK() != f.getK())
    return err() << "#tile.mma_desc states m/n/k that the fragment type does not";
  if (desc.getFamily() != f.getFamily())
    return err() << "#tile.mma_desc family \"" << desc.getFamily()
                 << "\" contradicts the fragment type's \"" << f.getFamily()
                 << "\"";
  if (desc.getAccType() != f.getAcc())
    return err() << "#tile.mma_desc acc \"" << desc.getAccType()
                 << "\" contradicts the fragment type's \"" << f.getAcc() << "\"";

  StringRef role = f.getRole();
  StringRef descElem, descLayout, which;
  if (role == "a") {
    descElem = desc.getAType(); descLayout = desc.getALayout(); which = "a";
  } else if (role == "b") {
    descElem = desc.getBType(); descLayout = desc.getBLayout(); which = "b";
  } else if (role == "acc") {
    descElem = desc.getAccType(); which = "acc";
  } else {
    // scale_a / scale_b: the descriptor has no per-scale element or layout
    // field, so there is nothing further to cross-check.
    return success();
  }
  if (descElem != f.getElem())
    return err() << "#tile.mma_desc " << which << " = \"" << descElem
                 << "\" contradicts the fragment type's elem \"" << f.getElem()
                 << "\" -- codegen selects the instruction from the descriptor";
  if (!descLayout.empty() && descLayout != f.getLayout())
    return err() << "#tile.mma_desc " << which << "_layout \"" << descLayout
                 << "\" contradicts the fragment type's layout \""
                 << f.getLayout()
                 << "\" -- codegen selects the register layout from the "
                    "descriptor";
  return success();
}

// The typed `tile.mma` contract, read from the OPERAND TYPES.
//
// This is W1.1's point. The legacy path below recovers the same facts by
// chasing each operand's defining op, which cannot cross a block-argument edge
// -- and a K-loop accumulator is an `scf.for` iter-arg by construction, so the
// typed form was unusable by every real GEMM (W1_1_TYPING_DESIGN.md §2). Read
// from the type and a block argument is just another value.
//
// The result type equals the accumulator type on purpose: that is what lets
// `scf.for`'s own iter-arg/yield equality close the loop, with no Tile-specific
// loop reasoning anywhere.
LogicalResult verifyMMAFromTypes(Operation *op, ArrayRef<Value> data) {
  auto err = [&]() -> InFlightDiagnostic { return op->emitOpError(); };

  // No mixing. A half-migrated op would otherwise get the weaker of the two
  // contracts on whichever operand still carried the bare type.
  for (auto [index, value] : llvm::enumerate(data))
    if (!typedFragment(value))
      return err() << "TILE_MMA_MIXED_FRAGMENT_FORMS: operand " << index
                   << " is not a parameterized !tile.fragment; the typed form "
                      "is all-or-nothing";

  FragmentType a = typedFragment(data[0]);
  const bool isNVFP4 = a.getElem() == "nvfp4" || a.getElem() == "fp4_e2m1";
  const size_t expected = isNVFP4 ? 5 : 3;
  if (data.size() != expected)
    return err() << "TILE_MMA_ARITY: expected " << expected
                 << " data operands (" 
                 << (isNVFP4 ? "A, B, accumulator, scale_a, scale_b"
                             : "A, B, accumulator")
                 << "), got " << data.size();

  static const char *kRoles[] = {"a", "b", "acc", "scale_a", "scale_b"};
  for (size_t i = 0; i < data.size(); ++i)
    if (typedFragment(data[i]).getRole() != kRoles[i])
      return err() << "TILE_MMA_OPERAND_ROLE: operand " << i
                   << " must have role \"" << kRoles[i] << "\", got \""
                   << typedFragment(data[i]).getRole() << "\"";

  // Shared instruction context. `elem` and `layout` are deliberately NOT here:
  // `#tile.mma_desc` has always carried aType/bType and a_layout/b_layout
  // separately, so A and B may legitimately differ in both.
  for (size_t i = 1; i < data.size(); ++i) {
    FragmentType f = typedFragment(data[i]);
    if (f.getM() != a.getM() || f.getN() != a.getN() || f.getK() != a.getK())
      return err() << "TILE_MMA_SHAPE_MISMATCH: operand " << i
                   << " states a different m/n/k than operand 0";
    if (f.getFamily() != a.getFamily())
      return err() << "TILE_MMA_FAMILY_MISMATCH: operand " << i << " states "
                      "family \"" << f.getFamily() << "\" but operand 0 states \""
                   << a.getFamily()
                   << "\". Family selects a physical register ABI (wave 32 for "
                      "RDNA/WMMA vs 64 for CDNA/MFMA), so these fragments are "
                      "not interchangeable";
    if (f.getAcc() != a.getAcc())
      return err() << "TILE_MMA_ACCUM_MISMATCH: operand " << i
                   << " states accumulator \"" << f.getAcc()
                   << "\" but operand 0 states \"" << a.getAcc()
                   << "\" (Decision #15a: one accumulator contract per MMA)";
  }

  // The accumulator's own element type IS the accumulator dtype. Without this a
  // producer could hand over a bf16-element fragment while every operand agreed
  // `acc = "f32"`, which is the accumulator-width confusion Decision #15a and
  // W1.3 both exist to prevent.
  FragmentType acc = typedFragment(data[2]);
  if (acc.getElem() != acc.getAcc())
    return err() << "TILE_MMA_ACCUM_ELEMENT: the accumulator fragment's elem (\""
                 << acc.getElem() << "\") must be its acc (\"" << acc.getAcc()
                 << "\")";

  if (op->getNumResults() != 1 || op->getResult(0).getType() != acc)
    return err() << "TILE_MMA_RESULT_TYPE: tile.mma returns exactly one value, "
                    "the updated accumulator, whose type must equal the "
                    "accumulator operand's";

  // The descriptor is now OPTIONAL -- the type carries everything except
  // `k_blocks`. When present it must agree, so a stale descriptor cannot
  // contradict the types it used to be the sole source of.
  if (auto desc = mmaDescAttr(op))
    for (Value v : data)
      if (failed(descriptorAgreesWithFragment(op, desc, typedFragment(v))))
        return failure();
  return success();
}

int64_t i64AttrOr(Operation *op, llvm::StringRef name, int64_t fallback) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return fallback;
}

int64_t bladeCountFor(int64_t p, int64_t q) {
  int64_t n = p + q;
  if (n < 0 || n > 30)
    return -1;
  return int64_t{1} << n;
}

bool isStaticF32RankedTensor(Type ty, RankedTensorType &out) {
  out = dyn_cast<RankedTensorType>(ty);
  return out && out.hasStaticShape() && out.getElementType().isF32();
}

LogicalResult verifyCliffordAttrs(Operation *op) {
  int64_t p = i64AttrOr(op, "p", -1);
  int64_t q = i64AttrOr(op, "q", -1);
  if (p != 3 || q != 0)
    return op->emitOpError("Clifford Tile value seam currently requires p=3, q=0");
  if (bladeCountFor(p, q) != 8)
    return op->emitOpError("Clifford Tile value seam expects cl30 coefficient axis");

  if (Attribute sigAttr = op->getAttr("signature")) {
    SmallVector<int64_t> signature;
    if (auto dense = dyn_cast<DenseI64ArrayAttr>(sigAttr)) {
      signature.append(dense.asArrayRef().begin(), dense.asArrayRef().end());
    } else if (auto array = dyn_cast<ArrayAttr>(sigAttr)) {
      for (Attribute element : array) {
        auto intAttr = dyn_cast<IntegerAttr>(element);
        if (!intAttr)
          return op->emitOpError("signature must be an i64 array");
        signature.push_back(intAttr.getInt());
      }
    } else {
      return op->emitOpError("signature must be an i64 array");
    }
    if (signature.size() != 3)
      return op->emitOpError("signature length must equal p+q");
    for (int64_t v : signature)
      if (v != 1 && v != -1)
        return op->emitOpError("signature entries must be +1 or -1");
  }
  return success();
}

LogicalResult verifyCliffordTensor(Operation *op, RankedTensorType ty,
                                   llvm::StringRef label) {
  if (!ty)
    return success();
  if (ty.getRank() < 1)
    return op->emitOpError() << label << " must include a coefficient axis";
  if (ty.getDimSize(ty.getRank() - 1) != 8)
    return op->emitOpError()
           << label << " coefficient axis must equal 8 for cl30";
  return success();
}

LogicalResult verifyGradeMask(Operation *op) {
  Attribute maskAttr = op->getAttr("grade_mask");
  if (!maskAttr)
    return success();
  SmallVector<int64_t> mask;
  if (auto dense = dyn_cast<DenseI64ArrayAttr>(maskAttr)) {
    mask.append(dense.asArrayRef().begin(), dense.asArrayRef().end());
  } else if (auto array = dyn_cast<ArrayAttr>(maskAttr)) {
    for (Attribute element : array) {
      auto intAttr = dyn_cast<IntegerAttr>(element);
      if (!intAttr)
        return op->emitOpError("grade_mask must be an i64 array");
      mask.push_back(intAttr.getInt());
    }
  } else {
    return op->emitOpError("grade_mask must be an i64 array");
  }
  if (mask.empty())
    return op->emitOpError("grade_mask must not be empty");
  for (int64_t grade : mask)
    if (grade < 0 || grade > 3)
      return op->emitOpError("grade_mask entries must be in [0, p+q]");
  return success();
}

LogicalResult verifyCliffordBinary(Operation *op, ValueRange inputs,
                                   ValueRange outputs,
                                   llvm::StringRef label) {
  if (inputs.size() != 2 || outputs.size() != 1)
    return op->emitOpError("expects exactly 2 inputs and 1 result");
  if (failed(verifyCliffordAttrs(op)) || failed(verifyGradeMask(op)))
    return failure();
  RankedTensorType lhs, rhs, res;
  if (!isStaticF32RankedTensor(inputs[0].getType(), lhs) ||
      !isStaticF32RankedTensor(inputs[1].getType(), rhs) ||
      !isStaticF32RankedTensor(outputs[0].getType(), res))
    return op->emitOpError()
           << label << " expects static fp32 tensors and result";
  if (failed(verifyCliffordTensor(op, lhs, "lhs")) ||
      failed(verifyCliffordTensor(op, rhs, "rhs")) ||
      failed(verifyCliffordTensor(op, res, "result")))
    return failure();
  if (!sameStaticShape(lhs, rhs) || !sameStaticShape(lhs, res))
    return op->emitOpError() << label << " shapes must match exactly";
  return success();
}

LogicalResult verifyCliffordUnary(Operation *op, ValueRange inputs,
                                  ValueRange outputs,
                                  llvm::StringRef label,
                                  bool allowNormDrop = false) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return op->emitOpError("expects exactly 1 input and 1 result");
  if (failed(verifyCliffordAttrs(op)) || failed(verifyGradeMask(op)))
    return failure();
  RankedTensorType input, res;
  if (!isStaticF32RankedTensor(inputs[0].getType(), input) ||
      !isStaticF32RankedTensor(outputs[0].getType(), res))
    return op->emitOpError()
           << label << " expects static fp32 tensors and result";
  if (failed(verifyCliffordTensor(op, input, "input")))
    return failure();
  if (allowNormDrop && res.getRank() == input.getRank() - 1) {
    for (int64_t i = 0, e = res.getRank(); i < e; ++i)
      if (res.getDimSize(i) != input.getDimSize(i))
        return op->emitOpError("norm result batch dimensions must match input");
    return success();
  }
  if (failed(verifyCliffordTensor(op, res, "result")))
    return failure();
  if (!sameStaticShape(input, res))
    return op->emitOpError() << label << " shapes must match exactly";
  return success();
}

// Shared rank-2 matmul contract (honors transposeA/transposeB discardable
// attrs). lhs/rhs/result must be rank-2; K, M, N consistent.
LogicalResult verifyRank2Matmul(Operation *op, ValueRange inputs,
                                ValueRange outputs) {
  if (inputs.size() != 2 || outputs.size() != 1)
    return op->emitOpError("expects exactly 2 inputs and 1 result");
  auto lhs = dyn_cast<RankedTensorType>(inputs[0].getType());
  auto rhs = dyn_cast<RankedTensorType>(inputs[1].getType());
  auto res = dyn_cast<RankedTensorType>(outputs[0].getType());
  if (!lhs || !rhs || !res)
    return success(); // unranked — nothing to check
  if (lhs.getRank() != 2 || rhs.getRank() != 2 || res.getRank() != 2)
    return op->emitOpError("expects rank-2 lhs, rhs, and result tensors");
  bool ta = boolAttr(op, "transposeA"), tb = boolAttr(op, "transposeB");
  int64_t lk = ta ? lhs.getDimSize(0) : lhs.getDimSize(1);
  int64_t rk = tb ? rhs.getDimSize(1) : rhs.getDimSize(0);
  auto dyn = [](int64_t d) { return ShapedType::isDynamic(d); };
  if (!dyn(lk) && !dyn(rk) && lk != rk)
    return op->emitOpError("contracting dimensions must match");
  int64_t m = ta ? lhs.getDimSize(1) : lhs.getDimSize(0);
  int64_t n = tb ? rhs.getDimSize(0) : rhs.getDimSize(1);
  if (!dyn(m) && !dyn(res.getDimSize(0)) && m != res.getDimSize(0))
    return op->emitOpError("result row dimension must equal lhs M");
  if (!dyn(n) && !dyn(res.getDimSize(1)) && n != res.getDimSize(1))
    return op->emitOpError("result column dimension must equal rhs N");
  return success();
}

LogicalResult verifyTileControl(Operation *op, ValueRange inputs,
                                ValueRange outputs) {
  auto verifyStatic = [&](ValueRange values) -> LogicalResult {
    for (Value value : values) {
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type || !type.hasStaticShape())
        return op->emitOpError(
            "NVIDIA control-flow Tile contract requires static ranked tensors");
    }
    return success();
  };
  if (failed(verifyStatic(inputs)) || failed(verifyStatic(outputs)))
    return failure();
  StringRef name = op->getName().getStringRef();
  if (!op->getAttrOfType<StringAttr>("source"))
    return op->emitOpError("requires a source Graph op attribute");

  if (name == "tile.control_for") {
    auto start = op->getAttrOfType<IntegerAttr>("start");
    auto stop = op->getAttrOfType<IntegerAttr>("stop");
    auto step = op->getAttrOfType<IntegerAttr>("step");
    if (!start || !stop || !step || step.getInt() == 0)
      return op->emitOpError("requires start/stop and a non-zero step");
    int64_t carry = 0;
    if (auto attr = op->getAttrOfType<IntegerAttr>("carry_arg_index"))
      carry = attr.getInt();
    if (carry < 0 || carry >= static_cast<int64_t>(inputs.size()) ||
        outputs.size() != 1 || outputs[0].getType() != inputs[carry].getType())
      return op->emitOpError("requires one shape-stable carried result");
  } else if (name == "tile.control_if") {
    auto flag = op->getAttrOfType<IntegerAttr>("flag_arg_index");
    if (!flag || flag.getInt() < 0 ||
        flag.getInt() >= static_cast<int64_t>(inputs.size()) || outputs.empty())
      return op->emitOpError("requires an in-range flag and branch result");
  } else if (name == "tile.control_while") {
    auto maxIters = op->getAttrOfType<IntegerAttr>("max_iters");
    auto carry = op->getAttrOfType<IntegerAttr>("carry_arg_index");
    if (!maxIters || maxIters.getInt() <= 0 || !carry || carry.getInt() < 0 ||
        carry.getInt() >= static_cast<int64_t>(inputs.size()) ||
        outputs.size() != 1 ||
        outputs[0].getType() != inputs[carry.getInt()].getType())
      return op->emitOpError(
          "requires max_iters>0 and one shape-stable carried result");
  } else if (name == "tile.control_scan") {
    auto trip = op->getAttrOfType<IntegerAttr>("trip");
    if (!trip || trip.getInt() <= 0 || inputs.size() < 2 ||
        outputs.size() != 2 || outputs[0].getType() != inputs[0].getType())
      return op->emitOpError(
          "requires trip>0, init/xs, and shape-stable carry/ys results");
    auto xs = cast<RankedTensorType>(inputs[1].getType());
    auto ys = cast<RankedTensorType>(outputs[1].getType());
    if (xs.getRank() < 1 || ys.getRank() < 1 ||
        xs.getDimSize(0) != trip.getInt() ||
        ys.getDimSize(0) != trip.getInt())
      return op->emitOpError("xs/ys leading dimension must equal trip");
  }
  return success();
}
} // namespace

LogicalResult MatmulOp::verify() {
  return verifyRank2Matmul(getOperation(), getInputs(), getOutputs());
}

LogicalResult GemmOp::verify() {
  return verifyRank2Matmul(getOperation(), getInputs(), getOutputs());
}

LogicalResult BatchedGemmOp::verify() {
  auto inputs = getInputs();
  auto outputs = getOutputs();
  if (inputs.size() != 2 || outputs.size() != 1)
    return emitOpError("expects exactly 2 inputs and 1 result");
  auto lhs = dyn_cast<RankedTensorType>(inputs[0].getType());
  auto rhs = dyn_cast<RankedTensorType>(inputs[1].getType());
  auto res = dyn_cast<RankedTensorType>(outputs[0].getType());
  if (!lhs || !rhs || !res)
    return success();
  if (lhs.getRank() != 3 || rhs.getRank() != 3 || res.getRank() != 3)
    return emitOpError("expects rank-3 lhs, rhs, and result tensors");
  auto dyn = [](int64_t d) { return ShapedType::isDynamic(d); };
  auto agree = [&](int64_t a, int64_t b) { return dyn(a) || dyn(b) || a == b; };
  if (!agree(lhs.getDimSize(0), rhs.getDimSize(0)) ||
      !agree(lhs.getDimSize(0), res.getDimSize(0)))
    return emitOpError("batch dimensions must match (no broadcasting)");
  if (!agree(lhs.getDimSize(2), rhs.getDimSize(1)))
    return emitOpError("contracting dimensions must match");
  if (!agree(res.getDimSize(1), lhs.getDimSize(1)))
    return emitOpError("result M must equal lhs M");
  if (!agree(res.getDimSize(2), rhs.getDimSize(2)))
    return emitOpError("result N must equal rhs N");
  return success();
}

LogicalResult ControlForOp::verify() {
  return verifyTileControl(getOperation(), getInputs(), getOutputs());
}

LogicalResult ControlIfOp::verify() {
  return verifyTileControl(getOperation(), getInputs(), getOutputs());
}

LogicalResult ControlWhileOp::verify() {
  return verifyTileControl(getOperation(), getInputs(), getOutputs());
}

LogicalResult ControlScanOp::verify() {
  return verifyTileControl(getOperation(), getInputs(), getOutputs());
}

LogicalResult PPOPolicyLossOp::verify() {
  auto inputs = getInputs();
  auto outputs = getOutputs();
  if (inputs.size() < 3 || inputs.size() > 6 || outputs.size() != 1)
    return emitOpError("expects 3 to 6 inputs and 1 result");
  auto next = dyn_cast<RankedTensorType>(inputs[0].getType());
  auto old = dyn_cast<RankedTensorType>(inputs[1].getType());
  auto adv = dyn_cast<RankedTensorType>(inputs[2].getType());
  auto res = dyn_cast<RankedTensorType>(outputs[0].getType());
  if (!next || !old || !adv || !res)
    return success();
  if (!next.hasStaticShape() || !old.hasStaticShape() ||
      !adv.hasStaticShape() || !res.hasStaticShape())
    return emitOpError("expects static fp32 operands and rank-0 fp32 result");
  if (!next.getElementType().isF32() || !old.getElementType().isF32() ||
      !adv.getElementType().isF32() || !res.getElementType().isF32())
    return emitOpError("expects fp32 operands and fp32 result");
  if (!sameStaticShape(next, old) || !sameStaticShape(next, adv))
    return emitOpError("input shapes must match exactly");
  for (auto input : inputs.drop_front(3)) {
    auto side = dyn_cast<RankedTensorType>(input.getType());
    if (!side)
      continue;
    if (!side.hasStaticShape() || !side.getElementType().isF32() ||
        !sameStaticShape(next, side))
      return emitOpError(
          "optional mask/ref_logp/entropy inputs must be static fp32 tensors "
          "with the same shape as logp_new");
  }
  if (res.getRank() != 0)
    return emitOpError("mean-reduction result must be rank-0 tensor");
  if (auto r = getOperation()->getAttrOfType<StringAttr>("reduction");
      r && r.getValue() != "mean")
    return emitOpError("only reduction=\"mean\" is executable in Tile PPO");
  if (auto c = getOperation()->getAttrOfType<FloatAttr>("clip_epsilon");
      c && c.getValueAsDouble() <= 0.0)
    return emitOpError("clip_epsilon must be positive");
  if (auto k = getOperation()->getAttrOfType<FloatAttr>("kl_coef");
      k && k.getValueAsDouble() < 0.0)
    return emitOpError("kl_coef must be non-negative for Tile PPO value mode");
  if (auto e = getOperation()->getAttrOfType<FloatAttr>("entropy_coef");
      e && e.getValueAsDouble() < 0.0)
    return emitOpError("entropy_coef must be non-negative for Tile PPO value mode");
  return success();
}

LogicalResult EBMEnergyQuadraticOp::verify() {
  auto inputs = getInputs();
  auto outputs = getOutputs();
  if (inputs.size() != 2 || outputs.size() != 1)
    return emitOpError("expects exactly 2 inputs and 1 result");
  auto x = dyn_cast<RankedTensorType>(inputs[0].getType());
  auto y = dyn_cast<RankedTensorType>(inputs[1].getType());
  auto energies = dyn_cast<RankedTensorType>(outputs[0].getType());
  if (!x || !y || !energies)
    return success();
  if (!x.hasStaticShape() || !y.hasStaticShape() ||
      !energies.hasStaticShape())
    return emitOpError("expects static fp32 rank-2 inputs and rank-1 result");
  if (!x.getElementType().isF32() || !y.getElementType().isF32() ||
      !energies.getElementType().isF32())
    return emitOpError("expects fp32 inputs and fp32 result");
  if (x.getRank() != 2 || y.getRank() != 2)
    return emitOpError("expects rank-2 x and y tensors");
  if (energies.getRank() != 1)
    return emitOpError("energies result must be rank-1");
  if (!sameStaticShape(x, y))
    return emitOpError("x and y shapes must match exactly");
  if (energies.getDimSize(0) != x.getDimSize(0))
    return emitOpError("energies length must equal batch dimension");
  return success();
}

LogicalResult EBMLangevinStepOp::verify() {
  auto inputs = getInputs();
  auto outputs = getOutputs();
  if (inputs.size() != 3 || outputs.size() != 1)
    return emitOpError("expects exactly 3 inputs and 1 result");
  if (failed(requireBoolAttr(getOperation(), "has_noise", true)))
    return failure();
  auto y = dyn_cast<RankedTensorType>(inputs[0].getType());
  auto grad = dyn_cast<RankedTensorType>(inputs[1].getType());
  auto noise = dyn_cast<RankedTensorType>(inputs[2].getType());
  auto out = dyn_cast<RankedTensorType>(outputs[0].getType());
  if (!y || !grad || !noise || !out)
    return success();
  if (!y.hasStaticShape() || !grad.hasStaticShape() ||
      !noise.hasStaticShape() || !out.hasStaticShape())
    return emitOpError("expects static fp32 operands and result");
  if (!y.getElementType().isF32() || !grad.getElementType().isF32() ||
      !noise.getElementType().isF32() || !out.getElementType().isF32())
    return emitOpError("expects fp32 operands and fp32 result");
  if (!sameStaticShape(y, grad) || !sameStaticShape(y, noise) ||
      !sameStaticShape(y, out))
    return emitOpError("y/grad/noise/result shapes must match exactly");
  if (auto eta = getOperation()->getAttrOfType<FloatAttr>("eta");
      !eta || eta.getValueAsDouble() <= 0.0)
    return emitOpError("eta must be present and positive");
  if (auto scale = getOperation()->getAttrOfType<FloatAttr>("noise_scale");
      scale && scale.getValueAsDouble() < 0.0)
    return emitOpError("noise_scale must be non-negative");
  return success();
}

LogicalResult EBMRefinementOp::verify() {
  auto inputs = getInputs();
  auto outputs = getOutputs();
  if (inputs.size() != 2 || outputs.size() != 1)
    return emitOpError("expects exactly 2 inputs and 1 result");
  auto y = dyn_cast<RankedTensorType>(inputs[0].getType());
  auto grad = dyn_cast<RankedTensorType>(inputs[1].getType());
  auto out = dyn_cast<RankedTensorType>(outputs[0].getType());
  if (!y || !grad || !out)
    return success();
  if (!y.hasStaticShape() || !grad.hasStaticShape() || !out.hasStaticShape())
    return emitOpError("expects static fp32 operands and result");
  if (!y.getElementType().isF32() || !grad.getElementType().isF32() ||
      !out.getElementType().isF32())
    return emitOpError("expects fp32 operands and fp32 result");
  if (!sameStaticShape(y, grad) || !sameStaticShape(y, out))
    return emitOpError("y/grad/result shapes must match exactly");
  if (auto eta = getOperation()->getAttrOfType<FloatAttr>("eta");
      !eta || eta.getValueAsDouble() <= 0.0)
    return emitOpError("eta must be present and positive");
  if (auto steps = getOperation()->getAttrOfType<IntegerAttr>("steps");
      !steps || steps.getInt() <= 0)
    return emitOpError("steps must be present and positive");
  return success();
}

LogicalResult EBMPartitionExactOp::verify() {
  auto inputs = getInputs();
  auto outputs = getOutputs();
  if (inputs.size() != 1 || outputs.size() != 1)
    return emitOpError("expects exactly 1 input and 1 result");
  auto energies = dyn_cast<RankedTensorType>(inputs[0].getType());
  auto out = dyn_cast<RankedTensorType>(outputs[0].getType());
  if (!energies || !out)
    return success();
  if (!energies.hasStaticShape() || !out.hasStaticShape())
    return emitOpError("expects static fp32 energies and scalar result");
  if (!energies.getElementType().isF32() || !out.getElementType().isF32())
    return emitOpError("expects fp32 energies and fp32 result");
  if (out.getRank() != 0)
    return emitOpError("partition result must be scalar for value execution");
  if (auto temperature = getOperation()->getAttrOfType<FloatAttr>("temperature");
      temperature && temperature.getValueAsDouble() <= 0.0)
    return emitOpError("temperature must be positive");
  if (auto reduction = getOperation()->getAttrOfType<StringAttr>("reduction");
      reduction && reduction.getValue() != "logsumexp")
    return emitOpError("only reduction=\"logsumexp\" is executable");
  return success();
}

LogicalResult CliffordGeometricProductOp::verify() {
  return verifyCliffordBinary(getOperation(), getInputs(), getOutputs(),
                              "clifford_geometric_product");
}

LogicalResult CliffordOuterProductOp::verify() {
  return verifyCliffordBinary(getOperation(), getInputs(), getOutputs(),
                              "clifford_outer_product");
}

LogicalResult CliffordInnerProductOp::verify() {
  return verifyCliffordBinary(getOperation(), getInputs(), getOutputs(),
                              "clifford_inner_product");
}

LogicalResult CliffordReverseOp::verify() {
  return verifyCliffordUnary(getOperation(), getInputs(), getOutputs(),
                             "clifford_reverse");
}

LogicalResult CliffordGradeProjectOp::verify() {
  return verifyCliffordUnary(getOperation(), getInputs(), getOutputs(),
                             "clifford_grade_project");
}

LogicalResult CliffordNormOp::verify() {
  return verifyCliffordUnary(getOperation(), getInputs(), getOutputs(),
                             "clifford_norm", /*allowNormDrop=*/true);
}

LogicalResult CliffordRotorSandwichOp::verify() {
  return verifyCliffordBinary(getOperation(), getInputs(), getOutputs(),
                              "clifford_rotor_sandwich");
}

LogicalResult ViewOp::verify() {
  if (getInputs().empty())
    return emitOpError("expects a source and optional index operands");
  if (!getOperation()->getAttrOfType<TileLayoutAttr>("tile.layout"))
    return emitOpError("requires a #tile.layout attribute");
  if (auto memory = getOperation()->getAttrOfType<TileMemoryLayoutAttr>("tile.memory")) {
    // The pointer-backed operand contract, defined HERE rather than per
    // backend (PR #510 review).
    //
    //   static leading_dim:  3 inputs, or 5 with (rowBound, colBound)
    //   dynamic leading_dim: 4 inputs, or 6 with bounds; the final operand is
    //                        the runtime leading dimension.
    //
    // This used to accept any count >= 3, which made a 4-operand view legal and
    // meaningless, and left "is the bounded form valid?" to whichever backend
    // happened to look. A shared-IR op whose accepted shape is decided by its
    // consumers is how the same valid Tile IR becomes backend-dependent.
    const size_t inputs = getInputs().size();
    const bool dynamic = memory.getLeadingDim() == 0;
    const bool linearBase = getOperation()->hasAttr("tile.linear_base");
    const size_t offset = linearBase ? 1 : 0;
    const bool valid = dynamic ? (inputs == 4 + offset || inputs == 6 + offset)
                               : (inputs == 3 + offset || inputs == 5 + offset);
    if (!valid)
      return emitOpError()
             << "TILE_VIEW_POINTER_ARITY: pointer-backed tile.view takes "
                "(base, rowOrigin, colOrigin), optionally followed by "
                "(rowBound, colBound), and requires a final SSA leading "
                "dimension exactly when tile.memory leading_dim is zero; "
                "tile.linear_base adds one precomputed row-major/column-major "
                "linear origin immediately after base; got "
             << inputs << " operands";
  }
  return success();
}

LogicalResult MaterializeComposedLayoutOp::verify() {
  TileComposedLayoutAttr layout = getLayout();
  SmallVector<ComposedLayoutMaterializationMode> modes;
  SmallVector<ComposedLayoutDynamicLeaf> dynamicLeaves;
  if (!getMaterializableComposedLayout(layout, modes, dynamicLeaves) ||
      getCoordinates().size() != modes.size() + dynamicLeaves.size())
    return emitOpError("TILE_COMPOSED_LAYOUT_NOT_MATERIALIZABLE: composed "
                       "layout must be a scalar-output coordinate map accepted "
                       "by the native layout authority, followed by one i64 "
                       "operand per dynamic leaf in canonical preorder");
  return success();
}

LogicalResult MaterializeComposedLayoutTupleOp::verify() {
  ArrayAttr layouts = getLayouts();
  if (layouts.empty() || getResults().size() != layouts.size())
    return emitOpError("TILE_COMPOSED_LAYOUT_TUPLE_ARITY: requires one i64 "
                       "result per scalar codomain component");
  std::optional<size_t> coordinateRank;
  for (Attribute attr : layouts) {
    auto layout = dyn_cast<TileComposedLayoutAttr>(attr);
    SmallVector<ComposedLayoutMaterializationMode> modes;
    SmallVector<ComposedLayoutDynamicLeaf> dynamicLeaves;
    if (!layout ||
        !getMaterializableComposedLayout(layout, modes, dynamicLeaves) ||
        !dynamicLeaves.empty())
      return emitOpError("TILE_COMPOSED_LAYOUT_TUPLE_NOT_MATERIALIZABLE: "
                         "every tuple component must be a statically proven "
                         "scalar composed layout");
    if (!coordinateRank)
      coordinateRank = modes.size();
    else if (*coordinateRank != modes.size())
      return emitOpError("TILE_COMPOSED_LAYOUT_TUPLE_DOMAIN: every tuple "
                         "component must consume the same coordinate rank");
  }
  if (getCoordinates().size() != *coordinateRank)
    return emitOpError("TILE_COMPOSED_LAYOUT_TUPLE_COORDINATES: coordinate "
                       "count must equal the shared domain rank");
  return success();
}

static LogicalResult verifyPackedAddressing(Operation *op, ValueRange inputs,
                                            bool store) {
  auto packed =
      op->getAttrOfType<TilePackedPhysicalViewAttr>("tile.packed_view");
  if (!packed)
    return op->emitOpError("requires a #tile.packed_view attribute");
  if (!op->getAttrOfType<TileLayoutAttr>("tile.layout"))
    return op->emitOpError("requires a #tile.layout attribute");
  if (!op->getAttrOfType<TileMemoryLayoutAttr>("tile.memory"))
    return op->emitOpError("requires a #tile.memory_layout attribute");

  bool scaled = packed.getScale().getDtype() != "none";
  unsigned prefix = store ? 2 : (scaled ? 2 : 1);
  unsigned expected = prefix + 4;
  if (inputs.size() != expected)
    return op->emitOpError()
           << (store ? "packed store" : scaled ? "scaled packed load"
                                               : "unscaled packed load")
           << " expects " << expected << " operands";
  if (store && !isa<TileValueType>(inputs.front().getType()))
    return op->emitOpError("packed store requires a leading !tile.tile value");
  if (store && scaled)
    return op->emitOpError(
        "scale-bearing packed stores require an explicit scale-production "
        "contract and are not supported");
  for (Value coordinate : inputs.drop_front(prefix))
    if (!coordinate.getType().isInteger(64))
      return op->emitOpError(
          "row/column origins and logical extents must be i64");
  return success();
}

LogicalResult PackedLoadOp::verify() {
  return verifyPackedAddressing(getOperation(), getInputs(), /*store=*/false);
}

LogicalResult PackedStoreOp::verify() {
  return verifyPackedAddressing(getOperation(), getInputs(), /*store=*/true);
}

// W1.1 step 2 — when the RESULT is a parameterized fragment, the type is the
// contract: `role` and `#tile.mma_desc` become redundant restatements, so they
// are optional and merely have to agree. This is the producer half of the same
// collapse `MMAOp::verify()` gets; without it the typed form is unusable,
// because every typed fragment has to come from one of these two ops.
static LogicalResult verifyFragmentProducerAgainstType(Operation *op,
                                                       FragmentType result) {
  if (auto role = op->getAttrOfType<StringAttr>("role"))
    if (role.getValue() != result.getRole())
      return op->emitOpError()
             << "TILE_FRAGMENT_ROLE_DISAGREES: role attribute \""
             << role.getValue() << "\" contradicts the result type's role \""
             << result.getRole() << "\"";
  if (auto desc = mmaDescAttr(op))
    return descriptorAgreesWithFragment(op, desc, result);
  return success();
}

LogicalResult FragmentPackOp::verify() {
  if (getInputs().size() != 1 || !isa<TileValueType>(getInputs().front().getType()))
    return emitOpError("expects exactly one !tile.tile input");
  if (getOperation()->getNumResults() == 1)
    if (FragmentType typed = typedFragment(getOperation()->getResult(0))) {
      if (failed(verifyFragmentProducerAgainstType(getOperation(), typed)))
        return failure();
      const bool isScale =
          typed.getRole() == "scale_a" || typed.getRole() == "scale_b";
      const bool isNVFP4 =
          typed.getElem() == "nvfp4" || typed.getElem() == "fp4_e2m1";
      if (isScale && !isNVFP4)
        return emitOpError("scale fragments require an NVFP4 element type");
      return success();
    }
  auto role = getOperation()->getAttrOfType<StringAttr>("role");
  if (!role || (role.getValue() != "a" && role.getValue() != "b" &&
                role.getValue() != "acc" && role.getValue() != "scale_a" &&
                role.getValue() != "scale_b"))
    return emitOpError("requires role = a, b, acc, scale_a, or scale_b");
  auto desc = mmaDescAttr(getOperation());
  if (!desc)
    return emitOpError("requires a #tile.mma_desc mma attribute");
  bool isScale = role.getValue() == "scale_a" || role.getValue() == "scale_b";
  bool isNVFP4 = desc.getAType() == "nvfp4" || desc.getAType() == "fp4_e2m1";
  if (isScale && !isNVFP4)
    return emitOpError("scale fragments require an NVFP4 MMA descriptor");
  return success();
}

LogicalResult FragmentZeroOp::verify() {
  if (getOperation()->getNumResults() == 1)
    if (FragmentType typed = typedFragment(getOperation()->getResult(0))) {
      if (typed.getRole() != "acc")
        return emitOpError()
               << "TILE_FRAGMENT_ZERO_ROLE: tile.fragment_zero produces the "
                  "accumulator, so its type must have role \"acc\", got \""
               << typed.getRole() << "\"";
      return verifyFragmentProducerAgainstType(getOperation(), typed);
    }
  auto role = getOperation()->getAttrOfType<StringAttr>("role");
  if (!role || role.getValue() != "acc")
    return emitOpError("requires role = acc");
  if (!mmaDescAttr(getOperation()))
    return emitOpError("requires a #tile.mma_desc mma attribute");
  return success();
}

//===----------------------------------------------------------------------===//
// tile.async_copy / tile.wait_async — typed-token form (W1.1)
//
// `!tile.async_token` was declared and never used anywhere in the dialect: a
// Decision #29 violation ("a declaration must have a consumer"). It was not an
// oversight -- the type was introduced for the NV warp-spec dependency model
// that replaced program-order reasoning -- it simply had no producer or
// consumer wired to it. These verifiers are that consumer.
//
// Same dual-form migration shape as MMAOp::verify(): the typed form is
// enforced when present, the legacy form is accepted unchanged so the existing
// producers keep working. The permissive branch is deleted only once W1.1's
// producer migration completes.
//===----------------------------------------------------------------------===//

// Shared by both sync ops: `stage` is an OPTIONAL legacy-form grouping key
// (see the ODS description — TileBufferReusePass and the Python spine consume
// it; a key-less op fails safe as "matches anything"). When it is present it
// must be well formed. This is the one check the deleted ScheduleOps.cpp
// stage-model verifier carried that was worth keeping; requiring the attribute
// outright was not — no in-tree producer satisfies that, including the
// production TileIRLoweringPass emitter.
static LogicalResult verifyOptionalStage(Operation *op) {
  if (auto stage = op->getAttrOfType<IntegerAttr>("stage"))
    if (stage.getInt() < 0)
      return op->emitOpError(
          "TILE_ASYNC_STAGE_NEGATIVE: 'stage' is an optional grouping key, "
          "but when present it must be >= 0; got ")
             << stage.getInt();
  return success();
}

LogicalResult AsyncCopyOp::verify() {
  if (failed(verifyOptionalStage(getOperation())))
    return failure();

  llvm::SmallVector<Value> tokens;
  for (Value out : getOutputs())
    if (isa<AsyncTokenType>(out.getType()))
      tokens.push_back(out);

  if (tokens.empty())
    return success();  // legacy form: ordering via barrier_id / depends_on

  if (tokens.size() != 1)
    return emitOpError("typed form yields exactly one !tile.async_token; got ")
           << tokens.size();
  if (!isa<AsyncTokenType>(getOutputs().back().getType()))
    return emitOpError(
        "the !tile.async_token must be the last result of tile.async_copy");
  if (getInputs().empty())
    return emitOpError("typed form expects at least a source operand");
  // NOTE: the established convention is ONE operand (the source) with the
  // copied tile returned alongside the token -- e.g.
  //   %tile, %tok = tile.async_copy %src : (tensor<..>) -> (tensor<..>, !tile.async_token)
  // An earlier version of this verifier required a destination operand too,
  // modelled on `tessera_rocm.async_copy(dst, src, bytes)`. That was wrong for
  // Tile level and the existing fixtures caught it. Do not re-tighten this
  // without re-reading tile_async_token_roundtrip.mlir and
  // warpspec_token_sync_legality.mlir.
  return success();
}

LogicalResult WaitAsyncOp::verify() {
  if (failed(verifyOptionalStage(getOperation())))
    return failure();

  for (Value in : getInputs()) {
    if (!isa<AsyncTokenType>(in.getType()))
      continue;
    Operation *producer = in.getDefiningOp();
    if (!producer)
      return emitOpError(
          "!tile.async_token operand must be produced by a tile.async_copy, "
          "not a block argument");
    if (!isa<AsyncCopyOp>(producer))
      return emitOpError("!tile.async_token operand must be produced by a "
                         "tile.async_copy; got '")
             << producer->getName().getStringRef() << "'";
  }
  return success();
}

LogicalResult MMAOp::verify() {
  SmallVector<Value> data = dataOperands(getOperation());
  const bool hasAnyFragment = llvm::any_of(data, [](Value value) {
    return isa<FragmentType>(value.getType());
  });
  if (hasAnyFragment) {
    // W1.1 step 5: there is no producer-chasing or bare-fragment contract any
    // more. A cooperative-matrix value states its complete ABI in its type,
    // including across block arguments and loop-carried accumulator edges.
    for (auto [index, value] : llvm::enumerate(data))
      if (auto fragment = dyn_cast<FragmentType>(value.getType());
          fragment && fragment.isUnknown())
        return emitOpError()
               << "TILE_MMA_BARE_FRAGMENT_REMOVED: operand " << index
               << " uses bare !tile.fragment; parameterize m/n/k, element and "
                  "accumulator dtypes, role, layout, and family";
    return verifyMMAFromTypes(getOperation(), data);
  }

  // Historical tensor-valued producers are a distinct, explicitly verified
  // value lane while W3 moves them to tile.matmul. They are not eligible for
  // fragment lowering, but retaining a shape-checked lane avoids replacing one
  // permissive compatibility path with a flag-day frontend break.
  if ((data.size() != 2 && data.size() != 3) || getOutputs().size() != 1)
    return emitOpError(
        "TILE_MMA_VALUE_ARITY: tensor value form expects A, B, optional "
        "accumulator, and exactly one result");
  SmallVector<Value> productInputs{data[0], data[1]};
  if (failed(verifyRank2Matmul(getOperation(), productInputs, getOutputs())))
    return failure();
  if (data.size() == 3 && data[2].getType() != getOutputs().front().getType())
    return emitOpError(
        "TILE_MMA_VALUE_ACCUMULATOR: tensor accumulator type must equal the "
        "result type");
  if (llvm::any_of(data, [](Value value) {
        return !isa<RankedTensorType>(value.getType());
      }))
    return emitOpError(
        "TILE_MMA_VALUE_TYPE: non-fragment value form requires ranked tensors");
  return success();
}

LogicalResult FragmentUnpackOp::verify() {
  if (getInputs().size() != 1 || !isa<FragmentType>(getInputs().front().getType()))
    return emitOpError("expects exactly one !tile.fragment input");

  // W1.1 step 2 (PR #503 review) — read the input TYPE when it carries the
  // contract. Without this the descriptor is only optional while the result is
  // left packed: the moment a typed `tile.mma` feeds the ordinary epilogue,
  // this verifier demanded `mmaDescAttr(producer)` and rejected the module.
  //
  // It also producer-chased, so it had the block-argument problem too: a K-loop
  // accumulator unpacked after the loop is an `scf.for` RESULT, whose defining
  // op is the loop and carries no descriptor.
  if (FragmentType typed = typedFragment(getInputs().front())) {
    if (typed.getRole() != "acc")
      return emitOpError()
             << "TILE_FRAGMENT_UNPACK_ROLE: only an accumulator fragment may be "
                "unpacked, got role \"" << typed.getRole() << "\"";
    if (auto desc = mmaDescAttr(getOperation()))
      if (failed(descriptorAgreesWithFragment(getOperation(), desc, typed)))
        return failure();
    if (!getOperation()->getAttrOfType<TileLayoutAttr>("tile.layout"))
      return emitOpError("requires a #tile.layout attribute");
    return success();
  }

  Operation *producer = getInputs().front().getDefiningOp();
  if (!producer || !mmaDescAttr(producer))
    return emitOpError("fragment must be produced by a descriptor-carrying Tile op");
  auto role = producer->getAttrOfType<StringAttr>("role");
  if (role && role.getValue() != "acc")
    return emitOpError("only an accumulator fragment may be unpacked");
  auto desc = mmaDescAttr(getOperation());
  if (desc && desc != mmaDescAttr(producer))
    return emitOpError("mma attribute must match the input fragment descriptor");
  if (!getOperation()->getAttrOfType<TileLayoutAttr>("tile.layout"))
    return emitOpError("requires a #tile.layout attribute");
  return success();
}

LogicalResult StoreOp::verify() {
  if (getInputs().empty() ||
      !isa<TileValueType>(getInputs().front().getType()))
    return emitOpError(
        "pointer-backed form expects tile, base, row origin, column origin");
  if (!getOperation()->getAttrOfType<TileLayoutAttr>("tile.layout"))
    return emitOpError("requires a #tile.layout attribute");
  auto memory =
      getOperation()->getAttrOfType<TileMemoryLayoutAttr>("tile.memory");
  if (!memory)
    return emitOpError("requires a #tile.memory_layout attribute");
  const size_t inputs = getInputs().size();
  const bool dynamic = memory.getLeadingDim() == 0;
  const bool valid = dynamic ? (inputs == 5 || inputs == 7)
                             : (inputs == 4 || inputs == 6);
  if (!valid)
    return emitOpError()
           << "TILE_STORE_POINTER_ARITY: pointer-backed tile.store takes "
              "(tile, base, rowOrigin, colOrigin), optionally followed by "
              "(rowBound, colBound), and requires a final SSA leading "
              "dimension exactly when tile.memory leading_dim is zero; got "
           << inputs << " operands";
  return success();
}

LogicalResult MatmulKernelOp::verify() {
  auto desc = getOperation()->getAttrOfType<TileMmaDescAttr>("mma");
  auto epilogue = getOperation()->getAttrOfType<TileEpilogueAttr>("epilogue");
  if (!desc)
    return emitOpError("requires a #tile.mma_desc mma attribute");
  if (!epilogue)
    return emitOpError("requires a #tile.epilogue epilogue attribute");
  bool blockScaled = desc.getAType() == "nvfp4" ||
                     desc.getAType() == "fp4_e2m1" ||
                     desc.getAType() == "e2m3" ||
                     desc.getAType() == "e3m2";
  bool residual = false;
  if (auto attr = getOperation()->getAttrOfType<BoolAttr>("residual"))
    residual = attr.getValue();
  if (blockScaled && (epilogue.getBias() || residual))
    return emitOpError("block-scaled launch-level matmul does not support fused bias/residual yet");
  if (residual) {
    auto order = getOperation()->getAttrOfType<StringAttr>("epilogue_order");
    if (!order || order.getValue() != "matmul_bias_activation_residual")
      return emitOpError("residual requires epilogue_order=matmul_bias_activation_residual");
  }
  unsigned pointerCount = blockScaled
      ? 5
      : 3 + unsigned(epilogue.getBias()) + unsigned(residual);
  unsigned compactExpected = pointerCount + 3;
  unsigned stridedExpected = pointerCount + 6;
  bool strided = !blockScaled && getInputs().size() == stridedExpected;
  if (getInputs().size() != compactExpected && !strided)
    return emitOpError() << (blockScaled
        ? "expects packed A, packed B, scale A, scale B, D, M, N, K operands"
        : "expects A, B, optional bias, optional residual, D, M, N, K, "
          "and optional LDA, LDB, LDD operands");
  for (Value dim : getInputs().drop_front(pointerCount))
    if (!dim.getType().isInteger(64))
      return emitOpError("M, N, K, and optional leading dimensions must be i64");
  for (Value pointer : getInputs().take_front(pointerCount))
    if (!isa<LLVM::LLVMPointerType>(pointer.getType()))
    return emitOpError("matrix, scale, optional bias, and D operands must be !llvm.ptr");
  int64_t warps = 1;
  if (auto attr = getOperation()->getAttrOfType<IntegerAttr>("warps"))
    warps = attr.getInt();
  if (warps != 1 && warps != 4)
    return emitOpError("warps must be 1 or 4");
  StringRef staging = "global";
  if (auto attr = getOperation()->getAttrOfType<StringAttr>("staging"))
    staging = attr.getValue();
  if (staging != "global" && staging != "shared")
    return emitOpError("staging must be global or shared");
  if (staging == "shared" && warps != 4)
    return emitOpError("shared staging currently requires four warps");
  if (auto canonical =
          getOperation()->getAttrOfType<BoolAttr>("tessera.canonical_k_loop");
      canonical && canonical.getValue()) {
    auto tileM = getOperation()->getAttrOfType<IntegerAttr>("tessera.tile_m");
    auto tileN = getOperation()->getAttrOfType<IntegerAttr>("tessera.tile_n");
    auto tileK = getOperation()->getAttrOfType<IntegerAttr>("tessera.tile_k");
    if (!tileM || !tileN || !tileK)
      return emitOpError(
          "canonical K-loop requires tessera.tile_m, tessera.tile_n, and "
          "tessera.tile_k");
    if (tileM.getInt() <= 0 || tileN.getInt() <= 0 || tileK.getInt() <= 0)
      return emitOpError("canonical K-loop tile sizes must be positive");
    if (tileM.getInt() != desc.getM() || tileN.getInt() != desc.getN() ||
        tileK.getInt() != desc.getK())
      return emitOpError(
          "canonical K-loop tile sizes must match the physical MMA descriptor");
    auto macroM = getOperation()->getAttrOfType<IntegerAttr>(
        "tessera.macro_tile_m");
    auto macroN = getOperation()->getAttrOfType<IntegerAttr>(
        "tessera.macro_tile_n");
    if (bool(macroM) != bool(macroN))
      return emitOpError(
          "canonical K-loop macro_tile_m and macro_tile_n must appear together");
    if (macroM &&
        (macroM.getInt() < desc.getM() || macroN.getInt() < desc.getN() ||
         macroM.getInt() % desc.getM() != 0 ||
         macroN.getInt() % desc.getN() != 0))
      return emitOpError(
          "canonical K-loop macro tile must be a positive multiple of the physical MMA tile");
    if (desc.getAccType() != "f32" && desc.getAccType() != "s32" &&
        desc.getAccType() != "int32" && desc.getAccType() != "f64")
      return emitOpError(
          "canonical K-loop requires FP32, INT32, or FP64 accumulation");
  }
  return success();
}

LogicalResult SoftmaxKernelOp::verify() {
  if (getInputs().size() != 4)
    return emitOpError("expects source, destination, rows, and columns operands");
  if (!isa<LLVM::LLVMPointerType>(getInputs()[0].getType()) ||
      !isa<LLVM::LLVMPointerType>(getInputs()[1].getType()))
    return emitOpError("source and destination operands must be !llvm.ptr");
  if (!getInputs()[2].getType().isInteger(64) ||
      !getInputs()[3].getType().isInteger(64))
    return emitOpError("rows and columns must be i64");
  auto storage = getOperation()->getAttrOfType<StringAttr>("storage");
  auto accum = getOperation()->getAttrOfType<StringAttr>("accum");
  if (!storage || (storage.getValue() != "f16" &&
                   storage.getValue() != "bf16" &&
                   storage.getValue() != "f32"))
    return emitOpError(
        "requires storage=\"f16\", storage=\"bf16\", or storage=\"f32\"");
  if (!accum || accum.getValue() != "f32")
    return emitOpError("requires accum=\"f32\"");
  auto axis = getOperation()->getAttrOfType<IntegerAttr>("axis");
  if (!axis || axis.getInt() != -1)
    return emitOpError("currently requires axis=-1");
  auto expMode = getOperation()->getAttrOfType<StringAttr>("exp_mode");
  if (!expMode || expMode.getValue().empty())
    return emitOpError("requires an explicit exp_mode");
  if (!getOperation()->getAttrOfType<BoolAttr>("ftz"))
    return emitOpError("requires an explicit ftz boolean");
  return success();
}

LogicalResult ReduceKernelOp::verify() {
  if (getInputs().size() != 5)
    return emitOpError("expects source, destination, outer, axis extent, and inner operands");
  if (!isa<LLVM::LLVMPointerType>(getInputs()[0].getType()) ||
      !isa<LLVM::LLVMPointerType>(getInputs()[1].getType()))
    return emitOpError("source and destination operands must be !llvm.ptr");
  for (Value dim : getInputs().drop_front(2))
    if (!dim.getType().isInteger(64))
      return emitOpError("reduction dimensions must be i64");
  auto storage = getOperation()->getAttrOfType<StringAttr>("storage");
  auto accum = getOperation()->getAttrOfType<StringAttr>("accum");
  auto kind = getOperation()->getAttrOfType<StringAttr>("kind");
  auto axis = getOperation()->getAttrOfType<IntegerAttr>("axis");
  auto keepdims = getOperation()->getAttrOfType<BoolAttr>("keepdims");
  auto schedule = getOperation()->getAttrOfType<StringAttr>("schedule");
  auto nanMode = getOperation()->getAttrOfType<StringAttr>("nan_mode");
  if (!storage || (storage.getValue() != "f16" &&
                   storage.getValue() != "bf16" &&
                   storage.getValue() != "f32"))
    return emitOpError(
        "requires storage=\"f16\", storage=\"bf16\", or storage=\"f32\"");
  if (!accum || accum.getValue() != "f32")
    return emitOpError("requires accum=\"f32\"");
  if (!kind || (kind.getValue() != "sum" && kind.getValue() != "mean" &&
                kind.getValue() != "max" && kind.getValue() != "min"))
    return emitOpError("requires kind in {sum, mean, max, min}");
  if (!axis || axis.getInt() < 0)
    return emitOpError("requires a normalized nonnegative axis");
  if (!keepdims)
    return emitOpError("requires an explicit keepdims boolean");
  if (!schedule || (schedule.getValue() != "serial" &&
                    schedule.getValue() != "cooperative_128"))
    return emitOpError("requires schedule=serial|cooperative_128");
  if (!nanMode || nanMode.getValue() != "propagate")
    return emitOpError("currently requires nan_mode=\"propagate\"");
  return success();
}

LogicalResult FFTKernelOp::verify() {
  if (getInputs().size() != 4)
    return emitOpError("expects source, destination, batch, and length operands");
  if (!isa<LLVM::LLVMPointerType>(getInputs()[0].getType()) ||
      !isa<LLVM::LLVMPointerType>(getInputs()[1].getType()))
    return emitOpError("source and destination operands must be !llvm.ptr");
  if (!getInputs()[2].getType().isInteger(64) ||
      !getInputs()[3].getType().isInteger(64))
    return emitOpError("batch and length must be i64");
  auto requiredString = [&](StringRef name) -> StringRef {
    auto attr = getOperation()->getAttrOfType<StringAttr>(name);
    return attr ? attr.getValue() : StringRef();
  };
  StringRef mode = requiredString("mode");
  StringRef realPolicy = requiredString("real_transform_policy");
  StringRef hermitianLayout = requiredString("hermitian_layout");
  StringRef hermitianWeight = requiredString("hermitian_weight");
  StringRef strategy = requiredString("strategy");
  StringRef radixPolicy = requiredString("radix_policy");
  if ((mode != "c2c" && mode != "r2c" && mode != "c2r") ||
      (strategy != "radix2" && strategy != "mixed_radix" &&
       strategy != "dft" && strategy != "bluestein") ||
      (radixPolicy != "radix2" && radixPolicy != "mixed_radix"))
    return emitOpError("requires explicit mode, strategy, and radix policy");
  if ((realPolicy != "not_applicable" &&
       realPolicy != "packed_even_n2_hermitian_v1" &&
       realPolicy != "native_cufft_r2c_c2r_v1" &&
       realPolicy != "full_complex_hermitian_fallback") ||
      (hermitianLayout != "full_complex" &&
       hermitianLayout != "half_spectrum_nyquist_explicit") ||
      (hermitianWeight != "none" && hermitianWeight != "half_interior" &&
       hermitianWeight != "double_interior"))
    return emitOpError("requires an explicit real-transform and Hermitian policy");
  if (requiredString("storage") != "complex64_interleaved_f32" ||
      requiredString("accum") != "f32" ||
      (requiredString("normalization") != "backward" &&
       requiredString("normalization") != "forward" &&
       requiredString("normalization") != "ortho") ||
      requiredString("twiddle_layout") != "interleaved_f32" ||
      requiredString("algorithm").empty() ||
      requiredString("workspace_policy").empty() ||
      requiredString("residency").empty() ||
      requiredString("twiddle_policy").empty() ||
      requiredString("kernel_family").empty())
    return emitOpError("requires the canonical complex64/f32 FFT package policy");
  if (mode == "c2c" && hermitianWeight != "none")
    return emitOpError("full-complex FFT cannot apply Hermitian weighting");
  if (mode == "r2c" && hermitianWeight == "half_interior")
    return emitOpError("r2c cannot apply input-side half-interior weighting");
  if (mode == "c2r" && hermitianWeight == "double_interior")
    return emitOpError("c2r cannot apply output-side double-interior weighting");
  auto sequence = getOperation()->getAttrOfType<DenseI64ArrayAttr>("radix_sequence");
  auto length = getOperation()->getAttrOfType<IntegerAttr>("length");
  auto physicalLength =
      getOperation()->getAttrOfType<IntegerAttr>("physical_length");
  auto batch = getOperation()->getAttrOfType<IntegerAttr>("batch");
  auto axis = getOperation()->getAttrOfType<IntegerAttr>("axis");
  auto workspace = getOperation()->getAttrOfType<IntegerAttr>("workspace_elems");
  auto workgroup = getOperation()->getAttrOfType<IntegerAttr>("tessera.workgroup_size");
  auto hash = getOperation()->getAttrOfType<StringAttr>("tessera.schedule_hash");
  if (!sequence || !length || length.getInt() <= 0 || !physicalLength ||
      physicalLength.getInt() <= 0 || !batch ||
      batch.getInt() <= 0 || !axis || axis.getInt() < 0 || !workspace ||
      workspace.getInt() < 0 || !workgroup || workgroup.getInt() <= 0 ||
      !hash || hash.getValue().size() != 64)
    return emitOpError("requires complete launch dimensions, workspace, and schedule identity");
  if (realPolicy == "packed_even_n2_hermitian_v1" &&
      (mode == "c2c" || length.getInt() % 2 != 0 ||
       physicalLength.getInt() != length.getInt() / 2))
    return emitOpError("has an invalid packed-real physical length");
  return success();
}

LogicalResult SpectralProgramKernelOp::verify() {
  auto kind = getOperation()->getAttrOfType<StringAttr>("kind");
  auto inputCount = getOperation()->getAttrOfType<IntegerAttr>("input_count");
  auto workspace = getOperation()->getAttrOfType<IntegerAttr>("workspace_bytes");
  auto hash = getOperation()->getAttrOfType<StringAttr>("tessera.schedule_hash");
  if (!kind || !inputCount || inputCount.getInt() < 1 ||
      inputCount.getInt() > 2 || !workspace || workspace.getInt() <= 0 ||
      !hash || hash.getValue().size() != 64)
    return emitOpError("requires kind, input count, workspace, and schedule identity");
  if (getInputs().size() != static_cast<size_t>(inputCount.getInt() + 2))
    return emitOpError("expects source pointers, destination pointer, and workspace bytes");
  for (int64_t index = 0; index <= inputCount.getInt(); ++index)
    if (!isa<LLVM::LLVMPointerType>(getInputs()[index].getType()))
      return emitOpError("source and destination operands must be !llvm.ptr");
  if (!getInputs().back().getType().isInteger(64))
    return emitOpError("workspace size operand must be i64");
  auto requiredString = [&](StringRef name) -> StringRef {
    auto attr = getOperation()->getAttrOfType<StringAttr>(name);
    return attr ? attr.getValue() : StringRef();
  };
  if (requiredString("target").empty() || requiredString("arch").empty() ||
      requiredString("input_shapes").empty() ||
      requiredString("input_signature").empty() ||
      requiredString("shape_bounds").empty() ||
      requiredString("template_digest").size() != 64 ||
      (requiredString("shape_policy") != "exact_runtime_specialization_v1" &&
       requiredString("shape_policy") != "bounded_runtime_specialization_v1") ||
      (requiredString("storage") != "f32" &&
       requiredString("storage") != "f16" &&
       requiredString("storage") != "bf16") ||
      requiredString("abi_storage") != "f32" ||
      (requiredString("storage_conversion") != "native_f32" &&
       requiredString("storage_conversion") !=
           "native_package_cast_f32_accumulate_cast_output_v1") ||
      (requiredString("axis_packing") != "none_contiguous" &&
       requiredString("axis_packing") != "native_package_host_pack_v1" &&
       requiredString("axis_packing") !=
           "native_runtime_stride_descriptor_v1") ||
      (requiredString("normalization") != "backward" &&
       requiredString("normalization") != "forward" &&
       requiredString("normalization") != "ortho") ||
      requiredString("complex_layout") != "interleaved_f32x2" ||
      requiredString("workspace_policy") != "persistent_artifact_workspace" ||
      requiredString("fusion_topology").empty() ||
      requiredString("mutation_lineage") !=
          "inputs_immutable_output_fresh_v1" ||
      requiredString("native_entry").empty())
    return emitOpError("requires complete spectral package policy");
  if (kind.getValue() == "tessera.stft" ||
      kind.getValue() == "tessera.istft") {
    auto transform = getOperation()->getAttrOfType<IntegerAttr>("transform_length");
    auto window = getOperation()->getAttrOfType<IntegerAttr>("window_length");
    auto onesided = getOperation()->getAttrOfType<BoolAttr>("onesided");
    auto windowBroadcast =
        getOperation()->getAttrOfType<StringAttr>("window_broadcast");
    if (!transform || !window || !onesided || transform.getInt() <= 0 ||
        window.getInt() <= 0 || transform.getInt() < window.getInt() ||
        !windowBroadcast ||
        windowBroadcast.getValue() != "trailing_batch_broadcast_v1")
      return emitOpError("requires transform/window length and spectrum policy");
  }
  return success();
}

LogicalResult SpectralBackwardKernelOp::verify() {
  auto inputCount = getOperation()->getAttrOfType<IntegerAttr>("input_count");
  auto outputCount = getOperation()->getAttrOfType<IntegerAttr>("output_count");
  auto hash = getOperation()->getAttrOfType<StringAttr>("tessera.schedule_hash");
  if (!inputCount || inputCount.getInt() < 2 || !outputCount ||
      outputCount.getInt() + 1 != inputCount.getInt() || !hash ||
      hash.getValue().size() != 64)
    return emitOpError("requires complete multi-output spectral adjoint identity");
  if (getInputs().size() != static_cast<size_t>(inputCount.getInt() +
                                                outputCount.getInt()))
    return emitOpError("expects source pointers followed by gradient destinations");
  for (Value operand : getInputs())
    if (!isa<LLVM::LLVMPointerType>(operand.getType()))
      return emitOpError("all launch operands must be !llvm.ptr");
  auto requiredString = [&](StringRef name) -> StringRef {
    auto attr = getOperation()->getAttrOfType<StringAttr>(name);
    return attr ? attr.getValue() : StringRef();
  };
  if (requiredString("kind").empty() ||
      (requiredString("pad_mode") != "constant" &&
       requiredString("pad_mode") != "reflect") ||
      requiredString("output_signature").empty() ||
      requiredString("mutation_lineage") !=
          "inputs_immutable_outputs_fresh_v1" ||
      ((requiredString("kind") == "tessera.stft" ||
        requiredString("kind") == "tessera.istft") &&
       requiredString("window_broadcast") != "trailing_batch_broadcast_v1"))
    return emitOpError("requires kind, output signature, and mutation lineage");
  return success();
}

LogicalResult ElementwiseKernelOp::verify() {
  auto family = getOperation()->getAttrOfType<StringAttr>("family");
  auto kind = getOperation()->getAttrOfType<StringAttr>("kind");
  auto storage = getOperation()->getAttrOfType<StringAttr>("storage");
  auto outputStorage =
      getOperation()->getAttrOfType<StringAttr>("output_storage");
  if (!family || !kind || !storage || !outputStorage)
    return emitOpError(
        "requires family, kind, storage, and output_storage attributes");
  bool unary = family.getValue() == "unary";
  bool binary = family.getValue() == "binary";
  bool predicate = family.getValue() == "predicate";
  bool compare = family.getValue() == "compare";
  bool logical = family.getValue() == "logical";
  bool bitwise = family.getValue() == "bitwise";
  bool where = family.getValue() == "where";
  bool transcendental = family.getValue() == "transcendental";
  bool binaryMath = family.getValue() == "binary_math";
  if (!unary && !binary && !predicate && !compare && !logical && !bitwise &&
      !where && !transcendental && !binaryMath)
    return emitOpError(
        "requires family in {unary, binary, predicate, compare, logical, bitwise, where, transcendental, binary_math}");
  bool binaryArity = binary || compare || binaryMath ||
                     (logical && kind.getValue() != "not") ||
                     (bitwise && kind.getValue() != "not" &&
                      kind.getValue() != "popcount");
  unsigned expected = where ? 5 : binaryArity ? 4 : 3;
  if (getInputs().size() != expected)
    return emitOpError() << family.getValue() << " expects " << expected
                         << " operands including destination and N";
  for (Value pointer : getInputs().drop_back())
    if (!isa<LLVM::LLVMPointerType>(pointer.getType()))
      return emitOpError("source and destination operands must be !llvm.ptr");
  if (!getInputs().back().getType().isInteger(64))
    return emitOpError("flattened element count N must be i64");
  StringRef requiredStorage = logical ? "i8" : bitwise ? "i32" : "f32";
  if (storage.getValue() != requiredStorage)
    return emitOpError() << family.getValue() << " requires storage=\""
                         << requiredStorage << "\"";
  if (where) {
    auto conditionStorage =
        getOperation()->getAttrOfType<StringAttr>("condition_storage");
    if (!conditionStorage || conditionStorage.getValue() != "i8")
      return emitOpError("where requires condition_storage=\"i8\"");
    if (outputStorage.getValue() != "f32" || kind.getValue() != "where")
      return emitOpError("where requires kind=where and output_storage=\"f32\"");
  } else if (predicate) {
    if (outputStorage.getValue() != "i8")
      return emitOpError("predicate requires output_storage=\"i8\"");
    if (kind.getValue() != "isnan" && kind.getValue() != "isinf" &&
        kind.getValue() != "isfinite")
      return emitOpError("predicate kind must be isnan|isinf|isfinite");
  } else if (compare) {
    if (outputStorage.getValue() != "i8")
      return emitOpError("compare requires output_storage=\"i8\"");
    if (kind.getValue() != "eq" && kind.getValue() != "ne" &&
        kind.getValue() != "lt" && kind.getValue() != "le" &&
        kind.getValue() != "gt" && kind.getValue() != "ge")
      return emitOpError("compare kind must be eq|ne|lt|le|gt|ge");
  } else if (logical) {
    if (outputStorage.getValue() != "i8")
      return emitOpError("logical requires output_storage=\"i8\"");
    if (kind.getValue() != "and" && kind.getValue() != "or" &&
        kind.getValue() != "xor" && kind.getValue() != "not")
      return emitOpError("logical kind must be and|or|xor|not");
  } else if (bitwise) {
    if (outputStorage.getValue() != "i32")
      return emitOpError("bitwise requires output_storage=\"i32\"");
    if (kind.getValue() != "and" && kind.getValue() != "or" &&
        kind.getValue() != "xor" && kind.getValue() != "not" &&
        kind.getValue() != "popcount")
      return emitOpError("bitwise kind must be and|or|xor|not|popcount");
  } else if (transcendental) {
    if (outputStorage.getValue() != "f32")
      return emitOpError("transcendental requires output_storage=\"f32\"");
    if (kind.getValue() != "exp" && kind.getValue() != "log" &&
        kind.getValue() != "tanh" && kind.getValue() != "sigmoid" &&
        kind.getValue() != "silu" && kind.getValue() != "gelu" &&
        kind.getValue() != "erf" && kind.getValue() != "softplus" &&
        kind.getValue() != "expm1" && kind.getValue() != "log1p" &&
        kind.getValue() != "cos" && kind.getValue() != "tan" &&
        kind.getValue() != "sinh" && kind.getValue() != "cosh" &&
        kind.getValue() != "asin" && kind.getValue() != "acos" &&
        kind.getValue() != "atan" && kind.getValue() != "erfc" &&
        kind.getValue() != "sin" && kind.getValue() != "lgamma" &&
        kind.getValue() != "digamma")
      return emitOpError("unsupported transcendental kind");
  } else if (binaryMath) {
    if (outputStorage.getValue() != "f32" ||
        (kind.getValue() != "pow" && kind.getValue() != "silu_mul"))
      return emitOpError("binary_math requires kind=pow|silu_mul and output_storage=\"f32\"");
  } else {
    if (outputStorage.getValue() != "f32")
      return emitOpError("unary/binary requires output_storage=\"f32\"");
    if (unary && kind.getValue() != "sqrt" && kind.getValue() != "rsqrt" &&
        kind.getValue() != "reciprocal" && kind.getValue() != "abs" &&
        kind.getValue() != "sign" && kind.getValue() != "floor" &&
        kind.getValue() != "ceil" && kind.getValue() != "trunc" &&
        kind.getValue() != "round")
      return emitOpError(
          "unary kind must be sqrt|rsqrt|reciprocal|abs|sign|floor|ceil|trunc|round");
    if (binary && kind.getValue() != "sub" && kind.getValue() != "div" &&
        kind.getValue() != "maximum" && kind.getValue() != "minimum" &&
        kind.getValue() != "add" && kind.getValue() != "mul" &&
        kind.getValue() != "mod" && kind.getValue() != "floor_div")
      return emitOpError(
          "binary kind must be sub|div|maximum|minimum|add|mul|mod|floor_div");
  }
  return success();
}

LogicalResult CudaIntrinsicKernelOp::verify() {
  if (getInputs().size() != 5)
    return emitOpError("expects A, B, C, destination, and N operands");
  for (Value pointer : getInputs().take_front(4))
    if (!isa<LLVM::LLVMPointerType>(pointer.getType()))
      return emitOpError("A/B/C/destination operands must be !llvm.ptr");
  if (!getInputs().back().getType().isInteger(64))
    return emitOpError("N must be i64");
  auto kind = getOperation()->getAttrOfType<StringAttr>("kind");
  auto input = getOperation()->getAttrOfType<StringAttr>("input_storage");
  auto output = getOperation()->getAttrOfType<StringAttr>("output_storage");
  auto rounding = getOperation()->getAttrOfType<StringAttr>("rounding");
  auto saturation = getOperation()->getAttrOfType<BoolAttr>("saturation");
  auto laneWidth = getOperation()->getAttrOfType<IntegerAttr>("lane_width");
  auto signedness = getOperation()->getAttrOfType<StringAttr>("signedness");
  auto predicate = getOperation()->getAttrOfType<StringAttr>("predicate_form");
  if (!kind || !input || !output || !rounding || !saturation || !laneWidth ||
      !signedness || !predicate)
    return emitOpError(
        "requires kind, input/output storage, rounding, saturation, "
        "lane_width, signedness, and predicate_form");
  StringRef value = kind.getValue();
  bool cast = value.starts_with("cvt_f32_i32_");
  bool supported =
      value == "abs_i32" || value == "min_i32" || value == "max_i32" ||
      value == "brev_u32" || value == "byte_perm_u32" ||
      value == "clz_u32" || value == "ffs_u32" || value == "popc_u32" ||
      value == "funnelshift_l_wrap_u32" ||
      value == "dp2a_lo_s32" || value == "dp4a_s32" ||
      value == "bitcast_f32_i32" || value == "bitcast_i32_f32" ||
      value == "vadd2_u16x2" || value == "vadd4_u8x4" ||
      value == "vadd4_s8x4" || value == "vaddss4_s8x4" ||
      value == "vsub4_u8x4" || value == "vsub4_s8x4" ||
      value == "vsubss4_s8x4" || value == "vabsdiff4_u8x4" ||
      value == "vcmpeq4_u8x4_mask" || value == "vcmpeq4_u8x4_pred" ||
      cast;
  if (!supported)
    return emitOpError("unsupported CUDA intrinsic kind");
  if (cast) {
    StringRef mode = value.drop_front(StringRef("cvt_f32_i32_").size());
    if (input.getValue() != "f32" || output.getValue() != "i32" ||
        (mode != "rn" && mode != "rd" && mode != "ru" && mode != "rz") ||
        rounding.getValue() != mode)
      return emitOpError("numeric casts require f32->i32 and matching RN/RD/RU/RZ");
  } else if (value == "bitcast_f32_i32") {
    if (input.getValue() != "f32" || output.getValue() != "i32")
      return emitOpError("float_as_int requires f32->i32 storage");
  } else if (value == "bitcast_i32_f32") {
    if (input.getValue() != "i32" || output.getValue() != "f32")
      return emitOpError("int_as_float requires i32->f32 storage");
  } else if (input.getValue() != "i32" || output.getValue() != "i32" ||
             rounding.getValue() != "none") {
    return emitOpError("integer/packed intrinsics require i32 bit containers and rounding=none");
  }
  bool packed = value.starts_with("v");
  if (!packed &&
      (laneWidth.getInt() != 0 || signedness.getValue() != "scalar" ||
       predicate.getValue() != "none" || saturation.getValue()))
    return emitOpError("scalar CUDA Math attributes must use lane_width=0, "
                       "signedness=scalar, predicate_form=none, and no saturation");
  return success();
}

LogicalResult AllocOp::verify() {
  if (getSpace() != "smem" && getSpace() != "tmem" &&
      getSpace() != "gmem")
    return emitOpError("space must be smem|tmem|gmem");
  if (getBytes() == 0)
    return emitOpError("bytes must be positive");
  // layout's #tile.layout typing is enforced by the ODS constraint since the
  // CAKE Phase 1 §5.2 row 5 hoist — rejected at parse, before this runs.
  if (getSpace() == "tmem" &&
      getOperation()->getAttrOfType<StringAttr>("target") &&
      getOperation()->getAttrOfType<StringAttr>("target").getValue() ==
          "nvidia_sm120")
    return emitOpError("SM120 has no TMEM physical allocation");
  return success();
}

LogicalResult DeallocOp::verify() {
  if (!getBuffer().getDefiningOp<AllocOp>())
    return emitOpError("buffer must be the SSA result of tile.alloc");
  return success();
}

LogicalResult PipelineInitOp::verify() {
  if (getDepth() == 0 || getStage() >= getDepth())
    return emitOpError("requires depth > 0 and stage < depth");
  if (getPhase() > 1)
    return emitOpError("phase must be 0 or 1");
  if (getRole() != "producer" && getRole() != "consumer")
    return emitOpError("role must be producer|consumer");
  uint64_t expected = getRole() == "producer" ? 1 : 0;
  if (getPhase() != expected)
    return emitOpError(
        "initial producer phase must be 1 and consumer phase must be 0");
  if (Value logicalRole = getLogicalRole()) {
    // A loop-carried !tile.role has no direct defining op; the shared
    // TileDataflowLegalityPass resolves that provenance across the back-edge.
    if (auto declaration = logicalRole.getDefiningOp<RoleOp>();
        declaration && declaration.getKind() != getRole())
      return emitOpError("logical role kind must match pipeline role");
  }
  return success();
}

LogicalResult PipelineAdvanceOp::verify() {
  if (getInputs().empty() ||
      !isa<PipelineStateType>(getInputs().front().getType()))
    return emitOpError(
        "first operand must be the prior !tile.pipeline_state");
  return success();
}

LogicalResult AllReduceOp::verify() {
  return verifyCollective(getOperation(), getInput(), getOutput(),
                          /*scalesAxis=*/false, /*multiply=*/false,
                          /*requiresReduction=*/true);
}

LogicalResult ReduceScatterOp::verify() {
  return verifyCollective(getOperation(), getInput(), getOutput(),
                          /*scalesAxis=*/true, /*multiply=*/false,
                          /*requiresReduction=*/true);
}

LogicalResult AllGatherOp::verify() {
  return verifyCollective(getOperation(), getInput(), getOutput(),
                          /*scalesAxis=*/true, /*multiply=*/true,
                          /*requiresReduction=*/false);
}

LogicalResult AllToAllOp::verify() {
  return verifyCollective(getOperation(), getInput(), getOutput(),
                          /*scalesAxis=*/false, /*multiply=*/false,
                          /*requiresReduction=*/false);
}

LogicalResult CollectivePermuteOp::verify() {
  if (getInput().getType() != getOutput().getType())
    return emitOpError("must preserve the payload type");
  if (getMeshAxis().empty())
    return emitOpError("requires a non-empty mesh_axis");
  auto sources = getSourcePeers();
  auto targets = getTargetPeers();
  if (sources.empty() || sources.size() != targets.size())
    return emitOpError("requires equal non-empty source and target peer maps");
  llvm::SmallDenseSet<int64_t> seenSources;
  llvm::SmallDenseSet<int64_t> seenTargets;
  int64_t worldSize = getWorldSize().value_or(0);
  if (worldSize && worldSize < 2)
    return emitOpError("world_size must be at least two when present");
  for (auto [source, target] : llvm::zip_equal(sources, targets)) {
    if (source < 0 || target < 0 || (worldSize &&
        (source >= worldSize || target >= worldSize)))
      return emitOpError("peer is outside the communicator");
    if (!seenSources.insert(source).second || !seenTargets.insert(target).second)
      return emitOpError("source and target peers must each be unique");
  }
  return success();
}

static LogicalResult verifyPointerAndI64Tail(Operation *op, ValueRange inputs,
                                             unsigned pointerCount,
                                             unsigned i64Count) {
  if (inputs.size() != pointerCount + i64Count)
    return op->emitOpError() << "expects " << pointerCount << " pointer and "
                             << i64Count << " i64 operands";
  for (Value value : inputs.take_front(pointerCount))
    if (!isa<LLVM::LLVMPointerType>(value.getType()))
      return op->emitOpError("buffer operands must be !llvm.ptr");
  for (Value value : inputs.drop_front(pointerCount))
    if (!value.getType().isInteger(64))
      return op->emitOpError("dimension operands must be i64");
  return success();
}

LogicalResult ArgReduceKernelOp::verify() {
  if (failed(verifyPointerAndI64Tail(getOperation(), getInputs(), 2, 2)))
    return failure();
  auto kind = getOperation()->getAttrOfType<StringAttr>("kind");
  auto storage = getOperation()->getAttrOfType<StringAttr>("storage");
  auto output = getOperation()->getAttrOfType<StringAttr>("output_storage");
  auto tie = getOperation()->getAttrOfType<StringAttr>("tie_break");
  if (!kind || (kind.getValue() != "argmax" && kind.getValue() != "argmin"))
    return emitOpError("requires kind=argmax|argmin");
  if (!storage || storage.getValue() != "f32" || !output ||
      output.getValue() != "i32")
    return emitOpError("requires f32 input and i32 output storage");
  if (!tie || tie.getValue() != "first")
    return emitOpError("requires tie_break=first");
  return success();
}

LogicalResult ScanKernelOp::verify() {
  if (failed(verifyPointerAndI64Tail(getOperation(), getInputs(), 2, 2)))
    return failure();
  auto kind = getOperation()->getAttrOfType<StringAttr>("kind");
  auto storage = getOperation()->getAttrOfType<StringAttr>("storage");
  auto inclusive = getOperation()->getAttrOfType<BoolAttr>("inclusive");
  if (!kind || (kind.getValue() != "sum" && kind.getValue() != "product" &&
                kind.getValue() != "max" && kind.getValue() != "min"))
    return emitOpError("requires kind=sum|product|max|min");
  if (!storage || storage.getValue() != "f32")
    return emitOpError("requires storage=f32");
  if (!inclusive || !inclusive.getValue())
    return emitOpError("currently requires inclusive=true");
  return success();
}

LogicalResult NormKernelOp::verify() {
  if (getInputs().size() != 5)
    return emitOpError("expects source, destination, rows, columns, and epsilon");
  if (!isa<LLVM::LLVMPointerType>(getInputs()[0].getType()) ||
      !isa<LLVM::LLVMPointerType>(getInputs()[1].getType()) ||
      !getInputs()[2].getType().isInteger(64) ||
      !getInputs()[3].getType().isInteger(64) ||
      !getInputs()[4].getType().isF32())
    return emitOpError("requires ptr, ptr, i64, i64, f32 operands");
  auto kind = getOperation()->getAttrOfType<StringAttr>("kind");
  auto storage = getOperation()->getAttrOfType<StringAttr>("storage");
  auto accum = getOperation()->getAttrOfType<StringAttr>("accum");
  auto axis = getOperation()->getAttrOfType<IntegerAttr>("axis");
  auto affine = getOperation()->getAttrOfType<BoolAttr>("affine");
  if (!kind || (kind.getValue() != "rmsnorm" && kind.getValue() != "layernorm"))
    return emitOpError("requires kind=rmsnorm|layernorm");
  if (!storage ||
      (storage.getValue() != "f16" && storage.getValue() != "bf16" &&
       storage.getValue() != "f32") ||
      !accum || accum.getValue() != "f32")
    return emitOpError("requires f16, bf16, or f32 storage and f32 accumulation");
  if (!axis || axis.getInt() != -1 || !affine || affine.getValue())
    return emitOpError("requires axis=-1 and affine=false");
  return success();
}

LogicalResult RopeKernelOp::verify() {
  if (failed(verifyPointerAndI64Tail(getOperation(), getInputs(), 3, 2)))
    return failure();
  auto storage = getOperation()->getAttrOfType<StringAttr>("storage");
  auto layout = getOperation()->getAttrOfType<StringAttr>("layout");
  if (!storage || storage.getValue() != "f32")
    return emitOpError("requires storage=f32");
  if (!layout || layout.getValue() != "interleaved_pairs")
    return emitOpError("requires layout=interleaved_pairs");
  return success();
}

LogicalResult AlibiKernelOp::verify() {
  if (failed(verifyPointerAndI64Tail(getOperation(), getInputs(), 2, 2)))
    return failure();
  auto storage = getOperation()->getAttrOfType<StringAttr>("storage");
  auto formula = getOperation()->getAttrOfType<StringAttr>("formula");
  if (!storage || storage.getValue() != "f32")
    return emitOpError("requires storage=f32");
  if (!formula || formula.getValue() != "slope_times_j_minus_i")
    return emitOpError("requires formula=slope_times_j_minus_i");
  return success();
}

LogicalResult X86ABIKernelOp::verify() {
  auto symbol = getOperation()->getAttrOfType<StringAttr>("symbol");
  auto abi = getOperation()->getAttrOfType<StringAttr>("abi");
  auto family = getOperation()->getAttrOfType<StringAttr>("family");
  auto effects = getOperation()->getAttrOfType<StringAttr>("effects");
  auto returnsStatus =
      getOperation()->getAttrOfType<BoolAttr>("returns_status");
  if (!symbol || !symbol.getValue().starts_with("tessera_x86_"))
    return emitOpError("requires symbol with tessera_x86_ prefix");
  if (!abi || !abi.getValue().starts_with("tessera.x86.") ||
      !abi.getValue().ends_with(".v1"))
    return emitOpError("requires a tessera.x86.*.v1 ABI identifier");
  if (!family || family.getValue().empty())
    return emitOpError("requires a non-empty family attribute");
  if (!returnsStatus)
    return emitOpError("requires an explicit returns_status boolean");
  if (!effects || (effects.getValue() != "readonly" &&
                   effects.getValue() != "writeonly" &&
                   effects.getValue() != "readwrite" &&
                   effects.getValue() != "stateful"))
    return emitOpError(
        "requires effects=readonly|writeonly|readwrite|stateful");
  if (getInputs().empty())
    return emitOpError("requires at least one typed ABI operand");
  for (Value value : getInputs()) {
    Type type = value.getType();
    if (!isa<LLVM::LLVMPointerType>(type) && !type.isInteger(64) &&
        !type.isInteger(32) && !type.isF32() && !type.isF64())
      return emitOpError(
          "operands must be !llvm.ptr, i64, i32, f32, or f64");
  }
  return success();
}

LogicalResult AttentionKernelOp::verify() {
  auto bias = getOperation()->getAttrOfType<BoolAttr>("bias");
  bool hasBias = bias && bias.getValue();
  auto lseCheckpoint =
      getOperation()->getAttrOfType<StringAttr>("lse_checkpoint");
  bool hasSavedLse = lseCheckpoint && lseCheckpoint.getValue() == "saved";
  if (!bias)
    return emitOpError("requires an explicit bias boolean");
  if (lseCheckpoint && (lseCheckpoint.getValue() != "recompute" &&
                        lseCheckpoint.getValue() != "saved"))
    return emitOpError("requires lse_checkpoint=\"recompute\" or \"saved\"");
  if (getInputs().size() != 11 + unsigned(hasBias) + unsigned(hasSavedLse))
    return emitOpError(
        "expects Q, K, V, optional bias, O, optional row_lse, B, Hq, Hkv, Sq, Sk, D, and Dv operands");
  unsigned pointerCount = 4 + unsigned(hasBias) + unsigned(hasSavedLse);
  for (Value pointer : getInputs().take_front(pointerCount))
    if (!isa<LLVM::LLVMPointerType>(pointer.getType()))
      return emitOpError("Q, K, V, optional bias, O, and optional row_lse operands must be !llvm.ptr");
  for (Value dim : getInputs().drop_front(pointerCount))
    if (!dim.getType().isInteger(64))
      return emitOpError("attention dimensions must be i64");
  auto storage = getOperation()->getAttrOfType<StringAttr>("storage");
  auto accum = getOperation()->getAttrOfType<StringAttr>("accum");
  auto scale = getOperation()->getAttrOfType<FloatAttr>("scale");
  auto causal = getOperation()->getAttrOfType<BoolAttr>("causal");
  auto windowLeft = getOperation()->getAttrOfType<IntegerAttr>("window_left");
  auto windowRight = getOperation()->getAttrOfType<IntegerAttr>("window_right");
  auto softcap = getOperation()->getAttrOfType<FloatAttr>("softcap");
  auto dropout = getOperation()->getAttrOfType<FloatAttr>("dropout_p");
  auto dropoutSeed = getOperation()->getAttrOfType<IntegerAttr>("dropout_seed");
  if (!storage || (storage.getValue() != "f16" &&
                   storage.getValue() != "bf16" &&
                   storage.getValue() != "f32"))
    return emitOpError(
        "requires storage=\"f16\", storage=\"bf16\", or storage=\"f32\"");
  if (!accum || accum.getValue() != "f32")
    return emitOpError("requires accum=\"f32\"");
  if (!scale || !scale.getValue().isFinite() || scale.getValueAsDouble() <= 0.0)
    return emitOpError("requires a finite positive f32 scale");
  if (!causal)
    return emitOpError("requires an explicit causal boolean");
  if (!windowLeft || !windowRight || windowLeft.getInt() < -1 ||
      windowRight.getInt() < -1)
    return emitOpError("requires window_left/window_right >= -1");
  if (!softcap || !softcap.getValue().isFinite() ||
      softcap.getValueAsDouble() < 0.0)
    return emitOpError("requires finite softcap >= 0");
  if (!dropout || !dropout.getValue().isFinite() ||
      dropout.getValueAsDouble() < 0.0 || dropout.getValueAsDouble() >= 1.0)
    return emitOpError("requires finite dropout_p in [0, 1)");
  if (!dropoutSeed)
    return emitOpError("requires an explicit dropout_seed");
  return success();
}

LogicalResult DepthAttentionKernelOp::verify() {
  if (getInputs().size() != 6)
    return emitOpError(
        "expects query, sources, output, source_count, rows, and width operands");
  for (Value pointer : getInputs().take_front(3))
    if (!isa<LLVM::LLVMPointerType>(pointer.getType()))
      return emitOpError("query, sources, and output must be !llvm.ptr");
  for (Value extent : getInputs().drop_front(3))
    if (!extent.getType().isInteger(64))
      return emitOpError("source_count, rows, and width must be i64");
  auto requiredString = [&](StringRef name) -> StringRef {
    auto attr = getOperation()->getAttrOfType<StringAttr>(name);
    return attr ? attr.getValue() : StringRef();
  };
  if (requiredString("storage") != "f32" ||
      requiredString("softmax") != "f32" ||
      requiredString("accum") != "f32")
    return emitOpError("initial physical boundary requires all-f32 numeric policy");
  if (requiredString("arch") != "gfx1151" &&
      requiredString("arch") != "zen5-avx512")
    return emitOpError("requires an exact owned architecture profile");
  auto eps = getOperation()->getAttrOfType<FloatAttr>("eps");
  auto sourceTile = getOperation()->getAttrOfType<IntegerAttr>("source_tile");
  auto workgroup =
      getOperation()->getAttrOfType<IntegerAttr>("tessera.workgroup_size");
  auto hash = getOperation()->getAttrOfType<StringAttr>("tessera.schedule_hash");
  auto reassociable =
      getOperation()->getAttrOfType<BoolAttr>("reassociable");
  if (!eps || !eps.getValue().isFinite() || eps.getValueAsDouble() <= 0.0 ||
      !sourceTile || sourceTile.getInt() <= 0 || !workgroup ||
      workgroup.getInt() <= 0 || !hash || hash.getValue().size() != 64 ||
      !reassociable || !reassociable.getValue())
    return emitOpError("requires complete epsilon, tiling, reassociation, and schedule identity");
  if (requiredString("statistics_recurrence") !=
          "rms_key_online_softmax_stats_v1" ||
      requiredString("merge_recurrence") !=
          "max_shifted_pairwise_merge_v1")
    return emitOpError("requires canonical depth-attention recurrences");
  return success();
}

LogicalResult AttentionBackwardKernelOp::verify() {
  auto bias = getOperation()->getAttrOfType<BoolAttr>("bias");
  bool hasBias = bias && bias.getValue();
  auto lseCheckpoint =
      getOperation()->getAttrOfType<StringAttr>("lse_checkpoint");
  bool hasSavedLse = lseCheckpoint && lseCheckpoint.getValue() == "saved";
  if (!bias)
    return emitOpError("requires an explicit bias boolean");
  if (lseCheckpoint && (lseCheckpoint.getValue() != "recompute" &&
                        lseCheckpoint.getValue() != "saved"))
    return emitOpError("requires lse_checkpoint=\"recompute\" or \"saved\"");
  if (getInputs().size() != 14 + unsigned(hasBias) + unsigned(hasSavedLse))
    return emitOpError(
        "expects dO, Q, K, V, optional bias, optional row_lse, dQ, dK, dV, B, Hq, Hkv, Sq, Sk, D, and Dv operands");
  unsigned pointerCount = 7 + unsigned(hasBias) + unsigned(hasSavedLse);
  for (Value pointer : getInputs().take_front(pointerCount))
    if (!isa<LLVM::LLVMPointerType>(pointer.getType()))
      return emitOpError(
          "dO, Q, K, V, optional bias, optional row_lse, dQ, dK, and dV operands must be !llvm.ptr");
  for (Value dim : getInputs().drop_front(pointerCount))
    if (!dim.getType().isInteger(64))
      return emitOpError("attention backward dimensions must be i64");
  auto storage = getOperation()->getAttrOfType<StringAttr>("storage");
  auto accum = getOperation()->getAttrOfType<StringAttr>("accum");
  auto scale = getOperation()->getAttrOfType<FloatAttr>("scale");
  auto causal = getOperation()->getAttrOfType<BoolAttr>("causal");
  auto windowLeft = getOperation()->getAttrOfType<IntegerAttr>("window_left");
  auto windowRight = getOperation()->getAttrOfType<IntegerAttr>("window_right");
  auto softcap = getOperation()->getAttrOfType<FloatAttr>("softcap");
  auto dropout = getOperation()->getAttrOfType<FloatAttr>("dropout_p");
  auto dropoutSeed = getOperation()->getAttrOfType<IntegerAttr>("dropout_seed");
  auto route = getOperation()->getAttrOfType<StringAttr>("route");
  auto deterministic = getOperation()->getAttrOfType<BoolAttr>("deterministic");
  auto workspace = getOperation()->getAttrOfType<IntegerAttr>("workspace_bytes");
  auto workspaceOwner =
      getOperation()->getAttrOfType<StringAttr>("workspace_owner");
  if (!storage || (storage.getValue() != "f16" &&
                   storage.getValue() != "bf16" &&
                   storage.getValue() != "f32"))
    return emitOpError(
        "deterministic reference route requires storage=\"f16\", "
        "storage=\"bf16\", or storage=\"f32\"");
  if (!accum || accum.getValue() != "f32")
    return emitOpError("requires accum=\"f32\"");
  if (!scale || !scale.getValue().isFinite() || scale.getValueAsDouble() <= 0.0)
    return emitOpError("requires a finite positive f32 scale");
  if (!causal)
    return emitOpError("requires an explicit causal boolean");
  if (!windowLeft || !windowRight || windowLeft.getInt() < -1 ||
      windowRight.getInt() < -1)
    return emitOpError("requires window_left/window_right >= -1");
  if (!softcap || !softcap.getValue().isFinite() ||
      softcap.getValueAsDouble() < 0.0)
    return emitOpError("requires finite softcap >= 0");
  if (!dropout || !dropout.getValue().isFinite() ||
      dropout.getValueAsDouble() < 0.0 || dropout.getValueAsDouble() >= 1.0)
    return emitOpError("requires finite dropout_p in [0, 1)");
  if (!dropoutSeed)
    return emitOpError("requires an explicit dropout_seed");
  if (!route)
    return emitOpError("requires an explicit backward route");
  if (!deterministic || !deterministic.getValue())
    return emitOpError("attention backward requires deterministic=true");
  if (!workspace || !workspaceOwner)
    return emitOpError("requires explicit workspace_bytes/workspace_owner");
  if (route.getValue() == "deterministic_direct") {
    if (workspace.getInt() != 0)
      return emitOpError("deterministic_direct requires workspace_bytes=0");
    if (workspaceOwner.getValue() != "output_element")
      return emitOpError(
          "deterministic_direct requires workspace_owner=\"output_element\"");
    return success();
  }
  if (route.getValue() != "deterministic_split_reduced")
    return emitOpError(
        "route must be \"deterministic_direct\" or "
        "\"deterministic_split_reduced\"");
  if (workspace.getInt() <= 0 ||
      workspaceOwner.getValue() != "program_launch")
    return emitOpError(
        "deterministic_split_reduced requires positive launch-owned workspace");
  auto splitCount =
      getOperation()->getAttrOfType<IntegerAttr>("split_count");
  auto reductionOrder =
      getOperation()->getAttrOfType<DenseI64ArrayAttr>("reduction_order");
  auto queryBlock =
      getOperation()->getAttrOfType<IntegerAttr>("query_block");
  auto keyBlock =
      getOperation()->getAttrOfType<IntegerAttr>("key_block");
  auto loopOrder = getOperation()->getAttrOfType<ArrayAttr>("loop_order");
  if (!splitCount || splitCount.getInt() < 2)
    return emitOpError("split/reduced route requires split_count >= 2");
  if (!reductionOrder ||
      reductionOrder.size() != splitCount.getInt())
    return emitOpError(
        "split/reduced route requires one reduction_order entry per split");
  for (auto [index, value] : llvm::enumerate(reductionOrder.asArrayRef()))
    if (value != static_cast<int64_t>(index))
      return emitOpError(
          "split/reduced partials must reduce in ascending split order");
  if (!queryBlock || queryBlock.getInt() <= 0 ||
      !keyBlock || keyBlock.getInt() <= 0)
    return emitOpError("split/reduced route requires positive block sizes");
  if (!loopOrder || loopOrder.size() != 5)
    return emitOpError(
        "split/reduced route requires five-level canonical loop_order");
  return success();
}

LogicalResult TrainingKernelOp::verify() {
  auto family = getOperation()->getAttrOfType<StringAttr>("family");
  if (!family)
    return emitOpError("requires a training family");
  bool lion = family.getValue() == "lion_vjp";
  bool optimizer = family.getValue() == "optimizer_vjp";
  bool sequenceMixer = family.getValue() == "sequence_mixer_backward";
  auto topology = getOperation()->getAttrOfType<StringAttr>("topology");
  bool factored = !lion && !optimizer && !sequenceMixer && topology &&
                  topology.getValue() == "factored";
  bool full = !lion && !optimizer && !sequenceMixer && topology &&
              topology.getValue() == "full";
  if (!lion && !optimizer && !sequenceMixer &&
      family.getValue() != "adafactor_vjp")
    return emitOpError(
        "training contract supports typed optimizer VJPs, Lion, Adafactor, and sequence mixers");
  unsigned optimizerPointers = 0;
  if (optimizer) {
    auto kind = getOperation()->getAttrOfType<StringAttr>("optimizer");
    if (!kind)
      return emitOpError("optimizer VJP requires an optimizer identity");
    if (kind.getValue() == "sgd") optimizerPointers = 5;
    else if (kind.getValue() == "momentum" || kind.getValue() == "nesterov")
      optimizerPointers = 8;
    else if (kind.getValue() == "adam" || kind.getValue() == "adamw")
      optimizerPointers = 11;
    else
      return emitOpError("optimizer VJP has an unsupported family");
  }
  unsigned pointerCount =
      sequenceMixer ? 13 : lion ? 8 : optimizer ? optimizerPointers
                                                : factored ? 9 : full ? 7 : 0;
  unsigned dimensionCount = sequenceMixer ? 5 : factored ? 2 : 1;
  if (!pointerCount || getInputs().size() != pointerCount + dimensionCount)
    return emitOpError("training operands disagree with family/state topology");
  for (Value pointer : getInputs().take_front(pointerCount))
    if (!isa<LLVM::LLVMPointerType>(pointer.getType()))
      return emitOpError("training buffers must be !llvm.ptr");
  for (Value dimension : getInputs().drop_front(pointerCount))
    if (!dimension.getType().isInteger(64))
      return emitOpError("training dimensions must be i64");
  auto storage = getOperation()->getAttrOfType<StringAttr>("storage");
  auto mutation = getOperation()->getAttrOfType<StringAttr>("mutation_mode");
  auto alias = getOperation()->getAttrOfType<StringAttr>("alias_policy");
  auto hash = getOperation()->getAttrOfType<StringAttr>("tessera.schedule_hash");
  if (!storage || storage.getValue() != "f32")
    return emitOpError("initial training contract requires f32 storage");
  if (lion) {
    auto derivative =
        getOperation()->getAttrOfType<StringAttr>("derivative_policy");
    if (!derivative || derivative.getValue() != "stop_gradient_through_sign")
      return emitOpError("requires the canonical stop-sign derivative policy");
  }
  if (!mutation || mutation.getValue() != "functional" || !alias ||
      alias.getValue() != "no_input_output_alias")
    return emitOpError("requires a functional no-alias state transition");
  if (!hash || hash.getValue().size() != 64)
    return emitOpError("requires a schedule lineage hash");
  if (sequenceMixer) {
    auto mixer = getOperation()->getAttrOfType<StringAttr>("mixer_family");
    auto workspace =
        getOperation()->getAttrOfType<StringAttr>("workspace_owner");
    auto phases = getOperation()->getAttrOfType<ArrayAttr>("phase_order");
    auto chunk = getOperation()->getAttrOfType<IntegerAttr>("chunk_size");
    if (!mixer || (mixer.getValue() != "gated_deltanet" &&
                   mixer.getValue() != "kimi_delta_attention" &&
                   mixer.getValue() != "modified_delta_attention"))
      return emitOpError("requires a proven sequence-mixer family");
    if (!workspace || workspace.getValue() != "program_launch" || !chunk ||
        chunk.getInt() <= 0)
      return emitOpError("requires positive chunks and launch-owned workspace");
    static constexpr StringLiteral expected[] = {
        "checkpoint", "chunk_summary", "chunk_prefix", "chunk_fill", "reverse"};
    if (!phases || phases.size() != 5)
      return emitOpError("requires five ordered sequence-mixer phases");
    for (auto [attribute, name] : llvm::zip_equal(phases, expected)) {
      auto value = dyn_cast<StringAttr>(attribute);
      if (!value || value.getValue() != name)
        return emitOpError("requires canonical sequence-mixer phase order");
    }
    return success();
  }
  auto transition =
      getOperation()->getAttrOfType<StringAttr>("state_transition");
  if (!transition)
    return emitOpError("requires an optimizer state transition");
  if (lion && transition.getValue() != "m@0-read-only;d_m@1-fresh")
    return emitOpError("requires the Lion state transition");
  if (!lion && transition.getValue() != "state@0-read-only;d_state@1-fresh")
    return emitOpError("requires the Adafactor state transition");
  if (optimizer) {
    for (StringRef name : {"learning_rate", "beta1", "beta2", "epsilon",
                           "momentum", "weight_decay"}) {
      auto value = getOperation()->getAttrOfType<FloatAttr>(name);
      if (!value || !value.getValue().isFinite())
        return emitOpError("requires finite optimizer coefficients");
    }
    return success();
  }
  SmallVector<StringRef> coefficientNames{"learning_rate", "beta2"};
  coefficientNames.push_back(lion ? "weight_decay" : "epsilon");
  for (StringRef name : coefficientNames) {
    auto value = getOperation()->getAttrOfType<FloatAttr>(name);
    if (!value || !value.getValue().isFinite())
      return emitOpError("requires finite Lion coefficients");
  }
  return success();
}

LogicalResult SolverIFTKernelOp::verify() {
  if (getInputs().size() != 7)
    return emitOpError("requires parameter/solution/cotangent/residual/linear-solution/parameter-cotangent/N");
  for (Value pointer : getInputs().take_front(6))
    if (!isa<LLVM::LLVMPointerType>(pointer.getType()))
      return emitOpError("IFT phase buffers must be !llvm.ptr");
  if (!getInputs().back().getType().isInteger(64))
    return emitOpError("IFT element count must be i64");
  auto model = getOperation()->getAttrOfType<StringAttr>("residual_model");
  auto solver = getOperation()->getAttrOfType<StringAttr>("linear_solver");
  auto productMode = getOperation()->getAttrOfType<StringAttr>("product_mode");
  auto transpose = getOperation()->getAttrOfType<BoolAttr>("transpose");
  auto wrt = getOperation()->getAttrOfType<StringAttr>("wrt");
  auto scale = getOperation()->getAttrOfType<FloatAttr>("adjoint_scale");
  auto storage = getOperation()->getAttrOfType<StringAttr>("storage");
  auto accum = getOperation()->getAttrOfType<StringAttr>("accum");
  auto hash = getOperation()->getAttrOfType<StringAttr>("tessera.schedule_hash");
  auto residualDigest =
      getOperation()->getAttrOfType<StringAttr>("residual_digest");
  if (!model || model.getValue() != "diagonal_sqrt_v1" || !solver ||
      solver.getValue() != "diagonal_matrix_free_v1" || !productMode ||
      !transpose || !wrt || wrt.getValue() != "parameter" ||
      !scale || scale.getValueAsDouble() != -1.0)
    return emitOpError("requires the canonical diagonal-sqrt IFT chain");
  if ((productMode.getValue() == "vjp" && !transpose.getValue()) ||
      (productMode.getValue() == "jvp" && transpose.getValue()) ||
      (productMode.getValue() != "vjp" && productMode.getValue() != "jvp"))
    return emitOpError("requires vjp/transpose or jvp/non-transpose solver products");
  if (!storage || storage.getValue() != "f32" || !accum ||
      accum.getValue() != "f32")
    return emitOpError("initial IFT package requires f32 storage/accumulation");
  if (!hash || hash.getValue().size() != 64 || !residualDigest ||
      residualDigest.getValue().size() != 64)
    return emitOpError("requires schedule and residual lineage digests");
  return success();
}

LogicalResult ESLowRankCorrectionKernelOp::verify() {
  if (getInputs().size() != 8)
    return emitOpError("requires x/member-ids/key/output pointers and P/rows/in/out dimensions");
  for (Value pointer : getInputs().take_front(4))
    if (!isa<LLVM::LLVMPointerType>(pointer.getType()))
      return emitOpError("ES data operands must be !llvm.ptr");
  for (Value dim : getInputs().drop_front(4))
    if (!dim.getType().isInteger(64))
      return emitOpError("ES launch dimensions must be i64");
  auto requiredString = [&](StringRef name) -> StringRef {
    auto attr = getOperation()->getAttrOfType<StringAttr>(name);
    return attr ? attr.getValue() : StringRef();
  };
  auto rank = getOperation()->getAttrOfType<IntegerAttr>("rank");
  auto epoch = getOperation()->getAttrOfType<IntegerAttr>("epoch");
  auto version = getOperation()->getAttrOfType<IntegerAttr>("rng_version");
  auto sigma = getOperation()->getAttrOfType<FloatAttr>("sigma");
  auto hash = getOperation()->getAttrOfType<StringAttr>("tessera.schedule_hash");
  StringRef architecture = requiredString("arch");
  if ((architecture != "gfx1151" && architecture != "zen5-avx512") ||
      requiredString("score") != "gaussian" ||
      requiredString("rng_algorithm") !=
          "splitmix64-philox4x32-boxmuller" ||
      requiredString("storage") != "f32" || requiredString("accum") != "f32" ||
      !rank || rank.getInt() != 1 || !epoch || epoch.getInt() < 0 ||
      !version || version.getInt() != 1 || !sigma ||
      !sigma.getValue().isFinite() || sigma.getValueAsDouble() <= 0.0 ||
      !hash || hash.getValue().size() != 64)
    return emitOpError(
        "requires the content-addressed gfx1151 or Zen 5 AVX-512 ES rank-1 contract");
  return success();
}

LogicalResult PagedKVReadKernelOp::verify() {
  if (getInputs().size() != 10)
    return emitOpError(
        "expects pages, page table, output, P, LP, page size, H, D, start, and tokens");
  for (Value pointer : getInputs().take_front(3))
    if (!isa<LLVM::LLVMPointerType>(pointer.getType()))
      return emitOpError("pages, page table, and output must be !llvm.ptr");
  for (Value dim : getInputs().drop_front(3))
    if (!dim.getType().isInteger(64))
      return emitOpError("paged-KV dimensions must be i64");
  auto storage = getOperation()->getAttrOfType<StringAttr>("storage");
  auto table = getOperation()->getAttrOfType<StringAttr>("table_storage");
  auto route = getOperation()->getAttrOfType<StringAttr>("route");
  if (!storage || storage.getValue() != "f32")
    return emitOpError("currently requires storage=\"f32\"");
  if (!table || table.getValue() != "i32")
    return emitOpError("requires table_storage=\"i32\"");
  if (!route || route.getValue() != "direct")
    return emitOpError("canonical materializer currently requires route=\"direct\"");
  return success();
}

LogicalResult PagedAttentionKernelOp::verify() {
  if (getInputs().size() != 14)
    return emitOpError(
        "expects Q, K pages, V pages, page table, token indices, O, P, LP, page size, H, Q length, token count, D, and causal offset");
  for (Value pointer : getInputs().take_front(6))
    if (!isa<LLVM::LLVMPointerType>(pointer.getType()))
      return emitOpError("paged-attention buffers must be !llvm.ptr");
  for (Value dim : getInputs().drop_front(6))
    if (!dim.getType().isInteger(64))
      return emitOpError("paged-attention dimensions and causal offset must be i64");
  auto storage = getOperation()->getAttrOfType<StringAttr>("storage");
  auto accum = getOperation()->getAttrOfType<StringAttr>("accum");
  auto table = getOperation()->getAttrOfType<StringAttr>("table_storage");
  auto index = getOperation()->getAttrOfType<StringAttr>("token_index_storage");
  auto scale = getOperation()->getAttrOfType<FloatAttr>("scale");
  auto causal = getOperation()->getAttrOfType<BoolAttr>("causal");
  auto route = getOperation()->getAttrOfType<StringAttr>("route");
  if (!storage || storage.getValue() != "f32" || !accum ||
      accum.getValue() != "f32")
    return emitOpError("requires f32 storage and accumulation");
  if (!table || table.getValue() != "i32" || !index ||
      index.getValue() != "i64")
    return emitOpError("requires i32 page-table and i64 token-index storage");
  if (!scale || !scale.getValue().isFinite() || scale.getValueAsDouble() <= 0.0)
    return emitOpError("requires a finite positive f32 scale");
  if (!causal)
    return emitOpError("requires an explicit causal boolean");
  if (!route || route.getValue() != "fused_direct")
    return emitOpError("requires route=\"fused_direct\"");
  return success();
}

LogicalResult ReplaySSMDecodeKernelOp::verify() {
  if (getInputs().size() != 11)
    return emitOpError("expects delta, x, B, S0, C, A, Y, batch, channels, state, and tokens");
  for (Value pointer : getInputs().take_front(7))
    if (!isa<LLVM::LLVMPointerType>(pointer.getType()))
      return emitOpError("ReplaySSM buffers must be !llvm.ptr");
  for (Value dim : getInputs().drop_front(7))
    if (!dim.getType().isInteger(64))
      return emitOpError("ReplaySSM dimensions must be i64");
  auto storage = getOperation()->getAttrOfType<StringAttr>("storage");
  auto route = getOperation()->getAttrOfType<StringAttr>("route");
  if (!storage || storage.getValue() != "f32")
    return emitOpError("currently requires storage=\"f32\"");
  if (!route || route.getValue() != "output_only")
    return emitOpError("decode requires route=\"output_only\"");
  return success();
}

LogicalResult ReplaySSMFlushKernelOp::verify() {
  if (getInputs().size() != 9)
    return emitOpError("expects delta, x, B, S0, A, batch, channels, state, and tokens");
  for (Value pointer : getInputs().take_front(5))
    if (!isa<LLVM::LLVMPointerType>(pointer.getType()))
      return emitOpError("ReplaySSM buffers must be !llvm.ptr");
  for (Value dim : getInputs().drop_front(5))
    if (!dim.getType().isInteger(64))
      return emitOpError("ReplaySSM dimensions must be i64");
  auto storage = getOperation()->getAttrOfType<StringAttr>("storage");
  auto route = getOperation()->getAttrOfType<StringAttr>("route");
  auto deterministic = getOperation()->getAttrOfType<BoolAttr>("deterministic");
  if (!storage || storage.getValue() != "f32")
    return emitOpError("currently requires storage=\"f32\"");
  if (!route || route.getValue() != "state_and_output")
    return emitOpError("flush requires route=\"state_and_output\"");
  if (!deterministic || !deterministic.getValue())
    return emitOpError("checkpoint fold requires deterministic=true");
  return success();
}

LogicalResult MoEDispatchKernelOp::verify() {
  if (getInputs().size() != 6)
    return emitOpError("expects X, token indices, O, T, S, and H");
  for (Value pointer : getInputs().take_front(3))
    if (!isa<LLVM::LLVMPointerType>(pointer.getType()))
      return emitOpError("MoE buffers must be !llvm.ptr");
  for (Value dim : getInputs().drop_front(3))
    if (!dim.getType().isInteger(64))
      return emitOpError("MoE dimensions must be i64");
  auto storage = getOperation()->getAttrOfType<StringAttr>("storage");
  auto index = getOperation()->getAttrOfType<StringAttr>("index_storage");
  if (!storage || (storage.getValue() != "f16" && storage.getValue() != "bf16" &&
                   storage.getValue() != "f32") ||
      !index || index.getValue() != "i32")
    return emitOpError("requires f16/bf16/f32 storage and index_storage=\"i32\"");
  return success();
}

LogicalResult MoECombineKernelOp::verify() {
  if (getInputs().size() != 7)
    return emitOpError("expects partials, token indices, weights, O, T, S, and H");
  for (Value pointer : getInputs().take_front(4))
    if (!isa<LLVM::LLVMPointerType>(pointer.getType()))
      return emitOpError("MoE buffers must be !llvm.ptr");
  for (Value dim : getInputs().drop_front(4))
    if (!dim.getType().isInteger(64))
      return emitOpError("MoE dimensions must be i64");
  auto storage = getOperation()->getAttrOfType<StringAttr>("storage");
  auto index = getOperation()->getAttrOfType<StringAttr>("index_storage");
  auto deterministic = getOperation()->getAttrOfType<BoolAttr>("deterministic");
  if (!storage || (storage.getValue() != "f16" && storage.getValue() != "bf16" &&
                   storage.getValue() != "f32") ||
      !index || index.getValue() != "i32")
    return emitOpError("requires f16/bf16/f32 storage and index_storage=\"i32\"");
  if (!deterministic || !deterministic.getValue())
    return emitOpError("canonical combine requires deterministic=true");
  return success();
}

LogicalResult GroupedGemmKernelOp::verify() {
  if (getInputs().size() != 8)
    return emitOpError("expects X, expert weights, group offsets, O, T, K, N, and E");
  for (Value pointer : getInputs().take_front(4))
    if (!isa<LLVM::LLVMPointerType>(pointer.getType()))
      return emitOpError("grouped GEMM buffers must be !llvm.ptr");
  for (Value dim : getInputs().drop_front(4))
    if (!dim.getType().isInteger(64))
      return emitOpError("grouped GEMM dimensions must be i64");
  auto storage = getOperation()->getAttrOfType<StringAttr>("storage");
  auto accum = getOperation()->getAttrOfType<StringAttr>("accum");
  auto index = getOperation()->getAttrOfType<StringAttr>("index_storage");
  if (!storage || (storage.getValue() != "f16" && storage.getValue() != "bf16" &&
                   storage.getValue() != "f32") || !accum || accum.getValue() != "f32" ||
      !index || index.getValue() != "i32")
    return emitOpError("requires f16/bf16/f32 storage, f32 accumulation, and index_storage=\"i32\"");
  return success();
}

LogicalResult TMADescriptorOp::verify() {
  auto rows = getOperation()->getAttrOfType<IntegerAttr>("tile_rows");
  auto columns = getOperation()->getAttrOfType<IntegerAttr>("tile_cols");
  auto slot = getOperation()->getAttrOfType<IntegerAttr>("slot");
  auto expect = getOperation()->getAttrOfType<IntegerAttr>("expect_tx");
  if (!rows || !columns || rows.getInt() <= 0 || columns.getInt() <= 0)
    return emitOpError("requires positive tile_rows and tile_cols");
  if (slot && slot.getInt() < 0)
    return emitOpError("requires slot >= 0 when assigned");
  if (expect && expect.getInt() <= 0)
    return emitOpError("requires expect_tx > 0 when assigned");
  if (static_cast<bool>(slot) != static_cast<bool>(expect))
    return emitOpError("requires slot and expect_tx to be assigned together");
  if (auto shape =
          getOperation()->getAttrOfType<DenseI64ArrayAttr>("source_shape")) {
    if (shape.empty() ||
        llvm::any_of(shape.asArrayRef(),
                     [](int64_t extent) { return extent <= 0; }))
      return emitOpError("source_shape must contain positive logical extents");
  }
  return success();
}

LogicalResult TMACopyAsyncOp::verify() {
  auto slot = getOperation()->getAttrOfType<IntegerAttr>("mbarrier_slot");
  auto expect = getOperation()->getAttrOfType<IntegerAttr>("expect_tx");
  if (!slot || slot.getInt() < 0)
    return emitOpError("requires mbarrier_slot >= 0");
  if (expect && expect.getInt() <= 0)
    return emitOpError("requires expect_tx > 0 when assigned");
  if (getBarrier() && !expect)
    return emitOpError(
        "an SSA mbarrier binding requires explicit expect_tx bytes");
  // A copy must be observable by *some* completion mechanism (CAKE Phase 1
  // §5.2 row 2): an SSA mbarrier, a typed !tile.async_token result, or a
  // legacy grouping key (tile.barrier_id / tile.barrier / stage /
  // tile.depends_on). A copy with none of these gates on nothing — no wait
  // can ever retire it — and used to verify silently.
  if (!getBarrier()) {
    bool producesToken = llvm::any_of(getOutputs(), [](Value out) {
      return isa<AsyncTokenType>(out.getType());
    });
    bool hasLegacyKey = getOperation()->hasAttr("tile.barrier_id") ||
                        getOperation()->hasAttr("tile.barrier") ||
                        getOperation()->hasAttr("stage") ||
                        getOperation()->hasAttr("tile.depends_on") ||
                        // AsyncCopyLowering's declared "barrier retrofit
                        // pending" marker (stripped by NVTMADescriptorPass
                        // when the SSA barrier is bound).
                        getOperation()->hasAttr("tile.pending_mbarrier");
    if (!producesToken && !hasLegacyKey)
      return emitOpError(
          "TILE_TMA_COPY_GATES_ON_NOTHING: a tile.tma.copy_async needs an SSA "
          "mbarrier operand, a !tile.async_token result, or a legacy grouping "
          "key (tile.barrier_id / stage); with none, no wait can retire it");
  }
  int64_t coordinateCount = 0;
  if (auto count =
          getOperation()->getAttrOfType<IntegerAttr>("coordinate_count"))
    coordinateCount = count.getInt();
  if (coordinateCount < 0 ||
      static_cast<size_t>(coordinateCount) > getDependencies().size())
    return emitOpError(
        "coordinate_count must describe a prefix of dependency operands");
  for (Value coordinate :
       getDependencies().take_front(static_cast<size_t>(coordinateCount)))
    if (!coordinate.getType().isIndex())
      return emitOpError("TMA coordinates must be index-typed SSA operands");
  return success();
}

LogicalResult RoleOp::verify() {
  auto name = getOperation()->getAttrOfType<StringAttr>("name");
  auto kind = getOperation()->getAttrOfType<StringAttr>("kind");
  auto members = getOperation()->getAttrOfType<ArrayAttr>("members");
  if (!name || name.getValue().empty())
    return emitOpError("requires a non-empty logical role name");
  if (!kind || !llvm::is_contained({"producer", "consumer"}, kind.getValue()))
    return emitOpError("kind must be producer or consumer");
  if (!members || members.empty())
    return emitOpError("requires at least one logical member");
  llvm::DenseSet<StringRef> seen;
  for (Attribute memberAttr : members) {
    auto member = dyn_cast<StringAttr>(memberAttr);
    if (!member || member.getValue().empty())
      return emitOpError(
          "logical role members must be non-empty symbolic names");
    if (!seen.insert(member.getValue()).second)
      return emitOpError("logical role members must be unique");
  }
  return success();
}

LogicalResult MBarrierInitOp::verify() {
  auto slots = getOperation()->getAttrOfType<IntegerAttr>("slots");
  auto phaseBits = getOperation()->getAttrOfType<IntegerAttr>("phase_bits");
  if (!slots || slots.getInt() <= 0)
    return emitOpError("requires slots > 0");
  if (!phaseBits || phaseBits.getInt() < 1 || phaseBits.getInt() > 32)
    return emitOpError("requires phase_bits in [1, 32]");
  if (!getRoles().empty()) {
    bool producer = false, consumer = false, unresolved = false;
    llvm::DenseSet<StringRef> names;
    for (Value value : getRoles()) {
      auto role = value.getDefiningOp<RoleOp>();
      // Loop-carried roles are legal at the operation boundary. The shared
      // dataflow verifier follows their init/back-edge sources and rejects an
      // unresolved or mixed-kind root before lowering.
      if (!role) {
        unresolved = true;
        continue;
      }
      StringRef name = role.getName();
      if (!names.insert(name).second)
        return emitOpError("logical role names must be unique per barrier");
      producer |= role.getKind() == "producer";
      consumer |= role.getKind() == "consumer";
    }
    if (!unresolved && (!producer || !consumer))
      return emitOpError(
          "a role-bearing barrier requires both producer and consumer roles");
  }
  return success();
}

LogicalResult MBarrierArriveExpectTxOp::verify() {
  auto slot = getOperation()->getAttrOfType<IntegerAttr>("slot");
  auto bytes = getOperation()->getAttrOfType<IntegerAttr>("bytes");
  if (!slot || slot.getInt() < 0)
    return emitOpError("requires slot >= 0");
  if (!bytes || bytes.getInt() <= 0)
    return emitOpError("requires bytes > 0");
  return success();
}

LogicalResult MBarrierWaitOp::verify() {
  auto slot = getOperation()->getAttrOfType<IntegerAttr>("slot");
  if (!slot || slot.getInt() < 0)
    return emitOpError("requires slot >= 0");
  // Typed sync contract (CAKE Phase 1 §5.2 row 1). Every dependency must be a
  // typed synchronization value — an arbitrary SSA value (the measured hole:
  // an i32 constant, a loop induction variable) is not a completion edge and
  // conveys nothing a backend can honor.
  for (Value dep : getDependencies())
    if (!isa<AsyncTokenType, MBarrierTokenType>(dep.getType()))
      return emitOpError(
                 "TILE_WAIT_UNTYPED_DEPENDENCY: dependency operands must be "
                 "!tile.async_token or !tile.mbarrier_token; got ")
             << dep.getType();
  // A wait must gate on something: the mbarrier token from
  // tile.mbarrier.arrive_expect_tx, at least one async-token dependency, or
  // the explicit retire-all marker (the declared keyless-wait_async legacy
  // semantics, stamped by AsyncCopyLoweringPass and replaced with the real
  // token set by NVTMADescriptorPass). Fails closed — the bare form used to
  // verify and gated on nothing.
  if (!getToken() && getDependencies().empty() &&
      !getOperation()->hasAttr("tile.retire_all"))
    return emitOpError(
        "TILE_WAIT_GATES_ON_NOTHING: a wait needs a !tile.mbarrier_token, "
        "at least one !tile.async_token dependency, or the explicit "
        "tile.retire_all marker; with none it releases against no completion "
        "edge");
  return success();
}

LogicalResult MBarrierTryWaitOp::verify() {
  auto slot = getOperation()->getAttrOfType<IntegerAttr>("slot");
  if (!slot || slot.getInt() < 0)
    return emitOpError("requires slot >= 0");
  return success();
}

LogicalResult TMEMAllocOp::verify() {
  auto bytes = getOperation()->getAttrOfType<IntegerAttr>("bytes");
  auto alignment = getOperation()->getAttrOfType<IntegerAttr>("alignment");
  if (!bytes || bytes.getInt() <= 0)
    return emitOpError("requires bytes > 0");
  if (!alignment || alignment.getInt() <= 0 ||
      (alignment.getInt() & (alignment.getInt() - 1)) != 0)
    return emitOpError("requires power-of-two alignment");
  return success();
}

LogicalResult TMEMLoadOp::verify() {
  return success();
}

LogicalResult TMEMStoreOp::verify() {
  return success();
}

LogicalResult TCGen05MMAOp::verify() {
  auto group = getOperation()->getAttrOfType<IntegerAttr>("cta_group");
  auto descriptor =
      getOperation()->getAttrOfType<TileMmaDescAttr>("mma");
  if (!group || group.getInt() < 1 || group.getInt() > 4)
    return emitOpError("requires cta_group in [1, 4]");
  if (!descriptor || descriptor.getFamily() != "tcgen05")
    return emitOpError("requires a #tile.mma_desc with family=\"tcgen05\"");
  FragmentType lhs = typedFragment(getLhs());
  FragmentType rhs = typedFragment(getRhs());
  if (!lhs || !rhs)
    return emitOpError(
        "TILE_TCGEN05_FRAGMENT_CONTRACT: lhs/rhs require parameterized "
        "!tile.fragment values; the bare migration type is illegal");
  if (lhs.getRole() != "a" || rhs.getRole() != "b")
    return emitOpError(
        "TILE_TCGEN05_FRAGMENT_CONTRACT: lhs/rhs fragment roles must be a/b");
  if (lhs.getFamily() != "tcgen05" || rhs.getFamily() != "tcgen05")
    return emitOpError(
        "TILE_TCGEN05_FRAGMENT_CONTRACT: lhs/rhs family must be tcgen05");
  if (lhs.getM() != rhs.getM() || lhs.getN() != rhs.getN() ||
      lhs.getK() != rhs.getK() || lhs.getAcc() != rhs.getAcc())
    return emitOpError(
        "TILE_TCGEN05_FRAGMENT_CONTRACT: lhs/rhs shape and accumulator "
        "contracts must agree");
  if (failed(descriptorAgreesWithFragment(getOperation(), descriptor, lhs)) ||
      failed(descriptorAgreesWithFragment(getOperation(), descriptor, rhs)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// W1.1 step 1 — !tile.fragment assembly
//
// Two spellings, one type:
//
//   !tile.fragment                      the legacy all-unknown form
//   !tile.fragment<m = 16, n = 16, k = 16, elem = "bf16", acc = "f32",
//                  role = "a", layout = "row_major", family = "auto">
//
// The bare form has to keep parsing: every fixture in the tree and every Python
// text emitter writes it, and W1.1 migrates producers one at a time (steps
// 3-4). Making the parameters mandatory on day one would have broken all of
// them at once for no gain -- and the plan's own risk table names exactly that
// ordering error.
//===----------------------------------------------------------------------===//

Type FragmentType::parse(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();
  SMLoc loc = parser.getCurrentLocation();
  // No `<` means the bare form. `parseOptionalLess` fails when absent, which is
  // not an error here -- it is the legacy spelling.
  if (parser.parseOptionalLess())
    return FragmentType::getUnknown(ctx);

  int64_t m = 0, n = 0, k = 0;
  std::string elem, acc, role, layout, family;

  auto intField = [&](StringRef name, int64_t &out) -> ParseResult {
    if (parser.parseKeyword(name) || parser.parseEqual() ||
        parser.parseInteger(out) || parser.parseComma())
      return failure();
    return success();
  };
  auto strField = [&](StringRef name, std::string &out,
                      bool last = false) -> ParseResult {
    if (parser.parseKeyword(name) || parser.parseEqual() ||
        parser.parseString(&out))
      return failure();
    if (!last && parser.parseComma())
      return failure();
    return success();
  };

  if (intField("m", m) || intField("n", n) || intField("k", k) ||
      strField("elem", elem) || strField("acc", acc) ||
      strField("role", role) || strField("layout", layout) ||
      strField("family", family, /*last=*/true) || parser.parseGreater())
    return Type();

  // getChecked, not get: the parser is the ONLY entry point for textual IR, so
  // an unchecked construction here would let a malformed contract into the IR
  // and only fail later (or not at all). PR #502 review.
  return FragmentType::getChecked([&] { return parser.emitError(loc); }, ctx, m,
                                  n, k, StringRef(elem), StringRef(acc),
                                  StringRef(role), StringRef(layout),
                                  StringRef(family));
}

void FragmentType::print(AsmPrinter &printer) const {
  // Round-trip: the all-unknown form prints as the bare mnemonic, so a file
  // that came in legacy goes out legacy and existing fixtures keep matching.
  if (isUnknown())
    return;
  // Print each string through a StringAttr rather than concatenating the raw
  // bytes between quotes. `parseString` DECODES escapes, so a value containing
  // a quote, backslash or newline would come back in and go straight out
  // unescaped -- emitting IR the second parse cannot read. PR #502 review; the
  // round-trip fixture used only clean values and would not have caught it.
  // The verifier now constrains role/layout/family to closed sets, but `elem`
  // and `acc` are open dtype names, and correctness here should not depend on
  // a neighbouring check.
  auto quoted = [&](StringRef s) {
    printer << StringAttr::get(getContext(), s);
  };
  printer << "<m = " << getM() << ", n = " << getN() << ", k = " << getK()
          << ", elem = ";
  quoted(getElem());
  printer << ", acc = ";
  quoted(getAcc());
  printer << ", role = ";
  quoted(getRole());
  printer << ", layout = ";
  quoted(getLayout());
  printer << ", family = ";
  quoted(getFamily());
  printer << ">";
}

} // namespace tile
} // namespace tessera
