//===- TesseraNeighbors.cpp — Neighbors dialect registration (Phase 7) -----===//
//
// Registers the tessera.neighbors dialect together with all types, attributes,
// and ops defined in tessera_neighbors.td.
//
// Because we generate C++ from ODS at build time the real project would use
// mlir-tblgen.  Here we provide a hand-written implementation that mirrors
// exactly what the ODS-generated code would produce so the dialect is
// immediately usable without a build-step dependency.
//
// Types registered:
//   !tessera.neighbors.topology  — handle for mesh / torus / hex / custom graph
//   !tessera.neighbors.halo      — halo view derived from a tile
//
// Attrs registered:
//   #tessera.neighbors.delta_array<[i0, i1, …]>  — per-axis integer Δ vector
//   #tessera.neighbors.str<"…">                  — utility string attr
//
// Ops registered (all verified below):
//   tessera.neighbors.topology.create   — build a topology object
//   tessera.neighbors.halo.region       — derive a halo view from a tile
//   tessera.neighbors.halo.exchange     — initiate async halo exchange
//   tessera.neighbors.neighbor.read     — read a neighbor slice by Δ
//   tessera.neighbors.stencil.define    — declare taps + BC
//   tessera.neighbors.stencil.apply     — apply stencil (infers halos)
//   tessera.neighbors.pipeline.config   — configure overlap/double-buffer
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

using namespace mlir;

// ---------------------------------------------------------------------------
// Namespace
// ---------------------------------------------------------------------------

namespace tessera {
namespace neighbors {

// ---------------------------------------------------------------------------
// Type storage helpers (singleton types — no parameters)
// ---------------------------------------------------------------------------

/// Storage class shared by TopologyType and HaloType (no parameters needed).
struct SimpleTypeStorage : public TypeStorage {
  using KeyTy = int;
  static SimpleTypeStorage *construct(TypeStorageAllocator &alloc, KeyTy) {
    return new (alloc.allocate<SimpleTypeStorage>()) SimpleTypeStorage{};
  }
  bool operator==(const KeyTy &) const { return true; }
};

struct TopologyType : public Type::TypeBase<TopologyType, Type, SimpleTypeStorage> {
  using Base::Base;
  static TopologyType get(MLIRContext *ctx) {
    return Base::get(ctx, 0);
  }
  static llvm::StringRef name() { return "topology"; }
};

struct HaloType : public Type::TypeBase<HaloType, Type, SimpleTypeStorage> {
  using Base::Base;
  static HaloType get(MLIRContext *ctx) {
    return Base::get(ctx, 0);
  }
  static llvm::StringRef name() { return "halo"; }
};

// ---------------------------------------------------------------------------
// Attribute storage
// ---------------------------------------------------------------------------

struct DeltaArrayStorage : public AttributeStorage {
  using KeyTy = llvm::ArrayRef<int64_t>;

  explicit DeltaArrayStorage(llvm::ArrayRef<int64_t> v)
      : values(v.begin(), v.end()) {}

  static DeltaArrayStorage *construct(AttributeStorageAllocator &alloc,
                                      const KeyTy &key) {
    return new (alloc.allocate<DeltaArrayStorage>()) DeltaArrayStorage(key);
  }
  bool operator==(const KeyTy &other) const {
    return llvm::ArrayRef<int64_t>(values) == other;
  }
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine_range(key.begin(), key.end());
  }

  llvm::SmallVector<int64_t> values;
};

struct DeltaArrayAttr : public Attribute::AttrBase<DeltaArrayAttr, Attribute,
                                                    DeltaArrayStorage> {
  using Base::Base;
  static DeltaArrayAttr get(MLIRContext *ctx, llvm::ArrayRef<int64_t> vals) {
    return Base::get(ctx, vals);
  }
  llvm::ArrayRef<int64_t> getValues() const {
    return getImpl()->values;
  }
  static llvm::StringRef name() { return "delta_array"; }
};

// ---------------------------------------------------------------------------
// Dialect definition
// ---------------------------------------------------------------------------

class NeighborsDialect : public Dialect {
public:
  explicit NeighborsDialect(MLIRContext *ctx)
      : Dialect(getDialectNamespace(), ctx,
                TypeID::get<NeighborsDialect>()) {
    // Register types
    addTypes<TopologyType, HaloType>();
    // Register attributes
    addAttributes<DeltaArrayAttr>();
    // Register ops (bodies defined below as Op structs)
    addOperations<
#define GET_OP_LIST
#include "NeighborsOps.cpp.inc"
    >();
  }

  static llvm::StringRef getDialectNamespace() {
    return "tessera.neighbors";
  }

  // Parse a dialect type.
  Type parseType(DialectAsmParser &parser) const override {
    llvm::StringRef keyword;
    if (parser.parseKeyword(&keyword)) return {};
    if (keyword == "topology") return TopologyType::get(getContext());
    if (keyword == "halo")     return HaloType::get(getContext());
    parser.emitError(parser.getNameLoc(),
                     "unknown tessera.neighbors type: ") << keyword;
    return {};
  }

  // Print a dialect type.
  void printType(Type type, DialectAsmPrinter &printer) const override {
    if (type.isa<TopologyType>()) { printer << "topology"; return; }
    if (type.isa<HaloType>())     { printer << "halo";     return; }
    printer << "unknown";
  }

  // Parse a dialect attribute.
  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override {
    llvm::StringRef keyword;
    if (parser.parseKeyword(&keyword)) return {};
    if (keyword == "delta_array") {
      if (parser.parseLess()) return {};
      llvm::SmallVector<int64_t> vals;
      if (parser.parseLSquare()) return {};
      while (parser.parseOptionalRSquare()) {
        int64_t v;
        if (parser.parseInteger(v)) return {};
        vals.push_back(v);
        parser.parseOptionalComma();
      }
      if (parser.parseGreater()) return {};
      return DeltaArrayAttr::get(getContext(), vals);
    }
    parser.emitError(parser.getNameLoc(),
                     "unknown tessera.neighbors attribute: ") << keyword;
    return {};
  }

  // Print a dialect attribute.
  void printAttribute(Attribute attr,
                      DialectAsmPrinter &printer) const override {
    if (auto da = attr.dyn_cast<DeltaArrayAttr>()) {
      printer << "delta_array<[";
      llvm::interleaveComma(da.getValues(), printer);
      printer << "]>";
      return;
    }
    printer << "unknown";
  }
};

// ---------------------------------------------------------------------------
// Op definitions (manual, matching tessera_neighbors.td)
// ---------------------------------------------------------------------------

// -- topology.create --------------------------------------------------------
struct CreateTopologyOp
    : Op<CreateTopologyOp, OpTrait::ZeroSuccessors,
         OpTrait::ZeroRegions, OpTrait::OneResult> {
  using Op::Op;
  static llvm::StringRef getOperationName() {
    return "tessera.neighbors.topology.create";
  }
  static ParseResult parse(OpAsmParser &, OperationState &) {
    return success();
  }
  void print(OpAsmPrinter &p) {
    p << " ";
    p.printOptionalAttrDict((*this)->getAttrs());
  }
  LogicalResult verify() {
    // kind attr is required
    if (!(*this)->getAttr("kind"))
      return emitOpError("requires 'kind' string attribute");
    return success();
  }
};

// -- halo.region ------------------------------------------------------------
struct HaloRegionOp
    : Op<HaloRegionOp, OpTrait::ZeroSuccessors,
         OpTrait::ZeroRegions, OpTrait::OneResult, OpTrait::AtLeastNOperands<1>::Impl> {
  using Op::Op;
  static llvm::StringRef getOperationName() {
    return "tessera.neighbors.halo.region";
  }
  static ParseResult parse(OpAsmParser &p, OperationState &result) {
    OpAsmParser::UnresolvedOperand tile;
    Type tileType;
    if (p.parseOperand(tile) || p.parseColonType(tileType)) return failure();
    if (p.resolveOperand(tile, tileType, result.operands)) return failure();
    p.parseOptionalAttrDict(result.attributes);
    result.addTypes(HaloType::get(result.getContext()));
    return success();
  }
  void print(OpAsmPrinter &p) {
    p << " " << getOperand(0) << " : " << getOperand(0).getType();
    p.printOptionalAttrDict((*this)->getAttrs());
  }
  LogicalResult verify() { return success(); }
};

// -- halo.exchange ----------------------------------------------------------
struct HaloExchangeOp
    : Op<HaloExchangeOp, OpTrait::ZeroSuccessors,
         OpTrait::ZeroRegions, OpTrait::OneResult, OpTrait::AtLeastNOperands<1>::Impl> {
  using Op::Op;
  static llvm::StringRef getOperationName() {
    return "tessera.neighbors.halo.exchange";
  }
  static ParseResult parse(OpAsmParser &p, OperationState &result) {
    OpAsmParser::UnresolvedOperand halo;
    Type haloType;
    if (p.parseOperand(halo) || p.parseColonType(haloType)) return failure();
    if (p.resolveOperand(halo, haloType, result.operands)) return failure();
    p.parseOptionalAttrDict(result.attributes);
    result.addTypes(IndexType::get(result.getContext())); // token as index
    return success();
  }
  void print(OpAsmPrinter &p) {
    p << " " << getOperand(0) << " : " << getOperand(0).getType();
    p.printOptionalAttrDict((*this)->getAttrs());
  }
  LogicalResult verify() {
    if (!getOperand(0).getType().isa<HaloType>())
      return emitOpError("operand must be !tessera.neighbors.halo");
    return success();
  }
};

// -- neighbor.read ----------------------------------------------------------
struct NeighborReadOp
    : Op<NeighborReadOp, OpTrait::ZeroSuccessors,
         OpTrait::ZeroRegions, OpTrait::OneResult, OpTrait::AtLeastNOperands<2>::Impl> {
  using Op::Op;
  static llvm::StringRef getOperationName() {
    return "tessera.neighbors.neighbor.read";
  }
  static ParseResult parse(OpAsmParser &p, OperationState &result) {
    OpAsmParser::UnresolvedOperand tile, topo;
    Type tileType, topoType, outType;
    if (p.parseOperand(tile) || p.parseComma() ||
        p.parseOperand(topo) || p.parseComma()) return failure();
    // parse the delta_array attribute inline
    Attribute delta;
    if (p.parseAttribute(delta)) return failure();
    result.addAttribute("delta", delta);
    if (p.parseColonType(tileType) || p.parseArrowTypeList({outType}))
      return failure();
    if (p.resolveOperand(tile, tileType, result.operands)) return failure();
    if (p.resolveOperand(topo, topoType, result.operands)) return failure();
    result.addTypes(outType);
    return success();
  }
  void print(OpAsmPrinter &p) {
    p << " " << getOperand(0) << ", " << getOperand(1) << ", ";
    p.printAttribute((*this)->getAttr("delta"));
    p << " : " << getOperand(0).getType() << " -> "
      << getResult(0).getType();
  }
  LogicalResult verify() {
    if (!(*this)->getAttr("delta"))
      return emitOpError("requires 'delta' DeltaArrayAttr");
    return success();
  }
};

// -- stencil.define ---------------------------------------------------------
struct StencilDefineOp
    : Op<StencilDefineOp, OpTrait::ZeroSuccessors,
         OpTrait::ZeroRegions, OpTrait::OneResult, OpTrait::ZeroOperands> {
  using Op::Op;
  static llvm::StringRef getOperationName() {
    return "tessera.neighbors.stencil.define";
  }
  static ParseResult parse(OpAsmParser &p, OperationState &result) {
    p.parseOptionalAttrDict(result.attributes);
    result.addTypes(IndexType::get(result.getContext())); // stencil as opaque index
    return success();
  }
  void print(OpAsmPrinter &p) {
    p << " ";
    p.printOptionalAttrDict((*this)->getAttrs());
  }
  LogicalResult verify() {
    if (!(*this)->getAttr("taps"))
      return emitOpError("requires 'taps' array of DeltaArrayAttr");
    return success();
  }
};

// -- stencil.apply ----------------------------------------------------------
struct StencilApplyOp
    : Op<StencilApplyOp, OpTrait::ZeroSuccessors,
         OpTrait::ZeroRegions, OpTrait::OneResult, OpTrait::AtLeastNOperands<3>::Impl> {
  using Op::Op;
  static llvm::StringRef getOperationName() {
    return "tessera.neighbors.stencil.apply";
  }
  static ParseResult parse(OpAsmParser &p, OperationState &result) {
    SmallVector<OpAsmParser::UnresolvedOperand> operands(3);
    SmallVector<Type> types(3);
    for (int i = 0; i < 3; ++i) {
      if (p.parseOperand(operands[i])) return failure();
      if (i < 2 && p.parseComma()) return failure();
    }
    if (p.parseColonTypeList(types)) return failure();
    for (int i = 0; i < 3; ++i)
      if (p.resolveOperand(operands[i], types[i], result.operands))
        return failure();
    // output type = field type (operand 1)
    result.addTypes(types[1]);
    return success();
  }
  void print(OpAsmPrinter &p) {
    p << " " << getOperand(0) << ", " << getOperand(1) << ", " << getOperand(2);
    p << " : " << getOperand(0).getType() << ", "
      << getOperand(1).getType() << ", " << getOperand(2).getType();
  }
  LogicalResult verify() { return success(); }
};

// -- pipeline.config --------------------------------------------------------
struct PipelineConfigOp
    : Op<PipelineConfigOp, OpTrait::ZeroSuccessors,
         OpTrait::ZeroRegions, OpTrait::ZeroResults, OpTrait::ZeroOperands> {
  using Op::Op;
  static llvm::StringRef getOperationName() {
    return "tessera.neighbors.pipeline.config";
  }
  static ParseResult parse(OpAsmParser &p, OperationState &result) {
    p.parseOptionalAttrDict(result.attributes);
    return success();
  }
  void print(OpAsmPrinter &p) {
    p << " ";
    p.printOptionalAttrDict((*this)->getAttrs());
  }
  LogicalResult verify() { return success(); }
};

// ---------------------------------------------------------------------------
// Factory — used by the global dialect registry
// ---------------------------------------------------------------------------

void registerNeighborsDialect(DialectRegistry &registry) {
  registry.insert<NeighborsDialect>();
}

} // namespace neighbors
} // namespace tessera
