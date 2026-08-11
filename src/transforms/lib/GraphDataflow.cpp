#include "Tessera/Transforms/GraphDataflow.h"

#include "Tessera/Transforms/Passes.h"
#include "Tessera/Transforms/SemanticEffects.h"

#include "tessera/Dialect/Collective/IR/CollectiveDialect.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/LivenessAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>

using namespace mlir;

namespace tessera {
namespace {

using ProductLattice = dataflow::Lattice<GraphValueFact>;

static void canonicalizeRoots(SmallVectorImpl<Value> &roots) {
  llvm::sort(roots, [](Value lhs, Value rhs) {
    return lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer();
  });
  roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
}

static void setShapeFromType(GraphValueFact &fact, Type type) {
  fact.shape.clear();
  if (auto shaped = dyn_cast<ShapedType>(type); shaped && shaped.hasRank()) {
    fact.shapeTop = false;
    fact.shape.append(shaped.getShape().begin(), shaped.getShape().end());
    return;
  }
  if (!isa<ShapedType>(type)) {
    fact.shapeTop = false; // A scalar has the known rank-zero shape.
    return;
  }
  fact.shapeTop = true;
}

static GraphValueFact entryFact(Value value) {
  GraphValueFact fact;
  fact.initialized = true;
  setShapeFromType(fact, value.getType());
  // Tensor arguments have value semantics at Graph IR. Memref arguments may
  // alias each other or externally visible storage unless the ABI explicitly
  // says otherwise, so they enter at top.
  fact.aliasTop = isa<BaseMemRefType>(value.getType());
  if (!fact.aliasTop)
    fact.aliasRoots.push_back(value);
  return fact;
}

static bool isExplicitAliasProducer(Operation *op) {
  if (isa<CastOpInterface, ViewLikeOpInterface>(op))
    return true;
  if (auto alias = op->getAttrOfType<StringAttr>("tessera.aliasing"))
    return alias.getValue() != "none";
  return false;
}

static bool hasStochasticIdentity(Operation *op) {
  if (auto identity =
          op->getAttrOfType<StringAttr>("tessera.stochastic_identity"))
    return identity.getValue() != "none";
  return getRegisteredSemanticEffect(op) == SemanticEffectLevel::Random;
}

class GraphProductAnalysis final
    : public dataflow::SparseForwardDataFlowAnalysis<ProductLattice> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GraphProductAnalysis)
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult visitOperation(
      Operation *op, ArrayRef<const ProductLattice *> operands,
      ArrayRef<ProductLattice *> results) override {
    for (ProductLattice *result : results) {
      GraphValueFact fact;
      fact.initialized = true;
      setShapeFromType(fact, result->getAnchor().getType());

      if (op->getNumRegions() != 0) {
        fact.aliasTop = true;
      } else if (isExplicitAliasProducer(op)) {
        ArrayRef<const ProductLattice *> aliasOperands = operands;
        if (auto alias =
                op->getAttrOfType<StringAttr>("tessera.aliasing")) {
          StringRef value = alias.getValue();
          unsigned index = 0;
          if (!value.consume_front("operand_") || value.getAsInteger(10, index) ||
              index >= operands.size()) {
            fact.aliasTop = true;
            propagateIfChanged(result, result->join(fact));
            continue;
          }
          aliasOperands = operands.slice(index, 1);
        }
        fact.aliasTop = aliasOperands.empty();
        for (const ProductLattice *operand : aliasOperands) {
          const GraphValueFact &operandFact = operand->getValue();
          if (!operandFact.initialized || operandFact.aliasTop) {
            fact.aliasTop = true;
            fact.aliasRoots.clear();
            break;
          }
          fact.aliasRoots.append(operandFact.aliasRoots.begin(),
                                 operandFact.aliasRoots.end());
        }
        canonicalizeRoots(fact.aliasRoots);
      } else if (getRegisteredSemanticEffect(op) ==
                     SemanticEffectLevel::Pure &&
                 isMemoryEffectFree(op)) {
        // Pure, non-view SSA results own fresh storage/value identity.
        fact.aliasTop = false;
        fact.aliasRoots.push_back(result->getAnchor());
      } else {
        // An unregistered or effectful producer can return an alias of any
        // operand or hidden state. Do not guess.
        fact.aliasTop = true;
      }
      propagateIfChanged(result, result->join(fact));
    }
    return success();
  }

  void setToEntryState(ProductLattice *lattice) override {
    propagateIfChanged(lattice,
                       lattice->join(entryFact(lattice->getAnchor())));
  }
};

struct MemoryAccess {
  bool read = false;
  bool write = false;
  bool global = false;
  Value value;
  SideEffects::Resource *resource = nullptr;
};

struct MemorySummary {
  bool top = false;
  SmallVector<MemoryAccess, 4> accesses;
};

static MemorySummary summarizeMemory(Operation *op) {
  MemorySummary summary;
  std::optional<SmallVector<MemoryEffects::EffectInstance>> effects =
      getEffectsRecursively(op);
  if (!effects) {
    if (getRegisteredSemanticEffect(op) != SemanticEffectLevel::Pure)
      summary.top = true;
    return summary;
  }
  for (const MemoryEffects::EffectInstance &effect : *effects) {
    MemoryAccess access;
    access.read = isa<MemoryEffects::Read>(effect.getEffect());
    access.write = isa<MemoryEffects::Write, MemoryEffects::Allocate,
                       MemoryEffects::Free>(effect.getEffect());
    access.value = effect.getValue();
    access.global = !access.value;
    access.resource = effect.getResource();
    if (!access.read && !access.write) {
      summary.top = true;
      return summary;
    }
    summary.accesses.push_back(access);
  }
  return summary;
}

static bool directlyDepends(Operation *lhs, Operation *rhs) {
  for (Value operand : lhs->getOperands())
    if (operand.getDefiningOp() == rhs)
      return true;
  for (Value operand : rhs->getOperands())
    if (operand.getDefiningOp() == lhs)
      return true;
  return false;
}

} // namespace

GraphValueFact GraphValueFact::top() {
  GraphValueFact fact;
  fact.initialized = true;
  return fact;
}

GraphValueFact GraphValueFact::join(const GraphValueFact &lhs,
                                    const GraphValueFact &rhs) {
  if (!lhs.initialized)
    return rhs;
  if (!rhs.initialized)
    return lhs;
  GraphValueFact result;
  result.initialized = true;
  result.shapeTop = lhs.shapeTop || rhs.shapeTop || lhs.shape != rhs.shape;
  if (!result.shapeTop)
    result.shape = lhs.shape;
  result.aliasTop = lhs.aliasTop || rhs.aliasTop;
  if (!result.aliasTop) {
    result.aliasRoots = lhs.aliasRoots;
    result.aliasRoots.append(rhs.aliasRoots.begin(), rhs.aliasRoots.end());
    canonicalizeRoots(result.aliasRoots);
  }
  return result;
}

bool GraphValueFact::operator==(const GraphValueFact &rhs) const {
  return initialized == rhs.initialized && shapeTop == rhs.shapeTop &&
         shape == rhs.shape && aliasTop == rhs.aliasTop &&
         aliasRoots == rhs.aliasRoots;
}

void GraphValueFact::print(raw_ostream &os) const {
  if (!initialized) {
    os << "bottom";
    return;
  }
  os << "shape=";
  if (shapeTop)
    os << "top";
  else
    llvm::interleaveComma(shape, os);
  os << ";alias=";
  if (aliasTop)
    os << "top";
  else
    os << aliasRoots.size() << " root(s)";
}

struct GraphDataflowAnalysis::Impl {
  explicit Impl(func::FuncOp function) : function(function) {}

  func::FuncOp function;
  std::unique_ptr<DataFlowSolver> solver;
  std::unique_ptr<SymbolTableCollection> symbolTables;
  bool valid = false;
};

GraphDataflowAnalysis::GraphDataflowAnalysis(func::FuncOp function)
    : impl(std::make_unique<Impl>(function)) {}
GraphDataflowAnalysis::~GraphDataflowAnalysis() = default;
GraphDataflowAnalysis::GraphDataflowAnalysis(GraphDataflowAnalysis &&) noexcept =
    default;
GraphDataflowAnalysis &
GraphDataflowAnalysis::operator=(GraphDataflowAnalysis &&) noexcept = default;

LogicalResult GraphDataflowAnalysis::run() {
  impl->solver = std::make_unique<DataFlowSolver>();
  impl->symbolTables = std::make_unique<SymbolTableCollection>();
  impl->solver->load<dataflow::DeadCodeAnalysis>();
  impl->solver->load<GraphProductAnalysis>();
  impl->solver->load<dataflow::LivenessAnalysis>(*impl->symbolTables);
  impl->valid = succeeded(impl->solver->initializeAndRun(impl->function));
  return success(impl->valid);
}

LogicalResult GraphDataflowAnalysis::recompute() {
  invalidate();
  return run();
}

void GraphDataflowAnalysis::invalidate() {
  if (impl->solver)
    impl->solver->eraseAllStates();
  impl->valid = false;
}

bool GraphDataflowAnalysis::isValid() const { return impl->valid; }

GraphValueFact GraphDataflowAnalysis::getValueFact(Value value) const {
  if (!impl->valid || !impl->solver)
    return GraphValueFact::top();
  const ProductLattice *lattice =
      impl->solver->lookupState<ProductLattice>(value);
  if (!lattice || !lattice->getValue().initialized)
    return GraphValueFact::top();
  return lattice->getValue();
}

bool GraphDataflowAnalysis::isLive(Value value) const {
  if (!impl->valid || !impl->solver)
    return true;
  const dataflow::Liveness *liveness =
      impl->solver->lookupState<dataflow::Liveness>(value);
  return !liveness || liveness->isLive;
}

bool GraphDataflowAnalysis::mayAlias(Value lhs, Value rhs) const {
  if (!impl->valid)
    return true;
  if (lhs == rhs)
    return true;
  GraphValueFact lhsFact = getValueFact(lhs);
  GraphValueFact rhsFact = getValueFact(rhs);
  if (lhsFact.aliasTop || rhsFact.aliasTop)
    return true;
  for (Value lhsRoot : lhsFact.aliasRoots)
    if (llvm::is_contained(rhsFact.aliasRoots, lhsRoot))
      return true;
  return false;
}

bool GraphDataflowAnalysis::hasMemoryDependence(Operation *lhs,
                                                Operation *rhs) const {
  if (!impl->valid || !lhs || !rhs)
    return true;
  MemorySummary lhsSummary = summarizeMemory(lhs);
  MemorySummary rhsSummary = summarizeMemory(rhs);
  if (lhsSummary.top || rhsSummary.top)
    return true;
  for (const MemoryAccess &a : lhsSummary.accesses) {
    for (const MemoryAccess &b : rhsSummary.accesses) {
      if (!a.write && !b.write)
        continue;
      if (a.resource != b.resource)
        continue;
      if (a.global || b.global || mayAlias(a.value, b.value))
        return true;
    }
  }
  return false;
}

bool GraphDataflowAnalysis::canMoveAcross(Operation *moving,
                                          Operation *across) const {
  if (!impl->valid || !moving || !across ||
      moving->getBlock() != across->getBlock())
    return false;
  if (moving->getNumRegions() != 0 || across->getNumRegions() != 0)
    return false;
  if (directlyDepends(moving, across))
    return false;
  if (hasStochasticIdentity(across) || isExplicitAliasProducer(across))
    return false;
  // Await is a registered ordered-collective operation. Its memory effects
  // are value-scoped and could otherwise appear disjoint from another await,
  // but communication order is part of the observable program contract.
  if (isa<tessera::collective::AwaitOp>(across))
    return false;
  SemanticEffectLevel effect = getRegisteredSemanticEffect(across);
  if (effect == SemanticEffectLevel::Collective ||
      effect == SemanticEffectLevel::State ||
      effect == SemanticEffectLevel::IO || effect == SemanticEffectLevel::Top)
    return false;
  return !hasMemoryDependence(moving, across);
}

GraphDataflowAnalysis::ActiveOpSet
GraphDataflowAnalysis::computeActivity(ValueRange roots) const {
  ActiveOpSet active;
  if (!impl->valid)
    return active;
  SmallVector<Value, 16> worklist(roots.begin(), roots.end());
  llvm::SmallDenseSet<Value, 32> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    Operation *producer = value.getDefiningOp();
    if (!producer || !active.insert(producer).second)
      continue;
    if (producer->getName().getStringRef() == "tessera.stop_gradient")
      continue;
    // Activity through an unknown region is represented by the parent op; its
    // internal activity cannot be proven until region adjoints land.
    if (producer->getNumRegions() != 0)
      continue;
    worklist.append(producer->operand_begin(), producer->operand_end());
  }
  return active;
}

namespace {

class GraphDataflowAnnotationPass final
    : public PassWrapper<GraphDataflowAnnotationPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GraphDataflowAnnotationPass)

  StringRef getArgument() const final { return "tessera-graph-dataflow"; }
  StringRef getDescription() const final {
    return "Materialize fail-closed Graph shape, alias, liveness, and activity "
           "facts for inspection";
  }

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    GraphDataflowAnalysis analysis(function);
    if (failed(analysis.run())) {
      function.emitError()
          << "tessera-graph-dataflow: fixed-point analysis failed";
      return signalPassFailure();
    }

    SmallVector<Value, 4> roots;
    function.walk([&](func::ReturnOp returnOp) {
      roots.append(returnOp.getOperands().begin(), returnOp.getOperands().end());
    });
    GraphDataflowAnalysis::ActiveOpSet active =
        analysis.computeActivity(roots);
    Builder builder(&getContext());

    function.walk([&](Operation *op) {
      if (op->getNumResults() == 0)
        return;
      SmallVector<Attribute, 4> shapes;
      SmallVector<Attribute, 4> aliases;
      SmallVector<Attribute, 4> aliasesOperands;
      SmallVector<Attribute, 4> live;
      for (Value result : op->getResults()) {
        GraphValueFact fact = analysis.getValueFact(result);
        if (fact.shapeTop) {
          shapes.push_back(builder.getStringAttr("top"));
        } else {
          SmallVector<Attribute, 4> dimensions;
          for (int64_t dimension : fact.shape)
            dimensions.push_back(ShapedType::isDynamic(dimension)
                                     ? Attribute(builder.getStringAttr("?"))
                                     : Attribute(builder.getI64IntegerAttr(
                                           dimension)));
          shapes.push_back(builder.getArrayAttr(dimensions));
        }
        aliases.push_back(fact.aliasTop
                              ? Attribute(builder.getStringAttr("top"))
                              : Attribute(builder.getI64IntegerAttr(
                                    fact.aliasRoots.size())));
        if (fact.aliasTop) {
          aliasesOperands.push_back(builder.getStringAttr("top"));
        } else {
          SmallVector<Attribute, 4> operandIndices;
          for (auto [index, operand] : llvm::enumerate(op->getOperands()))
            if (analysis.mayAlias(result, operand))
              operandIndices.push_back(builder.getI64IntegerAttr(index));
          aliasesOperands.push_back(builder.getArrayAttr(operandIndices));
        }
        live.push_back(builder.getBoolAttr(analysis.isLive(result)));
      }
      op->setAttr("tessera.dataflow.shape", builder.getArrayAttr(shapes));
      op->setAttr("tessera.dataflow.alias_roots", builder.getArrayAttr(aliases));
      op->setAttr("tessera.dataflow.aliases_operands",
                  builder.getArrayAttr(aliasesOperands));
      op->setAttr("tessera.dataflow.live", builder.getArrayAttr(live));
      op->setAttr("tessera.dataflow.activity",
                  builder.getStringAttr(active.contains(op) ? "active"
                                                            : "inactive"));
    });
    function->setAttr("tessera.dataflow.schema_version",
                      builder.getI64IntegerAttr(1));
  }
};

} // namespace

std::unique_ptr<Pass> createGraphDataflowAnnotationPass() {
  return std::make_unique<GraphDataflowAnnotationPass>();
}

} // namespace tessera
