// Shared proof boundary for memref reuse assignment and arena consumption.
#ifndef TESSERA_TRANSFORMS_TILEMEMREFLIFETIME_H
#define TESSERA_TRANSFORMS_TILEMEMREFLIFETIME_H
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <limits>
#include <optional>
#include <utility>

namespace tessera::memory {
inline bool isMarker(mlir::Operation *op) {
  auto name = op->getName().getStringRef();
  return name == "tile.alloc_shared" || name == "tile.tmem.alloc";
}
inline int64_t staticBytes(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type || !type.hasStaticShape() || !type.getLayout().isIdentity() ||
      !type.getElementType().isIntOrFloat()) return -1;
  int64_t bytes = (type.getElementType().getIntOrFloatBitWidth() + 7) / 8;
  for (int64_t dim : type.getShape()) {
    if (dim && bytes > std::numeric_limits<int64_t>::max() / dim) return -1;
    bytes *= dim;
  }
  return bytes;
}
inline mlir::Value viewSource(mlir::Operation *op) {
  if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(op)) return cast.getSource();
  if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(op)) return view.getViewSource();
  return {};
}
// Prove a token's current generation through forwarding, not merely a shared
// defining-op origin. Identity loop carries include the zero-trip init path;
// changing backedges remain unknown until an iteration-sensitive proof exists.
inline bool forwardedToken(mlir::Value value,
                           llvm::function_ref<bool(mlir::Value)> leaf,
                           llvm::SmallPtrSetImpl<mlir::Value> &active) {
  if (!active.insert(value).second) return false;
  bool result = leaf(value);
  if (!result) {
    if (auto output = mlir::dyn_cast<mlir::OpResult>(value)) {
      auto *owner = output.getOwner();
      unsigned i = output.getResultNumber();
      if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(owner)) {
        auto yes = mlir::cast<mlir::scf::YieldOp>(branch.thenBlock()->getTerminator());
        auto no = mlir::cast<mlir::scf::YieldOp>(branch.elseBlock()->getTerminator());
        result = forwardedToken(yes.getOperand(i), leaf, active) &&
                 forwardedToken(no.getOperand(i), leaf, active);
      } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(owner)) {
        auto yield = mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
        if (yield.getOperand(i) == loop.getRegionIterArgs()[i]) {
          result = forwardedToken(loop.getInitArgs()[i], leaf, active);
        } else {
          // A replacement token must name a generation defined outside this
          // loop. A token freshly issued by the body cannot stand for every
          // iteration merely because its defining operation is the same.
          auto invariantLeaf = [&](mlir::Value candidate) {
            auto *definition = candidate.getDefiningOp();
            return definition && !loop->isProperAncestor(definition) && leaf(candidate);
          };
          bool replaced = forwardedToken(yield.getOperand(i), invariantLeaf, active);
          llvm::APInt lower, upper, step;
          bool nonempty = mlir::matchPattern(loop.getLowerBound(), mlir::m_ConstantInt(&lower)) &&
              mlir::matchPattern(loop.getUpperBound(), mlir::m_ConstantInt(&upper)) &&
              mlir::matchPattern(loop.getStep(), mlir::m_ConstantInt(&step)) &&
              step.isStrictlyPositive() &&
              (loop.getUnsignedCmp() ? lower.ult(upper) : lower.slt(upper));
          result = replaced && (nonempty || forwardedToken(loop.getInitArgs()[i], leaf, active));
        }
      }
    } else if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(arg.getOwner()->getParentOp())) {
        if (arg.getArgNumber() > 0) {
          unsigned i = arg.getArgNumber() - 1;
          auto yield = mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
          if (yield.getOperand(i) == arg)
            result = forwardedToken(loop.getInitArgs()[i], leaf, active);
        }
      }
    }
  }
  active.erase(value);
  return result;
}
inline bool committedCopy(mlir::Value token, mlir::Operation *copy) {
  llvm::SmallPtrSet<mlir::Value, 16> groups;
  return forwardedToken(token, [&](mlir::Value candidate) {
    auto group = candidate.getDefiningOp<mlir::nvgpu::DeviceAsyncCreateGroupOp>();
    if (!group || group->getBlock() != copy->getBlock()) return false;
    return llvm::any_of(group.getInputTokens(), [&](mlir::Value input) {
      llvm::SmallPtrSet<mlir::Value, 16> copies;
      return forwardedToken(input, [&](mlir::Value v) { return v == copy->getResult(0); }, copies);
    });
  }, groups);
}
inline bool completes(mlir::Operation *wait, mlir::Operation *copy) {
  if (auto nativeWait = mlir::dyn_cast<mlir::nvgpu::DeviceAsyncWaitOp>(wait)) {
    if (!mlir::isa<mlir::nvgpu::DeviceAsyncCopyOp>(copy)) return false;
    if (auto pending = nativeWait.getNumGroupsAttr(); pending && pending.getInt() != 0) return false;
    return committedCopy(nativeWait.getAsyncDependencies(), copy);
  }
  if (wait->getName().getStringRef() != "tile.wait_async") return false;
  if (wait->getNumOperands()) {
    for (mlir::Value result : copy->getResults())
      if (llvm::is_contained(wait->getOperands(), result)) return true;
    return false;
  }
  for (auto key : {"stage", "tile.barrier_id"})
    if (auto value = wait->getAttr(key))
      if (copy->getAttr(key) != value) return false;
  return true;
}
// Body-derived borrowing summaries. Recursion, external symbols, returned
// aliases, retained descriptors and async accesses are not borrowed calls.
// No user-supplied ownership/noalias attribute can override this proof.
using BorrowSummaries = llvm::DenseMap<std::pair<mlir::Operation *, unsigned>, bool>;
inline bool borrowedArgument(mlir::func::FuncOp fn, unsigned arg,
                             llvm::SmallPtrSetImpl<mlir::Operation *> &active,
                             BorrowSummaries &summaries) {
  if (!fn || fn.isExternal() || !fn.isPrivate() || arg >= fn.getNumArguments()) return false;
  auto key = std::make_pair(fn.getOperation(), arg);
  if (auto cached = summaries.find(key); cached != summaries.end()) return cached->second;
  if (!active.insert(fn).second) return false;
  bool safe = true;
  llvm::SmallPtrSet<mlir::Value, 16> seen;
  llvm::SmallVector<mlir::Value> work{fn.getArgument(arg)};
  while (!work.empty() && safe) {
    auto value = work.pop_back_val();
    if (!seen.insert(value).second) continue;
    for (auto &use : value.getUses()) {
      auto *user = use.getOwner();
      if (viewSource(user) == value) {
        for (auto result : user->getResults()) work.push_back(result);
      } else if (auto call = mlir::dyn_cast<mlir::func::CallOp>(user)) {
        auto callee = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(call, call.getCalleeAttr());
        safe &= borrowedArgument(callee, use.getOperandNumber(), active, summaries);
      } else if (!mlir::isa<mlir::memref::LoadOp, mlir::memref::StoreOp,
                            mlir::memref::DimOp>(user)) safe = false;
    }
  }
  active.erase(fn);
  summaries[key] = safe;
  return safe;
}
inline bool borrowedCall(mlir::Operation *op, mlir::Value value,
                         bool requireWorkgroupABI = false) {
  auto call = mlir::dyn_cast<mlir::func::CallOp>(op);
  if (!call) return false;
  auto fn = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(call, call.getCalleeAttr());
  bool found = false;
  BorrowSummaries summaries;
  for (auto [i, operand] : llvm::enumerate(call.getOperands())) {
    if (operand != value) continue;
    found = true;
    auto type = mlir::dyn_cast<mlir::MemRefType>(operand.getType());
    if (requireWorkgroupABI) {
      auto space = type ? mlir::dyn_cast_or_null<mlir::IntegerAttr>(type.getMemorySpace()) : mlir::IntegerAttr();
      if (!space || space.getInt() != 3) return false;
    }
    llvm::SmallPtrSet<mlir::Operation *, 8> active;
    if (!borrowedArgument(fn, i, active, summaries)) return false;
  }
  return found;
}

inline bool workgroupUniform(mlir::Value value) {
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    auto *block = arg.getOwner();
    if (auto kernel = mlir::dyn_cast<mlir::gpu::GPUFuncOp>(block->getParentOp())) {
      // Only launch arguments of a registered kernel have the per-workgroup
      // value contract. Helper/function arguments and attribution buffers do not.
      return kernel.isKernel() && block == &kernel.getBody().front() &&
             arg.getArgNumber() < kernel.getFunctionType().getNumInputs() &&
             mlir::isa<mlir::IntegerType, mlir::IndexType, mlir::FloatType>(arg.getType());
    }
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(block->getParentOp())) {
      // All participating threads visit the same induction values when the
      // three loop controls agree. Loop-carried values need a separate proof.
      return value == loop.getInductionVar() &&
             workgroupUniform(loop.getLowerBound()) &&
             workgroupUniform(loop.getUpperBound()) &&
             workgroupUniform(loop.getStep());
    }
    return false;
  }
  auto *op = value.getDefiningOp();
  if (!op || !op->isRegistered()) return false;
  auto name = op->getName().getStringRef();
  if (name == "arith.constant" || name == "gpu.block_id" ||
      name == "gpu.grid_dim" || name == "gpu.block_dim") return true;
  if (op->getName().getDialectNamespace() != "arith" || op->getNumRegions()) return false;
  return llvm::all_of(op->getOperands(), workgroupUniform);
}

// Completion must occur on every selected path, including an empty else path.
inline bool completesIn(mlir::Region &region, mlir::Operation *copy, bool sync);
inline bool completionOp(mlir::Operation *op, mlir::Operation *copy, bool sync) {
  auto name = op->getName().getStringRef();
  if (sync ? (name == "tile.cta_sync" || name == "tile.sbarrier" || name == "gpu.barrier") : completes(op, copy)) return true;
  if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
    if (sync && !workgroupUniform(branch.getCondition())) return false;
    llvm::APInt condition;
    if (mlir::matchPattern(branch.getCondition(), mlir::m_ConstantInt(&condition)))
      return completesIn(condition.isZero() ? branch.getElseRegion() : branch.getThenRegion(), copy, sync);
    return completesIn(branch.getThenRegion(), copy, sync) &&
           completesIn(branch.getElseRegion(), copy, sync);
  }
  return false;
}
inline bool completesIn(mlir::Region &region, mlir::Operation *copy, bool sync) {
  if (!llvm::hasSingleElement(region)) return false;
  return llvm::any_of(region.front(), [&](mlir::Operation &op) { return completionOp(&op, copy, sync); });
}

struct Interval {
  int64_t start, end;
  bool reusable;
};
// Local intervals compose with all-path completion and uniform structured
// branch exclusivity. Region-local intervals must finish before their exit,
// including a loop backedge. Escapes and unknown CFG/region forms prevent
// coalescing. Views retain the entire backing allocation conservatively.
class Lifetimes {
  llvm::DenseMap<mlir::Operation *, int64_t> index;
  mlir::Operation *function;
  int64_t end = 0;
public:
  explicit Lifetimes(mlir::Operation *fn) : function(fn) {
    fn->walk([&](mlir::Operation *op) { index[op] = end++; });
  }
  // A changing token is a recurrence, not an invariant SSA origin. Prove a
  // seed, per-iteration wait/publication/read/release/refill, and final drain
  // for one fixed allocation slot. Different slots are proved independently.
  std::optional<Interval> rotating(mlir::Operation *marker) const {
    using namespace mlir;
    Value buffer = marker->getOperand(0);
    scf::ForOp loop;
    nvgpu::DeviceAsyncCopyOp seed, refill;
    llvm::SmallVector<Operation *> reads;
    for (Operation *user : buffer.getUsers()) {
      if (user == marker) continue;
      if (auto copy = dyn_cast<nvgpu::DeviceAsyncCopyOp>(user)) {
        if (copy.getDst() != buffer) return std::nullopt;
        if (copy->getBlock() == marker->getBlock()) {
          if (seed) return std::nullopt;
          seed = copy;
        } else {
          auto owner = dyn_cast<scf::ForOp>(copy->getParentOp());
          if (!owner || (loop && owner != loop) || refill) return std::nullopt;
          loop = owner; refill = copy;
        }
      } else if (isa<memref::LoadOp>(user)) reads.push_back(user);
      else return std::nullopt; // aliases/escapes/writers need separate proofs
    }
    if (!loop || !seed || !refill || reads.empty() ||
        loop->getBlock() != marker->getBlock() ||
        marker->getParentOp() != function ||
        !marker->isBeforeInBlock(seed) || !seed->isBeforeInBlock(loop) ||
        !workgroupUniform(loop.getLowerBound()) ||
        !workgroupUniform(loop.getUpperBound()) || !workgroupUniform(loop.getStep())) return std::nullopt;
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    for (unsigned i = 0; i < loop.getNumRegionIterArgs(); ++i) {
      if (!committedCopy(loop.getInitArgs()[i], seed) ||
          !committedCopy(yield.getOperand(i), refill)) continue;
      Operation *wait = nullptr, *publish = nullptr, *release = nullptr;
      for (auto &op : *loop.getBody()) {
        if (auto w = dyn_cast<nvgpu::DeviceAsyncWaitOp>(op)) {
          if (w.getAsyncDependencies() == loop.getRegionIterArgs()[i] &&
              (!w.getNumGroupsAttr() || w.getNumGroupsAttr().getInt() == 0)) wait = &op;
        }
        if (wait && !publish && isa<gpu::BarrierOp>(op)) publish = &op;
      }
      if (!publish) continue;
      bool valid = true;
      for (auto *read : reads)
        valid &= read->getBlock() == loop.getBody() &&
                 publish->isBeforeInBlock(read) && read->isBeforeInBlock(refill);
      if (!valid) continue;
      for (auto *op = publish->getNextNode(); op && op != refill; op = op->getNextNode())
        if (isa<gpu::BarrierOp>(op) && llvm::all_of(reads, [&](Operation *r) { return r->isBeforeInBlock(op); })) release = op;
      if (!release) continue;
      // Loop results select the seed on zero trips and the last refill on
      // nonzero trips. Draining that exact result covers both paths.
      Operation *finalWait = nullptr;
      for (auto *op = loop->getNextNode(); op; op = op->getNextNode()) {
        if (auto w = dyn_cast<nvgpu::DeviceAsyncWaitOp>(op))
          if (w.getAsyncDependencies() == loop.getResult(i) &&
              (!w.getNumGroupsAttr() || w.getNumGroupsAttr().getInt() == 0)) finalWait = op;
        if (finalWait && isa<gpu::BarrierOp>(op))
          return Interval{index.lookup(marker), index.lookup(op), true};
      }
    }
    return std::nullopt;
  }
  bool uniformNesting(mlir::Operation *marker) const {
    using namespace mlir;
    for (auto *parent = marker->getParentOp(); parent != function; parent = parent->getParentOp()) {
      if (!parent) return false;
      if (auto loop = dyn_cast<scf::ForOp>(parent)) {
        if (!workgroupUniform(loop.getLowerBound()) || !workgroupUniform(loop.getUpperBound()) ||
            !workgroupUniform(loop.getStep())) return false;
      } else if (auto branch = dyn_cast<scf::IfOp>(parent)) {
        if (!workgroupUniform(branch.getCondition())) return false;
      } else return false;
    }
    return true;
  }
  // Bounded dynamic slot permutation. A bijective backedge keeps N
  // allocations distinct on every trip (including zero trips). All operations
  // finish and collectively release before the backedge; no async generation
  // may cross it. This is deliberately separate from changing-token recurrence.
  std::optional<Interval> permuted(mlir::Operation *marker) const {
    using namespace mlir;
    Value root = marker->getOperand(0);
    scf::ForOp loop;
    for (auto *user : root.getUsers())
      if (auto candidate = dyn_cast<scf::ForOp>(user)) {
        if (loop && candidate != loop) return std::nullopt;
        loop = candidate;
      }
    if (!loop || !uniformNesting(marker) ||
        loop->getBlock() != marker->getBlock() || !marker->isBeforeInBlock(loop) ||
        !workgroupUniform(loop.getLowerBound()) || !workgroupUniform(loop.getUpperBound()) ||
        !workgroupUniform(loop.getStep())) return std::nullopt;
    llvm::SmallVector<unsigned> slots;
    for (auto [i, value] : llvm::enumerate(loop.getInitArgs()))
      if (isa<MemRefType>(value.getType())) slots.push_back(i);
    if (slots.size() < 2) return std::nullopt;
    llvm::SmallPtrSet<Value, 16> roots, backedges;
    for (unsigned i : slots)
      if (!roots.insert(loop.getInitArgs()[i]).second) return std::nullopt;
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    for (unsigned i : slots) {
      Value carried = yield.getOperand(i);
      if (!backedges.insert(carried).second || !llvm::any_of(slots, [&](unsigned j) {
        return carried == loop.getRegionIterArgs()[j];
      })) return std::nullopt;
    }
    Operation *lastAccess = nullptr;
    for (unsigned i : slots) {
      Value init = loop.getInitArgs()[i], arg = loop.getRegionIterArgs()[i];
      if (!init.getDefiningOp() || !isa<memref::AllocaOp, memref::AllocOp>(init.getDefiningOp()) ||
          init.getType() != root.getType() || !loop.getResult(i).use_empty()) return std::nullopt;
      // Each physical root must have its own dominating allocation marker.
      Operation *ownerMarker = nullptr;
      for (auto *user : init.getUsers()) if (isMarker(user)) {
        if (ownerMarker) return std::nullopt;
        ownerMarker = user;
      }
      if (!ownerMarker || ownerMarker->getBlock() != marker->getBlock() ||
          !ownerMarker->isBeforeInBlock(loop) || (init.getDefiningOp() && viewSource(init.getDefiningOp()))) return std::nullopt;
      for (auto *user : init.getUsers()) {
        if (user == ownerMarker || user == loop) continue;
        if (!isa<memref::StoreOp>(user) || user->getBlock() != loop->getBlock() ||
            !ownerMarker->isBeforeInBlock(user) || !user->isBeforeInBlock(loop)) return std::nullopt;
        bool published = false;
        for (auto *next = user->getNextNode(); next != loop; next = next->getNextNode())
          published |= isa<gpu::BarrierOp>(next);
        if (!published) return std::nullopt;
      }
      for (auto *user : arg.getUsers()) {
        if (user == yield) continue;
        if (user->getBlock() != loop.getBody()) return std::nullopt;
        if (auto copy = dyn_cast<nvgpu::DeviceAsyncCopyOp>(user)) {
          if (copy.getDst() != arg) return std::nullopt;
          Operation *wait = nullptr, *publication = nullptr;
          for (auto *next = user->getNextNode(); next != yield; next = next->getNextNode()) {
            if (completes(next, user)) wait = next;
            if (wait && isa<gpu::BarrierOp>(next)) { publication = next; break; }
            // No other access to either slot before completion/publication.
            for (unsigned j : slots)
              if (llvm::is_contained(next->getOperands(), loop.getRegionIterArgs()[j])) return std::nullopt;
          }
          if (!publication) return std::nullopt;
          if (!lastAccess || lastAccess->isBeforeInBlock(publication)) lastAccess = publication;
        } else if (!isa<memref::LoadOp, memref::StoreOp>(user)) return std::nullopt;
        if (!lastAccess || lastAccess->isBeforeInBlock(user)) lastAccess = user;
      }
    }
    if (!lastAccess) return std::nullopt;
    for (auto *next = lastAccess->getNextNode(); next != yield; next = next->getNextNode())
      if (isa<gpu::BarrierOp>(next))
        return Interval{index.lookup(marker), index.lookup(loop.getBody()->getTerminator()), true};
    return std::nullopt;
  }
  // A pending copy token and its destination slot cross the same backedge.
  // Prove their coupled permutation inductively, instead of collapsing either
  // SSA value to a static origin. The seed also covers the zero-trip path.
  std::optional<Interval> pendingSwap(mlir::Operation *marker) const {
    using namespace mlir;
    Value root = marker->getOperand(0);
    scf::ForOp loop;
    for (auto *user : root.getUsers()) if (auto candidate = dyn_cast<scf::ForOp>(user)) {
      if (loop && loop != candidate) return std::nullopt;
      loop = candidate;
    }
    if (!loop || !uniformNesting(marker) || loop->getBlock() != marker->getBlock() ||
        !marker->isBeforeInBlock(loop) || !workgroupUniform(loop.getLowerBound()) ||
        !workgroupUniform(loop.getUpperBound()) || !workgroupUniform(loop.getStep())) return std::nullopt;
    llvm::SmallVector<unsigned> slots;
    for (auto [i, v] : llvm::enumerate(loop.getInitArgs())) if (isa<MemRefType>(v.getType())) slots.push_back(i);
    if (slots.size() < 2) return std::nullopt;
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    llvm::SmallPtrSet<void *, 8> roots, backedges;
    for (unsigned i : slots) {
      if (!roots.insert(loop.getInitArgs()[i].getAsOpaquePointer()).second ||
          !backedges.insert(yield.getOperand(i).getAsOpaquePointer()).second ||
          !llvm::any_of(slots, [&](unsigned j) {
            return yield.getOperand(i) == loop.getRegionIterArgs()[j];
          })) return std::nullopt;
    }
    nvgpu::DeviceAsyncCopyOp seed, refill;
    unsigned readIndex = 0, writeIndex = 0;
    llvm::SmallVector<Operation *> reads;
    for (unsigned i : slots) {
      Value init = loop.getInitArgs()[i], arg = loop.getRegionIterArgs()[i];
      if (!init.getDefiningOp() || !isa<memref::AllocOp, memref::AllocaOp>(init.getDefiningOp()) ||
          init.getType() != root.getType() || !loop.getResult(i).use_empty()) return std::nullopt;
      Operation *ownerMarker = nullptr;
      for (auto *user : init.getUsers()) if (isMarker(user)) {
        if (ownerMarker) return std::nullopt;
        ownerMarker = user;
      }
      if (!ownerMarker || ownerMarker->getBlock() != loop->getBlock() || !ownerMarker->isBeforeInBlock(loop)) return std::nullopt;
      for (auto *user : init.getUsers()) {
        if (user == ownerMarker || user == loop) continue;
        auto copy = dyn_cast<nvgpu::DeviceAsyncCopyOp>(user);
        if (!copy || seed || copy.getDst() != init || copy->getBlock() != loop->getBlock() ||
            !ownerMarker->isBeforeInBlock(copy) || !copy->isBeforeInBlock(loop)) return std::nullopt;
        seed = copy; readIndex = i;
      }
      for (auto *user : arg.getUsers()) {
        if (user == yield) continue;
        if (user->getBlock() != loop.getBody()) return std::nullopt;
        if (auto copy = dyn_cast<nvgpu::DeviceAsyncCopyOp>(user)) {
          if (refill || copy.getDst() != arg) return std::nullopt;
          refill = copy; writeIndex = i;
        } else if (isa<memref::LoadOp>(user)) reads.push_back(user);
        else return std::nullopt;
      }
    }
    if (!seed || !refill || reads.empty() || readIndex == writeIndex ||
        yield.getOperand(readIndex) != loop.getRegionIterArgs()[writeIndex]) return std::nullopt;
    for (auto *read : reads)
      if (cast<memref::LoadOp>(read).getMemRef() != loop.getRegionIterArgs()[readIndex]) return std::nullopt;
    for (unsigned i = 0; i < loop.getNumRegionIterArgs(); ++i) {
      if (!committedCopy(loop.getInitArgs()[i], seed) || !committedCopy(yield.getOperand(i), refill)) continue;
      Operation *wait = nullptr, *publish = nullptr, *release = nullptr;
      for (auto &op : *loop.getBody()) {
        if (auto w = dyn_cast<nvgpu::DeviceAsyncWaitOp>(op))
          if (w.getAsyncDependencies() == loop.getRegionIterArgs()[i] &&
              (!w.getNumGroupsAttr() || w.getNumGroupsAttr().getInt() == 0)) wait = &op;
        if (wait && !publish && isa<gpu::BarrierOp>(op)) publish = &op;
      }
      if (!publish || !publish->isBeforeInBlock(refill) || !llvm::all_of(reads, [&](Operation *r) {
        return publish->isBeforeInBlock(r);
      })) continue;
      for (auto *op = publish->getNextNode(); op && op != yield; op = op->getNextNode())
        if (isa<gpu::BarrierOp>(op) && llvm::all_of(reads, [&](Operation *r) { return r->isBeforeInBlock(op); })) release = op;
      if (!release) continue;
      Operation *finalWait = nullptr;
      for (auto *op = loop->getNextNode(); op; op = op->getNextNode()) {
        if (auto w = dyn_cast<nvgpu::DeviceAsyncWaitOp>(op))
          if (w.getAsyncDependencies() == loop.getResult(i) &&
              (!w.getNumGroupsAttr() || w.getNumGroupsAttr().getInt() == 0)) finalWait = op;
        if (finalWait && isa<gpu::BarrierOp>(op)) return Interval{index.lookup(marker), index.lookup(op), true};
      }
    }
    return std::nullopt;
  }
  Interval get(mlir::Operation *marker) const {
    if (auto pending = pendingSwap(marker)) return *pending;
    if (auto slots = permuted(marker)) return *slots;
    if (auto recurrence = rotating(marker)) return *recurrence;
    Interval live{index.lookup(marker), index.lookup(marker), true};
    auto *block = marker->getBlock();
    auto *owner = block->getParentOp();
    bool localRegion = mlir::isa<mlir::scf::IfOp, mlir::scf::ForOp>(owner);
    if (!localRegion && (block != &function->getRegion(0).front() ||
        !llvm::hasSingleElement(function->getRegion(0)))) live.reusable = false;
    // All enclosing control decisions must be uniform for collective storage.
    for (auto *parent = owner; parent != function; parent = parent->getParentOp()) {
      if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(parent))
        live.reusable &= workgroupUniform(branch.getCondition());
      else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent))
        live.reusable &= workgroupUniform(loop.getLowerBound()) &&
                         workgroupUniform(loop.getUpperBound()) && workgroupUniform(loop.getStep());
      else live.reusable = false;
    }
    llvm::SmallPtrSet<mlir::Value, 16> seen;
    llvm::SmallVector<mlir::Value> work{marker->getOperand(0)};
    // If the marker is itself a view, account for all sibling aliases too.
    for (mlir::Value root = work.front(); root.getDefiningOp();) {
      auto source = viewSource(root.getDefiningOp());
      if (!source) break;
      work.push_back(source);
      root = source;
    }
    while (!work.empty()) {
      auto value = work.pop_back_val();
      if (!seen.insert(value).second) continue;
      for (mlir::Operation *user : value.getUsers()) {
        if (user == marker) continue;
        live.end = std::max(live.end, index.lookup(user));
        if (user->getBlock() != block || index.lookup(user) < live.start)
          live.reusable = false;
        if (viewSource(user) == value) {
          for (auto result : user->getResults()) work.push_back(result);
          continue;
        }
        auto name = user->getName().getStringRef();
        if (name == "tile.async_copy" || name == "tile.tma.copy_async" || mlir::isa<mlir::nvgpu::DeviceAsyncCopyOp>(user)) {
          int64_t completion = end;
          for (mlir::Operation *next = user->getNextNode(); next; next = next->getNextNode())
            if (completionOp(next, user, false)) {
              if (mlir::isa<mlir::nvgpu::DeviceAsyncCopyOp>(user)) {
                // NVGPU waits are per-thread; shared storage also requires a
                // workgroup rendezvous before publication/reuse.
                for (auto *publish = next->getNextNode(); publish; publish = publish->getNextNode())
                  if (mlir::isa<mlir::gpu::BarrierOp>(publish)) { completion = index.lookup(publish); break; }
              } else completion = index.lookup(next);
              break;
            }
          live.end = std::max(live.end, completion);
        } else if (mlir::isa<mlir::memref::LoadOp, mlir::memref::StoreOp>(user) || borrowedCall(user, value)) {
          // Workgroup storage cannot be reassigned until all threads finish
          // synchronous accesses. Program order alone is not a rendezvous.
          int64_t completion = end;
          for (mlir::Operation *next = user->getNextNode(); next; next = next->getNextNode()) {
            if (completionOp(next, nullptr, true)) {
              completion = index.lookup(next); break;
            }
          }
          live.end = std::max(live.end, completion);
        } else if (!isMarker(user) && !mlir::isa<mlir::memref::DimOp>(user)) {
          live.reusable = false;
        }
      }
    }
    // Region-local storage must be released before leaving the region: in a
    // loop this also proves that the next iteration cannot overwrite it.
    if (localRegion && live.end >= index.lookup(block->getTerminator())) live.reusable = false;
    if (!live.reusable) { live.start = 0; live.end = end; }
    return live;
  }
  bool disjoint(mlir::Operation *a, mlir::Operation *b) const {
    if (a->getName() != b->getName() ||
        a->getOperand(0).getType() != b->getOperand(0).getType()) return false;
    auto x = get(a), y = get(b);
    if (!x.reusable || !y.reusable) return false;
    if (a->getBlock() == b->getBlock()) return x.end < y.start || y.end < x.start;
    // Opposite arms of one uniform branch never coexist. Both intervals have
    // independently proved that no access or DMA escapes their arm.
    for (auto *parent = a->getParentOp(); parent && parent != function; parent = parent->getParentOp()) {
      auto branch = mlir::dyn_cast<mlir::scf::IfOp>(parent);
      if (!branch || !workgroupUniform(branch.getCondition())) continue;
      auto in = [](mlir::Region &region, mlir::Operation *op) {
        return op->getParentRegion() == &region || region.isAncestor(op->getParentRegion());
      };
      if ((in(branch.getThenRegion(), a) && in(branch.getElseRegion(), b)) ||
          (in(branch.getElseRegion(), a) && in(branch.getThenRegion(), b))) return true;
    }
    return false;
  }
};
} // namespace tessera::memory
#endif
