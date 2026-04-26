//===- ScheduleVerifiers.cpp (v1.2) ----------------------------------------===//
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinOps.h"
#include <unordered_set>
#include <set>
using namespace mlir;
namespace tessera { namespace schedule {

// Keep a map of roles seen per schedule region (simple scan)
static LogicalResult checkRoleUniqueness(Operation *container) {
  std::set<std::string> seen;
  for (Operation &op : container->getRegion(0).front()) {
    if (op.getName().getStringRef().endswith(".warp")) {
      auto role = op.getAttrOfType<StringAttr>("role");
      if (!role) continue;
      auto r = role.getValue().str();
      if (seen.count(r)) return op.emitOpError("duplicate warp role '") << r << "'";
      seen.insert(r);
    }
  }
  return success();
}

LogicalResult verifyWarp(Operation *op, StringRef role, int64_t count) {
  if (role.empty())
    return op->emitOpError("role must be a non-empty string");
  if (count <= 0)
    return op->emitOpError("count must be > 0");
  // Uniqueness checked at container-level by pass; ok here.
  return success();
}

LogicalResult verifyPipe(Operation *op, DictionaryAttr buffering) {
  if (!buffering) return success();
  std::unordered_set<std::string> allowed = {"K","V","S","O"};
  for (auto it : buffering) {
    auto key = it.getName().str();
    if (!allowed.count(key))
      return op->emitOpError("buffering key '") << key << "' not in {K,V,S,O}";
    if (auto intAttr = it.getValue().dyn_cast<IntegerAttr>()) {
      if (intAttr.getInt() <= 0)
        return op->emitOpError("buffering value for '") << key << "' must be > 0";
    } else {
      return op->emitOpError("buffering value for '") << key << "' must be integer";
    }
  }
  return success();
}

LogicalResult verifyPolicy(Operation *op, StringRef kind, int64_t maxCTA, StringRef queue) {
  if (kind != "persistent" && kind != "grid")
    return op->emitOpError("kind must be 'persistent' or 'grid'");
  if (kind == "persistent" && maxCTA != 1)
    return op->emitOpError("persistent policy requires max_cta_per_sm == 1");
  if (queue.empty())
    return op->emitOpError("tile_queue must be provided");
  return success();
}

}} // ns
