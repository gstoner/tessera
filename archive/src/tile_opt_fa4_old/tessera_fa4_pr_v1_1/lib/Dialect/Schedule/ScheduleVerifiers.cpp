//===- ScheduleVerifiers.cpp (v1.1) ----------------------------------------===//
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include <unordered_set>
using namespace mlir;
namespace tessera { namespace schedule {

// Warp: role non-empty, count>0
LogicalResult verifyWarp(Operation *op, StringRef role, int64_t count) {
  if (role.empty())
    return op->emitOpError("role must be a non-empty string");
  if (count <= 0)
    return op->emitOpError("count must be > 0");
  return success();
}

// Pipe: buffering keys must be subset of {K,V,S,O} with positive ints
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

// Policy: kind must be 'persistent' or 'grid'; if persistent then max_cta_per_sm==1
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
