#pragma once
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <limits>
#include <optional>
namespace tessera {
// Compact row-major source and destination; every axis and byte range is
// derivable without a runtime layout guess.
inline std::optional<int64_t> staticPermutationElements(
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> axes) {
  if (shape.empty() || shape.size() > 8 || axes.size() != shape.size())
    return std::nullopt;
  int64_t count = 1;
  llvm::SmallVector<bool> seen(shape.size(), false);
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] <= 0 || count > std::numeric_limits<int64_t>::max() / shape[i] ||
        axes[i] < 0 || uint64_t(axes[i]) >= shape.size() || seen[axes[i]])
      return std::nullopt;
    seen[axes[i]] = true;
    count *= shape[i];
  }
  if (count > std::numeric_limits<int64_t>::max() / 4) return std::nullopt;
  return count;
}
} // namespace tessera
