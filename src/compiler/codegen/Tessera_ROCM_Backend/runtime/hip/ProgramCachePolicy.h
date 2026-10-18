// Bounded native program retention; independent of kernel schedules.
#pragma once
#include <algorithm>
#include <cstdint>
#include <string_view>

namespace tessera::rocm {
inline uint64_t programIdleByteBudget(std::string_view architecture,
                                      uint64_t deviceBytes) {
  constexpr uint64_t legacy = uint64_t(128) << 20;
  constexpr uint64_t ceiling = uint64_t(512) << 20;
  if (architecture != "gfx1201" || !deviceBytes) return legacy;
  return std::min(ceiling, std::max(legacy, deviceBytes / 32));
}
inline bool programFrameFitsIdleBudget(uint64_t idleBytes, uint64_t frameBytes,
                                       uint64_t limit) {
  return idleBytes <= limit && frameBytes <= limit - idleBytes;
}
} // namespace tessera::rocm
