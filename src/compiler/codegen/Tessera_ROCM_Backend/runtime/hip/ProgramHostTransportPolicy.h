// Host transport selection only; kernel arithmetic and image identity are unchanged.
#ifndef TESSERA_ROCM_PROGRAM_HOST_TRANSPORT_POLICY_H
#define TESSERA_ROCM_PROGRAM_HOST_TRANSPORT_POLICY_H
#include <cstdint>
#include <cstring>
#include <string_view>

namespace tessera::rocm {
inline bool usePinnedProgramHostTransport(std::string_view architecture,
                                         uint64_t inputBytes,
                                         uint64_t outputBytes,
                                         const char *overrideMode = nullptr) {
  if (overrideMode)
    return std::strcmp(overrideMode, "1") == 0;
  constexpr uint64_t maxSlabBytes = 8 * 1024 * 1024;
  // Cap both operands before adding: even malformed capacities cannot overflow.
  // Upload and completed-output readback both pay pageable transfer overhead.
  return architecture == "gfx1201" && inputBytes <= maxSlabBytes &&
         outputBytes <= maxSlabBytes &&
         inputBytes + outputBytes >= 256 * 1024;
}
} // namespace tessera::rocm
#endif
