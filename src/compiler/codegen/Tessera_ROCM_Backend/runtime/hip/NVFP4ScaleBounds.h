// Host admission of encoded scale bytes; no tensor numerical outputs.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace tessera::rocm {
inline int nvfp4ScaleBounds(const uint8_t *scales, uint64_t bytes,
    uint64_t rows, uint64_t columns, const int64_t *offsets,
    uint64_t offsetCount, uint8_t *bounds, uint64_t boundsBytes) {
  if (!scales || !offsets || !bounds || !rows || !columns ||
      rows > uint64_t(INT64_MAX) || columns > UINT64_MAX / rows ||
      bytes != rows * columns || bytes > SIZE_MAX || offsetCount < 2 ||
      offsetCount - 1 > rows || offsetCount > SIZE_MAX / sizeof(int64_t) ||
      offsetCount - 1 > SIZE_MAX / 2 || boundsBytes != 2 * (offsetCount - 1))
    return 1;
  auto overlaps = [](const void *a, uint64_t n, const void *b, uint64_t m) {
    auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return n > UINTPTR_MAX - x || m > UINTPTR_MAX - y ||
           (x < y + m && y < x + n);
  };
  if (overlaps(scales, bytes, bounds, boundsBytes) ||
      overlaps(offsets, offsetCount * sizeof(int64_t), bounds, boundsBytes))
    return 1;
  if (offsets[0] != 0 || offsets[offsetCount - 1] != int64_t(rows)) return 1;
  for (uint64_t i = 1; i < offsetCount; ++i)
    if (offsets[i] <= offsets[i - 1] || uint64_t(offsets[i]) > rows) return 1;
  std::vector<uint8_t> result(size_t(boundsBytes), uint8_t(0));
  for (uint64_t p = 0; p + 1 < offsetCount; ++p) {
    uint8_t low = 127, high = 0;
    uint64_t first = uint64_t(offsets[p]) * columns;
    uint64_t last = uint64_t(offsets[p + 1]) * columns;
    for (uint64_t i = first; i < last; ++i) {
      uint8_t code = scales[i];
      if (code == 127 || code > 128) return 2;
      // Both signed zeros remain admissible and do not bound underflow.
      if (code && code < 127) {
        low = std::min(low, code); high = std::max(high, code);
      }
    }
    result[2 * p] = low; result[2 * p + 1] = high;
  }
  std::memcpy(bounds, result.data(), size_t(boundsBytes));
  return 0;
}
} // namespace tessera::rocm
