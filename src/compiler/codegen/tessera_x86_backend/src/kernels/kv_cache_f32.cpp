#include <algorithm>
#include <cstdint>
#include <cstring>

namespace {

bool valid_shape(int64_t max_seq, int64_t row_len) {
  return max_seq >= 0 && row_len > 0;
}

} // namespace

extern "C" int tessera_x86_kv_cache_append_f32(
    float *cache, int64_t max_seq, int64_t row_len, int64_t start,
    const float *rows, int64_t row_count) {
  if (!cache || !rows || !valid_shape(max_seq, row_len) || row_count < 0 ||
      start < 0 || start > max_seq || row_count > max_seq - start)
    return 1;
  if (row_count == 0)
    return 0;
  std::memcpy(cache + start * row_len, rows,
              static_cast<size_t>(row_count * row_len) * sizeof(float));
  return 0;
}

extern "C" int tessera_x86_kv_cache_read_f32(
    const float *cache, int64_t max_seq, int64_t row_len, int64_t start,
    int64_t end, float *output) {
  if (!cache || !output || !valid_shape(max_seq, row_len) || start < 0 ||
      end < start || end > max_seq)
    return 1;
  if (end == start)
    return 0;
  std::memcpy(output, cache + start * row_len,
              static_cast<size_t>((end - start) * row_len) * sizeof(float));
  return 0;
}

extern "C" int tessera_x86_kv_cache_prune_f32(
    float *cache, int64_t max_seq, int64_t row_len, int64_t current_seq,
    int64_t limit) {
  if (!cache || !valid_shape(max_seq, row_len) || current_seq < 0 ||
      current_seq > max_seq || limit < 0)
    return 1;
  if (limit >= current_seq)
    return 0;
  const int64_t keep = std::min(limit, current_seq);
  const int64_t first = current_seq - keep;
  if (keep > 0)
    std::memmove(cache, cache + first * row_len,
                 static_cast<size_t>(keep * row_len) * sizeof(float));
  std::fill(cache + keep * row_len, cache + current_seq * row_len, 0.0f);
  return 0;
}
