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

// ── ODS-WIRE-3: speculative-decode cursor ops over a cache HANDLE ─────────
//
// `tessera.cache.commit` / `tessera.cache.rollback` thread an opaque
// `!tessera.kv_cache` value-to-value (`cache -> updated`). The artifact-only
// `tessera_x86_kv_cache_op(kind)` bridge cannot lower them: it carries no
// handle and no count, and refuses ops whose result is used. This is the
// handle ABI `LowerKVCacheCursorToX86` (TileToX86Pass.cpp) lowers to: handle
// and count in, updated handle out. The update is linear and in place (the
// returned handle is the argument), so a chain commit -> rollback lowers to
// a chain of calls with no copies.
//
// Semantics match the Python references (`tessera/__init__.py` cache_commit /
// cache_rollback on a KVCacheHandle):
//   commit(n):   keep the first n tokens (`speculative.advance_kv`); zero
//                rows [n, current_seq); current_seq = n. n must be in
//                [0, current_seq].
//   rollback(n): drop the newest n tokens (`KVCacheHandle.trim`); n is
//                clamped to current_seq; zero the dropped rows.
// Rejected arguments return NULL and leave the handle untouched.

// Plain C layout; mirrored by `_X86KVCacheF32Handle` in tessera/runtime.py
// (drift-gated by tests/unit/test_x86_kv_cache_cursor.py).
struct tessera_x86_kv_cache_f32_handle {
  int64_t abi_version; // TESSERA_X86_KV_CACHE_F32_HANDLE_ABI
  float *keys;         // (max_seq, row_len) contiguous f32
  float *values;       // (max_seq, row_len) contiguous f32
  int64_t max_seq;
  int64_t row_len;
  int64_t current_seq;
};

namespace {

constexpr int64_t kKvCacheF32HandleAbi = 1;

bool valid_handle(const tessera_x86_kv_cache_f32_handle *handle) {
  return handle && handle->abi_version == kKvCacheF32HandleAbi &&
         handle->keys && handle->values &&
         valid_shape(handle->max_seq, handle->row_len) &&
         handle->current_seq >= 0 && handle->current_seq <= handle->max_seq;
}

tessera_x86_kv_cache_f32_handle *
truncate_to(tessera_x86_kv_cache_f32_handle *handle, int64_t new_seq) {
  const int64_t first = new_seq * handle->row_len;
  const int64_t last = handle->current_seq * handle->row_len;
  std::fill(handle->keys + first, handle->keys + last, 0.0f);
  std::fill(handle->values + first, handle->values + last, 0.0f);
  handle->current_seq = new_seq;
  return handle;
}

} // namespace

extern "C" int64_t tessera_x86_kv_cache_f32_handle_abi() {
  return kKvCacheF32HandleAbi;
}

extern "C" tessera_x86_kv_cache_f32_handle *
tessera_x86_kv_cache_commit_f32(tessera_x86_kv_cache_f32_handle *handle,
                                int64_t accepted_length) {
  if (!valid_handle(handle) || accepted_length < 0 ||
      accepted_length > handle->current_seq)
    return nullptr;
  return truncate_to(handle, accepted_length);
}

extern "C" tessera_x86_kv_cache_f32_handle *
tessera_x86_kv_cache_rollback_f32(tessera_x86_kv_cache_f32_handle *handle,
                                  int64_t num_rejected) {
  if (!valid_handle(handle) || num_rejected < 0)
    return nullptr;
  const int64_t dropped = std::min(num_rejected, handle->current_seq);
  return truncate_to(handle, handle->current_seq - dropped);
}
