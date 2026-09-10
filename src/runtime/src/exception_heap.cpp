#include "tessera/exception_heap.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

struct tsr_exception_heap {
  struct Node {
    uint32_t kind = 0, offset = 0, size = 0, generation = 0;
    uint64_t cause = 0, context = 0;
    bool live = false, root = false, marked = false;
  };
  std::mutex mutex;
  std::vector<Node> nodes;
  std::vector<unsigned char> payload;
  std::vector<uint32_t> pending;
  std::vector<std::pair<uint32_t,uint32_t>> ranges;
  uint32_t generation = 0;
  tsr_exception_heap(uint32_t n, uint32_t b) : nodes(n), payload(b) {
    pending.reserve(n); ranges.reserve(n);
  }
  Node *find(uint64_t h) {
    if (!h || uint32_t(h) >= nodes.size()) return nullptr;
    auto &n = nodes[uint32_t(h)];
    return n.live && n.generation == (h >> 32) ? &n : nullptr;
  }
  bool edge(uint64_t h) { return !h || find(h); }
};
extern "C" int tsr_exception_heap_create(uint32_t nodes, uint32_t bytes, tsr_exception_heap **out) {
  if (!out) return 1;
  *out = nullptr;
  if (!nodes || nodes > 65536 || !bytes || bytes > (1u << 28)) return 1;
  try { *out = new tsr_exception_heap(nodes, bytes); return 0; }
  catch (const std::bad_alloc &) { return 3; }
}
extern "C" void tsr_exception_heap_destroy(tsr_exception_heap *h) { delete h; }
extern "C" int tsr_exception_heap_alloc(tsr_exception_heap *h, uint32_t kind,
    const void *payload, uint32_t size, uint64_t cause, uint64_t context,
    int rooted, uint64_t *out) {
  if (!out) return 1;
  *out = 0;
  if (!h || (!payload && size) || (rooted != 0 && rooted != 1)) return 1;
  std::lock_guard<std::mutex> lock(h->mutex);
  if (!h->edge(cause) || !h->edge(context)) return 1;
  if (h->generation == std::numeric_limits<uint32_t>::max()) return 2;
  uint32_t slot = h->nodes.size();
  h->ranges.clear();
  for (uint32_t i = 0; i < h->nodes.size(); ++i) {
    auto &n = h->nodes[i];
    if (!n.live) { if (slot == h->nodes.size()) slot = i; }
    else if (n.size) h->ranges.emplace_back(n.offset, n.size);
  }
  if (slot == h->nodes.size() || size > h->payload.size()) return 2;
  std::sort(h->ranges.begin(), h->ranges.end());
  uint32_t offset = 0;
  for (auto [start, length] : h->ranges) {
    if (size <= start - offset) break;
    offset = start + length;
  }
  if (size > h->payload.size() - offset) return 2;
  auto &n = h->nodes[slot];
  if (size) std::memcpy(h->payload.data() + offset, payload, size);
  n = {kind, offset, size, ++h->generation, cause, context, true, rooted != 0, false};
  *out = (uint64_t(n.generation) << 32) | slot;
  return 0;
}
extern "C" int tsr_exception_heap_edges(tsr_exception_heap *h, uint64_t handle,
    uint64_t cause, uint64_t context) {
  if (!h) return 1;
  std::lock_guard<std::mutex> lock(h->mutex);
  auto n = h->find(handle);
  if (!n || !h->edge(cause) || !h->edge(context)) return 1;
  n->cause = cause; n->context = context;
  return 0;
}
extern "C" int tsr_exception_heap_root(tsr_exception_heap *h, uint64_t handle, int rooted) {
  if (!h || (rooted != 0 && rooted != 1)) return 1;
  std::lock_guard<std::mutex> lock(h->mutex);
  auto n = h->find(handle);
  if (!n) return 1;
  n->root = rooted != 0;
  return 0;
}
extern "C" int tsr_exception_heap_collect(tsr_exception_heap *h, uint32_t *collected) {
  if (!h || !collected) return 1;
  std::lock_guard<std::mutex> lock(h->mutex);
  h->pending.clear();
  for (uint32_t i = 0; i < h->nodes.size(); ++i) {
    auto &n = h->nodes[i];
    n.marked = n.live && n.root;
    if (n.marked) h->pending.push_back(i);
  }
  while (!h->pending.empty()) {
    auto &n = h->nodes[h->pending.back()]; h->pending.pop_back();
    for (auto edge : {n.cause, n.context}) {
      auto next = h->find(edge);
      if (next && !next->marked) { next->marked = true; h->pending.push_back(uint32_t(edge)); }
    }
  }
  *collected = 0;
  for (auto &n : h->nodes) if (n.live && !n.marked) { n.live = false; ++*collected; }
  return 0;
}
extern "C" int tsr_exception_heap_read(tsr_exception_heap *h, uint64_t handle,
    uint32_t *kind, uint64_t *cause, uint64_t *context, void *payload,
    uint32_t capacity, uint32_t *size) {
  if (!h || !kind || !cause || !context || !size) return 1;
  std::lock_guard<std::mutex> lock(h->mutex);
  auto n = h->find(handle);
  if (!n) return 1;
  *size = n->size;
  if (capacity < n->size || (!payload && n->size)) return 2;
  *kind = n->kind; *cause = n->cause; *context = n->context;
  if (n->size) std::memcpy(payload, h->payload.data() + n->offset, n->size);
  return 0;
}
