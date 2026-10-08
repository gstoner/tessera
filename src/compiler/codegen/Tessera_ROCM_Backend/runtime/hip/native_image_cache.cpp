// Native module/function ownership for checked compiler-generated images.
// The caller holds a lease until its launches have completed. Explicit clear
// precedes context destruction/device reset. No HIP calls run at process exit.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unistd.h>

namespace {
constexpr size_t maxModules = 16, maxPayloadBytes = 32 * 1024 * 1024;
const pid_t process = getpid();
using Identity = std::tuple<int, uintptr_t, std::string>;
using Key = std::pair<Identity, std::string>;
struct Entry {
  hipModule_t module{};
  Identity owner;
  std::map<std::string, hipFunction_t> functions;
  size_t leases = 0, bytes = 0;
  uint64_t touch = 0;
  bool cached = false;
  ~Entry() { if (module) hipModuleUnload(module); }
};
struct Lease { std::shared_ptr<Entry> entry; };
struct Cache {
  std::mutex mutex;
  std::map<Key, std::shared_ptr<Entry>> entries;
  uint64_t clock = 0, loads = 0, hits = 0, lookups = 0, unloads = 0;
};
// Intentionally process-lived: HIP may already be shut down during C++
// destruction. Explicit clear is the deterministic lifecycle boundary.
Cache &cache() { static Cache *value = new Cache; return *value; }
int identity(Identity &out) {
  if (getpid() != process) return 2; // inherited HIP contexts are not reusable
  int device = 0;
  hipCtx_t context{};
  hipDeviceProp_t props{};
  if (hipGetDevice(&device) != hipSuccess ||
      hipCtxGetCurrent(&context) != hipSuccess ||
      hipGetDeviceProperties(&props, device) != hipSuccess) return 2;
  out = {device, reinterpret_cast<uintptr_t>(context), props.gcnArchName};
  return 0;
}
int lookup(Cache &c, Entry &entry, const char *name, hipFunction_t &function) {
  auto found = entry.functions.find(name);
  if (found != entry.functions.end()) { function = found->second; return 0; }
  if (hipModuleGetFunction(&function, entry.module, name) != hipSuccess) return 4;
  entry.functions.emplace(name, function);
  ++c.lookups;
  return 0;
}
bool makeRoom(Cache &c, const Identity &owner, size_t newBytes) {
  if (newBytes > maxPayloadBytes) return false;
  for (;;) {
    size_t count = 0, bytes = 0;
    auto victim = c.entries.end();
    for (auto it = c.entries.begin(); it != c.entries.end(); ++it) {
      if (it->first.first != owner) continue;
      ++count; bytes += it->second->bytes;
      if (!it->second->leases &&
          (victim == c.entries.end() ||
           it->second->touch < victim->second->touch)) victim = it;
    }
    if (count < maxModules && bytes + newBytes <= maxPayloadBytes) return true;
    if (victim == c.entries.end()) return false;
    if (hipModuleUnload(victim->second->module) != hipSuccess) return false;
    ++c.unloads;
    victim->second->module = nullptr;
    c.entries.erase(victim);
  }
}
} // namespace

// Status: 1 malformed request, 2 wrong process/context, 3 module load,
// 4 missing function, 5 active lease, 6 module unload.
extern "C" int tessera_rocm_image_acquire(
    const void *payload, size_t bytes, const char *name, void **lease,
    void **module, void **function, int *hit) try {
  if (!payload || bytes < 4 || !name || !*name || !lease || !module ||
      !function || !hit || std::memcmp(payload, "\177ELF", 4)) return 1;
  *lease = nullptr; *module = nullptr; *function = nullptr; *hit = 0;
  Identity owner;
  if (identity(owner)) return 2;
  Cache &c = cache();
  std::lock_guard<std::mutex> guard(c.mutex);
  std::string binary(static_cast<const char *>(payload), bytes);
  Key key{owner, binary};
  auto found = c.entries.find(key);
  std::shared_ptr<Entry> entry;
  if (found != c.entries.end()) { entry = found->second; *hit = 1; }
  else {
    entry = std::make_shared<Entry>();
    if (hipModuleLoadData(&entry->module, payload) != hipSuccess) return 3;
    ++c.loads;
    // Module loading may initialize the primary context. Bind the resulting
    // identity, never a pre-initialization null handle.
    if (identity(owner)) return 2;
    entry->owner = owner; entry->bytes = bytes;
    key.first = owner;
    auto initialized = c.entries.find(key);
    if (initialized != c.entries.end()) {
      hipModuleUnload(entry->module); entry->module = nullptr; ++c.unloads;
      entry = initialized->second; *hit = 1;
    }
  }
  hipFunction_t fn{};
  int rc = lookup(c, *entry, name, fn);
  if (rc) {
    if (!*hit) {
      hipModuleUnload(entry->module); entry->module = nullptr; ++c.unloads;
    }
    return rc;
  }
  if (!*hit && makeRoom(c, owner, bytes)) {
    entry->cached = true;
    c.entries.emplace(std::move(key), entry);
  }
  auto held = std::make_unique<Lease>(Lease{entry});
  entry->touch = ++c.clock;
  ++entry->leases;
  if (*hit) ++c.hits;
  *lease = held.release();
  *module = reinterpret_cast<void *>(entry->module);
  *function = reinterpret_cast<void *>(fn);
  return 0;
} catch (...) { return 7; }
extern "C" int tessera_rocm_image_lookup(
    void *lease, const char *name, void **function) try {
  if (!lease || !name || !*name || !function) return 1;
  auto &entry = *static_cast<Lease *>(lease)->entry;
  Identity current;
  if (identity(current) || current != entry.owner) return 2;
  Cache &c = cache();
  std::lock_guard<std::mutex> guard(c.mutex);
  hipFunction_t fn{};
  int rc = lookup(c, entry, name, fn);
  if (!rc) *function = reinterpret_cast<void *>(fn);
  return rc;
} catch (...) { return 7; }
extern "C" int tessera_rocm_image_release(void *lease) try {
  if (!lease || getpid() != process) return 1;
  auto *held = static_cast<Lease *>(lease);
  auto entry = held->entry;
  Identity current;
  if (identity(current) || current != entry->owner) return 2;
  Cache &c = cache();
  std::lock_guard<std::mutex> guard(c.mutex);
  if (!entry->cached) {
    if (hipModuleUnload(entry->module) != hipSuccess) return 6;
    ++c.unloads;
    entry->module = nullptr;
  }
  --entry->leases;
  delete held;
  return 0;
} catch (...) { return 7; }
extern "C" int tessera_rocm_image_clear_current() try {
  Identity owner;
  if (identity(owner)) return 2;
  Cache &c = cache();
  std::lock_guard<std::mutex> guard(c.mutex);
  for (const auto &pair : c.entries)
    if (pair.first.first == owner && pair.second->leases) return 5;
  for (auto it = c.entries.begin(); it != c.entries.end();) {
    if (it->first.first != owner) { ++it; continue; }
    if (hipModuleUnload(it->second->module) != hipSuccess) return 6;
    ++c.unloads;
    it->second->module = nullptr;
    it = c.entries.erase(it);
  }
  return 0;
} catch (...) { return 7; }
extern "C" int tessera_rocm_image_stats(uint64_t *loads, uint64_t *hits,
    uint64_t *lookups, uint64_t *unloads) try {
  if (!loads || !hits || !lookups || !unloads || getpid() != process) return 1;
  Cache &c = cache();
  std::lock_guard<std::mutex> guard(c.mutex);
  *loads=c.loads; *hits=c.hits; *lookups=c.lookups; *unloads=c.unloads;
  return 0;
} catch (...) { return 7; }
