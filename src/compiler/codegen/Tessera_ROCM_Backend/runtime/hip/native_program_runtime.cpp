// Program execution owns bytes and lifetime, never numerical kernel bodies.
#include "native_program_runtime.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>
#include <unistd.h>
extern "C" int tessera_rocm_image_acquire(const void *, size_t, const char *,
                                        void **, void **, void **, int *);
extern "C" int tessera_rocm_image_release(void *);
namespace {
const pid_t process = getpid();
struct Memref { void *allocated, *aligned; int64_t offset, elements, stride; };
struct Stage {
  void *lease = nullptr;
  hipFunction_t function{};
  std::array<Memref, 7> refs{};
  std::array<int64_t, 8> scalars{};
  std::array<void *, 43> argv{};
  std::array<unsigned, 6> geometry{};
};
struct Program {
  std::string cacheKey;
  uint64_t allocationBytes = 0;
  int device = 0;
  hipCtx_t context{};
  hipStream_t stream{};
  bool pinned = false;
  void *pinnedInputs = nullptr, *pinnedReadback = nullptr;
  uint64_t pinnedReadbackBytes = 0;
  std::vector<uint64_t> inputOffsets;
  uint32_t arguments = 0;
  std::vector<TesseraRocmProgramBuffer> contract;
  std::vector<void *> buffers;
  std::vector<std::vector<unsigned char>> snapshots;
  std::vector<Stage> stages;
  std::vector<unsigned char> readback;
  std::vector<hipEvent_t> events;
  std::vector<hipGraph_t> graphs;
  std::vector<hipGraphExec_t> graphExecutables;
  uint64_t generation = 0;
  bool poisoned = true, ready = false, output = false;
};
struct State {
  std::mutex mutex;
  std::map<uint64_t, std::unique_ptr<Program>> programs;
  std::vector<std::unique_ptr<Program>> idle;
  uint64_t idleBytes = 0, hits = 0, misses = 0;
  uint64_t next = 1;
};
// A failed completion retains allocations and host snapshots; implicit library
// teardown must not release storage still referenced by an asynchronous copy.
State &state() { static auto *s = new State; return *s; }
bool cacheEnabled() {
  const char *value = std::getenv("TESSERA_ROCM_PROGRAM_CACHE");
  return !value || std::strcmp(value, "0");
}
template<class T> void keyValue(std::string &key, const T &value) {
  key.append(reinterpret_cast<const char *>(&value), sizeof(value));
}
std::string programKey(const std::string &arch, uint32_t arguments,
                       uint32_t buffers, const TesseraRocmProgramBuffer *contract,
                       uint32_t steps, const TesseraRocmProgramStep *plan) {
  // Exact bytes, not caller pointers or a collision-prone hash. Input values
  // are uploaded on every acquire; only immutable image/ABI ownership is reused.
  std::string key = arch;
  keyValue(key, arguments); keyValue(key, buffers); keyValue(key, steps);
  for (uint32_t i = 0; i < buffers; ++i) {
    const auto &b = contract[i];
    keyValue(key,b.bytes); keyValue(key,b.elements);
    keyValue(key,b.first_write); keyValue(key,b.last_read);
    keyValue(key,b.ownership); keyValue(key,b.reserved);
  }
  for (uint32_t i = 0; i < steps; ++i) {
    const auto &p = plan[i];
    keyValue(key,p.image_bytes);
    key.append(static_cast<const char *>(p.image), p.image_bytes);
    size_t length = std::strlen(p.entry); keyValue(key,length); key.append(p.entry,length);
    keyValue(key,p.input_count); keyValue(key,p.scalar_count); keyValue(key,p.output);
    for (unsigned j=0;j<p.input_count;++j) keyValue(key,p.inputs[j]);
    for (unsigned j=0;j<p.scalar_count;++j) keyValue(key,p.scalars[j]);
    for (unsigned j=0;j<6;++j) keyValue(key,p.geometry[j]);
  }
  return key;
}
bool identity(const Program &p) {
  int d = 0; hipCtx_t context{};
  return getpid() == process && hipGetDevice(&d) == hipSuccess &&
         hipCtxGetCurrent(&context) == hipSuccess &&
         d == p.device && context == p.context;
}
int clean(Program &p) {
  if (p.stream && hipStreamSynchronize(p.stream) != hipSuccess) return 7;
  while (!p.graphExecutables.empty()) {
    if (hipGraphExecDestroy(p.graphExecutables.back()) != hipSuccess) return 9;
    p.graphExecutables.pop_back();
  }
  while (!p.graphs.empty()) {
    if (hipGraphDestroy(p.graphs.back()) != hipSuccess) return 9;
    p.graphs.pop_back();
  }
  while (!p.events.empty()) {
    if (hipEventDestroy(p.events.back()) != hipSuccess) return 9;
    p.events.pop_back();
  }
  if (p.pinnedInputs && hipHostFree(p.pinnedInputs) != hipSuccess) return 9;
  p.pinnedInputs = nullptr;
  if (p.pinnedReadback && hipHostFree(p.pinnedReadback) != hipSuccess) return 9;
  p.pinnedReadback = nullptr;
  for (auto &buffer : p.buffers) {
    if (buffer && hipFree(buffer) != hipSuccess) return 9;
    buffer = nullptr;
  }
  for (auto &stage : p.stages) {
    if (stage.lease && tessera_rocm_image_release(stage.lease)) return 8;
    stage.lease = nullptr;
  }
  if (p.stream && hipStreamDestroy(p.stream) != hipSuccess) return 9;
  p.stream = nullptr;
  return 0;
}
bool inputContract(const Program &p, const void *const *inputs,
                   const uint64_t *bytes) {
  if (!inputs || !bytes) return false;
  for (uint32_t i = 0; i < p.arguments; ++i)
    if (!inputs[i] || bytes[i] != p.contract[i].bytes) return false;
  return true;
}
int upload(Program &p, const void *const *inputs, const uint64_t *bytes) {
  if (!inputContract(p, inputs, bytes)) return 1;
  std::vector<std::vector<unsigned char>> next;
  if (!p.pinned) {
    next.resize(p.arguments);
    for (uint32_t i = 0; i < p.arguments; ++i) {
      auto *begin = static_cast<const unsigned char *>(inputs[i]);
      next[i].assign(begin, begin + bytes[i]);
    }
  }
  // Reused pinned staging may still be referenced by prior work; completion
  // precedes overwriting it, and failed completion retains the allocation.
  if (hipStreamSynchronize(p.stream) != hipSuccess) { p.poisoned = true; return 7; }
  p.ready = false; p.output = false;
  if (p.pinned) {
    for (uint32_t i = 0; i < p.arguments; ++i)
      std::memcpy(static_cast<unsigned char *>(p.pinnedInputs) + p.inputOffsets[i],
                  inputs[i], bytes[i]);
  } else {
    p.snapshots = std::move(next);
  }
  for (uint32_t i = 0; i < p.arguments; ++i) {
    const void *source = p.pinned
        ? static_cast<unsigned char *>(p.pinnedInputs) + p.inputOffsets[i]
        : p.snapshots[i].data();
    if (hipMemcpyAsync(p.buffers[i], source, p.contract[i].bytes,
                       hipMemcpyHostToDevice, p.stream) != hipSuccess) {
      p.poisoned = true; return 5;
    }
  }
  if (hipStreamSynchronize(p.stream) != hipSuccess) { p.poisoned = true; return 7; }
  p.ready = true;
  return 0;
}
} // namespace

extern "C" int tessera_rocm_program_pack_host_view(
    const void *source, uint64_t sourceSpan, uint32_t rank,
    const uint64_t *shape, const uint64_t *strides, uint32_t itemBytes,
    void *destination, uint64_t destinationBytes) {
  constexpr uint64_t limit = INT64_MAX;
  if (!source || !destination || !shape || !strides || rank < 2 || rank > 32 ||
      !itemBytes || itemBytes > 8 || sourceSpan > limit ||
      destinationBytes > limit) return 1;
  uint64_t count = 1, span = itemBytes;
  for (uint32_t axis = 0; axis < rank; ++axis) {
    if (!shape[axis] || shape[axis] > limit || !strides[axis] ||
        strides[axis] % itemBytes || strides[axis] > limit ||
        count > limit / shape[axis]) return 1;
    count *= shape[axis];
    if (shape[axis] - 1 > (limit - span) / strides[axis]) return 1;
    span += (shape[axis] - 1) * strides[axis];
  }
  if (count > limit / itemBytes || count * itemBytes != destinationBytes ||
      span > sourceSpan) return 1;
  uintptr_t src = reinterpret_cast<uintptr_t>(source);
  uintptr_t dst = reinterpret_cast<uintptr_t>(destination);
  if (src > UINTPTR_MAX - sourceSpan || dst > UINTPTR_MAX - destinationBytes ||
      (src < dst + destinationBytes && dst < src + sourceSpan)) return 1;
  const auto *input = static_cast<const unsigned char *>(source);
  auto *output = static_cast<unsigned char *>(destination);
  // Collapse an actually contiguous suffix; singleton axes contribute no
  // offset. The validated total byte count bounds every block multiplication.
  uint32_t prefix = rank;
  uint64_t blockBytes = itemBytes;
  while (prefix && (shape[prefix-1] == 1 || strides[prefix-1] == blockBytes)) {
    blockBytes *= shape[--prefix];
  }
  if (!prefix) {
    std::memcpy(output, input, destinationBytes);
    return 0;
  }
  const uint64_t columns = shape[prefix-1];
  const uint64_t rows = destinationBytes / blockBytes / columns;
  for (uint64_t row = 0; row < rows; ++row) {
    uint64_t remaining = row, offset = 0;
    for (uint32_t axis = prefix-1; axis-- > 0;) {
      offset += (remaining % shape[axis]) * strides[axis];
      remaining /= shape[axis];
    }
    auto *outRow = output + row * columns * blockBytes;
    const auto *inRow = input + offset;
    // Fixed-width memcpy keeps unaligned byte inputs legal while avoiding a
    // dynamic libc call for each interleaved scalar. Only storage is moved.
    auto fixed = [&](auto width) {
      constexpr size_t bytes = decltype(width)::value;
      for (uint64_t column = 0; column < columns; ++column)
        std::memcpy(outRow + column * bytes,
                    inRow + column * strides[prefix-1], bytes);
    };
    switch (blockBytes) {
    case 1: fixed(std::integral_constant<size_t, 1>{}); break;
    case 2: fixed(std::integral_constant<size_t, 2>{}); break;
    case 4: fixed(std::integral_constant<size_t, 4>{}); break;
    case 8: fixed(std::integral_constant<size_t, 8>{}); break;
    default:
      for (uint64_t column = 0; column < columns; ++column)
        std::memcpy(outRow + column * blockBytes,
                    inRow + column * strides[prefix-1], blockBytes);
    }
  }
  return 0;
}

extern "C" int tessera_rocm_program_prepare(
    const char *architecture, uint32_t arguments, uint32_t buffers,
    const TesseraRocmProgramBuffer *contract, uint32_t steps,
    const TesseraRocmProgramStep *plan, const void *const *inputs,
    const uint64_t *inputBytes, uint64_t *handle) try {
  if (handle) *handle = 0;
  if (!handle || !architecture || !contract || !plan || !inputs || !inputBytes ||
      !arguments || arguments > 128 || !steps || steps > 128 ||
      buffers != arguments + steps || buffers > 256) return 1;
  if (getpid() != process) return 2;
  std::string arch(architecture);
  if (arch != "gfx1201" && arch != "gfx1151") return 1;
  // Derive lifetimes from the actual SSA edge plan; caller claims must agree.
  std::vector<int64_t> reads(buffers, -1);
  uint64_t total = 0;
  unsigned returned = 0;
  for (uint32_t i = 0; i < buffers; ++i) {
    const auto &b = contract[i];
    if (!b.bytes || !b.elements || b.bytes % b.elements ||
        b.bytes / b.elements > 8 || b.elements > uint64_t(INT64_MAX) ||
        b.reserved || b.ownership > 2 ||
        (i < arguments ? b.ownership != 0 || b.first_write != -1 :
         b.ownership == 0 || b.first_write != int64_t(i - arguments)) ||
        b.bytes > (uint64_t(1) << 31) - total) return 1;
    total += b.bytes;
    if (i < arguments && (!inputs[i] || inputBytes[i] != b.bytes)) return 1;
    if (b.ownership == 2) ++returned;
    if (i >= arguments) reads[i] = i - arguments;
  }
  if (!returned) return 1;
  for (uint32_t i = 0; i < steps; ++i) {
    const auto &s = plan[i];
    if (!s.image || s.image_bytes < 4 || s.image_bytes > (uint64_t(1) << 28) ||
        std::memcmp(s.image, "\177ELF", 4) || !s.entry || !*s.entry ||
        !s.input_count || s.input_count > 6 || s.scalar_count > 8 ||
        s.output != arguments + i) return 1;
    for (uint32_t j = 0; j < s.input_count; ++j) {
      if (s.inputs[j] >= s.output) return 1;
      reads[s.inputs[j]] = i;
    }
    uint64_t block = 1;
    for (uint32_t j = 0; j < 6; ++j) {
      if (!s.geometry[j] || s.geometry[j] > 2147483647U) return 1;
      if (j >= 3) {
        if (s.geometry[j] > 1024 || block > 1024 / s.geometry[j]) return 1;
        block *= s.geometry[j];
      }
    }
  }
  for (uint32_t i = 0; i < buffers; ++i) {
    if (contract[i].ownership == 2) reads[i] = steps;
    if (contract[i].last_read != reads[i]) return 1;
  }
  auto p = std::make_unique<Program>();
  p->arguments = arguments;
  p->contract.assign(contract, contract + buffers);
  p->buffers.resize(buffers, nullptr);
  p->stages.resize(steps);
  p->events.reserve(2);
  if (hipInit(0) != hipSuccess || hipGetDevice(&p->device) != hipSuccess ||
      hipCtxGetCurrent(&p->context) != hipSuccess) return 2;
  hipDeviceProp_t properties{};
  if (hipGetDeviceProperties(&properties, p->device) != hipSuccess) return 2;
  std::string liveArch(properties.gcnArchName);
  liveArch = liveArch.substr(0, liveArch.find(':'));
  if (liveArch != arch) return 2;
  auto &pool = state();
  std::lock_guard<std::mutex> guard(pool.mutex);
  if (!pool.next) return 12;
  uint64_t inputTotal = 0, outputMaximum = 0;
  for (uint32_t i = 0; i < arguments; ++i) {
    p->inputOffsets.push_back(inputTotal);
    inputTotal += contract[i].bytes;
  }
  for (uint32_t i = arguments; i < buffers; ++i)
    if (contract[i].ownership == 2)
      outputMaximum = std::max(outputMaximum, contract[i].bytes);
  // Owning gfx1201 transfer measurements favor pinned staging for moderate
  // uploads. Bound default locked memory and retain explicit mode overrides.
  const char *pinnedMode = std::getenv("TESSERA_ROCM_PROGRAM_PINNED");
  const bool automaticPinned = arch == "gfx1201" &&
      inputTotal >= 256 * 1024 && inputTotal <= 8 * 1024 * 1024 &&
      outputMaximum <= 8 * 1024 * 1024;
  p->pinned = pinnedMode ? !std::strcmp(pinnedMode, "1") : automaticPinned;
  p->cacheKey = programKey(arch, arguments, buffers, contract, steps, plan);
  keyValue(p->cacheKey, p->pinned);
  p->pinnedReadbackBytes = outputMaximum;
  p->allocationBytes = total + p->cacheKey.size() +
      (p->pinned ? inputTotal + outputMaximum : 0);
  if (cacheEnabled()) {
    for (auto it = pool.idle.begin(); it != pool.idle.end(); ++it) {
      if ((*it)->poisoned || (*it)->cacheKey != p->cacheKey || !identity(**it)) continue;
      auto reused = std::move(*it);
      pool.idleBytes -= reused->allocationBytes;
      pool.idle.erase(it);
      uint64_t id = pool.next++;
      pool.programs.emplace(id, std::move(reused));
      *handle = id;
      ++pool.hits;
      auto &owned = *pool.programs.at(id);
      // An upload failure quarantines this active owner until explicit close.
      return upload(owned, inputs, inputBytes);
    }
  }
  ++pool.misses;
  uint64_t id = pool.next++;
  pool.programs.emplace(id, std::move(p));
  *handle = id;
  auto &owned = *pool.programs.at(id);
  if (hipStreamCreateWithFlags(&owned.stream, hipStreamNonBlocking) != hipSuccess) return 4;
  for (uint32_t i = 0; i < steps; ++i) {
    auto &s = owned.stages[i];
    void *module = nullptr, *function = nullptr; int hit = 0;
    if (tessera_rocm_image_acquire(plan[i].image, plan[i].image_bytes,
         plan[i].entry, &s.lease, &module, &function, &hit)) return 3;
    s.function = static_cast<hipFunction_t>(function);
    std::copy_n(plan[i].geometry, 6, s.geometry.begin());
    std::copy_n(plan[i].scalars, plan[i].scalar_count, s.scalars.begin());
  }
  for (uint32_t i = 0; i < buffers; ++i)
    if (hipMalloc(&owned.buffers[i], contract[i].bytes) != hipSuccess) return 4;
  if (owned.pinned) {
    if (hipHostMalloc(&owned.pinnedInputs, inputTotal, hipHostMallocDefault) != hipSuccess ||
        hipHostMalloc(&owned.pinnedReadback, outputMaximum, hipHostMallocDefault) != hipSuccess)
      return 4;
  }
  for (uint32_t i = 0; i < steps; ++i) {
    auto &s = owned.stages[i]; size_t arg = 0;
    for (uint32_t j = 0; j <= plan[i].input_count; ++j) {
      uint32_t slot = j < plan[i].input_count ? plan[i].inputs[j] : plan[i].output;
      auto &r = s.refs[j];
      r = {owned.buffers[slot], owned.buffers[slot], 0,
           int64_t(contract[slot].elements), 1};
      s.argv[arg++] = &r.allocated; s.argv[arg++] = &r.aligned;
      s.argv[arg++] = &r.offset; s.argv[arg++] = &r.elements; s.argv[arg++] = &r.stride;
    }
    for (uint32_t j = 0; j < plan[i].scalar_count; ++j) s.argv[arg++] = &s.scalars[j];
  }
  int rc = upload(owned, inputs, inputBytes);
  if (rc) return rc;
  owned.poisoned = false;
  return 0;
} catch (...) { return 12; }

extern "C" int tessera_rocm_program_update(
    uint64_t handle, const void *const *inputs, const uint64_t *bytes) try {
  if (getpid() != process) return 2;
  auto &pool = state(); std::lock_guard<std::mutex> guard(pool.mutex);
  auto it = pool.programs.find(handle); if (it == pool.programs.end()) return 1;
  auto &p = *it->second;
  if (!identity(p)) return 2;
  if (p.poisoned) return 10;
  return upload(p, inputs, bytes);
} catch (...) { return 12; }

extern "C" int tessera_rocm_program_invoke(
    uint64_t handle, uint32_t repeats, uint64_t *generation, float *elapsed) try {
  if (generation) *generation = 0;
  if (elapsed) *elapsed = 0;
  if (!generation || !repeats || repeats > 1048576) return 1;
  if (getpid() != process) return 2;
  auto &pool = state(); std::lock_guard<std::mutex> guard(pool.mutex);
  auto it = pool.programs.find(handle); if (it == pool.programs.end()) return 1;
  auto &p = *it->second;
  if (!identity(p)) return 2;
  if (p.poisoned || !p.ready || p.generation == UINT64_MAX) return 10;
  p.output = false;
  hipEvent_t begin{}, end{};
  if (elapsed) {
    if (hipEventCreate(&begin) != hipSuccess) return 4;
    p.events.push_back(begin);
    if (hipEventCreate(&end) != hipSuccess) { p.poisoned = true; return 4; }
    p.events.push_back(end);
    if (hipEventRecord(begin, p.stream) != hipSuccess) { p.poisoned = true; return 6; }
  }
  int status = 0;
  for (uint32_t i = 0; i < repeats && !status; ++i)
    for (auto &s : p.stages)
      if (hipModuleLaunchKernel(s.function, s.geometry[0], s.geometry[1], s.geometry[2],
          s.geometry[3], s.geometry[4], s.geometry[5], 0, p.stream, s.argv.data(),
          nullptr) != hipSuccess) { status = 6; break; }
  if (!status && elapsed && hipEventRecord(end, p.stream) != hipSuccess) status = 6;
  // Completion precedes event destruction, output publication and any readback.
  if (hipStreamSynchronize(p.stream) != hipSuccess) { p.poisoned = true; return 7; }
  if (!status && elapsed && hipEventElapsedTime(elapsed, begin, end) != hipSuccess) status = 7;
  while (!p.events.empty()) {
    if (hipEventDestroy(p.events.back()) != hipSuccess) { p.poisoned = true; return 9; }
    p.events.pop_back();
  }
  if (status) { p.poisoned = true; return status; }
  if (elapsed) *elapsed /= repeats;
  p.output = true; *generation = ++p.generation;
  return 0;
} catch (...) { return 12; }

// Diagnostic member windows run the immutable SSA members in prefix order.
// Repetition of a pure member writes the same fresh output; consumers only run
// after producer completion. Captured launch groups remove CPU enqueue gaps;
// windows still include device graph dispatch, not instruction-counter time.
extern "C" int tessera_rocm_program_profile_members(
    uint64_t handle, uint32_t repeats, uint32_t members,
    uint64_t *generation, float *elapsed) try {
  if (generation) *generation = 0;
  if (!generation || !elapsed || !repeats || repeats > 65536) return 1;
  if (getpid() != process) return 2;
  auto &pool = state(); std::lock_guard<std::mutex> guard(pool.mutex);
  auto it = pool.programs.find(handle); if (it == pool.programs.end()) return 1;
  auto &p = *it->second;
  if (!identity(p)) return 2;
  if (p.poisoned || !p.ready || p.generation == UINT64_MAX) return 10;
  if (members != p.stages.size()) return 1;
  p.graphs.reserve(1); p.graphExecutables.reserve(1);
  p.events.reserve(2);
  p.output = false;
  // Quarantine on any operation failure or exception. Only complete successful
  // profiling publishes a fresh output generation and restores a usable owner.
  p.poisoned = true;
  for (uint32_t member = 0; member < members; ++member) {
    auto &s = p.stages[member];
    if (hipStreamBeginCapture(p.stream, hipStreamCaptureModeThreadLocal) != hipSuccess)
      return 6;
    int status = 0;
    for (uint32_t i = 0; i < repeats; ++i)
      if (hipModuleLaunchKernel(s.function, s.geometry[0], s.geometry[1], s.geometry[2],
          s.geometry[3], s.geometry[4], s.geometry[5], 0, p.stream, s.argv.data(),
          nullptr) != hipSuccess) { status = 6; break; }
    hipGraph_t graph{};
    auto ended = hipStreamEndCapture(p.stream, &graph);
    if (graph) p.graphs.push_back(graph);
    if (status || ended != hipSuccess || !graph) return 6;
    hipGraphExec_t executable{};
    auto instantiated = hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0);
    if (executable) p.graphExecutables.push_back(executable);
    if (instantiated != hipSuccess || !executable) return 4;
    hipEvent_t begin{}, end{};
    if (hipEventCreate(&begin) != hipSuccess) return 4;
    p.events.push_back(begin);
    if (hipEventCreate(&end) != hipSuccess) return 4;
    p.events.push_back(end);
    if (hipEventRecord(begin, p.stream) != hipSuccess ||
        hipGraphLaunch(executable, p.stream) != hipSuccess ||
        hipEventRecord(end, p.stream) != hipSuccess) return 6;
    if (hipStreamSynchronize(p.stream) != hipSuccess) return 7;
    if (hipEventElapsedTime(&elapsed[member], begin, end) != hipSuccess) return 7;
    elapsed[member] /= repeats;
    // clean first synchronizes, then releases graph/event resources. Here
    // release only these resources; immutable buffers/image leases stay owned.
    if (hipGraphExecDestroy(executable) != hipSuccess) return 9;
    p.graphExecutables.pop_back();
    if (hipGraphDestroy(graph) != hipSuccess) return 9;
    p.graphs.pop_back();
    while (!p.events.empty()) {
      if (hipEventDestroy(p.events.back()) != hipSuccess) return 9;
      p.events.pop_back();
    }
  }
  p.poisoned = false; p.output = true; *generation = ++p.generation;
  return 0;
} catch (...) { return 12; }

extern "C" int tessera_rocm_program_read(
    uint64_t handle, uint32_t slot, uint64_t generation, void *output,
    uint64_t bytes) try {
  if (!output) return 1;
  if (getpid() != process) return 2;
  auto &pool = state(); std::lock_guard<std::mutex> guard(pool.mutex);
  auto it = pool.programs.find(handle); if (it == pool.programs.end()) return 1;
  auto &p = *it->second;
  if (!identity(p)) return 2;
  if (slot >= p.contract.size() || p.contract[slot].ownership != 2 ||
      bytes != p.contract[slot].bytes) return 1;
  if (p.poisoned || !p.output || !generation || generation != p.generation) return 10;
  if (!p.pinned) p.readback.resize(bytes);
  if (p.pinned && bytes > p.pinnedReadbackBytes) return 1;
  void *staging = p.pinned ? p.pinnedReadback : p.readback.data();
  if (hipMemcpyAsync(staging, p.buffers[slot], bytes,
                     hipMemcpyDeviceToHost, p.stream) != hipSuccess) {
    p.poisoned = true; return 5;
  }
  if (hipStreamSynchronize(p.stream) != hipSuccess) { p.poisoned = true; return 7; }
  std::memcpy(output, staging, bytes);
  return 0;
} catch (...) { return 12; }

extern "C" int tessera_rocm_program_close(uint64_t handle) try {
  if (getpid() != process) return 2;
  auto &pool = state(); std::lock_guard<std::mutex> guard(pool.mutex);
  auto it = pool.programs.find(handle); if (it == pool.programs.end()) return 1;
  auto &p = *it->second;
  if (!identity(p)) return 2;
  // Completion and identity precede reuse. A closed handle is always removed;
  // a later acquire gets a fresh handle and retains its generation counter.
  const uint64_t idleLimit = uint64_t(128) << 20;
  if (!p.poisoned && cacheEnabled() && p.events.empty() && p.graphs.empty() &&
      p.graphExecutables.empty() &&
      p.allocationBytes <= idleLimit) {
    if (hipStreamSynchronize(p.stream) != hipSuccess) { p.poisoned = true; return 7; }
    // Close appends and acquire removes, so the first idle owner is least
    // recently used. Only completed owners in this device/context may be
    // evicted; failed cleanup remains quarantined and budgeted.
    while (pool.idle.size() >= 4 ||
           p.allocationBytes > idleLimit - pool.idleBytes) {
      auto victim = pool.idle.begin();
      for (; victim != pool.idle.end(); ++victim)
        if (!(*victim)->poisoned && identity(**victim)) break;
      if (victim == pool.idle.end()) break;
      auto &retired = **victim;
      retired.poisoned = true;
      int rc = clean(retired);
      if (rc) continue;
      pool.idleBytes -= retired.allocationBytes;
      pool.idle.erase(victim);
    }
    if (pool.idle.size() >= 4 ||
        p.allocationBytes > idleLimit - pool.idleBytes) {
      p.poisoned = true;
      int rc = clean(p);
      if (!rc) pool.programs.erase(it);
      return rc;
    }
    p.ready = false; p.output = false;
    p.snapshots.clear(); p.readback.clear();
    pool.idleBytes += p.allocationBytes;
    pool.idle.push_back(std::move(it->second));
    pool.programs.erase(it);
    return 0;
  }
  p.poisoned = true;
  int rc = clean(p);
  if (!rc) pool.programs.erase(it);
  return rc;
} catch (...) { return 12; }

extern "C" int tessera_rocm_program_cache_stats(
    uint64_t *hits, uint64_t *misses, uint64_t *entries, uint64_t *bytes) try {
  if (!hits || !misses || !entries || !bytes) return 1;
  if (getpid() != process) return 2;
  auto &pool=state(); std::lock_guard<std::mutex> guard(pool.mutex);
  *hits=pool.hits; *misses=pool.misses;
  *entries=pool.idle.size(); *bytes=pool.idleBytes;
  return 0;
} catch (...) { return 12; }

extern "C" int tessera_rocm_program_cache_clear() try {
  if (getpid() != process) return 2;
  auto &pool=state(); std::lock_guard<std::mutex> guard(pool.mutex);
  // Foreign-context owners cannot be destroyed from the current context.
  for (auto &p:pool.idle) if (!identity(*p)) return 2;
  while (!pool.idle.empty()) {
    auto &p=*pool.idle.back();
    p.poisoned=true;
    int rc=clean(p); if (rc) return rc;
    pool.idleBytes-=p.allocationBytes;
    pool.idle.pop_back();
  }
  return 0;
} catch (...) { return 12; }
