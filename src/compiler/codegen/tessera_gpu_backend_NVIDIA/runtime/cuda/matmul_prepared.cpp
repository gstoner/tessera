// Native ownership of checked static/bounded Schedule/Tile/LLVM SM120 matmul images.
// No source/kernel reconstruction; synchronous host storage is copied into a
// context-owned scratch. Each handle pins one module and physical ABI;
// the global invocation mutex and synchronous completion permit scratch reuse.
#include "tessera_nvidia_ptx_launch.h"
#include <cuda.h>
#include <unistd.h>
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
namespace {
std::mutex mutex;
const pid_t process = getpid();
thread_local std::string error;
CUcontext primary = nullptr;
uint64_t nextHandle = 1;
int bad(const char *message) { error = message; return 1; }
bool ok(CUresult status, const char *operation) {
  if (status == CUDA_SUCCESS) return true;
  const char *name = nullptr;
  cuGetErrorName(status, &name);
  error = std::string(operation) + ": " + (name ? name : "CUDA failure");
  return false;
}
struct Scratch {
  CUcontext context = nullptr;
  unsigned long long identity = 0;
  CUdeviceptr base = 0;
  size_t capacity = 0, allocations = 0;
  uint64_t generation = 0;
  void *host = nullptr;
  size_t hostCapacity = 0;
  ~Scratch() {
    if (getpid() != process || !context || cuCtxPushCurrent(context) != CUDA_SUCCESS) return;
    unsigned long long current = 0;
    if (cuCtxGetId(context, &current) == CUDA_SUCCESS && current == identity) {
      if (base) cuMemFree(base);
      if (host) cuMemFreeHost(host);
    }
    CUcontext prior = nullptr;
    cuCtxPopCurrent(&prior);
  }
  bool growHost(size_t bytes) {
    if (host && hostCapacity >= bytes) return true;
    void *replacement = nullptr;
    CUresult status = cuMemAllocHost(&replacement, bytes);
    if (status == CUDA_ERROR_OUT_OF_MEMORY && host) {
      if (!ok(cuMemFreeHost(host), "release idle pinned staging")) return false;
      host = nullptr; hostCapacity = 0;
      status = cuMemAllocHost(&replacement, bytes);
    }
    if (!ok(status, "grow pinned tensor staging")) return false;
    if (host && !ok(cuMemFreeHost(host), "retire pinned tensor staging")) {
      cuMemFreeHost(replacement); return false;
    }
    host = replacement; hostCapacity = bytes;
    return true;
  }
  bool grow(size_t bytes) {
    if (base && capacity >= bytes) return true;
    CUdeviceptr replacement = 0;
    CUresult status = cuMemAlloc(&replacement, bytes);
    // All prior synchronous frames are retired under mutex. Release idle old
    // storage and retry once when transient double residency is the OOM cause.
    if (status == CUDA_ERROR_OUT_OF_MEMORY && base) {
      if (!ok(cuMemFree(base), "release idle matmul scratch")) return false;
      base = 0; capacity = 0;
      status = cuMemAlloc(&replacement, bytes);
    }
    if (!ok(status, "grow retained matmul scratch")) return false;
    if (base && !ok(cuMemFree(base), "retire old matmul scratch")) {
      cuMemFree(replacement); return false;
    }
    base = replacement; capacity = bytes; ++allocations; ++generation;
    return true;
  }
};
std::map<unsigned long long, std::weak_ptr<Scratch>> scratch;
struct RowProducer {
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  bool cooperative = false;
};
struct Owner {
  CUcontext context = nullptr;
  unsigned long long identity = 0;
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  CUmodule producerModule = nullptr;
  CUfunction producer = nullptr;
  std::vector<RowProducer> followingProducers, rhsProducers;
  CUdeviceptr producerScratch = 0, rhsEdge = 0, rhsScratch = 0, ownedLhsEdge = 0, ownedResult = 0;
  bool cooperativeProducer = false, invoked = false, rowSymbol = false;
  int dynamicAxes = 0, storage = 0;
  bool macro = false;
  bool bias = false, residual = false, rowB = false, halfOutput = false;
  CUstream stream = nullptr;
  std::shared_ptr<Scratch> arena;
  std::array<CUdeviceptr, 5> buffers{};
  std::array<size_t, 5> bytes{};
  std::array<TesseraNvidiaMatmulHostView, 5> expected{};
  std::array<int64_t, 3> dims{};
  size_t count = 0;
  uint64_t hostGeneration = 0;
  CUdeviceptr hostBase = 0, hostEdge = 0;
  std::array<int64_t, 3> activeDims{};
  ~Owner() {
    if (getpid() != process || !context || cuCtxPushCurrent(context) != CUDA_SUCCESS) return;
    unsigned long long current = 0;
    if (cuCtxGetId(context, &current) == CUDA_SUCCESS && current == identity) {
      if (stream) cuStreamSynchronize(stream);
      if (stream) cuStreamDestroy(stream);
      if (producerScratch) cuMemFree(producerScratch);
      if (rhsEdge) cuMemFree(rhsEdge);
      if (ownedLhsEdge) cuMemFree(ownedLhsEdge);
      if (ownedResult) cuMemFree(ownedResult);
      if (rhsScratch) cuMemFree(rhsScratch);
      for (auto &stage : rhsProducers)
        if (stage.module) cuModuleUnload(stage.module);
      for (auto &stage : followingProducers)
        if (stage.module) cuModuleUnload(stage.module);
      if (producerModule) cuModuleUnload(producerModule);
      if (module) cuModuleUnload(module);
    }
    CUcontext prior = nullptr;
    cuCtxPopCurrent(&prior);
  }
};
std::map<uint64_t, std::unique_ptr<Owner>> owners;
bool current(CUcontext &context) {
  if (!ok(cuInit(0), "cuInit") || !ok(cuCtxGetCurrent(&context), "get context")) return false;
  if (!context) {
    CUdevice device = 0;
    if (!ok(cuDeviceGet(&device, 0), "get device")) return false;
    if (!primary && !ok(cuDevicePrimaryCtxRetain(&primary, device), "retain context")) return false;
    context = primary;
    if (!ok(cuCtxSetCurrent(context), "set context")) return false;
  }
  CUdevice device; int major = 0, minor = 0;
  return ok(cuCtxGetDevice(&device), "context device") &&
    ok(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "major") &&
    ok(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "minor") &&
    major == 12 && minor == 0;
}
bool view(Owner &owner, int dtype, int rank, int64_t x, int64_t y, bool column) {
  const size_t width = dtype == 1 ? 4 : 2;
  const __int128 bytes = (__int128)x * y * width;
  if (x <= 0 || y <= 0 || bytes > std::numeric_limits<size_t>::max() ||
      bytes > std::numeric_limits<int64_t>::max()) return false;
  const size_t index = owner.count++;
  auto &expected = owner.expected[index];
  expected.bytes = static_cast<size_t>(bytes);
  expected.dtype = dtype; expected.rank = rank;
  expected.shape[0] = x; expected.shape[1] = rank == 1 ? 0 : y;
  expected.strides[0] = column ? width : y * width;
  expected.strides[1] = rank == 1 ? 0 : column ? x * width : width;
  owner.bytes[index] = expected.bytes;
  return true;
}
// Host-staged and resident execution share the same sealed member kernel ABIs.
bool submit(Owner &owner, const std::array<CUdeviceptr, 5> &buffers,
            CUdeviceptr edge, std::array<int64_t, 3> dims, CUstream stream, int64_t rhsLeading = 0,
            int repeats = 1, float *stageTimes = nullptr) {
  size_t timingIndex = 0;
  auto launch = [&](CUfunction function, unsigned x, unsigned y, unsigned threads,
                    void **args, const char *message) {
    if (!stageTimes)
      return ok(cuLaunchKernel(function,x,y,1,threads,1,1,0,stream,args,nullptr),message);
    CUevent start = nullptr, end = nullptr;
    bool status = ok(cuEventCreate(&start,0),"create stage start event") &&
                  ok(cuEventCreate(&end,0),"create stage end event") &&
                  ok(cuEventRecord(start,stream),"record stage start");
    for (int i = 0; status && i < repeats; ++i)
      status = ok(cuLaunchKernel(function,x,y,1,threads,1,1,0,stream,args,nullptr),message);
    float elapsed = 0;
    status = status && ok(cuEventRecord(end,stream),"record stage end") &&
             ok(cuEventSynchronize(end),"synchronize stage end") &&
             ok(cuEventElapsedTime(&elapsed,start,end),"stage elapsed time");
    if (start) cuEventDestroy(start);
    if (end) cuEventDestroy(end);
    if (status) stageTimes[timingIndex++] = elapsed / repeats;
    return status;
  };
  if (owner.producer) {
    int64_t rows = dims[0], columns = dims[2];
    CUdeviceptr source = buffers[0];
    const size_t stages = 1 + owner.followingProducers.size();
    for (size_t index = 0; index < stages; ++index) {
      CUdeviceptr destination = ((stages - 1 - index) % 2 == 0) ? edge : owner.producerScratch;
      if (!destination || source == destination)
        return bad("prepared producer chain lost disjoint scratch"), false;
      CUfunction function = index == 0 ? owner.producer : owner.followingProducers[index - 1].function;
      bool cooperative = index == 0 ? owner.cooperativeProducer : owner.followingProducers[index - 1].cooperative;
      void *producerArgs[] = {&source, &destination, &rows, &columns};
      if (!launch(function,unsigned(cooperative ? rows : (rows + 127) / 128),
          1,128,producerArgs,"launch prepared tensor producer")) return false;
      source = destination;
    }
  }
  CUdeviceptr input = owner.producer ? edge : buffers[0];
  std::array<CUdeviceptr, 5> pointers = buffers;
  if (!owner.rhsProducers.empty()) {
    int64_t rows = dims[2], columns = dims[1];
    CUdeviceptr source = buffers[1];
    const size_t stages = owner.rhsProducers.size();
    for (size_t index = 0; index < stages; ++index) {
      CUdeviceptr destination = ((stages - 1 - index) % 2 == 0) ? owner.rhsEdge : owner.rhsScratch;
      if (!destination || source == destination)
        return bad("prepared RHS chain lost disjoint scratch"), false;
      auto &stage = owner.rhsProducers[index];
      void *producerArgs[] = {&source, &destination, &rows, &columns};
      if (!launch(stage.function,unsigned(stage.cooperative ? rows : (rows + 127) / 128),
          1,128,producerArgs,"launch prepared RHS tensor producer")) return false;
      source = destination;
    }
    pointers[1] = owner.rhsEdge;
    rhsLeading = dims[1];
  }
  void *args[11] = {}; size_t arg = 0;
  args[arg++] = &input;
  for (size_t i = 1; i < owner.count; ++i) args[arg++] = &pointers[i];
  for (auto &dimension : dims) args[arg++] = &dimension;
  std::array<int64_t, 3> leading{dims[2], rhsLeading ? rhsLeading : owner.rowB ? dims[1] : dims[2], dims[1]};
  if (owner.dynamicAxes)
    for (auto &dimension : leading) args[arg++] = &dimension;
  return launch(owner.function,unsigned((dims[1] + (owner.macro ? 31 : 7)) / (owner.macro ? 32 : 8)),
      unsigned((dims[0] + (owner.macro ? 31 : 15)) / (owner.macro ? 32 : 16)),
      owner.macro ? 128 : 32,args,"launch prepared matmul");
}

}
extern "C" const char *tessera_nvidia_matmul_last_error() { return error.c_str(); }
extern "C" int tessera_nvidia_matmul_prepare(
    const void *image, size_t imageBytes, const char *entry, const int64_t *dims,
    int storage, int bias, int residual, int rowB, int halfOutput, uint64_t *handle) {
  error.clear(); if (handle) *handle = 0;
  if (getpid() != process) return bad("prepared matmul cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);
  if (!image || !imageBytes || !entry || !dims || !handle ||
      std::strncmp(entry, "nvidia_sm120_scheduled_matmul_", 29) ||
      (std::strstr(entry, "_macro_kernel") && rowB) || (storage != 2 && storage != 3) ||
      (bias != 0 && bias != 1) || (residual != 0 && residual != 1) ||
      (rowB != 0 && rowB != 1) || (halfOutput != 0 && halfOutput != 1))
    return bad("invalid static matmul image or ABI");
  for (int i = 0; i < 3; ++i)
    if (dims[i] <= 0 || dims[i] >= (1LL << 31)) return bad("matmul extent outside ABI");
  try {
    auto owner = std::make_unique<Owner>();
    if (!current(owner->context) || !ok(cuCtxGetId(owner->context, &owner->identity), "context identity"))
      return bad("prepared matmul requires live SM120 context");
    owner->dims = {dims[0], dims[1], dims[2]};
    owner->storage = storage; owner->bias = bias; owner->residual = residual;
    owner->macro = std::strstr(entry, "_macro_kernel") != nullptr;
    owner->rowB = rowB; owner->halfOutput = halfOutput;
    owner->rowSymbol = std::strstr(entry, "_row_rhs_kernel") != nullptr;
    const int64_t m = dims[0], n = dims[1], k = dims[2];
    if (!view(*owner, storage, 2, m, k, false) ||
        !view(*owner, storage, 2, k, n, !rowB) ||
        (bias && !view(*owner, 1, 1, n, 1, false)) ||
        (residual && !view(*owner, 1, 2, m, n, false)) ||
        !view(*owner, halfOutput ? 2 : 1, 2, m, n, false))
      return bad("matmul capacity overflow");
    std::string ptx(static_cast<const char *>(image), imageBytes);
    if (!ok(cuModuleLoadData(&owner->module, ptx.c_str()), "load compiler PTX") ||
        !ok(cuModuleGetFunction(&owner->function, owner->module, entry), "resolve compiler entry") ||
        !ok(cuStreamCreate(&owner->stream, CU_STREAM_NON_BLOCKING), "create stream")) return 1;
    owner->arena = scratch[owner->identity].lock();
    if (!owner->arena) {
      owner->arena = std::make_shared<Scratch>();
      owner->arena->context = owner->context;
      owner->arena->identity = owner->identity;
      scratch[owner->identity] = owner->arena;
    }
    const uint64_t id = nextHandle++;
    owners.emplace(id, std::move(owner)); *handle = id; return 0;
  } catch (...) { return bad("prepared matmul allocation failed"); }
}
// Capacities remain immutable. Dynamic invocation derives compact active
// pitches only after verifying every shape against these declared axis bounds.
extern "C" int tessera_nvidia_matmul_set_dynamic_axes(uint64_t handle, int axes) {
  error.clear();
  if (getpid() != process) return bad("prepared dynamic matmul cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);
  auto found = owners.find(handle);
  if (found == owners.end() || axes <= 0 || axes > 7)
    return bad("invalid dynamic matmul axes");
  Owner &owner = *found->second;
  if (owner.invoked || owner.producer || !owner.rhsProducers.empty() || owner.dynamicAxes)
    return bad("dynamic axes require an unused owner");
  if (owner.macro) return bad("dynamic axes require a strided typed consumer");
  if (owner.rowB != owner.rowSymbol)
    return bad("dynamic RHS storage differs from compiler entry");
  // The native strided kernel has ordered pointer parameters followed by
  // M/N/K/LDA/LDB/LDD, all 64-bit. Never reinterpret a static kernel ABI.
  for (size_t i = 0; i < owner.count + 6; ++i) {
    size_t offset = 0, bytes = 0;
    if (!ok(cuFuncGetParamInfo(owner.function, i, &offset, &bytes),
            "inspect dynamic matmul parameter") || offset != i * 8 || bytes != 8)
      return bad("dynamic matmul parameter ABI mismatch");
  }
  size_t offset = 0, bytes = 0;
  if (cuFuncGetParamInfo(owner.function, owner.count + 6, &offset, &bytes) !=
      CUDA_ERROR_INVALID_VALUE)
    return bad("dynamic matmul parameter count mismatch");
  owner.dynamicAxes = axes;
  return 0;
}
// Attach exactly one compiler-owned shape-preserving row producer before use.
// Both pointer ABIs are retained by one synchronous owner; no edge escapes.
static int attachProducer(uint64_t handle, const void *image, size_t imageBytes,
    const char *entry, int cooperative, bool append, bool rhs = false) {
  error.clear();
  if (getpid() != process) return bad("prepared tensor edge cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);
  auto found = owners.find(handle);
  if (found == owners.end() || !image || !imageBytes || !entry ||
      (cooperative != 0 && cooperative != 1))
    return bad("invalid prepared tensor producer");
  Owner &owner = *found->second;
  if (rhs) {
    if (!owner.rowB || owner.macro)
      return bad("RHS producers require a row-major typed matmul consumer");
    if ((!append && !owner.rhsProducers.empty()) || (append && owner.rhsProducers.empty()))
      return bad("prepared RHS producer attachment order differs");
  } else {
    if (!append && owner.producer)
      return bad("prepared tensor producer already attached");
    if (append && !owner.producer)
      return bad("prepared tensor producer attachment order differs");
  }
  if (size_t(bool(owner.producer)) + owner.followingProducers.size() + owner.rhsProducers.size() >= 63)
    return bad("prepared tensor producer chain exceeds its bound");
  if (owner.invoked) return bad("tensor producer must attach before first invocation");
  const bool f16 = owner.expected[0].dtype == 2;
  const std::string type = f16 ? "f16" : "bf16";
  const std::string rms = "tessera_tile_norm_rmsnorm_" + type + "_";
  const std::string layer = "tessera_tile_norm_layernorm_" + type + "_";
  const std::string softmax = "tessera_tile_softmax_" + type;
  const bool norm = std::strncmp(entry, rms.c_str(), rms.size()) == 0 ||
                    std::strncmp(entry, layer.c_str(), layer.size()) == 0;
  const bool softmaxSerial = softmax == entry;
  const bool softmaxCooperative = softmax + "_cooperative_128" == entry;
  const bool expectedCooperative = norm
      ? bool(std::strstr(entry, "_cooperative_128_"))
      : softmaxCooperative;
  if ((!norm && !softmaxSerial && !softmaxCooperative) ||
      expectedCooperative != bool(cooperative) ||
      (rhs ? owner.dims[2] : owner.dims[0]) > (1LL << 31) / (rhs ? owner.dims[1] : owner.dims[2]))
    return bad("prepared tensor producer ABI or geometry mismatch");
  CUcontext context = nullptr; unsigned long long identity = 0;
  if (!ok(cuCtxGetCurrent(&context), "get producer context")) return 1;
  if (!context && !ok(cuCtxSetCurrent(owner.context), "restore producer context")) return 1;
  if ((context && context != owner.context) ||
      !ok(cuCtxGetId(owner.context, &identity), "producer context identity") ||
      identity != owner.identity) return bad("prepared tensor producer context changed");
  CUmodule module = nullptr; CUfunction function = nullptr;
  try {
    std::string ptx(static_cast<const char *>(image), imageBytes);
    if (!ok(cuModuleLoadData(&module, ptx.c_str()), "load compiler producer PTX")) return 1;
    if (!ok(cuModuleGetFunction(&function, module, entry), "resolve compiler producer")) {
      cuModuleUnload(module); return 1;
    }
    if (rhs) {
      // RHS row kernels use KxN, independently of the LHS MxK frame.
      // Two disjoint capacity buffers permit a chain without overwriting roots.
      if (!append && owner.dynamicAxes) {
        size_t total = 0;
        for (size_t i = 0; i < owner.count; ++i) {
          if (owner.bytes[i] > SIZE_MAX - total - 255) {
            cuModuleUnload(module); return bad("bounded DAG staging overflow");
          }
          total = (total + owner.bytes[i] + 255) & ~size_t(255);
        }
        const size_t hostBytes = total;
        if (owner.producer) {
          if (owner.bytes[0] > SIZE_MAX - total - 255) {
            cuModuleUnload(module); return bad("bounded DAG LHS edge overflow");
          }
          total = (total + owner.bytes[0] + 255) & ~size_t(255);
        }
        if (!owner.arena->grow(total) || !owner.arena->growHost(hostBytes)) {
          cuModuleUnload(module); return 1;
        }
      }
      CUdeviceptr replacement = 0;
      bool needsScratch = append && !owner.rhsScratch;
      bool needsEdge = !append;
      if ((needsScratch || needsEdge) && !ok(cuMemAlloc(&replacement, owner.bytes[1]),
                                            "allocate RHS producer capacity")) {
        cuModuleUnload(module); return 1;
      }
      try {
        owner.rhsProducers.push_back(RowProducer{module,function,bool(cooperative)});
      } catch (...) {
        if (replacement) cuMemFree(replacement);
        throw;
      }
      if (needsEdge) owner.rhsEdge = replacement;
      if (needsScratch) owner.rhsScratch = replacement;
    } else if (append) {
      // Bounded chains retain their maximum input/output staging frame from
      // the first attachment. A smaller first invocation must not introduce
      // a new allocation when the active shape later reaches its capacity.
      if (owner.dynamicAxes && owner.followingProducers.empty()) {
        size_t total = 0;
        for (size_t i = 0; i < owner.count; ++i) {
          if (owner.bytes[i] > SIZE_MAX - total - 255) {
            cuModuleUnload(module); return bad("bounded tensor staging overflow");
          }
          total = (total + owner.bytes[i] + 255) & ~size_t(255);
        }
        const size_t hostBytes = total;
        if (owner.bytes[0] > SIZE_MAX - total - 255) {
          cuModuleUnload(module); return bad("bounded tensor edge staging overflow");
        }
        total = (total + owner.bytes[0] + 255) & ~size_t(255);
        if (!owner.arena->grow(total) || !owner.arena->growHost(hostBytes)) {
          cuModuleUnload(module); return 1;
        }
      }
      CUdeviceptr scratch = 0;
      if (!owner.producerScratch && !ok(cuMemAlloc(&scratch, owner.bytes[0]),
                                       "allocate producer chain scratch")) {
        cuModuleUnload(module); return 1;
      }
      try {
        owner.followingProducers.push_back(RowProducer{module, function, bool(cooperative)});
      } catch (...) {
        if (scratch) cuMemFree(scratch);
        throw;
      }
      if (scratch) owner.producerScratch = scratch;
    } else {
      owner.producerModule = module; owner.producer = function;
      owner.cooperativeProducer = cooperative;
    }
    return 0;
  } catch (...) {
    if (module) cuModuleUnload(module);
    return bad("prepared tensor producer allocation failed");
  }
}
extern "C" int tessera_nvidia_matmul_attach_producer(
    uint64_t handle, const void *image, size_t imageBytes, const char *entry, int cooperative) {
  return attachProducer(handle, image, imageBytes, entry, cooperative, false);
}
extern "C" int tessera_nvidia_matmul_append_producer(
    uint64_t handle, const void *image, size_t imageBytes, const char *entry, int cooperative) {
  return attachProducer(handle, image, imageBytes, entry, cooperative, true);
}
extern "C" int tessera_nvidia_matmul_attach_rhs_producer(
    uint64_t handle, const void *image, size_t imageBytes, const char *entry,
    int cooperative, int append) {
  if (append != 0 && append != 1) return bad("invalid RHS append mode");
  return attachProducer(handle,image,imageBytes,entry,cooperative,bool(append),true);
}
extern "C" int tessera_nvidia_matmul_context_identity(uint64_t *identity) {
  error.clear(); if (identity) *identity = 0;
  if (getpid() != process) return bad("prepared tensor cache cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);
  if (!identity) return bad("prepared tensor context output is null");
  CUcontext context = nullptr; unsigned long long value = 0;
  if (!current(context) || !ok(cuCtxGetId(context, &value), "cache context identity"))
    return bad("prepared tensor cache requires live SM120 context");
  *identity = value; return 0;
}
extern "C" int tessera_nvidia_matmul_invoke(
    uint64_t handle, const TesseraNvidiaMatmulHostView *views, size_t count) {
  error.clear();
  if (getpid() != process) return bad("prepared matmul cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);
  auto found = owners.find(handle);
  if (found == owners.end()) return bad("prepared matmul is closed or unknown");
  Owner &owner = *found->second;
  if (!views || count != owner.count) return bad("prepared matmul buffer arity");
  CUcontext context = nullptr; unsigned long long identity = 0;
  if (!ok(cuCtxGetCurrent(&context), "get context")) return 1;
  if (!context && !ok(cuCtxSetCurrent(owner.context), "restore owning context")) return 1;
  if ((context && context != owner.context) ||
      !ok(cuCtxGetId(owner.context, &identity), "context identity") || identity != owner.identity)
    return bad("prepared matmul context changed");
  Owner frame;
  frame.dims = owner.dims;
  if (owner.dynamicAxes) {
    frame.dims = {views[0].shape[0], views[1].shape[1], views[0].shape[1]};
    for (size_t axis = 0; axis < 3; ++axis)
      if (frame.dims[axis] <= 0 || frame.dims[axis] > owner.dims[axis] ||
          (!(owner.dynamicAxes & (1 << axis)) && frame.dims[axis] != owner.dims[axis]))
        return bad("prepared dynamic matmul extent outside bound");
    const auto m = frame.dims[0], n = frame.dims[1], k = frame.dims[2];
    if (!view(frame, owner.storage, 2, m, k, false) ||
        !view(frame, owner.storage, 2, k, n, !owner.rowB) ||
        (owner.bias && !view(frame, 1, 1, n, 1, false)) ||
        (owner.residual && !view(frame, 1, 2, m, n, false)) ||
        !view(frame, owner.halfOutput ? 2 : 1, 2, m, n, false))
      return bad("prepared dynamic frame capacity overflow");
  } else {
    frame.count = owner.count; frame.bytes = owner.bytes; frame.expected = owner.expected;
  }
  // Singleton-axis strides never participate in addressing. Accept their
  // NumPy C/F/broadcast representations without relaxing non-singleton pitch.
  for (size_t i = 0; i < count; ++i) {
    const auto &actual = views[i]; const auto &expected = frame.expected[i];
    const size_t width = expected.dtype == 1 ? 4 : 2;
    if (!actual.data || reinterpret_cast<uintptr_t>(actual.data) % width ||
        actual.bytes != expected.bytes || actual.dtype != expected.dtype ||
        actual.rank != expected.rank || actual.shape[0] != expected.shape[0] ||
        actual.shape[1] != expected.shape[1] ||
        (expected.shape[0] > 1 && actual.strides[0] != expected.strides[0]) ||
        (expected.rank == 2 && expected.shape[1] > 1 && actual.strides[1] != expected.strides[1]) ||
        (expected.rank == 1 && actual.strides[1] != 0))
      return bad("prepared matmul host shape/stride/dtype/capacity mismatch");
  }
  size_t total = 0;
  for (size_t i = 0; i < count; ++i) {
    if (frame.bytes[i] > SIZE_MAX - total - 255) return bad("matmul scratch overflow");
    total = (total + frame.bytes[i] + 255) & ~size_t(255);
  }
  const size_t edgeOffset = total;
  if (owner.producer) {
    if (frame.bytes[0] > SIZE_MAX - total - 255) return bad("tensor edge scratch overflow");
    total = (total + frame.bytes[0] + 255) & ~size_t(255);
  }
  if (!owner.arena->grow(total) || !owner.arena->growHost(edgeOffset)) return 1;
  size_t offset = 0;
  for (size_t i = 0; i < count; ++i) {
    owner.buffers[i] = owner.arena->base + offset;
    offset = (offset + frame.bytes[i] + 255) & ~size_t(255);
  }
  owner.invoked = true;
  owner.hostGeneration = 0;
  ++owner.arena->generation;
  // Retained pinned staging is leased with device scratch. All transfers and
  // both kernels use the owner's stream; no default-stream copy can race the
  // producer. Copy all host inputs before any caller output is written.
  auto *host = static_cast<unsigned char *>(owner.arena->host);
  std::array<size_t, 5> hostOffsets{}; size_t hostOffset = 0;
  for (size_t i = 0; i < count; ++i) {
    hostOffsets[i] = hostOffset;
    hostOffset = (hostOffset + frame.bytes[i] + 255) & ~size_t(255);
  }
  for (size_t i = 0; i + 1 < count; ++i)
    std::memcpy(host + hostOffsets[i], views[i].data, frame.bytes[i]);
  bool submitted = true;
  for (size_t i = 0; i + 1 < count && submitted; ++i)
    submitted = ok(cuMemcpyHtoDAsync(owner.buffers[i], host + hostOffsets[i],
        frame.bytes[i], owner.stream), "upload prepared tensor input");
  if (submitted)
    submitted = submit(owner, owner.buffers,
        owner.producer ? owner.arena->base + edgeOffset : owner.buffers[0],
        frame.dims, owner.stream);
  if (submitted)
    submitted = ok(cuMemcpyDtoHAsync(host + hostOffsets[count - 1], owner.buffers[count - 1],
        frame.bytes[count - 1], owner.stream), "download prepared tensor output");
  // Every failure after an upload or producer submission drains the stream
  // before either pinned or device scratch can be leased by another owner.
  const std::string launchError = error;
  const bool completed = ok(cuStreamSynchronize(owner.stream), "complete prepared tensor/matmul");
  if (!submitted) { error = launchError; return 1; }
  if (!completed) return 1;
  std::memcpy(views[count - 1].data, host + hostOffsets[count - 1], frame.bytes[count - 1]);
  owner.hostGeneration = owner.arena->generation;
  owner.hostBase = owner.arena->base;
  owner.hostEdge = owner.producer ? owner.arena->base + edgeOffset : owner.buffers[0];
  owner.activeDims = frame.dims;
  return 0;
}

extern "C" int tessera_nvidia_matmul_profile(uint64_t handle, int repeats,
    float *stageMs, size_t stageCount, float *programMs,
    TesseraNvidiaMatmulHostView *output) {
  error.clear();
  if (getpid() != process) return bad("prepared profiling cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);
  auto found = owners.find(handle);
  if (found == owners.end()) return bad("prepared profiling owner is closed");
  Owner &owner = *found->second;
  const size_t stages = size_t(bool(owner.producer)) + owner.followingProducers.size() +
                        owner.rhsProducers.size() + 1;
  if (!stageMs || !programMs || !output || repeats <= 0 || repeats > 1000000 || stageCount != stages)
    return bad("prepared profiling stage/count ABI mismatch");
  if (!owner.hostGeneration || owner.hostGeneration != owner.arena->generation ||
      owner.hostBase != owner.arena->base)
    return bad("prepared profiling shared arena lease is stale");
  const size_t width = owner.halfOutput ? 2 : 4;
  const auto m = owner.activeDims[0], n = owner.activeDims[1];
  const size_t bytes = size_t(m)*size_t(n)*width;
  if (!output->data || reinterpret_cast<uintptr_t>(output->data) % width ||
      output->dtype != (owner.halfOutput ? 2 : 1) || output->rank != 2 ||
      output->shape[0] != m || output->shape[1] != n || output->bytes != bytes ||
      (m > 1 && output->strides[0] != int64_t(n*width)) ||
      (n > 1 && output->strides[1] != int64_t(width)))
    return bad("prepared profiling output shape/stride/storage mismatch");
  CUcontext context = nullptr; unsigned long long identity = 0;
  if (!ok(cuCtxGetCurrent(&context),"get profiling context")) return 1;
  if (!context && !ok(cuCtxSetCurrent(owner.context),"restore profiling context")) return 1;
  if ((context && context != owner.context) ||
      !ok(cuCtxGetId(owner.context,&identity),"profiling context identity") || identity != owner.identity)
    return bad("prepared profiling context changed");
  CUevent start = nullptr, end = nullptr;
  bool status = ok(cuEventCreate(&start,0),"create program start event") &&
                ok(cuEventCreate(&end,0),"create program end event") &&
                ok(cuEventRecord(start,owner.stream),"record program start");
  for (int i = 0; status && i < repeats; ++i)
    status = submit(owner,owner.buffers,owner.hostEdge,owner.activeDims,owner.stream);
  float elapsed = 0;
  status = status && ok(cuEventRecord(end,owner.stream),"record program end") &&
           ok(cuEventSynchronize(end),"synchronize program end") &&
           ok(cuEventElapsedTime(&elapsed,start,end),"program elapsed time");
  if (start) cuEventDestroy(start);
  if (end) cuEventDestroy(end);
  if (status) {
    *programMs = elapsed / repeats;
    status = submit(owner,owner.buffers,owner.hostEdge,owner.activeDims,owner.stream,
                    0,repeats,stageMs);
  }
  if (status)
    status = ok(cuMemcpyDtoH(output->data,owner.buffers[owner.count-1],bytes),
                "copy profiled output");
  if (!status) {
    const std::string saved = error;
    cuStreamSynchronize(owner.stream);
    owner.hostGeneration = 0;
    error = saved;
    return 1;
  }
  return 0;
}

// The final view is the caller's private edge allocation; the earlier views
// follow the consumer ABI with source replacing its LHS. Native completion
// retires all reads/writes before allocations may be released or reused.
static int invokeResident(
    uint64_t handle, const TesseraNvidiaMatmulHostView *views, size_t count,
    void *launchStream, bool ownedEdge,
    const uint64_t *producerStreams = nullptr, size_t producerCount = 0,
    int profileRepeats = 0, float *stageTimes = nullptr, size_t stageCount = 0,
    float *programMs = nullptr,
    const TesseraNvidiaMatmulHostView *hostOutput = nullptr) {
  error.clear();
  if (getpid() != process) return bad("prepared resident tensor cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);
  auto found = owners.find(handle);
  if (found == owners.end()) return bad("prepared resident tensor is closed or unknown");
  Owner &owner = *found->second;
  if (!owner.producer || !views ||
      count != owner.count + (ownedEdge ? 0 : 1) - (hostOutput ? 1 : 0) ||
      (!launchStream && !hostOutput) ||
      (ownedEdge && owner.rhsProducers.empty()) || (hostOutput && !ownedEdge))
    return bad("prepared resident tensor buffer/stream arity");
  const size_t stages = size_t(bool(owner.producer)) + owner.followingProducers.size() +
                        owner.rhsProducers.size() + 1;
  if ((profileRepeats || stageTimes || stageCount || programMs) &&
      (!producerStreams || profileRepeats <= 0 || profileRepeats > 1000000 ||
       !stageTimes || !programMs || stageCount != stages))
    return bad("ordered resident profiling stage/count ABI mismatch");
  CUcontext context = nullptr, streamContext = nullptr;
  unsigned long long identity = 0;
  CUstream stream = hostOutput ? owner.stream : static_cast<CUstream>(launchStream);
  if (!ok(cuCtxGetCurrent(&context), "get resident context") ||
      context != owner.context ||
      !ok(cuCtxGetId(context, &identity), "resident context identity") ||
      identity != owner.identity ||
      !ok(cuStreamGetCtx(stream, &streamContext), "resident stream context") ||
      streamContext != context) return bad("prepared resident tensor context changed");
  Owner frame;
  frame.dims = {views[0].shape[0], views[1].shape[1], views[0].shape[1]};
  for (size_t axis = 0; axis < 3; ++axis)
    if (frame.dims[axis] <= 0 || frame.dims[axis] > owner.dims[axis] ||
        (!(owner.dynamicAxes & (1 << axis)) && frame.dims[axis] != owner.dims[axis]))
      return bad("prepared resident extent outside bound");
  const auto m = frame.dims[0], n = frame.dims[1], k = frame.dims[2];
  if (!view(frame, owner.storage, 2, m, k, false) ||
      !view(frame, owner.storage, 2, k, n, !owner.rowB) ||
      (owner.bias && !view(frame, 1, 1, n, 1, false)) ||
      (owner.residual && !view(frame, 1, 2, m, n, false)) ||
      !view(frame, owner.halfOutput ? 2 : 1, 2, m, n, false))
    return bad("prepared resident frame capacity overflow");
  std::array<TesseraNvidiaMatmulHostView, 6> resultViews{};
  if (hostOutput) {
    const auto &expected = frame.expected[owner.count-1];
    const size_t width = expected.dtype == 1 ? 4 : 2;
    if (!hostOutput->data || reinterpret_cast<uintptr_t>(hostOutput->data) % width ||
        hostOutput->bytes != expected.bytes || hostOutput->dtype != expected.dtype ||
        hostOutput->rank != expected.rank || hostOutput->shape[0] != expected.shape[0] ||
        hostOutput->shape[1] != expected.shape[1] ||
        (expected.shape[0] > 1 && hostOutput->strides[0] != expected.strides[0]) ||
        (expected.shape[1] > 1 && hostOutput->strides[1] != expected.strides[1]))
      return bad("ordered resident host result storage mismatch");
    if (!owner.ownedResult && !ok(cuMemAlloc(&owner.ownedResult,owner.bytes[owner.count-1]),
                                  "allocate native resident result capacity")) return 1;
    for (size_t i = 0; i < count; ++i) resultViews[i] = views[i];
    resultViews[count] = expected;
    resultViews[count].data = reinterpret_cast<void *>(owner.ownedResult);
    ++count; views = resultViews.data();
  }
  std::array<TesseraNvidiaMatmulHostView, 6> ownedViews{};
  if (ownedEdge) {
    if (!owner.ownedLhsEdge && !ok(cuMemAlloc(&owner.ownedLhsEdge,owner.bytes[0]),
                                  "allocate owned resident LHS capacity")) return 1;
    for (size_t i = 0; i < count; ++i) ownedViews[i] = views[i];
    ownedViews[count] = frame.expected[0];
    ownedViews[count].data = reinterpret_cast<void *>(owner.ownedLhsEdge);
    views = ownedViews.data(); ++count;
  }
  std::array<CUdeviceptr, 6> pointers{};
  std::array<size_t, 6> spans{};
  int64_t rhsLeading = owner.rowB ? n : k;
  for (size_t i = 0; i < count; ++i) {
    const auto &actual = views[i];
    const auto &expected = frame.expected[i == owner.count ? 0 : i];
    const size_t width = expected.dtype == 1 ? 4 : 2;
    // The row producer and private edge retain their compact ABI. Only a
    // strided consumer RHS may vary pitch; its minor dimension stays dense.
    const bool pitchedRhs = i == 1 && owner.dynamicAxes && owner.rhsProducers.empty();
    if (!actual.data || reinterpret_cast<uintptr_t>(actual.data) % width ||
        actual.bytes != expected.bytes || actual.dtype != expected.dtype ||
        actual.rank != expected.rank || actual.shape[0] != expected.shape[0] ||
        actual.shape[1] != expected.shape[1] ||
        (!pitchedRhs && expected.shape[0] > 1 && actual.strides[0] != expected.strides[0]) ||
        (!pitchedRhs && expected.rank == 2 && expected.shape[1] > 1 && actual.strides[1] != expected.strides[1]) ||
        (expected.rank == 1 && actual.strides[1] != 0))
      return bad("prepared resident shape/stride/dtype/capacity mismatch");
    spans[i] = actual.bytes;
    if (pitchedRhs) {
      const int minorAxis = owner.rowB ? 1 : 0;
      const int majorAxis = 1 - minorAxis;
      const int64_t minorExtent = actual.shape[minorAxis];
      const int64_t majorExtent = actual.shape[majorAxis];
      const int64_t pitch = actual.strides[majorAxis];
      if ((minorExtent > 1 && actual.strides[minorAxis] != int64_t(width)) ||
          pitch <= 0 || pitch % width || pitch / int64_t(width) < minorExtent)
        return bad("prepared resident RHS pitch or orientation mismatch");
      const __int128 span = (__int128)(majorExtent - 1) * pitch + (__int128)minorExtent * width;
      if (span <= 0 || span > SIZE_MAX || span > INT64_MAX)
        return bad("prepared resident RHS storage span overflow");
      spans[i] = size_t(span);
      rhsLeading = pitch / int64_t(width);
    }
    pointers[i] = reinterpret_cast<CUdeviceptr>(actual.data);
    CUcontext allocationContext = nullptr;
    unsigned int memoryType = 0;
    CUdeviceptr base = 0; size_t capacity = 0;
    if (!ok(cuPointerGetAttribute(&allocationContext, CU_POINTER_ATTRIBUTE_CONTEXT,
          pointers[i]), "resident allocation context") ||
        !ok(cuPointerGetAttribute(&memoryType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
          pointers[i]), "resident allocation type") ||
        allocationContext != context || memoryType != CU_MEMORYTYPE_DEVICE ||
        !ok(cuMemGetAddressRange(&base, &capacity, pointers[i]), "resident allocation capacity") ||
        pointers[i] < base || pointers[i] - base > capacity ||
        spans[i] > capacity - size_t(pointers[i] - base))
      return bad("prepared resident allocation context or capacity mismatch");
    // Read-only inputs may alias. The edge and output must be disjoint from
    // every other live buffer and from each other.
    for (size_t j = 0; j < i; ++j) {
      if (i != owner.count && i != owner.count - 1 &&
          j != owner.count - 1) continue;
      if (pointers[i] < pointers[j] + spans[j] &&
          pointers[j] < pointers[i] + spans[i])
        return bad("prepared resident output/edge aliases live storage");
    }
  }
  // Validate and order every external read after its declared producer.
  // Keep events live through synchronous completion, including launch failure.
  // All allocation/context/shape/alias checks precede event submission.
  struct ProducerEvents {
    std::vector<CUevent> events;
    ~ProducerEvents() { for (CUevent event : events) cuEventDestroy(event); }
  } ordering;
  if (producerStreams) {
    if (!ownedEdge || producerCount != owner.count - 1)
      return bad("ordered resident producer stream arity mismatch");
    std::vector<CUstream> producers;
    for (size_t i = 0; i < producerCount; ++i) {
      if (!producerStreams[i])
        return bad("ordered resident producer stream must be explicit");
      CUstream producer = reinterpret_cast<CUstream>(uintptr_t(producerStreams[i]));
      CUcontext producerContext = nullptr;
      if (!ok(cuStreamGetCtx(producer, &producerContext), "producer stream context") ||
          producerContext != context)
        return bad("ordered resident producer context mismatch");
      if (producer == stream) continue;
      bool duplicate = false;
      for (CUstream prior : producers) duplicate |= prior == producer;
      if (!duplicate) producers.push_back(producer);
    }
    for (CUstream producer : producers) {
      CUevent event = nullptr;
      if (!ok(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "create producer ordering event"))
        return 1;
      ordering.events.push_back(event);
      if (!ok(cuEventRecord(event, producer), "record producer ordering event") ||
          !ok(cuStreamWaitEvent(stream, event, 0), "wait for resident producer"))
        return 1;
    }
  }
  std::array<CUdeviceptr, 5> buffers{};
  for (size_t i = 0; i < owner.count; ++i) buffers[i] = pointers[i];
  owner.invoked = true;
  bool submitted = true;
  if (profileRepeats) {
    CUevent start = nullptr, end = nullptr;
    submitted = ok(cuEventCreate(&start, 0), "create resident program start") &&
                ok(cuEventCreate(&end, 0), "create resident program end");
    if (start) ordering.events.push_back(start);
    if (end) ordering.events.push_back(end);
    submitted = submitted && ok(cuEventRecord(start, stream), "record resident program start");
    for (int i = 0; submitted && i < profileRepeats; ++i)
      submitted = submit(owner, buffers, pointers[owner.count], frame.dims, stream, rhsLeading);
    float elapsed = 0;
    submitted = submitted && ok(cuEventRecord(end, stream), "record resident program end") &&
                ok(cuEventSynchronize(end), "complete resident program window") &&
                ok(cuEventElapsedTime(&elapsed, start, end), "resident program elapsed");
    if (submitted) {
      *programMs = elapsed / profileRepeats;
      submitted = submit(owner, buffers, pointers[owner.count], frame.dims, stream,
                         rhsLeading, profileRepeats, stageTimes);
    }
  } else {
    submitted = submit(owner, buffers, pointers[owner.count], frame.dims, stream, rhsLeading);
  }
  const std::string launchError = error;
  const bool completed = ok(cuStreamSynchronize(stream), "complete resident tensor/matmul");
  if (!submitted) { error = launchError; return 1; }
  if (!completed) return 1;
  if (hostOutput && !ok(cuMemcpyDtoH(hostOutput->data,owner.ownedResult,hostOutput->bytes),
                        "copy completed ordered resident result")) return 1;
  return 0;
}
extern "C" int tessera_nvidia_matmul_invoke_resident(
    uint64_t handle, const TesseraNvidiaMatmulHostView *views, size_t count, void *stream) {
  return invokeResident(handle,views,count,stream,false);
}
extern "C" int tessera_nvidia_matmul_invoke_dag_resident(
    uint64_t handle, const TesseraNvidiaMatmulHostView *views, size_t count, void *stream) {
  return invokeResident(handle,views,count,stream,true);
}
extern "C" int tessera_nvidia_matmul_invoke_dag_resident_ordered(
    uint64_t handle, const TesseraNvidiaMatmulHostView *views, size_t count,
    const uint64_t *producerStreams, size_t producerCount, void *stream) {
  if (!producerStreams || !producerCount) {
    error.clear();
    return bad("ordered resident producer streams are required");
  }
  return invokeResident(handle, views, count, stream, true, producerStreams, producerCount);
}
extern "C" int tessera_nvidia_matmul_invoke_dag_resident_to_host_ordered(
    uint64_t handle, const TesseraNvidiaMatmulHostView *roots, size_t count,
    const uint64_t *producerStreams, size_t producerCount,
    const TesseraNvidiaMatmulHostView *output) {
  if (!producerStreams || !producerCount || !output) {
    error.clear();
    return bad("ordered resident roots, streams and host result are required");
  }
  return invokeResident(handle,roots,count,nullptr,true,producerStreams,producerCount,
                        0,nullptr,0,nullptr,output);
}
extern "C" int tessera_nvidia_matmul_profile_dag_resident_ordered(
    uint64_t handle, const TesseraNvidiaMatmulHostView *views, size_t count,
    const uint64_t *producerStreams, size_t producerCount, void *stream,
    int repeats, float *stageMs, size_t stageCount, float *programMs) {
  if (!producerStreams || !producerCount) {
    error.clear();
    return bad("ordered resident producer streams are required");
  }
  return invokeResident(handle, views, count, stream, true, producerStreams,
                        producerCount, repeats, stageMs, stageCount, programMs);
}
extern "C" int tessera_nvidia_matmul_close(uint64_t handle) {
  error.clear();
  if (getpid() != process) return bad("prepared matmul cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);
  auto found = owners.find(handle);
  if (found == owners.end()) return bad("prepared matmul is closed or unknown");
  owners.erase(found); return 0;
}
extern "C" int tessera_nvidia_matmul_scratch_stats(
    uint64_t handle, size_t *capacity, size_t *allocations) {
  error.clear();
  if (getpid() != process) return bad("prepared matmul cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);
  auto found = owners.find(handle);
  if (found == owners.end() || !capacity || !allocations)
    return bad("invalid prepared matmul scratch query");
  if (!ok(cuStreamQuery(found->second->stream), "query retired matmul stream")) return 1;
  const Owner &owner = *found->second;
  *capacity = owner.arena->capacity + (owner.ownedResult ? owner.bytes[owner.count-1] : 0);
  *allocations = owner.arena->allocations + size_t(bool(owner.ownedResult));
  return 0;
}
