// Synchronous host movement over compiler-generated HSACOs. No kernel source
// or numerical semantics live here. Explicit clear precedes context teardown.
#include <hip/hip_runtime.h>
#include "MovementPhysicalSpan.h"
#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

extern "C" int tessera_rocm_image_acquire(const void *, size_t, const char *,
                                         void **, void **, void **, int *);
extern "C" int tessera_rocm_image_release(void *);

namespace {
using tessera::rocm::pagedPhysicalSpan;
int clearResidentCurrent();
using Identity = std::tuple<int, uintptr_t, std::string>;
constexpr size_t maxRetainedBytes = 128 * 1024 * 1024;
const pid_t process = getpid();
struct Arena {
  std::array<void *, 3> buffers{};
  std::array<size_t, 3> capacities{};
  std::vector<void *> retired;
  void *pendingLease = nullptr;
  bool poisoned = false;
};
struct State {
  std::mutex mutex;
  std::map<Identity, std::unique_ptr<Arena>> arenas;
  uint64_t allocations = 0, frees = 0, reuses = 0, launches = 0;
};
// Process-lived, since HIP may be shut down before static destructors.
State &state() { static auto *value = new State; return *value; }
int identity(Identity &owner) {
  if (getpid() != process) return 2;
  int device = 0;
  hipCtx_t context{};
  hipDeviceProp_t properties{};
  if (hipGetDevice(&device) != hipSuccess ||
      hipCtxGetCurrent(&context) != hipSuccess ||
      hipGetDeviceProperties(&properties, device) != hipSuccess) return 2;
  owner = {device, reinterpret_cast<uintptr_t>(context), properties.gcnArchName};
  return 0;
}
int clearArena(State &s, Arena &arena, bool completed = false) {
  // Completion must be established before releasing either buffers or image.
  if (!completed && hipDeviceSynchronize() != hipSuccess) return 7;
  if (arena.pendingLease) {
    if (tessera_rocm_image_release(arena.pendingLease)) return 8;
    arena.pendingLease = nullptr;
  }
  for (size_t i = 0; i < 3; ++i) {
    if (arena.buffers[i]) {
      if (hipFree(arena.buffers[i]) != hipSuccess) return 9;
      arena.buffers[i] = nullptr;
      arena.capacities[i] = 0;
      ++s.frees;
    }
  }
  while (!arena.retired.empty()) {
    if (hipFree(arena.retired.back()) != hipSuccess) return 9;
    arena.retired.pop_back();
    ++s.frees;
  }
  arena.poisoned = false;
  return 0;
}
bool product(std::initializer_list<int64_t> dimensions, size_t elementBytes,
             size_t &bytes) {
  bytes = elementBytes;
  for (int64_t dimension : dimensions) {
    if (dimension <= 0 || uint64_t(dimension) >
        std::numeric_limits<size_t>::max() / bytes) return false;
    bytes *= size_t(dimension);
  }
  return bytes <= size_t(INT64_MAX);
}
int grow(State &s, Arena &arena, size_t index, size_t bytes) {
  if (arena.buffers[index] && arena.capacities[index] >= bytes) {
    ++s.reuses;
    return 0;
  }
  void *next = nullptr;
  if (hipMalloc(&next, bytes) != hipSuccess) return 4;
  ++s.allocations;
  if (arena.buffers[index] && hipFree(arena.buffers[index]) != hipSuccess) {
    arena.retired.push_back(next);
    arena.poisoned = true;
    return 9;
  }
  if (arena.buffers[index]) ++s.frees;
  arena.buffers[index] = next;
  arena.capacities[index] = bytes;
  return 0;
}
struct Memref {
  void *allocated, *aligned;
  int64_t offset, elements, stride;
};
} // namespace

// family 0: compact paged KV; family 1: MoE gather; family 2: strided paged KV.
// Paged KV uses f32 pages and i32 indices; family 2 appends four element strides.
// Status: 1 request, 2 identity, 3 image lease, 4 allocation, 5 copy,
// 6 launch, 7 completion, 8 lease release, 9 free, 10 quarantined, 12 exception.
// reuse=0 is an independently controlled allocation baseline.
extern "C" int tessera_rocm_movement_launch(
    const void *image, size_t imageBytes, const char *entry, const char *architecture,
    int family, const void *input, size_t inputBytes, const int32_t *indices,
    size_t indexBytes, void *output, size_t outputBytes, const int64_t *dimensions,
    size_t dimensionCount, int reuse) try {
  if (!image || imageBytes < 4 || std::memcmp(image, "\177ELF", 4) ||
      !entry || !*entry || !architecture || !input || !indices || !output ||
      !dimensions || family < 0 || family > 2 || (reuse != 0 && reuse != 1) ||
      dimensionCount != (family == 0 ? 7u : family == 2 ? 11u : 3u)) return 1;
  if (reinterpret_cast<uintptr_t>(input) % alignof(float) ||
      reinterpret_cast<uintptr_t>(indices) % alignof(int32_t) ||
      reinterpret_cast<uintptr_t>(output) % alignof(float) ||
      reinterpret_cast<uintptr_t>(dimensions) % alignof(int64_t)) return 1;
  if (family == 2) {
    uintptr_t source = reinterpret_cast<uintptr_t>(input);
    uintptr_t table = reinterpret_cast<uintptr_t>(indices);
    uintptr_t destination = reinterpret_cast<uintptr_t>(output);
    if (source > UINTPTR_MAX-inputBytes || table > UINTPTR_MAX-indexBytes ||
        destination > UINTPTR_MAX-outputBytes) return 1;
    if ((destination < source+inputBytes && source < destination+outputBytes) ||
        (destination < table+indexBytes && table < destination+outputBytes)) return 1;
  }
  if (getpid() != process) return 2;
  size_t expectedInput = 0, expectedIndices = 0, expectedOutput = 0;
  if (family == 0 || family == 2) {
    auto p=dimensions[0], lp=dimensions[1], page=dimensions[2],
         h=dimensions[3], d=dimensions[4], start=dimensions[5], tokens=dimensions[6];
    size_t capacity = 0;
    if (!product({p,page,h,d},4,expectedInput) ||
        !product({lp},4,expectedIndices) ||
        !product({tokens,h,d},4,expectedOutput) ||
        !product({lp,page},1,capacity) || start < 0 || uint64_t(start) > capacity ||
        uint64_t(tokens) > capacity - size_t(start)) return 1;
    if (family == 2 && !pagedPhysicalSpan(dimensions, expectedInput)) return 1;
    if (inputBytes != expectedInput || indexBytes != expectedIndices ||
        outputBytes != expectedOutput) return 1;
    for (int64_t i=0; i<lp; ++i)
      if (indices[i] < 0 || indices[i] >= p) return 1;
  } else {
    auto t=dimensions[0], slots=dimensions[1], h=dimensions[2];
    if (!product({t,h},4,expectedInput) || !product({slots},4,expectedIndices) ||
        !product({slots,h},4,expectedOutput)) return 1;
    if (inputBytes != expectedInput || indexBytes != expectedIndices ||
        outputBytes != expectedOutput) return 1;
    for (int64_t i=0; i<slots; ++i)
      if (indices[i] < 0 || indices[i] >= t) return 1;
  }
  size_t elements = outputBytes/4;
  if ((elements+255)/256 > INT_MAX ||
      inputBytes > std::numeric_limits<size_t>::max() - indexBytes ||
      inputBytes+indexBytes > std::numeric_limits<size_t>::max()-outputBytes) return 1;
  if (hipInit(0) != hipSuccess || hipDeviceSynchronize() != hipSuccess) return 7;
  Identity owner;
  if (identity(owner)) return 2;
  std::string arch = std::get<2>(owner);
  arch.resize(arch.find(':') == std::string::npos ? arch.size() : arch.find(':'));
  if ((arch != "gfx1151" && arch != "gfx1201") || arch != architecture) return 2;
  State &s = state();
  std::lock_guard<std::mutex> guard(s.mutex);
  auto &slot = s.arenas[owner];
  if (!slot) slot = std::make_unique<Arena>();
  Arena &arena = *slot;
  if (arena.poisoned || arena.pendingLease) return 10;
  bool retain = reuse && inputBytes+indexBytes+outputBytes <= maxRetainedBytes;
  if (!retain) {
    if (int rc = clearArena(s,arena,true)) return rc;
  } else {
    size_t retained = 0;
    for (size_t i=0; i<3; ++i)
      retained += std::max(arena.capacities[i],
                          std::array<size_t,3>{inputBytes,indexBytes,outputBytes}[i]);
    if (retained > maxRetainedBytes)
      if (int rc = clearArena(s,arena,true)) return rc;
  }
  void *lease=nullptr, *module=nullptr, *function=nullptr;
  int hit=0;
  if (tessera_rocm_image_acquire(image,imageBytes,entry,&lease,&module,&function,&hit))
    return 3;
  // Also retain the lease on allocation/vector exceptions; explicit clear
  // must establish completion before recovering this arena.
  arena.pendingLease = lease;
  auto finish = [&](int status, bool completed = false) {
    if (!completed && hipDeviceSynchronize() != hipSuccess) {
      arena.pendingLease = lease;
      arena.poisoned = true;
      return 7;
    }
    if (tessera_rocm_image_release(lease)) {
      arena.pendingLease = lease;
      arena.poisoned = true;
      return 8;
    }
    arena.pendingLease = nullptr;
    if (!retain || arena.poisoned) {
      if (int rc = clearArena(s,arena,true)) return rc;
    }
    return status;
  };
  for (size_t i=0; i<3; ++i)
    if (int rc=grow(s,arena,i,std::array<size_t,3>{inputBytes,indexBytes,outputBytes}[i]))
      return finish(rc);
  if (hipMemcpy(arena.buffers[0],input,inputBytes,hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(arena.buffers[1],indices,indexBytes,hipMemcpyHostToDevice) != hipSuccess)
    return finish(5);
  std::array<Memref,3> refs;
  std::vector<void *> arguments;
  for (size_t i=0; i<3; ++i) {
    refs[i]={arena.buffers[i],arena.buffers[i],0,
             int64_t(std::array<size_t,3>{inputBytes,indexBytes,outputBytes}[i]/4),1};
    arguments.insert(arguments.end(), {&refs[i].allocated,&refs[i].aligned,
        &refs[i].offset,&refs[i].elements,&refs[i].stride});
  }
  for (size_t i=0; i<dimensionCount; ++i)
    arguments.push_back(const_cast<int64_t *>(dimensions+i));
  if (hipModuleLaunchKernel(reinterpret_cast<hipFunction_t>(function),
        unsigned((elements+255)/256),1,1,256,1,1,0,nullptr,arguments.data(),nullptr)
      != hipSuccess) return finish(6);
  ++s.launches;
  if (hipDeviceSynchronize() != hipSuccess) {
    arena.pendingLease=lease;
    arena.poisoned=true;
    return 7;
  }
  if (hipMemcpy(output,arena.buffers[2],outputBytes,hipMemcpyDeviceToHost) != hipSuccess)
    return finish(5,true);
  return finish(0,true);
} catch (...) { return 12; }

// Native math host staging. Compiler descriptors validate semantic roles and
// storage; this service checks physical spans, dimensions and owner identity.
// family 0 unary, 1 binary, 2 last-axis scan; computation/output are f32.
extern "C" int tessera_rocm_math_launch(
    const void *image, size_t imageBytes, const char *entry, const char *architecture,
    int family, int inputElementBytes, const void *lhs, size_t lhsBytes,
    const void *rhs, size_t rhsBytes, void *output, size_t outputBytes,
    const int64_t *dimensions, size_t dimensionCount, int reuse) try {
  if (!image || imageBytes < 4 || std::memcmp(image,"\177ELF",4) ||
      !entry || !*entry || !architecture || family < 0 || family > 2 ||
      (inputElementBytes != 2 && inputElementBytes != 4) ||
      !lhs || !output || !dimensions || (reuse != 0 && reuse != 1) ||
      dimensionCount != (family == 2 ? 2u : 1u) ||
      uintptr_t(dimensions)%alignof(int64_t) || uintptr_t(lhs)%inputElementBytes ||
      uintptr_t(output)%alignof(float) ||
      (family == 1 ? (!rhs || uintptr_t(rhs)%inputElementBytes) : (rhs || rhsBytes))) return 1;
  if (getpid() != process) return 2;
  size_t elements = 0, inputBytes = 0, expectedOutput = 0;
  if (family == 2) {
    if (!product({dimensions[0],dimensions[1]},1,elements)) return 1;
  } else if (!product({dimensions[0]},1,elements)) return 1;
  if (elements > size_t(INT64_MAX)/4) return 1;
  inputBytes = elements*size_t(inputElementBytes); expectedOutput = elements*4;
  if (lhsBytes != inputBytes || outputBytes != expectedOutput ||
      (family == 1 && rhsBytes != inputBytes) ||
      uintptr_t(lhs) > UINTPTR_MAX-lhsBytes ||
      uintptr_t(output) > UINTPTR_MAX-outputBytes ||
      (family == 1 && uintptr_t(rhs) > UINTPTR_MAX-rhsBytes)) return 1;
  auto overlaps = [&](const void *pointer, size_t bytes) {
    return uintptr_t(output) < uintptr_t(pointer)+bytes &&
           uintptr_t(pointer) < uintptr_t(output)+outputBytes;
  };
  if (overlaps(lhs,lhsBytes) || (family == 1 && overlaps(rhs,rhsBytes))) return 1;
  size_t blocks = family == 2 ? size_t(dimensions[0]) : (elements+255)/256;
  if (blocks > INT_MAX || lhsBytes > SIZE_MAX-rhsBytes ||
      lhsBytes+rhsBytes > SIZE_MAX-outputBytes) return 1;
  // Reserve host argument storage before acquiring resources that need cleanup.
  std::vector<void*> arguments;
  arguments.reserve((family == 1 ? 15 : 10)+dimensionCount);
  // Every successful arena call already completes before releasing its lease.
  // A pending/poisoned arena is refused below, so no global pre-launch sync is needed.
  if (hipInit(0) != hipSuccess) return 2;
  Identity owner;
  if (identity(owner)) return 2;
  auto arch=std::get<2>(owner);
  if (auto suffix=arch.find(':'); suffix!=std::string::npos) arch.resize(suffix);
  if ((arch!="gfx1151" && arch!="gfx1201") || arch!=architecture) return 2;
  auto &s=state();
  std::lock_guard<std::mutex> guard(s.mutex);
  auto &slot=s.arenas[owner];
  if (!slot) slot=std::make_unique<Arena>();
  auto &arena=*slot;
  if (arena.poisoned || arena.pendingLease) return 10;
  std::array<size_t,3> bytes={lhsBytes,rhsBytes,outputBytes};
  bool retain=reuse && lhsBytes+rhsBytes+outputBytes<=maxRetainedBytes;
  size_t retained=0;
  for(size_t i=0;i<3;++i) retained+=std::max(arena.capacities[i],bytes[i]);
  if (!retain || retained>maxRetainedBytes)
    if (int rc=clearArena(s,arena,true)) return rc;
  void *lease=nullptr,*module=nullptr,*function=nullptr;
  int hit=0;
  if(tessera_rocm_image_acquire(image,imageBytes,entry,&lease,&module,&function,&hit)) return 3;
  arena.pendingLease=lease;
  auto finish=[&](int status,bool completed=false) {
    if(!completed && hipDeviceSynchronize()!=hipSuccess) {
      arena.poisoned=true;return 7;
    }
    if(tessera_rocm_image_release(lease)) {arena.poisoned=true;return 8;}
    arena.pendingLease=nullptr;
    if(!retain || arena.poisoned)
      if(int rc=clearArena(s,arena,true)) return rc;
    return status;
  };
  for(size_t i=0;i<3;++i)
    if(bytes[i]) if(int rc=grow(s,arena,i,bytes[i])) return finish(rc);
  if(hipMemcpy(arena.buffers[0],lhs,lhsBytes,hipMemcpyHostToDevice)!=hipSuccess ||
     (family==1 && hipMemcpy(arena.buffers[1],rhs,rhsBytes,hipMemcpyHostToDevice)!=hipSuccess))
    return finish(5);
  std::array<Memref,3> refs;
  for(size_t i=0;i<3;++i) if(bytes[i]) {
    refs[i]={arena.buffers[i],arena.buffers[i],0,
             int64_t(bytes[i]/(i==2?4:inputElementBytes)),1};
    auto &ref=refs[i];
    arguments.insert(arguments.end(),{&ref.allocated,&ref.aligned,&ref.offset,&ref.elements,&ref.stride});
  }
  for(size_t i=0;i<dimensionCount;++i) arguments.push_back(const_cast<int64_t*>(dimensions+i));
  if(hipModuleLaunchKernel(reinterpret_cast<hipFunction_t>(function),unsigned(blocks),1,1,
      256,1,1,0,nullptr,arguments.data(),nullptr)!=hipSuccess) return finish(6);
  ++s.launches;
  if(hipDeviceSynchronize()!=hipSuccess) {arena.poisoned=true;return 7;}
  if(hipMemcpy(output,arena.buffers[2],outputBytes,hipMemcpyDeviceToHost)!=hipSuccess)
    return finish(5,true);
  return finish(0,true);
} catch (...) { return 12; }

extern "C" int tessera_rocm_movement_clear_current() try {
  if (int status=clearResidentCurrent()) return status;
  Identity owner;
  if (identity(owner)) return 2;
  State &s=state();
  std::lock_guard<std::mutex> guard(s.mutex);
  auto found=s.arenas.find(owner);
  if (found==s.arenas.end()) return 0;
  int rc=clearArena(s,*found->second);
  if (!rc) s.arenas.erase(found);
  return rc;
} catch (...) { return 12; }

extern "C" int tessera_rocm_movement_stats(uint64_t *allocations,uint64_t *frees,
                                           uint64_t *reuses,uint64_t *launches) try {
  if (!allocations || !frees || !reuses || !launches || getpid()!=process) return 1;
  State &s=state();
  std::lock_guard<std::mutex> guard(s.mutex);
  *allocations=s.allocations; *frees=s.frees;
  *reuses=s.reuses; *launches=s.launches;
  return 0;
} catch (...) { return 12; }


namespace {
struct PreparedMovement {
  Identity owner;
  std::vector<unsigned char> image;
  std::string entry, architecture;
  int family;
  std::vector<int64_t> dimensions;
  std::array<std::vector<int64_t>,3> shapes;
  std::array<size_t,3> bytes;
};
struct PreparedState {
  std::mutex mutex;
  uint64_t next = 1;
  std::map<uint64_t,std::shared_ptr<const PreparedMovement>> calls;
};
PreparedState &preparedState() { static auto *value=new PreparedState; return *value; }
bool preparedShape(PreparedMovement &call) {
  auto &d=call.dimensions;
  if (call.family==0 || call.family==2) {
    if (d.size()!=(call.family==2?11u:7u)) return false;
    call.shapes={std::vector<int64_t>{d[0],d[2],d[3],d[4]},
                 std::vector<int64_t>{d[1]},
                 std::vector<int64_t>{d[6],d[3],d[4]}};
    size_t capacity=0;
    if (!product({d[1],d[2]},1,capacity) || d[5]<0 ||
        uint64_t(d[5])>capacity || d[6]<=0 ||
        uint64_t(d[6])>capacity-size_t(d[5])) return false;
  } else {
    if (call.family!=1 || d.size()!=3 || call.architecture!="gfx1151") return false;
    call.shapes={std::vector<int64_t>{d[0],d[2]},
                 std::vector<int64_t>{d[1]},
                 std::vector<int64_t>{d[1],d[2]}};
  }
  for (size_t i=0;i<3;++i) {
    size_t size=4;
    for (int64_t dim:call.shapes[i]) {
      if (dim<=0 || uint64_t(dim)>size_t(INT64_MAX)/size) return false;
      size*=size_t(dim);
    }
    call.bytes[i]=size;
  }
  if (call.family==2 && !pagedPhysicalSpan(d.data(),call.bytes[0])) return false;
  return (call.bytes[2]/4+255)/256<=INT_MAX;
}
} // namespace

// Fixed host view ABI: metadata is checked natively before any pointer read.
struct TesseraMovementHostView {
  void *data;
  size_t bytes;
  int32_t dtype; // 1 native f32, 2 native i32, all other codes refused.
  int32_t rank;
  int64_t shape[4], strides[4]; // Byte strides; ranks are at most four.
};

// Preparation copies the compiler image, entry and sealed static ABI.
// Handles are monotonically assigned IDs, never caller-dereferenced pointers.
extern "C" int tessera_rocm_movement_prepare(
    const void *image,size_t imageBytes,const char *entry,const char *architecture,
    int family,const int64_t *dimensions,size_t dimensionCount,uint64_t *handle) try {
  if (!handle || uintptr_t(handle)%alignof(uint64_t)) return 1;
  *handle=0;
  if (!image || imageBytes<4 || std::memcmp(image,"\177ELF",4) ||
      !entry || !*entry || !architecture || !dimensions ||
      (family<0 || family>2) ||
      dimensionCount!=(family==0?7u:family==2?11u:3u) ||
      reinterpret_cast<uintptr_t>(dimensions)%alignof(int64_t) ||
      getpid()!=process) return 1;
  auto call=std::make_shared<PreparedMovement>();
  call->entry=entry; call->architecture=architecture; call->family=family;
  if (call->architecture!="gfx1151" && call->architecture!="gfx1201") return 1;
  call->dimensions.assign(dimensions,dimensions+dimensionCount);
  if (!preparedShape(*call)) return 1;
  if (hipInit(0)!=hipSuccess || hipDeviceSynchronize()!=hipSuccess) return 7;
  if (identity(call->owner)) return 2;
  auto arch=std::get<2>(call->owner);
  if (auto suffix=arch.find(':');suffix!=std::string::npos) arch.resize(suffix);
  if (arch!=call->architecture) return 2;
  call->image.assign(static_cast<const unsigned char*>(image),
                     static_cast<const unsigned char*>(image)+imageBytes);
  auto &s=preparedState();
  std::lock_guard<std::mutex> guard(s.mutex);
  if (!s.next || s.next==UINT64_MAX) return 12;
  auto id=s.next;
  s.calls.emplace(id,std::move(call));
  ++s.next;
  *handle=id;
  return 0;
} catch (...) { return 12; }

extern "C" int tessera_rocm_movement_invoke(
    uint64_t handle,const TesseraMovementHostView *views,size_t viewCount,int reuse) try {
  if (!handle || !views || uintptr_t(views)%alignof(TesseraMovementHostView) ||
      viewCount!=3 || getpid()!=process ||
      (reuse!=0 && reuse!=1)) return 1;
  std::shared_ptr<const PreparedMovement> call;
  {
    auto &s=preparedState();
    std::lock_guard<std::mutex> guard(s.mutex);
    auto found=s.calls.find(handle);
    if (found==s.calls.end()) return 1;
    call=found->second;
  }
  for (size_t i=0;i<3;++i) {
    const auto &v=views[i]; const auto &shape=call->shapes[i];
    if (!v.data || uintptr_t(v.data)%4 || v.bytes!=call->bytes[i] ||
        v.dtype!=(i==1?2:1) || v.rank!=int32_t(shape.size()) ||
        uintptr_t(v.data)>UINTPTR_MAX-v.bytes) return 1;
    int64_t stride=4;
    for (int j=v.rank-1;j>=0;--j) {
      if (v.shape[j]!=shape[j]) return 1;
      if (i==0 && call->family==2) {
        if (uint64_t(call->dimensions[7+j])>uint64_t(INT64_MAX)/4 ||
            v.strides[j]!=call->dimensions[7+j]*4) return 1;
      } else if (shape[j]>1 && v.strides[j]!=stride) return 1;
      stride*=shape[j]; // preparedShape already checked the byte product.
    }
    for (size_t j=0;i==2 && j<i;++j)
      if (uintptr_t(v.data)<uintptr_t(views[j].data)+views[j].bytes &&
          uintptr_t(views[j].data)<uintptr_t(v.data)+v.bytes) return 1;
  }
  Identity owner;
  if (identity(owner) || owner!=call->owner) return 2;
  // shared_ptr retains the copied image/ABI through concurrent close.
  return tessera_rocm_movement_launch(call->image.data(),call->image.size(),
      call->entry.c_str(),call->architecture.c_str(),call->family,
      views[0].data,views[0].bytes,static_cast<const int32_t*>(views[1].data),
      views[1].bytes,views[2].data,views[2].bytes,call->dimensions.data(),
      call->dimensions.size(),reuse);
} catch (...) { return 12; }

extern "C" int tessera_rocm_movement_close(uint64_t handle) try {
  if (!handle || getpid()!=process) return 1;
  auto &s=preparedState();
  std::lock_guard<std::mutex> guard(s.mutex);
  return s.calls.erase(handle)==1?0:1;
} catch (...) { return 12; }

// Resident execution borrows no external device pointers. The compiler package
// fixes the ABI; this service owns allocation, stream and image lifetime.
namespace {
struct ResidentMovement {
  ~ResidentMovement();
  std::shared_ptr<const PreparedMovement> call;
  std::mutex mutex;
  std::array<void *,4> buffers{};
  std::array<Memref,3> refs;
  std::vector<void*> arguments;
  void *lease=nullptr, *module=nullptr, *function=nullptr;
  hipStream_t stream=nullptr;
  hipEvent_t begin=nullptr, end=nullptr;
  hipGraph_t graph=nullptr;
  hipGraphExec_t graphExec=nullptr;
  bool capturedReady=false;
  void *consumerLease=nullptr, *consumerModule=nullptr, *consumerFunction=nullptr;
  hipEvent_t consumerBegin=nullptr, consumerEnd=nullptr;
  std::array<Memref,2> consumerRefs;
  std::vector<void*> consumerArguments;
  int64_t rows=0, columns=0;
  bool ready=false, closing=false, published=false;
  uint64_t serial=0, outputGeneration=0;
};
struct ResidentState {
  std::mutex mutex;
  uint64_t next=1;
  std::map<uint64_t,std::shared_ptr<ResidentMovement>> calls;
};
ResidentState &residentState() { static auto *s=new ResidentState; return *s; }
std::shared_ptr<ResidentMovement> residentLookup(uint64_t handle) {
  if (!handle || getpid()!=process) return {};
  auto &s=residentState();
  std::lock_guard<std::mutex> guard(s.mutex);
  auto found=s.calls.find(handle);
  return found==s.calls.end() || !found->second->published?nullptr:found->second;
}
bool residentContext(const ResidentMovement &r) {
  if (getpid()!=process) return false;
  int device=0; hipCtx_t context{};
  return hipGetDevice(&device)==hipSuccess &&
         hipCtxGetCurrent(&context)==hipSuccess &&
         device==std::get<0>(r.call->owner) &&
         uintptr_t(context)==std::get<1>(r.call->owner);
}
bool residentView(const ResidentMovement &r,const TesseraMovementHostView &v,size_t role) {
  const auto &shape=r.call->shapes[role];
  if (!v.data || uintptr_t(v.data)%4 || v.bytes!=r.call->bytes[role] ||
      v.dtype!=(role==1?2:1) || v.rank!=int32_t(shape.size()) ||
      uintptr_t(v.data)>UINTPTR_MAX-v.bytes) return false;
  int64_t stride=4;
  for (int j=v.rank-1;j>=0;--j) {
    if (v.shape[j]!=shape[j]) return false;
    if (role==0 && r.call->family==2) {
      if (uint64_t(r.call->dimensions[7+j])>uint64_t(INT64_MAX)/4 ||
          v.strides[j]!=r.call->dimensions[7+j]*4) return false;
    } else if (shape[j]>1 && v.strides[j]!=stride) return false;
    stride*=shape[j];
  }
  return true;
}
int residentRelease(ResidentMovement &r) {
  r.closing=true; r.ready=false; r.outputGeneration=0;
  if (r.stream && hipStreamSynchronize(r.stream)!=hipSuccess) return 7;
  r.capturedReady=false;
  // Executables retain pointers, module functions and events. Retire graph
  // resources before releasing those dependencies; failed cleanup is retryable.
  if (r.graphExec) {
    if (hipGraphExecDestroy(r.graphExec)!=hipSuccess) return 9;
    r.graphExec=nullptr;
  }
  if (r.graph) {
    if (hipGraphDestroy(r.graph)!=hipSuccess) return 9;
    r.graph=nullptr;
  }
  if (r.begin) {
    if (hipEventDestroy(r.begin)!=hipSuccess) return 9;
    r.begin=nullptr;
  }
  if (r.end) {
    if (hipEventDestroy(r.end)!=hipSuccess) return 9;
    r.end=nullptr;
  }
  if (r.consumerBegin) {
    if (hipEventDestroy(r.consumerBegin)!=hipSuccess) return 9;
    r.consumerBegin=nullptr;
  }
  if (r.consumerEnd) {
    if (hipEventDestroy(r.consumerEnd)!=hipSuccess) return 9;
    r.consumerEnd=nullptr;
  }
  for (auto &buffer:r.buffers) if (buffer) {
    if (hipFree(buffer)!=hipSuccess) return 9;
    buffer=nullptr;
  }
  if (r.stream) {
    if (hipStreamDestroy(r.stream)!=hipSuccess) return 9;
    r.stream=nullptr;
  }
  if (r.lease) {
    if (tessera_rocm_image_release(r.lease)) return 8;
    r.lease=nullptr;
  }
  if (r.consumerLease) {
    if (tessera_rocm_image_release(r.consumerLease)) return 8;
    r.consumerLease=nullptr;
  }
  return 0;
}
ResidentMovement::~ResidentMovement() {
  if (call && residentContext(*this)) residentRelease(*this);
}
int clearResidentCurrent() {
  if (getpid()!=process) return 2;
  auto &s=residentState();
  std::vector<std::pair<uint64_t,std::shared_ptr<ResidentMovement>>> owned;
  {
    std::lock_guard<std::mutex> guard(s.mutex);
    for (auto &entry:s.calls) if(residentContext(*entry.second)) owned.push_back(entry);
  }
  for (auto &entry:owned) {
    std::lock_guard<std::mutex> guard(entry.second->mutex);
    if(int status=residentRelease(*entry.second)) return status;
    std::lock_guard<std::mutex> stateGuard(s.mutex);
    s.calls.erase(entry.first);
  }
  return 0;
}
}
extern "C" int tessera_rocm_movement_resident_prepare(
    uint64_t preparedHandle,uint64_t *handle) try {
  if (!handle || uintptr_t(handle)%alignof(uint64_t) || getpid()!=process) return 1;
  *handle=0;
  auto r=std::make_shared<ResidentMovement>();
  {
    auto &s=preparedState(); std::lock_guard<std::mutex> guard(s.mutex);
    auto found=s.calls.find(preparedHandle);
    if (found==s.calls.end()) return 1;
    r->call=found->second;
  }
  if (!residentContext(*r)) return 2;
  std::lock_guard<std::mutex> builderGuard(r->mutex);
  // Reserve all CPU-owned argument storage before acquiring GPU resources.
  r->arguments.reserve(15+r->call->dimensions.size());
  auto &s=residentState();
  uint64_t id=0;
  {
    std::lock_guard<std::mutex> guard(s.mutex);
    if (!s.next || s.next==UINT64_MAX) return 12;
    id=s.next++;
    s.calls.emplace(id,r);
  }
  int hit=0;
  if (tessera_rocm_image_acquire(r->call->image.data(),r->call->image.size(),
      r->call->entry.c_str(),&r->lease,&r->module,&r->function,&hit)) {
    std::lock_guard<std::mutex> guard(s.mutex);s.calls.erase(id);return 3;
  }
  int status=0;
  if (hipStreamCreateWithFlags(&r->stream,hipStreamNonBlocking)!=hipSuccess) status=4;
  for (size_t i=0;!status && i<3;++i)
    if (hipMalloc(&r->buffers[i],r->call->bytes[i])!=hipSuccess) status=4;
  if (!status && (hipEventCreate(&r->begin)!=hipSuccess || hipEventCreate(&r->end)!=hipSuccess)) status=4;
  // Failed completion/cleanup retains a quarantined owner for explicit retry.
  if (status) {
    if (residentRelease(*r)) {std::lock_guard<std::mutex> guard(s.mutex);r->published=true;*handle=id;}
    else {std::lock_guard<std::mutex> guard(s.mutex);s.calls.erase(id);}
    return status;
  }
  for (size_t i=0;i<3;++i) {
    r->refs[i]={r->buffers[i],r->buffers[i],0,int64_t(r->call->bytes[i]/4),1};
    auto &ref=r->refs[i];
    r->arguments.insert(r->arguments.end(),{&ref.allocated,&ref.aligned,&ref.offset,&ref.elements,&ref.stride});
  }
  for (auto &d:r->call->dimensions) r->arguments.push_back(const_cast<int64_t*>(&d));
  {std::lock_guard<std::mutex> guard(s.mutex);r->published=true;*handle=id;}
  return 0;
} catch (...) { return 12; }

extern "C" int tessera_rocm_movement_resident_upload(
    uint64_t handle,const TesseraMovementHostView *views,size_t count) try {
  if (!views || uintptr_t(views)%alignof(TesseraMovementHostView) || count!=2 || getpid()!=process) return 1;
  auto r=residentLookup(handle);if (!r) return 1;
  std::lock_guard<std::mutex> guard(r->mutex);
  if (!residentContext(*r)) return 2;
  if (r->closing) return 10;
  if (!residentView(*r,views[0],0) || !residentView(*r,views[1],1)) return 1;
  auto indices=static_cast<const int32_t*>(views[1].data);
  auto bound=r->call->dimensions[0];
  for (size_t i=0;i<r->call->bytes[1]/4;++i)
    if (indices[i]<0 || indices[i]>=bound) return 1;
  r->ready=false;
  for (size_t i=0;i<2;++i)
    if (hipMemcpyAsync(r->buffers[i],views[i].data,views[i].bytes,
                      hipMemcpyHostToDevice,r->stream)!=hipSuccess) {
      if (hipStreamSynchronize(r->stream)!=hipSuccess) r->closing=true;
      return 5;
    }
  if (hipStreamSynchronize(r->stream)!=hipSuccess) {r->closing=true;return 7;}
  r->ready=true;return 0;
} catch (...) { return 12; }

namespace {
hipError_t residentEnqueue(ResidentMovement &r,bool timed=true) {
  auto status=timed?hipEventRecord(r.begin,r.stream):hipSuccess;
  auto elements=r.call->bytes[2]/4;
  if (status==hipSuccess)
    status=hipModuleLaunchKernel(reinterpret_cast<hipFunction_t>(r.function),
      unsigned((elements+255)/256),1,1,256,1,1,0,r.stream,r.arguments.data(),nullptr);
  if (status==hipSuccess && timed) status=hipEventRecord(r.end,r.stream);
  if (status==hipSuccess && r.consumerLease) {
    if (timed) status=hipEventRecord(r.consumerBegin,r.stream);
    if (status==hipSuccess)
      status=hipModuleLaunchKernel(reinterpret_cast<hipFunction_t>(r.consumerFunction),
        unsigned(r.rows),1,1,256,1,1,0,r.stream,r.consumerArguments.data(),nullptr);
    if (status==hipSuccess && timed) status=hipEventRecord(r.consumerEnd,r.stream);
  }
  return status;
}
int residentInvoke(uint64_t handle,uint64_t *generation,float *kernelMs,
                   float *producerMs,float *consumerMs,bool captured=false) {
  if (!generation || uintptr_t(generation)%alignof(uint64_t) ||
      (kernelMs && uintptr_t(kernelMs)%alignof(float)) ||
      (producerMs && uintptr_t(producerMs)%alignof(float)) ||
      (consumerMs && uintptr_t(consumerMs)%alignof(float)) ||
      getpid()!=process) return 1;
  *generation=0;if(kernelMs)*kernelMs=0;
  if(producerMs)*producerMs=0;if(consumerMs)*consumerMs=0;
  auto r=residentLookup(handle);if (!r) return 1;
  std::lock_guard<std::mutex> guard(r->mutex);
  if (!residentContext(*r)) return 2;
  if (!r->ready || r->closing || r->serial==UINT64_MAX) return 10;
  if (consumerMs && !r->consumerLease && !captured) return 10;
  if (captured && !r->capturedReady) return 10;
  r->outputGeneration=0;
  hipError_t status=hipSuccess;
  if (captured) {
    status=hipEventRecord(r->begin,r->stream);
    if (status==hipSuccess) status=hipGraphLaunch(r->graphExec,r->stream);
    if (status==hipSuccess) status=hipEventRecord(r->end,r->stream);
  } else status=residentEnqueue(*r);
  if (status!=hipSuccess) {
    if (hipStreamSynchronize(r->stream)!=hipSuccess) r->closing=true;
    return 6;
  }
  if (hipStreamSynchronize(r->stream)!=hipSuccess) {r->closing=true;return 7;}
  float producer=0,consumer=0;
  if (hipEventElapsedTime(&producer,r->begin,r->end)!=hipSuccess || !(producer>0)) return 7;
  if (r->consumerLease && !captured &&
      (hipEventElapsedTime(&consumer,r->consumerBegin,r->consumerEnd)!=hipSuccess || !(consumer>0))) return 7;
  r->outputGeneration=++r->serial;*generation=r->outputGeneration;
  if(kernelMs)*kernelMs=producer+consumer;
  if(producerMs)*producerMs=producer;if(consumerMs)*consumerMs=consumer;
  return 0;
}
}
extern "C" int tessera_rocm_movement_resident_invoke(
    uint64_t handle,uint64_t *generation,float *kernelMs) try {
  return residentInvoke(handle,generation,kernelMs,nullptr,nullptr);
} catch (...) { return 12; }

extern "C" int tessera_rocm_movement_resident_invoke_softmax(
    uint64_t handle,uint64_t *generation,float *producerMs,float *consumerMs) try {
  if (!producerMs || !consumerMs) return 1;
  return residentInvoke(handle,generation,nullptr,producerMs,consumerMs);
} catch (...) { return 12; }

// Capture the compiler-owned ABI and current private addresses once. Uploads
// replace contents, never addresses. All replay and cleanup share the same
// owner lock, context check, module leases and synchronous completion policy.
extern "C" int tessera_rocm_movement_resident_capture(
    uint64_t handle,uint64_t *kernelNodes) try {
  if (!kernelNodes || uintptr_t(kernelNodes)%alignof(uint64_t) ||
      getpid()!=process) return 1;
  *kernelNodes=0;
  auto r=residentLookup(handle);if (!r) return 1;
  std::lock_guard<std::mutex> guard(r->mutex);
  if (!residentContext(*r)) return 2;
  if (!r->ready || r->closing) return 10;
  const size_t expected=r->consumerLease?2:1;
  if (r->capturedReady) {*kernelNodes=expected;return 0;}
  // A failed prior construction stays owned for explicit close/cleanup retry.
  if (r->graph || r->graphExec) return 10;
  if (hipStreamBeginCapture(r->stream,hipStreamCaptureModeThreadLocal)!=hipSuccess) return 6;
  auto submitted=residentEnqueue(*r,false);
  auto ended=hipStreamEndCapture(r->stream,&r->graph);
  if (submitted!=hipSuccess || ended!=hipSuccess || !r->graph) {
    r->closing=true;return 6;
  }
  size_t count=0;
  if (hipGraphGetNodes(r->graph,nullptr,&count)!=hipSuccess ||
      count!=expected) {r->closing=true;return 6;}
  std::vector<hipGraphNode_t> nodes(count);
  if (hipGraphGetNodes(r->graph,nodes.data(),&count)!=hipSuccess) {r->closing=true;return 6;}
  size_t kernels=0;
  for (auto node:nodes) {
    hipGraphNodeType type{};
    if (hipGraphNodeGetType(node,&type)!=hipSuccess) {r->closing=true;return 6;}
    if (type==hipGraphNodeTypeKernel) ++kernels;
    else {r->closing=true;return 6;}
  }
  if (kernels!=expected ||
      hipGraphInstantiateWithFlags(&r->graphExec,r->graph,0)!=hipSuccess) {
    r->closing=true;return 6;
  }
  r->capturedReady=true;*kernelNodes=kernels;return 0;
} catch (...) { return 12; }

extern "C" int tessera_rocm_movement_resident_invoke_captured(
    uint64_t handle,uint64_t *generation,float *kernelMs) try {
  if (!kernelMs) return 1;
  return residentInvoke(handle,generation,kernelMs,nullptr,nullptr,true);
} catch (...) { return 12; }

extern "C" int tessera_rocm_movement_resident_read(
    uint64_t handle,uint64_t generation,const TesseraMovementHostView *output) try {
  if (!output || uintptr_t(output)%alignof(TesseraMovementHostView) || getpid()!=process) return 1;
  auto r=residentLookup(handle);if (!r) return 1;
  std::lock_guard<std::mutex> guard(r->mutex);
  if (!residentContext(*r)) return 2;
  if (r->closing || !generation || generation!=r->outputGeneration) return 10;
  if (!residentView(*r,*output,2)) return 1;
  if (hipMemcpyAsync(output->data,r->buffers[r->consumerLease?3:2],output->bytes,
      hipMemcpyDeviceToHost,r->stream)!=hipSuccess) {
    if (hipStreamSynchronize(r->stream)!=hipSuccess) r->closing=true;
    return 5;
  }
  if (hipStreamSynchronize(r->stream)!=hipSuccess) {r->closing=true;return 7;}
  return 0;
} catch (...) { return 12; }

extern "C" int tessera_rocm_movement_resident_close(uint64_t handle) try {
  if (getpid()!=process) return 1;
  auto r=residentLookup(handle);if (!r) return 1;
  std::lock_guard<std::mutex> guard(r->mutex);
  if (!residentContext(*r)) return 2;
  if (int status=residentRelease(*r)) return status;
  auto &s=residentState();std::lock_guard<std::mutex> stateGuard(s.mutex);
  return s.calls.erase(handle)==1?0:1;
} catch (...) { return 12; }


// A named f32 last-axis softmax edge consumes the movement allocation directly.
// The native runtime owns both image leases and the entire allocation lifetime.
extern "C" int tessera_rocm_movement_resident_prepare_softmax(
    uint64_t preparedHandle,const void *image,size_t imageBytes,const char *entry,
    int64_t rows,int64_t columns,uint64_t *handle) try {
  if (!handle || uintptr_t(handle)%alignof(uint64_t) || !image || !imageBytes ||
      !entry || !*entry || rows<=0 || rows>UINT32_MAX || columns<=0 ||
      uint64_t(rows)>UINT64_MAX/uint64_t(columns) || getpid()!=process) return 1;
  *handle=0;
  uint64_t id=0;
  int status=tessera_rocm_movement_resident_prepare(preparedHandle,&id);
  if (status) {*handle=id;return status;}
  auto r=residentLookup(id);
  if (!r) return 1;
  {
    std::lock_guard<std::mutex> guard(r->mutex);
    if (r->closing || !residentContext(*r)) status=2;
    // This first edge is paged read, preserving its last axis and full extent.
    else if ((r->call->family!=0 && r->call->family!=2) || r->call->shapes[2].back()!=columns ||
             uint64_t(rows)*uint64_t(columns)!=r->call->bytes[2]/4) status=1;
    else try {
      r->consumerArguments.reserve(12);
      r->rows=rows;r->columns=columns;
      int hit=0;
      if (tessera_rocm_image_acquire(image,imageBytes,entry,
          &r->consumerLease,&r->consumerModule,&r->consumerFunction,&hit)) status=3;
      if (!status && hipMalloc(&r->buffers[3],r->call->bytes[2])!=hipSuccess) status=4;
      if (!status && (hipEventCreate(&r->consumerBegin)!=hipSuccess ||
                     hipEventCreate(&r->consumerEnd)!=hipSuccess)) status=4;
      if (!status) {
        for (size_t i=0;i<2;++i) {
          r->consumerRefs[i]={r->buffers[i+2],r->buffers[i+2],0,
                             int64_t(r->call->bytes[2]/4),1};
          auto &ref=r->consumerRefs[i];
          r->consumerArguments.insert(r->consumerArguments.end(),
            {&ref.allocated,&ref.aligned,&ref.offset,&ref.elements,&ref.stride});
        }
        r->consumerArguments.insert(r->consumerArguments.end(),{&r->rows,&r->columns});
      }
    } catch (...) {status=12;}
  }
  if (status) {
    if (tessera_rocm_movement_resident_close(id)) *handle=id;
    return status;
  }
  *handle=id;return 0;
} catch (...) { return 12; }
