#pragma once
#include <atomic>
// Included in matmul_prepared.cpp after shared ownership/ordering helpers.
// Interprets compiler-projected dataflow and launch arguments, never op names.
namespace {
std::atomic<uint64_t> nativeFrameNextThread{1};
uint64_t nativeFrameThreadIdentity() {
  thread_local const uint64_t identity=nativeFrameNextThread.fetch_add(1,std::memory_order_relaxed);
  return identity;
}
struct NativeFrameStage {
  CUfunction function = nullptr;
  uint32_t inputCount = 0, scalarCount = 0;
  std::array<uint32_t,9> buffers{};
  std::array<int64_t,8> scalars{};
  std::array<uint32_t,3> grid{}, block{};
};
struct NativeFrameOwner {
  CUcontext context = nullptr;
  const uint64_t thread=nativeFrameThreadIdentity();
  unsigned long long identity = 0;
  CUstream stream = nullptr;
  std::shared_ptr<Scratch> arena;
  std::vector<TesseraNvidiaMatmulHostView> descriptors;
  std::vector<uint32_t> roots, outputs;
  std::vector<size_t> offsets;
  std::vector<CUmodule> modules;
  std::vector<NativeFrameStage> stages;
  std::vector<CUdeviceptr> bindings;
  std::vector<unsigned long long> allocationIds;
  std::vector<uint64_t> producerStreams;
  size_t bytes = 0;
  uint64_t generation = 0;
  bool resident = false, poisoned = false;
  ~NativeFrameOwner() {
    if (getpid()!=process || !context || cuCtxPushCurrent(context)!=CUDA_SUCCESS) return;
    unsigned long long live=0;
    if (cuCtxGetId(context,&live)==CUDA_SUCCESS && live==identity) {
      if (stream) cuStreamSynchronize(stream);
      if (stream) cuStreamDestroy(stream);
      for (CUmodule module : modules) if (module) cuModuleUnload(module);
    }
    CUcontext previous=nullptr;cuCtxPopCurrent(&previous);
  }
};
std::map<uint64_t,std::unique_ptr<NativeFrameOwner>> nativeFrames;
bool nativeFrameContext(NativeFrameOwner &owner) {
  if (nativeFrameThreadIdentity()!=owner.thread)
    return bad("native frame thread changed"),false;
  if (owner.poisoned) return bad("native frame requires context teardown after failed completion"),false;
  CUcontext active=nullptr;unsigned long long live=0;
  if (!ok(cuCtxGetCurrent(&active),"get native frame context")) return false;
  if (!active && !ok(cuCtxSetCurrent(owner.context),"restore native frame context")) return false;
  if ((active && active!=owner.context) ||
      !ok(cuCtxGetId(owner.context,&live),"native frame context identity") || live!=owner.identity)
    return bad("native frame context changed"),false;
  return true;
}
bool nativeFrameDrain(NativeFrameOwner &owner, bool status) {
  const std::string saved=error;
  bool completed=ok(cuStreamSynchronize(owner.stream),"complete native frame");
  if (!completed) owner.poisoned=true;
  if (!status) error=saved;
  return status && completed;
}
bool nativeFrameView(const TesseraNvidiaMatmulHostView &actual,
                     const TesseraNvidiaMatmulHostView &expected) {
  size_t width=expected.dtype==1?4:2;
  return actual.data && reinterpret_cast<uintptr_t>(actual.data)%width==0 &&
      actual.bytes==expected.bytes && actual.dtype==expected.dtype &&
      actual.rank==2 && actual.shape[0]==expected.shape[0] &&
      actual.shape[1]==expected.shape[1] &&
      (actual.shape[0]==1 || actual.strides[0]==expected.strides[0]) &&
      (actual.shape[1]==1 || actual.strides[1]==expected.strides[1]) &&
      reinterpret_cast<uintptr_t>(actual.data)<=UINTPTR_MAX-actual.bytes;
}
bool nativeFrameOverlap(uintptr_t a,size_t ab,uintptr_t b,size_t bb) {
  return a<b+bb && b<a+ab;
}
bool nativeFrameViews(NativeFrameOwner &owner,
    const TesseraNvidiaMatmulHostView *roots,size_t rootCount,
    const TesseraNvidiaMatmulHostView *outputs,size_t outputCount,bool resident) {
  if (!roots || !outputs || rootCount!=owner.roots.size() || outputCount!=owner.outputs.size())
    return bad("native frame root/output count differs"),false;
  for (size_t i=0;i<rootCount;++i)
    if (!nativeFrameView(roots[i],owner.descriptors[owner.roots[i]]))
      return bad("native frame root shape/stride/storage differs"),false;
  for (size_t i=0;i<outputCount;++i) {
    if (!nativeFrameView(outputs[i],owner.descriptors[owner.outputs[i]]))
      return bad("native frame output shape/stride/storage differs"),false;
    uintptr_t a=reinterpret_cast<uintptr_t>(outputs[i].data);
    if (!resident) for (size_t j=0;j<rootCount;++j)
      if (nativeFrameOverlap(a,outputs[i].bytes,
          reinterpret_cast<uintptr_t>(roots[j].data),roots[j].bytes))
        return bad("native frame host output aliases a root"),false;
    for (size_t j=0;j<i;++j)
      if (nativeFrameOverlap(a,outputs[i].bytes,
          reinterpret_cast<uintptr_t>(outputs[j].data),outputs[j].bytes))
        return bad("native frame outputs overlap"),false;
  }
  return true;
}
bool submitNativeFrameStage(NativeFrameOwner &owner,size_t index) {
  auto &stage=owner.stages[index];
  std::array<CUdeviceptr,9> pointers{};
  std::array<void *,17> arguments{};
  size_t cursor=0;
  for (size_t i=0;i<=stage.inputCount;++i) {
    pointers[i]=owner.bindings[stage.buffers[i]];
    arguments[cursor++]=&pointers[i];
  }
  for (size_t i=0;i<stage.scalarCount;++i) arguments[cursor++]=&stage.scalars[i];
  return ok(cuLaunchKernel(stage.function,stage.grid[0],stage.grid[1],stage.grid[2],
      stage.block[0],stage.block[1],stage.block[2],0,owner.stream,arguments.data(),nullptr),
      "launch native program member");
}
bool submitNativeFrame(NativeFrameOwner &owner) {
  for (size_t i=0;i<owner.stages.size();++i)
    if (!submitNativeFrameStage(owner,i)) return false;
  return true;
}
int invokeNativeFrame(uint64_t handle,const TesseraNvidiaMatmulHostView *roots,size_t rootCount,
    const uint64_t *streams,size_t producerCount,
    const TesseraNvidiaMatmulHostView *outputs,size_t outputCount) {
  auto found=nativeFrames.find(handle);
  if (found==nativeFrames.end()) return bad("invalid native frame handle");
  auto &owner=*found->second;
  bool resident=streams!=nullptr;
  if ((resident && producerCount!=rootCount) || (!resident && producerCount))
    return bad("native frame producer stream count differs");
  if (!nativeFrameViews(owner,roots,rootCount,outputs,outputCount,resident) ||
      !nativeFrameContext(owner)) return 1;
  std::vector<CUdeviceptr> bindings(owner.descriptors.size());
  std::vector<unsigned long long> allocationIds(rootCount);
  std::vector<uint64_t> nextStreams;
  if (resident) nextStreams.assign(streams,streams+producerCount);
  for (size_t i=0;i<bindings.size();++i) bindings[i]=owner.arena->base+owner.offsets[i];
  if (resident) {
    for (size_t i=0;i<rootCount;++i) {
      auto pointer=static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(roots[i].data));
      if (!residentAllocation(pointer,roots[i].bytes,owner.context) ||
          nativeFrameOverlap(uintptr_t(pointer),roots[i].bytes,
              uintptr_t(owner.arena->base),owner.arena->capacity) ||
          !ok(cuPointerGetAttribute(&allocationIds[i],CU_POINTER_ATTRIBUTE_BUFFER_ID,pointer),
              "native frame root allocation identity"))
        return bad("native frame resident root capacity/context/private span differs");
      bindings[owner.roots[i]]=pointer;
    }
  }
  // Validate every producer before committing the previously valid frame.
  ProducerEvents ordering;
  if (resident && !ordering.wait(owner.context,owner.stream,streams,producerCount)) {
    nativeFrameDrain(owner,false);return 1;
  }
  owner.generation=0;
  ++owner.arena->generation;
  owner.bindings=std::move(bindings);
  auto *host=static_cast<unsigned char *>(owner.arena->host);
  bool status=true;
  if (!resident) {
    // Snapshot all host roots before any upload or caller output is committed.
    for (size_t i=0;i<rootCount;++i)
      std::memcpy(host+owner.offsets[owner.roots[i]],roots[i].data,roots[i].bytes);
    for (size_t i=0;i<rootCount && status;++i) {
      size_t id=owner.roots[i];
      status=ok(cuMemcpyHtoDAsync(owner.bindings[id],host+owner.offsets[id],roots[i].bytes,
          owner.stream),"upload native program root");
    }
  }
  if (status) status=submitNativeFrame(owner);
  for (size_t i=0;i<outputCount && status;++i) {
    size_t id=owner.outputs[i];
    status=ok(cuMemcpyDtoHAsync(host+owner.offsets[id],owner.bindings[id],outputs[i].bytes,
        owner.stream),"stage native program output");
  }
  if (!nativeFrameDrain(owner,status)) return 1;
  for (size_t i=0;i<outputCount;++i)
    std::memcpy(outputs[i].data,host+owner.offsets[owner.outputs[i]],outputs[i].bytes);
  owner.resident=resident;
  owner.producerStreams=std::move(nextStreams);
  owner.allocationIds=std::move(allocationIds);
  owner.generation=owner.arena->generation;
  return 0;
}
} // namespace

extern "C" int tessera_nvidia_native_frame_prepare_v1(
    const TesseraNvidiaMatmulHostView *buffers,const uint8_t *roles,size_t count,
    const TesseraNvidiaNativeFrameStepV1 *steps,size_t stageCount,uint64_t *handle) {
  error.clear();if (handle) *handle=0;
  if (getpid()!=process) return bad("native frame cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);
  if (!buffers || !roles || !steps || !handle || count<2 || count>128 ||
      stageCount<1 || stageCount>64) return bad("invalid native frame metadata extent");
  try {
    auto owner=std::make_unique<NativeFrameOwner>();
    std::vector<bool> initialized(count,false),read(count,false);
    for (size_t i=0;i<count;++i) {
      const auto &buffer=buffers[i];
      size_t width=buffer.dtype==1?4:2;
      if (buffer.data || buffer.rank!=2 || buffer.dtype<1 || buffer.dtype>3 ||
          roles[i]>2 || buffer.shape[0]<=0 || buffer.shape[1]<=0 ||
          buffer.shape[0]>INT32_MAX || buffer.shape[1]>INT32_MAX)
        return bad("native frame physical buffer capacity/layout differs");
      const __int128 bytes=(__int128)buffer.shape[0]*buffer.shape[1]*width;
      if (bytes>INT64_MAX || bytes!=buffer.bytes ||
          buffer.strides[0]!=int64_t(buffer.shape[1]*width) ||
          buffer.strides[1]!=int64_t(width) ||
          owner->bytes>SIZE_MAX-255 || buffer.bytes>SIZE_MAX-owner->bytes-255)
        return bad("native frame physical buffer capacity/layout differs");
      owner->descriptors.push_back(buffer);
      owner->offsets.push_back(owner->bytes);
      owner->bytes=(owner->bytes+buffer.bytes+255)&~size_t(255);
      if (roles[i]==0) {owner->roots.push_back(i);initialized[i]=true;}
      if (roles[i]==2) owner->outputs.push_back(i);
    }
    if (owner->roots.empty() || owner->outputs.empty()) return bad("native frame needs roots and outputs");
    for (size_t i=0;i<stageCount;++i) {
      const auto &source=steps[i];
      if (!source.image || !source.image_bytes || source.image_bytes>16*1024*1024 ||
          !source.entry || !source.entry[0] || source.input_count<1 ||
          source.input_count>8 || source.scalar_count>8)
        return bad("native frame member metadata differs");
      NativeFrameStage stage;
      stage.inputCount=source.input_count;stage.scalarCount=source.scalar_count;
      for (size_t j=0;j<=stage.inputCount;++j) {
        size_t id=source.buffers[j];
        if (id>=count || (j<stage.inputCount ? !initialized[id] : initialized[id]))
          return bad("native frame reads uninitialized storage or overwrites live storage");
        stage.buffers[j]=id;
        if (j<stage.inputCount) read[id]=true;
      }
      initialized[source.buffers[source.input_count]]=true;
      uint64_t threads=1;
      for (size_t axis=0;axis<3;++axis) {
        if (!source.grid[axis] || source.grid[axis]>(axis?65535u:2147483647u) ||
            !source.block[axis] || source.block[axis]>(axis==2?64u:1024u))
          return bad("native frame launch geometry exceeds device bounds");
        stage.grid[axis]=source.grid[axis];stage.block[axis]=source.block[axis];
        threads*=source.block[axis];
      }
      if (threads>1024) return bad("native frame block exceeds thread capacity");
      for (size_t j=0;j<stage.scalarCount;++j) {
        if (source.scalars[j]<=0 || source.scalars[j]>INT32_MAX)
          return bad("native frame scalar exceeds its positive static extent ABI");
        stage.scalars[j]=source.scalars[j];
      }
      owner->stages.push_back(stage);
    }
    for (size_t i=0;i<count;++i)
      if (!initialized[i] || (roles[i]==1 && !read[i]))
        return bad("native frame has unwritten or unconsumed private storage");
    if (!current(owner->context) ||
        !ok(cuCtxGetId(owner->context,&owner->identity),"native frame context identity"))
      return bad("native frame requires an exact live SM120 context");
    owner->modules.reserve(stageCount);
    std::map<std::string,size_t> images;
    for (size_t i=0;i<stageCount;++i) {
      std::string image(static_cast<const char *>(steps[i].image),steps[i].image_bytes);
      if (image.find('\0')!=std::string::npos || image.find(".target sm_120a")==std::string::npos)
        return bad("native frame requires a complete SM120a PTX image");
      auto imageFound=images.find(image);
      size_t id=0;
      if (imageFound==images.end()) {
        id=owner->modules.size();owner->modules.push_back(nullptr);
        if (!ok(cuModuleLoadData(&owner->modules[id],image.c_str()),"load native program image")) return 1;
        images.emplace(std::move(image),id);
      } else id=imageFound->second;
      if (!ok(cuModuleGetFunction(&owner->stages[i].function,owner->modules[id],steps[i].entry),
          "resolve native program entry")) return 1;
      auto &stage=owner->stages[i];
      size_t params=stage.inputCount+1+stage.scalarCount;
      for (size_t j=0;j<params;++j) {
        size_t offset=0,bytes=0;
        if (!ok(cuFuncGetParamInfo(stage.function,j,&offset,&bytes),"native member parameter ABI") ||
            offset!=j*8 || bytes!=8) return bad("native member pointer/scalar parameter ABI differs");
      }
      size_t offset=0,bytes=0;
      if (cuFuncGetParamInfo(stage.function,params,&offset,&bytes)!=CUDA_ERROR_INVALID_VALUE)
        return bad("native member parameter count differs");
    }
    if (!ok(cuStreamCreate(&owner->stream,CU_STREAM_NON_BLOCKING),"create native program stream")) return 1;
    // Reuse the allocator implementation with dedicated program storage.
    // Other prepared owners cannot invalidate this frame's profile or outputs.
    owner->arena=std::make_shared<Scratch>();
    owner->arena->context=owner->context;owner->arena->identity=owner->identity;
    if (!owner->arena->grow(owner->bytes) || !owner->arena->growHost(owner->bytes)) return 1;
    owner->bindings.resize(count);
    const uint64_t id=nextHandle++;
    nativeFrames.emplace(id,std::move(owner));*handle=id;return 0;
  } catch (...) {return bad("native frame preparation failed");}
}
extern "C" int tessera_nvidia_native_frame_invoke_v1(uint64_t handle,
    const TesseraNvidiaMatmulHostView *roots,size_t rootCount,
    const TesseraNvidiaMatmulHostView *outputs,size_t outputCount) {
  error.clear();if (getpid()!=process) return bad("native frame cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);
  try {return invokeNativeFrame(handle,roots,rootCount,nullptr,0,outputs,outputCount);}
  catch (...) {
    auto found=nativeFrames.find(handle);
    if (found!=nativeFrames.end()) {
      found->second->generation=0;nativeFrameDrain(*found->second,false);
    }
    return bad("native frame invocation failed");
  }
}
extern "C" int tessera_nvidia_native_frame_resident_v1(uint64_t handle,
    const TesseraNvidiaMatmulHostView *roots,size_t rootCount,
    const uint64_t *streams,size_t producerCount,
    const TesseraNvidiaMatmulHostView *outputs,size_t outputCount) {
  error.clear();if (getpid()!=process) return bad("native frame cannot cross fork");
  if (!streams || !producerCount) return bad("resident native frame requires explicit producer streams");
  std::lock_guard<std::mutex> lock(mutex);
  try {return invokeNativeFrame(handle,roots,rootCount,streams,producerCount,outputs,outputCount);}
  catch (...) {
    auto found=nativeFrames.find(handle);
    if (found!=nativeFrames.end()) {
      found->second->generation=0;nativeFrameDrain(*found->second,false);
    }
    return bad("resident native frame invocation failed");
  }
}
extern "C" int tessera_nvidia_native_frame_profile_v1(uint64_t handle,int repeats,int captured,
    float *stageMs,size_t stageCount,float *programMs) {
  error.clear();if (getpid()!=process) return bad("native frame cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);
  auto found=nativeFrames.find(handle);
  if (found==nativeFrames.end()) return bad("invalid native frame handle");
  auto &owner=*found->second;
  if (!stageMs || !programMs || stageCount!=owner.stages.size() ||
      repeats<1 || repeats>(captured?4096:1000000) || (captured!=0 && captured!=1) ||
      (captured && uint64_t(repeats)*stageCount>65536) ||
      !owner.generation || owner.generation!=owner.arena->generation)
    return bad("native frame profile requires its last completed frame and valid repeat count");
  if (!nativeFrameContext(owner)) return 1;
  try {
    ProducerEvents ordering;
    if (owner.resident) {
      for (size_t i=0;i<owner.roots.size();++i) {
        auto pointer=owner.bindings[owner.roots[i]];
        unsigned long long identity=0;
        if (!residentAllocation(pointer,owner.descriptors[owner.roots[i]].bytes,owner.context) ||
            !ok(cuPointerGetAttribute(&identity,CU_POINTER_ATTRIBUTE_BUFFER_ID,pointer),
                "profile native root allocation identity") || identity!=owner.allocationIds[i])
          return bad("native frame profiled allocation was retired or replaced");
      }
      if (!ordering.wait(owner.context,owner.stream,owner.producerStreams.data(),owner.producerStreams.size())) {
        nativeFrameDrain(owner,false);return 1;
      }
    }
    float program=0;std::vector<float> stages(stageCount);
    bool status=measureLaunches(owner.stream,repeats,bool(captured),
        [&]() {return submitNativeFrame(owner);},&program);
    for (size_t i=0;i<stageCount && status;++i)
      status=measureLaunches(owner.stream,repeats,bool(captured),
          [&]() {return submitNativeFrameStage(owner,i);},&stages[i]);
    if (status) status=submitNativeFrame(owner);
    if (!nativeFrameDrain(owner,status)) {owner.generation=0;return 1;}
    std::memcpy(stageMs,stages.data(),stageCount*sizeof(float));*programMs=program;
    return 0;
  } catch (...) {owner.generation=0;nativeFrameDrain(owner,false);return bad("native frame profiling failed");}
}
extern "C" int tessera_nvidia_native_frame_stats_v1(uint64_t handle,uint64_t *values,size_t count) {
  error.clear();if (getpid()!=process) return bad("native frame cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);
  auto found=nativeFrames.find(handle);
  if (found==nativeFrames.end() || !values || count!=6) return bad("invalid native frame stats request");
  auto &owner=*found->second;
  if (!nativeFrameContext(owner)) return 1;
  uint64_t result[]={owner.arena->capacity,owner.arena->allocations,owner.modules.size(),
      owner.stages.size(),owner.generation,owner.bytes};
  std::memcpy(values,result,sizeof(result));return 0;
}
extern "C" int tessera_nvidia_native_frame_close_v1(uint64_t handle) {
  error.clear();if (getpid()!=process) return bad("native frame cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);
  auto found=nativeFrames.find(handle);
  if (found==nativeFrames.end()) return bad("invalid native frame handle");
  auto &owner=*found->second;
  if (nativeFrameThreadIdentity()!=owner.thread) return bad("native frame thread changed");
  unsigned long long identity=0;
  CUresult live=cuCtxGetId(owner.context,&identity);
  if (live==CUDA_SUCCESS && identity==owner.identity) {
    CUcontext previous=nullptr;
    if (!ok(cuCtxPushCurrent(owner.context),"close native frame context")) return 1;
    bool complete=ok(cuStreamSynchronize(owner.stream),"retire native frame readers");
    cuCtxPopCurrent(&previous);
    if (!complete) {owner.poisoned=true;return 1;}
  } else if (live!=CUDA_ERROR_CONTEXT_IS_DESTROYED && live!=CUDA_ERROR_INVALID_CONTEXT &&
             live!=CUDA_ERROR_DEINITIALIZED)
    return bad("native frame context retirement cannot be confirmed");
  nativeFrames.erase(found);return 0;
}
