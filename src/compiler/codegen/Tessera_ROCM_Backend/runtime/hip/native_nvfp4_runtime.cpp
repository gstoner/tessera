// Native lifecycle for the compiler-owned static NVFP4 three-stage program.
// No numerical kernel or physical schedule is authored by this runtime.
#include <hip/hip_runtime.h>
#include <array>
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <cstring>
#include <cmath>
#include <limits>
#include <string>
#include <unistd.h>
extern "C" int tessera_rocm_image_acquire(const void *, size_t, const char *, void **, void **, void **, int *);
extern "C" int tessera_rocm_image_release(void *);
namespace {
const pid_t ownerProcess=getpid();
struct Memref {void *allocated,*aligned; int64_t offset,elements,stride;};
struct Stage {
  void *lease=nullptr; hipFunction_t function{};
  std::array<Memref,6> refs{};
  std::array<int64_t,3> scalars{};
  std::array<void*,33> argv{};
  std::array<unsigned,6> geometry{};
};
struct Graph {
  hipGraph_t graph{};hipGraphExec_t executable{};size_t nodes=0;
};
int releaseGraph(Graph &g) {
  if(g.executable && hipGraphExecDestroy(g.executable)!=hipSuccess)return 9;
  g.executable=nullptr;
  if(g.graph && hipGraphDestroy(g.graph)!=hipSuccess)return 9;
  g.graph=nullptr;
  return 0;
}
struct Program {
  int device=0; hipCtx_t context{}; hipStream_t stream{};
  std::array<void*,11> buffers{};
  std::array<size_t,11> bytes{},elements{};
  std::array<std::vector<unsigned char>,5> host{};
  std::array<Stage,3> stages{};
  bool weights=false,activations=false,output=false,poisoned=true;
  uint64_t generation=0;
  std::vector<hipEvent_t> events;
  std::vector<unsigned char> readback;
  std::map<std::pair<int,int>,Graph> graphs;
  Graph *activeCapture=nullptr;
};
struct CacheKey {
  std::array<int64_t,4> dims;
  std::array<unsigned,18> geometry;
  std::array<std::string,3> entries;
  std::array<std::vector<unsigned char>,3> images;
  bool operator==(const CacheKey &rhs) const {
    return dims==rhs.dims && geometry==rhs.geometry &&
           entries==rhs.entries && images==rhs.images;
  }
};
struct IdleProgram {CacheKey key;std::unique_ptr<Program> program;size_t bytes;};
constexpr size_t cacheLimit=64*1024*1024;
constexpr size_t cacheEntries=4;
struct State {
  std::mutex mutex;
  std::map<uint64_t,std::unique_ptr<Program>> programs;
  uint64_t nextHandle=1;
  std::map<uint64_t,CacheKey> cacheKeys;
  std::vector<IdleProgram> idle;
  size_t idleBytes=0;
};
// HIP can outlive library/static destruction. Quarantined owners must never
// free asynchronous upload/readback storage during implicit teardown.
State &state() {static auto *value=new State;return *value;}
auto &mutex=state().mutex;
auto &programs=state().programs;
auto &nextHandle=state().nextHandle;
bool product(size_t &out,std::initializer_list<int64_t> dims,size_t size) {
  out=size;
  for(auto d:dims) {
    if(d<=0 || uint64_t(d)>size_t(std::numeric_limits<int64_t>::max())/out)return false;
    out*=size_t(d);
  }
  return true;
}
bool identity(const Program &p) {
  int device=0; hipCtx_t ctx{};
  return getpid()==ownerProcess && hipGetDevice(&device)==hipSuccess &&
    hipCtxGetCurrent(&ctx)==hipSuccess && device==p.device && ctx==p.context;
}
int clean(Program &p) {
  if(p.stream) {
    hipStreamCaptureStatus capture{};
    if(hipStreamIsCapturing(p.stream,&capture)!=hipSuccess)return 7;
    if(capture!=hipStreamCaptureStatusNone) {
      if(!p.activeCapture)return 7;
      // EndCapture can report invalidation while still ending capture.
      int rc=hipStreamEndCapture(p.stream,&p.activeCapture->graph);
      if(rc!=hipSuccess &&
          (hipStreamIsCapturing(p.stream,&capture)!=hipSuccess ||
           capture!=hipStreamCaptureStatusNone))return 7;
    }
    p.activeCapture=nullptr;
  }
  if(p.stream && hipStreamSynchronize(p.stream)!=hipSuccess)return 7;
  for(auto it=p.graphs.begin();it!=p.graphs.end();) {
    if(releaseGraph(it->second))return 9;
    it=p.graphs.erase(it);
  }
  while(!p.events.empty()) {
    if(hipEventDestroy(p.events.back())!=hipSuccess)return 9;
    p.events.pop_back();
  }
  for(auto &b:p.buffers) {
    if(b && hipFree(b)!=hipSuccess)return 9;
    b=nullptr;
  }
  for(auto &s:p.stages) {
    if(s.lease && tessera_rocm_image_release(s.lease))return 8;
    s.lease=nullptr;
  }
  if(p.stream && hipStreamDestroy(p.stream)!=hipSuccess)return 9;
  p.stream=nullptr;
  return 0;
}
bool values(const Program &p,const void *a,const void *scale) {
  if(!a || !scale || uintptr_t(scale)%alignof(float))return false;
  auto codes=static_cast<const unsigned char*>(a);
  auto scales=static_cast<const float*>(scale);
  for(size_t i=0;i<p.elements[3];++i)if((codes[i]&127)==127)return false;
  for(size_t i=0;i<p.elements[4];++i)
    if(!std::isfinite(scales[i]) || scales[i]<0)return false;
  return true;
}
int launch(Program &p,int stage) {
  auto &s=p.stages[stage];
  return hipModuleLaunchKernel(s.function,s.geometry[0],s.geometry[1],s.geometry[2],
      s.geometry[3],s.geometry[4],s.geometry[5],0,p.stream,s.argv.data(),nullptr)==hipSuccess?0:6;
}
} // namespace

// Status: 1 malformed request, 2 process/context, 3 image, 4 allocation,
// 5 copy, 6 launch, 7 completion, 8 lease, 9 cleanup, 10 state, 12 exception.
// A nonzero preparation status can still return a retained handle: close it.
extern "C" int tessera_rocm_nvfp4_prepare(
    const void *const *images,const size_t *imageBytes,const char *const *entries,
    const int64_t *dims,const unsigned *geometry,const void *const *inputs,
    const size_t *inputBytes,uint64_t *handle) try {
  if(handle)*handle=0;
  if(!handle || !images || !imageBytes || !entries || !dims || !geometry ||
      !inputs || !inputBytes)return 1;
  if(getpid()!=ownerProcess)return 2;
  int64_t m=dims[0],n=dims[1],k=dims[2],segments=dims[3];
  if(m<=0 || n<=0 || n%16 || k<=0 || k%64 || segments<=0 || segments>n)return 1;
  auto p=std::make_unique<Program>();
  if(!product(p->bytes[0],{n,k/2},1) || !product(p->bytes[1],{n,k/16},1) ||
     !product(p->bytes[2],{segments},8) || !product(p->bytes[3],{m,k},1) ||
     !product(p->bytes[4],{m},4) || !product(p->bytes[5],{n,k/2},1) ||
     !product(p->bytes[6],{k/32,n},1) || !product(p->bytes[7],{n,k/32,2},8) ||
     !product(p->bytes[8],{n,k/2},1) || !product(p->bytes[9],{k/32+1,n},1) ||
     !product(p->bytes[10],{m,n},2))return 1;
  constexpr size_t sizes[11]={1,1,8,1,4,1,1,8,1,1,2};
  for(size_t i=0;i<11;++i)p->elements[i]=p->bytes[i]/sizes[i];
  for(size_t i=0;i<5;++i)if(!inputs[i] || inputBytes[i]!=p->bytes[i])return 1;
  if(uintptr_t(inputs[2])%alignof(double) || !values(*p,inputs[3],inputs[4]))return 1;
  auto gs=static_cast<const double*>(inputs[2]);
  for(int64_t i=0;i<segments;++i)if(!std::isfinite(gs[i]) || gs[i]<=0)return 1;
  auto scales=static_cast<const unsigned char*>(inputs[1]);
  for(size_t i=0;i<p->bytes[1];++i)if(((scales[i]&128) && (scales[i]&127)) || (scales[i]&127)==127)return 1;
  for(size_t stage=0;stage<3;++stage) {
    if(!images[stage] || imageBytes[stage]<4 || std::memcmp(images[stage],"\177ELF",4) ||
       !entries[stage] || !*entries[stage])return 1;
    for(size_t i=0;i<6;++i) {
      if(!geometry[stage*6+i] || (i>=3 && geometry[stage*6+i]>1024))return 1;
      p->stages[stage].geometry[i]=geometry[stage*6+i];
    }
    uint64_t block=uint64_t(geometry[stage*6+3])*geometry[stage*6+4]*geometry[stage*6+5];
    if(block>1024)return 1;
  }
  if(hipInit(0)!=hipSuccess || hipGetDevice(&p->device)!=hipSuccess ||
      hipCtxGetCurrent(&p->context)!=hipSuccess)return 2;
  hipDeviceProp_t properties{};
  if(hipGetDeviceProperties(&properties,p->device)!=hipSuccess ||
      std::string(properties.gcnArchName).substr(0,7)!="gfx1201")return 2;
  for(size_t i=0;i<5;++i) {
    auto begin=static_cast<const unsigned char*>(inputs[i]);
    p->host[i].assign(begin,begin+p->bytes[i]);
  }
  std::lock_guard<std::mutex> lock(mutex);
  if(!nextHandle)return 12;
  uint64_t id=nextHandle++;
  programs.emplace(id,std::move(p));*handle=id;
  auto &owned=*programs.at(id);
  owned.events.reserve(2);
  if(hipStreamCreateWithFlags(&owned.stream,hipStreamNonBlocking)!=hipSuccess)return 4;
  for(size_t stage=0;stage<3;++stage) {
    void *module=nullptr,*function=nullptr;int hit=0;
    if(tessera_rocm_image_acquire(images[stage],imageBytes[stage],entries[stage],
        &owned.stages[stage].lease,&module,&function,&hit))return 3;
    owned.stages[stage].function=static_cast<hipFunction_t>(function);
  }
  for(size_t i=0;i<11;++i)
    if(hipMalloc(&owned.buffers[i],owned.bytes[i])!=hipSuccess)return 4;
  for(size_t i=0;i<5;++i)
    if(hipMemcpyAsync(owned.buffers[i],owned.host[i].data(),owned.bytes[i],
        hipMemcpyHostToDevice,owned.stream)!=hipSuccess)return 5;
  constexpr int slots[3][6]={{0,1,2,5,6,7},{5,6,8,9,-1,-1},{3,8,4,9,10,-1}};
  constexpr int counts[3]={6,4,5};
  for(size_t stage=0;stage<3;++stage) {
    auto &s=owned.stages[stage];size_t arg=0;
    for(int i=0;i<counts[stage];++i) {
      int slot=slots[stage][i];auto &r=s.refs[i];
      r={owned.buffers[slot],owned.buffers[slot],0,int64_t(owned.elements[slot]),1};
      s.argv[arg++]=&r.allocated;s.argv[arg++]=&r.aligned;
      s.argv[arg++]=&r.offset;s.argv[arg++]=&r.elements;s.argv[arg++]=&r.stride;
    }
    if(stage==2) {
      s.scalars={m,n,k};
      for(auto &scalar:s.scalars)s.argv[arg++]=&scalar;
    }
  }
  if(hipStreamSynchronize(owned.stream)!=hipSuccess) {owned.poisoned=true;return 7;}
  owned.activations=true;owned.poisoned=false;
  return 0;
} catch(...) {return 12;}

extern "C" int tessera_rocm_nvfp4_update(
    uint64_t handle,const void *a,size_t aBytes,const void *scale,size_t scaleBytes) try {
  if(getpid()!=ownerProcess)return 2;
  std::lock_guard<std::mutex> lock(mutex);
  auto found=programs.find(handle);if(found==programs.end())return 1;
  auto &p=*found->second;
  if(!identity(p))return 2;
  if(p.poisoned)return 10;
  if(aBytes!=p.bytes[3] || scaleBytes!=p.bytes[4] || !values(p,a,scale))return 1;
  std::vector<unsigned char> nextA(static_cast<const unsigned char*>(a),
      static_cast<const unsigned char*>(a)+aBytes);
  std::vector<unsigned char> nextScale(static_cast<const unsigned char*>(scale),
      static_cast<const unsigned char*>(scale)+scaleBytes);
  if(hipStreamSynchronize(p.stream)!=hipSuccess){p.poisoned=true;return 7;}
  p.activations=false;p.output=false;
  p.host[3]=std::move(nextA);p.host[4]=std::move(nextScale);
  for(size_t i=3;i<5;++i)
    if(hipMemcpyAsync(p.buffers[i],p.host[i].data(),p.bytes[i],hipMemcpyHostToDevice,p.stream)!=hipSuccess) {
      p.poisoned=true;return 5;
    }
  if(hipStreamSynchronize(p.stream)!=hipSuccess){p.poisoned=true;return 7;}
  p.activations=true;return 0;
} catch(...) {return 12;}

// Full rebinding validates and snapshots all inputs before changing ownership.
extern "C" int tessera_rocm_nvfp4_update_inputs(
    uint64_t handle,const void *const *inputs,const size_t *bytes) try {
  if(!inputs || !bytes)return 1;
  if(getpid()!=ownerProcess)return 2;
  std::lock_guard<std::mutex> lock(mutex);
  auto found=programs.find(handle);if(found==programs.end())return 1;
  auto &p=*found->second;
  if(!identity(p))return 2;
  if(p.poisoned)return 10;
  for(size_t i=0;i<5;++i)if(!inputs[i] || bytes[i]!=p.bytes[i])return 1;
  if(uintptr_t(inputs[2])%alignof(double) || !values(p,inputs[3],inputs[4]))return 1;
  auto globals=static_cast<const double*>(inputs[2]);
  for(size_t i=0;i<p.elements[2];++i)if(!std::isfinite(globals[i]) || globals[i]<=0)return 1;
  auto scales=static_cast<const unsigned char*>(inputs[1]);
  for(size_t i=0;i<p.bytes[1];++i)
    if(((scales[i]&128) && (scales[i]&127)) || (scales[i]&127)==127)return 1;
  std::array<std::vector<unsigned char>,5> snapshots;
  for(size_t i=0;i<5;++i) {
    auto begin=static_cast<const unsigned char*>(inputs[i]);
    snapshots[i].assign(begin,begin+bytes[i]);
  }
  if(hipStreamSynchronize(p.stream)!=hipSuccess){p.poisoned=true;return 7;}
  p.weights=false;p.activations=false;p.output=false;
  p.host=std::move(snapshots);
  for(size_t i=0;i<5;++i)
    if(hipMemcpyAsync(p.buffers[i],p.host[i].data(),p.bytes[i],
                      hipMemcpyHostToDevice,p.stream)!=hipSuccess) {
      p.poisoned=true;return 5;
    }
  if(hipStreamSynchronize(p.stream)!=hipSuccess){p.poisoned=true;return 7;}
  p.activations=true;return 0;
} catch(...) {return 12;}

// stage 0/1/2 are individual kernels; 3 ingest; 4 combined.
// Repeated submissions and event ownership live below Python.
extern "C" int tessera_rocm_nvfp4_invoke(
    uint64_t handle,int stage,int repeats,uint64_t *generation,float *elapsed) try {
  if(!generation || stage<0 || stage>4 || repeats<=0 || repeats>1048576)return 1;
  *generation=0;if(elapsed)*elapsed=0;
  if(getpid()!=ownerProcess)return 2;
  std::lock_guard<std::mutex> lock(mutex);
  auto found=programs.find(handle);if(found==programs.end())return 1;
  auto &p=*found->second;
  if(!identity(p))return 2;
  if(p.poisoned || !p.activations || ((stage==1 || stage==2) && !p.weights))return 10;
  int first=stage==3 || stage==4?0:stage,last=stage==3?1:stage==4?2:stage;
  if(first==0)p.weights=false;
  p.output=false;
  if(!elapsed) {
    for(int i=0;i<repeats;++i)
      for(int s=first;s<=last;++s) {
        int rc=launch(p,s);
        if(rc){p.poisoned=true;return rc;}
      }
    // Submission is ordered on the private stream. Read/update/close establish
    // completion before exposing host values or releasing ownership.
    if(stage==3 || stage==4)p.weights=true;
    if(stage==2 || stage==4){p.output=true;++p.generation;}
    *generation=p.generation;return 0;
  }
  hipEvent_t begin{},end{};
  int status=0;
  if(hipEventCreate(&begin)!=hipSuccess)return 4;
  p.events.push_back(begin);
  if(hipEventCreate(&end)!=hipSuccess) {p.poisoned=true;return 4;}
  p.events.push_back(end);
  if(hipEventRecord(begin,p.stream)!=hipSuccess)status=6;
  for(int i=0;i<repeats && !status;++i)
    for(int s=first;s<=last && !status;++s)status=launch(p,s);
  if(!status && hipEventRecord(end,p.stream)!=hipSuccess)status=6;
  if(!status && hipEventSynchronize(end)!=hipSuccess)status=7;
  if(!status && hipEventElapsedTime(elapsed,begin,end)!=hipSuccess)status=7;
  // Even a partially submitted sequence must complete before event destruction.
  if(hipStreamSynchronize(p.stream)!=hipSuccess){p.poisoned=true;return 7;}
  while(!p.events.empty()) {
    if(hipEventDestroy(p.events.back())!=hipSuccess){p.poisoned=true;return 9;}
    p.events.pop_back();
  }
  if(status){p.poisoned=true;return status;}
  *elapsed/=repeats;
  if(stage==3 || stage==4)p.weights=true;
  if(stage==2 || stage==4){p.output=true;++p.generation;}
  *generation=p.generation;return 0;
} catch(...) {return 12;}

extern "C" int tessera_rocm_nvfp4_read(
    uint64_t handle,int slot,uint64_t generation,void *output,size_t bytes) try {
  if(!output || slot<5 || slot>10)return 1;
  if(getpid()!=ownerProcess)return 2;
  std::lock_guard<std::mutex> lock(mutex);
  auto found=programs.find(handle);if(found==programs.end())return 1;
  auto &p=*found->second;
  if(!identity(p))return 2;
  if(p.poisoned || (slot==10?(!p.output || generation!=p.generation):!p.weights))return 10;
  if(bytes!=p.bytes[slot])return 1;
  p.readback.resize(bytes);
  if(hipMemcpyAsync(p.readback.data(),p.buffers[slot],bytes,hipMemcpyDeviceToHost,p.stream)!=hipSuccess) {
    p.poisoned=true;return 5;
  }
  if(hipStreamSynchronize(p.stream)!=hipSuccess){p.poisoned=true;return 7;}
  std::memcpy(output,p.readback.data(),bytes);
  return 0;
} catch(...) {return 12;}

extern "C" int tessera_rocm_nvfp4_close(uint64_t handle) try {
  if(getpid()!=ownerProcess)return 2;
  std::lock_guard<std::mutex> lock(mutex);
  auto found=programs.find(handle);if(found==programs.end())return 1;
  if(!identity(*found->second))return 2;
  found->second->poisoned=true;
  int rc=clean(*found->second);
  if(!rc){state().cacheKeys.erase(handle);programs.erase(found);}
  return rc;
} catch(...) {return 12;}

// Cached graph replay owns capture/exec handles until stream completion.
// stage/repeat keys bind the existing native argv and immutable allocation set.
extern "C" int tessera_rocm_nvfp4_graph(
    uint64_t handle,int stage,int repeats,uint64_t *generation,
    uint64_t *nodes,float *elapsed) try {
  if(!generation || !nodes || stage<0 || stage>4 ||
      repeats<=0 || repeats>4096)return 1;
  *generation=0;*nodes=0;if(elapsed)*elapsed=0;
  if(getpid()!=ownerProcess)return 2;
  std::lock_guard<std::mutex> lock(mutex);
  auto found=programs.find(handle);if(found==programs.end())return 1;
  auto &p=*found->second;
  if(!identity(p))return 2;
  if(p.poisoned || !p.activations || ((stage==1 || stage==2) && !p.weights))return 10;
  auto key=std::make_pair(stage,repeats);
  auto entry=p.graphs.find(key);
  if(entry==p.graphs.end()) {
    if(hipStreamSynchronize(p.stream)!=hipSuccess){p.poisoned=true;return 7;}
    if(p.graphs.size()>=8) {
      auto old=p.graphs.begin();
      if(releaseGraph(old->second)){p.poisoned=true;return 9;}
      p.graphs.erase(old);
    }
    entry=p.graphs.emplace(key,Graph{}).first;
    auto &g=entry->second;
    p.activeCapture=&g;
    if(hipStreamBeginCapture(p.stream,hipStreamCaptureModeThreadLocal)!=hipSuccess) {
      p.poisoned=true;return 6;
    }
    int first=stage>=3?0:stage,last=stage==3?1:stage==4?2:stage;
    int status=0;
    for(int i=0;i<repeats && !status;++i)
      for(int s=first;s<=last && !status;++s)status=launch(p,s);
    int endStatus=hipStreamEndCapture(p.stream,&g.graph);
    if(status || endStatus!=hipSuccess){p.poisoned=true;return 6;}
    p.activeCapture=nullptr;
    size_t count=0;
    if(hipGraphGetNodes(g.graph,nullptr,&count)!=hipSuccess ||
        count!=size_t(repeats)*(stage==3?2:stage==4?3:1)) {
      p.poisoned=true;return 6;
    }
    g.nodes=count;
    if(hipGraphInstantiateWithFlags(&g.executable,g.graph,0)!=hipSuccess) {
      p.poisoned=true;return 6;
    }
  }
  auto &g=entry->second;
  p.output=false;
  if(stage==0 || stage>=3)p.weights=false;
  if(!elapsed) {
    if(hipGraphLaunch(g.executable,p.stream)!=hipSuccess){p.poisoned=true;return 6;}
    if(stage>=3)p.weights=true;
    if(stage==2 || stage==4){p.output=true;++p.generation;}
    *generation=p.generation;*nodes=g.nodes;return 0;
  }
  // Warm upload/first replay is outside the recorded event interval.
  if(hipGraphLaunch(g.executable,p.stream)!=hipSuccess){p.poisoned=true;return 6;}
  hipEvent_t begin{},end{};
  if(hipEventCreate(&begin)!=hipSuccess){p.poisoned=true;return 4;}
  p.events.push_back(begin);
  if(hipEventCreate(&end)!=hipSuccess){p.poisoned=true;return 4;}
  p.events.push_back(end);
  int status=0;
  if(hipEventRecord(begin,p.stream)!=hipSuccess ||
      hipGraphLaunch(g.executable,p.stream)!=hipSuccess ||
      hipEventRecord(end,p.stream)!=hipSuccess)status=6;
  if(!status && (hipEventSynchronize(end)!=hipSuccess ||
                hipEventElapsedTime(elapsed,begin,end)!=hipSuccess))status=7;
  if(hipStreamSynchronize(p.stream)!=hipSuccess){p.poisoned=true;return 7;}
  while(!p.events.empty()) {
    if(hipEventDestroy(p.events.back())!=hipSuccess){p.poisoned=true;return 9;}
    p.events.pop_back();
  }
  if(status){p.poisoned=true;return status;}
  if(stage>=3)p.weights=true;
  if(stage==2 || stage==4){p.output=true;++p.generation;}
  *generation=p.generation;*nodes=g.nodes;
  return 0;
} catch(...) {return 12;}


// Bounded native idle ownership. Checkout assigns a fresh token so stale
// handles cannot access reused allocations. Keys include complete images.
extern "C" int tessera_rocm_nvfp4_prepare_cached(
    const void *const *images,const size_t *imageBytes,const char *const *entries,
    const int64_t *dims,const unsigned *geometry,const void *const *inputs,
    const size_t *inputBytes,uint64_t *handle,int *hit) try {
  if(handle)*handle=0;if(hit)*hit=0;
  if(!handle || !hit || !images || !imageBytes || !entries || !dims || !geometry ||
     !inputs || !inputBytes)return 1;
  if(getpid()!=ownerProcess)return 2;
  CacheKey key;
  for(size_t i=0;i<3;++i) {
    if(!images[i] || imageBytes[i]<4 || !entries[i] || !*entries[i])return 1;
    if(imageBytes[i]>cacheLimit)
      return tessera_rocm_nvfp4_prepare(images,imageBytes,entries,dims,geometry,inputs,inputBytes,handle);
    auto begin=static_cast<const unsigned char*>(images[i]);
    key.images[i].assign(begin,begin+imageBytes[i]);key.entries[i]=entries[i];
  }
  std::copy_n(dims,4,key.dims.begin());std::copy_n(geometry,18,key.geometry.begin());
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto &pool=state();
    for(auto it=pool.idle.begin();it!=pool.idle.end();++it) {
      if(!(it->key==key) || !identity(*it->program))continue;
      if(!nextHandle)return 12;
      uint64_t id=nextHandle++;
      try {
        programs.try_emplace(id,nullptr);
        pool.cacheKeys.emplace(id,std::move(key));
      } catch(...) {programs.erase(id);pool.cacheKeys.erase(id);return 12;}
      programs.at(id)=std::move(it->program);pool.idleBytes-=it->bytes;
      pool.idle.erase(it);*handle=id;*hit=1;break;
    }
  }
  if(*handle)return tessera_rocm_nvfp4_update_inputs(*handle,inputs,inputBytes);
  int rc=tessera_rocm_nvfp4_prepare(images,imageBytes,entries,dims,geometry,inputs,inputBytes,handle);
  if(!rc) {
    std::lock_guard<std::mutex> lock(mutex);
    state().cacheKeys.emplace(*handle,std::move(key));
  }
  return rc;
} catch(...) {return 12;}

extern "C" int tessera_rocm_nvfp4_release_cached(uint64_t handle) try {
  if(getpid()!=ownerProcess)return 2;
  std::lock_guard<std::mutex> lock(mutex);
  auto found=programs.find(handle);if(found==programs.end())return 1;
  auto &p=*found->second;
  if(!identity(p))return 2;
  auto &pool=state();auto key=pool.cacheKeys.find(handle);
  size_t bytes=0;
  auto account=[&](size_t n) {
    if(n>cacheLimit-bytes)return false;
    bytes+=n;return true;
  };
  bool fits=!p.poisoned && p.graphs.empty() && p.events.empty() &&
            key!=pool.cacheKeys.end() && pool.idle.size()<cacheEntries;
  if(fits) {
    for(auto n:p.bytes)if(!account(n)){fits=false;break;}
    for(auto &v:p.host)if(!account(v.capacity())){fits=false;break;}
    if(!account(p.readback.capacity()))fits=false;
    for(auto &v:key->second.images)if(!account(v.capacity())){fits=false;break;}
    if(bytes>cacheLimit-pool.idleBytes)fits=false;
  }
  if(fits) {
    if(hipStreamSynchronize(p.stream)!=hipSuccess){p.poisoned=true;return 7;}
    pool.idle.emplace_back(); // allocate before moving any retained owner
    p.weights=false;p.activations=false;p.output=false;
    auto &idle=pool.idle.back();idle.bytes=bytes;
    idle.key=std::move(key->second);idle.program=std::move(found->second);
    pool.idleBytes+=bytes;pool.cacheKeys.erase(key);programs.erase(found);
    return 0;
  }
  p.poisoned=true;int rc=clean(p);if(rc)return rc;
  pool.cacheKeys.erase(handle);programs.erase(found);return 0;
} catch(...) {return 12;}

extern "C" int tessera_rocm_nvfp4_cache_clear() try {
  if(getpid()!=ownerProcess)return 2;
  std::lock_guard<std::mutex> lock(mutex);
  auto &pool=state();
  for(auto it=pool.idle.begin();it!=pool.idle.end();) {
    if(!identity(*it->program))return 2;
    it->program->poisoned=true;
    int rc=clean(*it->program);if(rc)return rc;
    pool.idleBytes-=it->bytes;it=pool.idle.erase(it);
  }
  return 0;
} catch(...) {return 12;}
