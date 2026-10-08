// Native ownership of verified MLIR/LLVM saved-LSE forward/JVP images.
#include "tessera_nvidia_ptx_launch.h"
#include <cuda.h>
#include <dlfcn.h>
#include <unistd.h>
#include <array>
#include <climits>
#include <cstring>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
namespace {
std::mutex ownerMutex;
thread_local std::string ownerError;
const pid_t ownerProcess=getpid();
CUcontext retainedPrimary=nullptr;
uint64_t nextOwner=1;
bool ok(CUresult code,const char *op) {
  if(code==CUDA_SUCCESS)return true;
  const char *name=nullptr;cuGetErrorName(code,&name);
  ownerError=std::string(op)+": "+(name?name:"CUDA failure");return false;
}
int bad(const char *reason){ownerError=reason;return 1;}
struct Owner {
  CUcontext context=nullptr;
  unsigned long long contextIdentity=0;
  CUmodule forwardModule=nullptr,tangentModule=nullptr;
  CUfunction forward=nullptr,tangent=nullptr;
  CUdeviceptr arena=0;
  std::array<CUdeviceptr,12> buffers{};
  std::array<size_t,12> bytes{};
  std::array<int64_t,7> dims{};
  std::array<int,4> mapping{},roles{};
  size_t activeCount=0;
  bool reverse=false,bias=false;
  CUstream stream=nullptr;
  std::array<int64_t,4> biasShape{};
  unsigned reverseGrid=0,reverseThreads=128;
  unsigned forwardGrid=0,tangentGrid=0,shared=0;
  std::array<CUevent,4> events{};
  void *sizerLibrary=nullptr;
  ~Owner(){
    if(getpid()!=ownerProcess)return; // Never touch inherited CUDA handles.
    if(sizerLibrary){dlclose(sizerLibrary);sizerLibrary=nullptr;}
    if(!context || cuCtxPushCurrent(context)!=CUDA_SUCCESS)return;
    unsigned long long currentIdentity=0;
    if(cuCtxGetId(context,&currentIdentity)!=CUDA_SUCCESS || currentIdentity!=contextIdentity){
      CUcontext prior=nullptr;cuCtxPopCurrent(&prior);return;
    }
    cuCtxSynchronize();
    for(auto event:events)if(event)cuEventDestroy(event);
    if(arena)cuMemFree(arena);
    if(stream)cuStreamDestroy(stream);
    if(tangentModule)cuModuleUnload(tangentModule);
    if(forwardModule)cuModuleUnload(forwardModule);
    if(sizerLibrary)dlclose(sizerLibrary);
    CUcontext prior=nullptr;cuCtxPopCurrent(&prior);
  }
};
std::map<uint64_t,std::unique_ptr<Owner>> owners;
bool extent(std::initializer_list<int64_t> values,size_t &bytes){
  bytes=4;for(auto n:values){
    if(n<=0 || n>=(1LL<<31) || bytes>size_t(LLONG_MAX)/size_t(n))return false;
    bytes*=size_t(n);
  }return true;
}
bool currentContext(CUcontext &context){
  if(!ok(cuInit(0),"cuInit") || !ok(cuCtxGetCurrent(&context),"cuCtxGetCurrent"))return false;
  if(!context){
    CUdevice device;
    if(!ok(cuDeviceGet(&device,0),"cuDeviceGet"))return false;
    if(!retainedPrimary && !ok(cuDevicePrimaryCtxRetain(&retainedPrimary,device),"retain primary"))return false;
    context=retainedPrimary;if(!ok(cuCtxSetCurrent(context),"set context"))return false;
  }
  CUdevice device;int major=0,minor=0;
  return ok(cuCtxGetDevice(&device),"get device") &&
    ok(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device),"capability major") &&
    ok(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device),"capability minor") &&
    major==12 && minor==0;
}
}
extern "C" const char *tessera_nvidia_attention_jvp_last_error(){return ownerError.c_str();}
static int prepareAttentionJvp(
  const void *fimage,size_t fbytes,const char *fentry,
  const void *timage,size_t tbytes,const char *tentry,
  const char *sizerPath,const char *sizerEntry,const int64_t *dims,
  const int64_t *biasShape,const int *mapping,const int *roles,size_t activeCount,uint64_t *handle){
  ownerError.clear();if(handle)*handle=0;
  if(getpid()!=ownerProcess)return bad("prepared attention cannot cross fork");
  std::lock_guard<std::mutex> lock(ownerMutex);
  if(!handle || !fimage || !fbytes || !fentry || !*fentry ||
     !timage || !tbytes || !tentry || !*tentry || !sizerPath || !*sizerPath ||
     !sizerEntry || !*sizerEntry || !dims || !mapping || !roles ||
     activeCount<1 || activeCount>4)return bad("invalid prepared image or ABI");
  try{
    auto state=std::make_unique<Owner>();
    state->bias=biasShape!=nullptr;
    const size_t primals=3+unsigned(state->bias),slots=state->bias?11:9;
    bool mapped[4]{},active[4]{};
    for(size_t i=0;i<primals;++i){
      if(mapping[i]<0 || mapping[i]>=int(primals) || mapped[mapping[i]])return bad("invalid frontend permutation");
      mapped[mapping[i]]=true;state->mapping[i]=mapping[i];
    }
    for(size_t i=0;i<activeCount;++i){
      if(roles[i]<0 || roles[i]>=int(primals) || active[roles[i]])return bad("invalid tangent roles");
      active[roles[i]]=true;state->roles[i]=roles[i];
    }
    state->activeCount=activeCount;
    for(size_t i=0;i<7;++i){
      if(dims[i]<=0 || dims[i]>65536)return bad("invalid prepared shape");state->dims[i]=dims[i];
    }
    auto [b,hq,hkv,sq,sk,d,dv]=state->dims;
    if(hq%hkv || !extent({b,hq,sq,d},state->bytes[0]) ||
       !extent({b,hkv,sk,d},state->bytes[1]) || !extent({b,hkv,sk,dv},state->bytes[2]) ||
       !extent({b,hq,sq,dv},state->bytes[3]) || !extent({b,hq,sq},state->bytes[4]))
      return bad("invalid prepared extent");
    state->bytes[5]=state->bytes[0];state->bytes[6]=state->bytes[1];
    state->bytes[7]=state->bytes[2];state->bytes[8]=state->bytes[3];
    if(state->bias){
      const int64_t scores[4]={b,hq,sq,sk};
      for(size_t i=0;i<4;++i){
        if(biasShape[i]!=1 && biasShape[i]!=scores[i])return bad("invalid prepared bias shape");
        state->biasShape[i]=biasShape[i];
      }
      if(!extent({biasShape[0],biasShape[1],biasShape[2],biasShape[3]},state->bytes[8]))
        return bad("invalid prepared bias extent");
      state->bytes[9]=state->bytes[8];state->bytes[10]=state->bytes[3];
    }
    uint64_t rows=uint64_t(b)*hq*sq,outputs=state->bytes[3]/4;
    if(rows>INT_MAX || (outputs+127)/128>UINT_MAX)return bad("grid exceeds bounds");
    state->tangentGrid=unsigned(rows);state->forwardGrid=unsigned((outputs+127)/128);
    size_t total=0;std::array<size_t,11> offsets{};
    for(size_t i=0;i<slots;++i){
      if(total>std::numeric_limits<size_t>::max()-255)return bad("alignment overflow");
      total=(total+255)&~size_t(255);offsets[i]=total;
      if(state->bytes[i]>std::numeric_limits<size_t>::max()-total)return bad("arena overflow");
      total+=state->bytes[i];
    }
    if(!currentContext(state->context))return bad("prepared attention requires current SM120");
    if(!ok(cuCtxGetId(state->context,&state->contextIdentity),"context identity"))return 3;
    state->sizerLibrary=dlopen(sizerPath,RTLD_NOW|RTLD_LOCAL);
    if(!state->sizerLibrary)return bad("native sizing library unavailable");
    auto symbol=dlsym(state->sizerLibrary,sizerEntry);
    if(!symbol)return bad("native sizing entry unavailable");
    // Compiler-produced sizing is pure scalar shape arithmetic. The two
    // signatures preserve the old ABI and the explicit biased pointer arity.
    using Sizer=int64_t(*)(void*,void*,void*,void*,void*,void*,void*,void*,void*,int64_t);
    using BiasSizer=int64_t(*)(void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,int64_t);
    int64_t shared=state->bias?
      reinterpret_cast<BiasSizer>(symbol)(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,128):
      reinterpret_cast<Sizer>(symbol)(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,128);
    if(shared<0 || shared>INT_MAX)return bad("invalid native shared extent");
    if(!ok(cuModuleLoadData(&state->forwardModule,fimage),"load forward") ||
       !ok(cuModuleGetFunction(&state->forward,state->forwardModule,fentry),"resolve forward") ||
       !ok(cuModuleLoadData(&state->tangentModule,timage),"load tangent") ||
       !ok(cuModuleGetFunction(&state->tangent,state->tangentModule,tentry),"resolve tangent"))return 3;
    CUdevice device;int limit=0,staticBytes=0;
    if(!ok(cuCtxGetDevice(&device),"get device") ||
       !ok(cuDeviceGetAttribute(&limit,CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,device),"shared limit") ||
       !ok(cuFuncGetAttribute(&staticBytes,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,state->tangent),"static shared"))return 3;
    if(shared>limit-staticBytes)return bad("native shared exceeds device");
    state->shared=unsigned(shared);
    if(shared && !ok(cuFuncSetAttribute(state->tangent,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,int(shared)),"dynamic shared"))return 3;
    if(!ok(cuMemAlloc(&state->arena,total),"allocate arena") ||
       !ok(cuMemsetD8(state->arena,0,total),"initialize arena"))return 3;
    for(size_t i=0;i<slots;++i)state->buffers[i]=state->arena+offsets[i];
    for(auto &event:state->events)if(!ok(cuEventCreate(&event,CU_EVENT_DEFAULT),"create event"))return 3;
    if(nextOwner==0)return bad("owner identity exhausted");
    uint64_t id=nextOwner++;owners.emplace(id,std::move(state));*handle=id;return 0;
  }catch(...){ownerError="native owner allocation failed";return 3;}
}

extern "C" int tessera_nvidia_attention_jvp_prepare(
  const void *fimage,size_t fbytes,const char *fentry,
  const void *timage,size_t tbytes,const char *tentry,
  const char *sizerPath,const char *sizerEntry,const int64_t *dims,
  const int *mapping,const int *roles,size_t activeCount,uint64_t *handle){
  return prepareAttentionJvp(fimage,fbytes,fentry,timage,tbytes,tentry,
    sizerPath,sizerEntry,dims,nullptr,mapping,roles,activeCount,handle);
}
extern "C" int tessera_nvidia_attention_jvp_prepare_bias(
  const void *fimage,size_t fbytes,const char *fentry,
  const void *timage,size_t tbytes,const char *tentry,
  const char *sizerPath,const char *sizerEntry,const int64_t *dims,
  const int64_t *biasShape,const int *mapping,const int *roles,
  size_t activeCount,uint64_t *handle){
  if(!biasShape){if(handle)*handle=0;return bad("missing prepared bias shape");}
  return prepareAttentionJvp(fimage,fbytes,fentry,timage,tbytes,tentry,
    sizerPath,sizerEntry,dims,biasShape,mapping,roles,activeCount,handle);
}
extern "C" int tessera_nvidia_attention_jvp_invoke(
  uint64_t handle,const void *const *inputs,const size_t *inputBytes,size_t inputCount,
  void *const *outputs,const size_t *outputBytes,float *deviceMilliseconds){
  ownerError.clear();
  if(getpid()!=ownerProcess)return bad("prepared attention cannot cross fork");
  std::lock_guard<std::mutex> lock(ownerMutex);
  auto found=owners.find(handle);if(found==owners.end() || found->second->reverse)return bad("prepared attention is closed or wrong product");
  auto &s=*found->second;CUcontext context=nullptr;
  if(!ok(cuCtxGetCurrent(&context),"get current context"))return 3;
  if(context!=s.context)return bad("prepared attention context disagrees");
  unsigned long long identity=0;
  if(!ok(cuCtxGetId(context,&identity),"context identity"))return 3;
  if(identity!=s.contextIdentity)return bad("prepared attention context generation disagrees");
  const size_t primals=3+unsigned(s.bias),slots=s.bias?11:9,resultSlot=s.bias?10:8;
  auto primalSlot=[&](size_t role){return role==3?size_t(8):role;};
  auto tangentSlot=[&](size_t role){return role==3?size_t(9):5+role;};
  if(!inputs || !inputBytes || inputCount!=primals+s.activeCount || !outputs ||
     !outputs[0] || !outputs[1] || !outputBytes || outputBytes[0]!=s.bytes[3] ||
     outputBytes[1]!=s.bytes[resultSlot])return bad("prepared host ABI disagrees");
  uintptr_t a=reinterpret_cast<uintptr_t>(outputs[0]),b=reinterpret_cast<uintptr_t>(outputs[1]);
  if(a>UINTPTR_MAX-outputBytes[0] || b>UINTPTR_MAX-outputBytes[1] ||
     (a<b+outputBytes[1] && b<a+outputBytes[0]))return bad("prepared outputs overlap");
  for(size_t i=0;i<primals;++i)
    if(!inputs[s.mapping[i]] || inputBytes[s.mapping[i]]!=s.bytes[primalSlot(i)])
      return bad("primal extent disagrees");
  for(size_t i=0;i<s.activeCount;++i)
    if(!inputs[primals+i] || inputBytes[primals+i]!=s.bytes[tangentSlot(s.roles[i])])
      return bad("tangent extent disagrees");
  for(size_t i=0;i<primals;++i)
    if(!ok(cuMemcpyHtoD(s.buffers[primalSlot(i)],inputs[s.mapping[i]],s.bytes[primalSlot(i)]),"upload primal"))return 3;
  for(size_t i=0;i<s.activeCount;++i)
    if(!ok(cuMemcpyHtoD(s.buffers[tangentSlot(s.roles[i])],inputs[primals+i],s.bytes[tangentSlot(s.roles[i])]),"upload tangent"))return 3;
  void *fa[17];size_t arg=0;
  for(size_t i=0;i<3;++i)fa[arg++]=&s.buffers[i];
  if(s.bias)fa[arg++]=&s.buffers[8];
  fa[arg++]=&s.buffers[3];fa[arg++]=&s.buffers[4];
  for(size_t i=0;i<7;++i)fa[arg++]=&s.dims[i];
  if(s.bias){
    const int64_t scores[4]={s.dims[0],s.dims[1],s.dims[3],s.dims[4]};
    bool broadcast=false;for(size_t i=0;i<4;++i)broadcast|=s.biasShape[i]!=scores[i];
    if(broadcast)for(size_t i=0;i<4;++i)fa[arg++]=&s.biasShape[i];
  }
  void *ta[12];for(size_t i=0;i<slots;++i)ta[i]=&s.buffers[i];
  int64_t scratch=128;ta[slots]=&scratch;
  if(!ok(cuEventRecord(s.events[0],nullptr),"start forward") ||
     !ok(cuLaunchKernel(s.forward,s.forwardGrid,1,1,128,1,1,0,nullptr,fa,nullptr),"launch forward") ||
     !ok(cuEventRecord(s.events[1],nullptr),"end forward") ||
     !ok(cuEventRecord(s.events[2],nullptr),"start tangent") ||
     !ok(cuLaunchKernel(s.tangent,s.tangentGrid,1,1,128,1,1,s.shared,nullptr,ta,nullptr),"launch tangent") ||
     !ok(cuEventRecord(s.events[3],nullptr),"end tangent") ||
     !ok(cuCtxSynchronize(),"complete product") ||
     !ok(cuMemcpyDtoH(outputs[0],s.buffers[3],s.bytes[3]),"download primal") ||
     !ok(cuMemcpyDtoH(outputs[1],s.buffers[resultSlot],s.bytes[resultSlot]),"download tangent"))return 3;
  if(deviceMilliseconds &&
     (!ok(cuEventElapsedTime(&deviceMilliseconds[0],s.events[0],s.events[1]),"forward elapsed") ||
      !ok(cuEventElapsedTime(&deviceMilliseconds[1],s.events[2],s.events[3]),"tangent elapsed")))return 3;
  return 0;
}
extern "C" int tessera_nvidia_attention_jvp_close(uint64_t handle){
  ownerError.clear();
  if(getpid()!=ownerProcess)return bad("prepared attention cannot cross fork");
  std::lock_guard<std::mutex> lock(ownerMutex);
  auto found=owners.find(handle);if(found==owners.end() || found->second->reverse)return bad("prepared attention is closed or wrong product");
  owners.erase(found);return 0;
}

// Reverse products share the module/context/arena retirement machinery above.
// Their private output/LSE generation never escapes a synchronous invocation.
extern "C" const char *tessera_nvidia_attention_vjp_last_error(){return ownerError.c_str();}
extern "C" int tessera_nvidia_attention_vjp_prepare(
  const void *fimage,size_t fbytes,const char *fentry,
  const void *bimage,size_t bbytes,const char *bentry,
  const int64_t *dims,const int64_t *biasShape,const int *mapping,
  const int *roles,size_t activeCount,uint64_t *handle){
  ownerError.clear();if(handle)*handle=0;
  if(getpid()!=ownerProcess)return bad("prepared attention cannot cross fork");
  std::lock_guard<std::mutex> lock(ownerMutex);
  if(!handle || !fimage || !fbytes || !fentry || !bimage || !bbytes || !bentry ||
     !dims || !mapping || !roles || activeCount<1 || activeCount>4)
    return bad("invalid reverse images or ABI");
  unsigned mask=0,bias=0,biasGradient=0,logical=0,threads=0;
  const char *prefix="tessera_tile_attention_backward_lse_output_compact_m";
  if(std::strncmp(bentry,prefix,std::strlen(prefix)))return bad("unsupported reverse entry contract");
  const char *cursor=bentry+std::strlen(prefix);
  auto number=[&](unsigned &value,unsigned bound){
    if(*cursor<'0' || *cursor>'9')return false;
    if(*cursor=='0' && cursor[1]>='0' && cursor[1]<='9')return false;
    value=0;
    while(*cursor>='0' && *cursor<='9'){
      const unsigned digit=unsigned(*cursor++-'0');
      if(value>bound/10 || (value==bound/10 && digit>bound%10))return false;
      value=value*10+digit;
    }
    return true;
  };
  auto field=[&](const char *tag,unsigned &value,unsigned bound){
    if(std::strncmp(cursor,tag,2))return false;
    cursor+=2;return number(value,bound);
  };
  if(!number(mask,15) || !field("_b",bias,1) || !field("_g",biasGradient,1) ||
     !field("_l",logical,1) || !field("_t",threads,128) || *cursor++!='_' ||
     std::strlen(cursor)!=10 || !mask || mask>(biasGradient?15u:7u) ||
     (biasGradient && !bias) || (threads!=64 && threads!=128) ||
     std::strncmp(fentry,"tessera_tile_attention_lse_",std::strlen("tessera_tile_attention_lse_")))
    return bad("unsupported reverse entry contract");
  for(const char *p=cursor;*p;++p)
    if(!(*p>='0' && *p<='9') && !(*p>='a' && *p<='f'))return bad("invalid reverse digest suffix");
  try{
    auto state=std::make_unique<Owner>();
    state->reverse=true;state->bias=bias;state->activeCount=activeCount;
    state->reverseThreads=threads;
    const size_t primals=3+bias;
    bool mapped[4]{},active[4]{};unsigned requestedMask=0;
    for(size_t i=0;i<primals;++i){
      if(mapping[i]<0 || mapping[i]>=int(primals) || mapped[mapping[i]])
        return bad("invalid reverse frontend permutation");
      mapped[mapping[i]]=true;state->mapping[i]=mapping[i];
    }
    for(size_t i=0;i<activeCount;++i){
      if(roles[i]<0 || roles[i]>=int(3+biasGradient) || active[roles[i]])
        return bad("invalid reverse gradient roles");
      active[roles[i]]=true;state->roles[i]=roles[i];requestedMask|=1u<<roles[i];
    }
    if(requestedMask!=mask)return bad("reverse native mask differs from requested roles");
    for(size_t i=0;i<7;++i){
      if(dims[i]<=0 || dims[i]>=(1LL<<31))return bad("invalid reverse shape");
      state->dims[i]=dims[i];
    }
    auto [b,hq,hkv,sq,sk,d,dv]=state->dims;
    if(hq%hkv || !extent({b,hq,sq,d},state->bytes[0]) ||
       !extent({b,hkv,sk,d},state->bytes[1]) || !extent({b,hkv,sk,dv},state->bytes[2]) ||
       !extent({b,hq,sq,dv},state->bytes[3]) || !extent({b,hq,sq},state->bytes[4]))
      return bad("invalid reverse extent");
    state->bytes[5]=state->bytes[3];
    if(bias){
      if(!biasShape)return bad("missing reverse bias shape");
      const int64_t scores[4]={b,hq,sq,sk};
      for(size_t i=0;i<4;++i){
        if(biasShape[i]!=1 && biasShape[i]!=scores[i])return bad("invalid reverse bias extent");
        state->biasShape[i]=biasShape[i];
      }
      if(!extent({biasShape[0],biasShape[1],biasShape[2],biasShape[3]},state->bytes[6]))
        return bad("invalid reverse bias size");
    }else if(biasShape)return bad("unexpected reverse bias shape");
    for(size_t i=0;i<4;++i)if(active[i])state->bytes[7+i]=state->bytes[i==3?6:i];
    uint64_t forwardElements=state->bytes[3]/4;
    __uint128_t reverseElements=0;
    for(size_t i=0;i<4;++i)
      if(active[i] || (logical && i<3+biasGradient))reverseElements+=state->bytes[i==3?6:i]/4;
    if(!reverseElements || reverseElements>uint64_t(INT_MAX)*threads ||
       forwardElements>=(1ULL<<31))return bad("reverse grid exceeds bounds");
    state->forwardGrid=unsigned((forwardElements+127)/128);
    state->reverseGrid=unsigned((reverseElements+threads-1)/threads);
    size_t total=0;std::array<size_t,12> offsets{};
    for(size_t i=0;i<12;++i){
      if(total>SIZE_MAX-255)return bad("reverse alignment overflow");
      total=(total+255)&~size_t(255);offsets[i]=total;
      if(state->bytes[i]>SIZE_MAX-total)return bad("reverse arena overflow");
      total+=state->bytes[i];
    }
    if(!currentContext(state->context))return bad("prepared reverse requires current SM120");
    if(!ok(cuCtxGetId(state->context,&state->contextIdentity),"reverse context identity") ||
       !ok(cuModuleLoadData(&state->forwardModule,fimage),"load reverse forward") ||
       !ok(cuModuleGetFunction(&state->forward,state->forwardModule,fentry),"resolve reverse forward") ||
       !ok(cuModuleLoadData(&state->tangentModule,bimage),"load backward") ||
       !ok(cuModuleGetFunction(&state->tangent,state->tangentModule,bentry),"resolve backward") ||
       !ok(cuStreamCreate(&state->stream,CU_STREAM_NON_BLOCKING),"create reverse stream") ||
       !ok(cuMemAlloc(&state->arena,total),"allocate reverse arena"))return 3;
    for(size_t i=0;i<12;++i)state->buffers[i]=state->arena+offsets[i];
    for(auto &event:state->events)if(!ok(cuEventCreate(&event,CU_EVENT_DEFAULT),"create reverse event"))return 3;
    if(nextOwner==0)return bad("owner identity exhausted");
    uint64_t id=nextOwner++;owners.emplace(id,std::move(state));*handle=id;return 0;
  }catch(...){ownerError="native reverse owner allocation failed";return 3;}
}
extern "C" int tessera_nvidia_attention_vjp_invoke(
  uint64_t handle,const void *const *inputs,const size_t *inputBytes,size_t inputCount,
  void *const *outputs,const size_t *outputBytes,size_t outputCount,float *deviceMilliseconds){
  ownerError.clear();
  if(getpid()!=ownerProcess)return bad("prepared attention cannot cross fork");
  std::lock_guard<std::mutex> lock(ownerMutex);
  auto found=owners.find(handle);
  if(found==owners.end() || !found->second->reverse)return bad("prepared reverse is closed or wrong product");
  auto &s=*found->second;CUcontext context=nullptr;unsigned long long identity=0;
  if(!ok(cuCtxGetCurrent(&context),"reverse current context") ||
     !ok(cuCtxGetId(s.context,&identity),"reverse context generation"))return 3;
  if(context!=s.context || identity!=s.contextIdentity)return bad("prepared reverse context generation disagrees");
  const size_t primals=3+unsigned(s.bias);
  if(!inputs || !inputBytes || inputCount!=primals+1 || !outputs || !outputBytes ||
     outputCount!=s.activeCount)return bad("reverse host ABI disagrees");
  auto validSpan=[](const void *p,size_t n){return p && n && reinterpret_cast<uintptr_t>(p)<=UINTPTR_MAX-n;};
  auto overlap=[](const void *a,size_t an,const void *b,size_t bn){
    auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    return x<y+bn && y<x+an;
  };
  for(size_t i=0;i<primals;++i)
    if(!validSpan(inputs[s.mapping[i]],inputBytes[s.mapping[i]]) ||
       inputBytes[s.mapping[i]]!=s.bytes[i==3?6:i])return bad("reverse primal extent disagrees");
  if(!validSpan(inputs[primals],inputBytes[primals]) || inputBytes[primals]!=s.bytes[5])
    return bad("reverse cotangent extent disagrees");
  for(size_t i=0;i<outputCount;++i){
    if(!validSpan(outputs[i],outputBytes[i]) || outputBytes[i]!=s.bytes[7+s.roles[i]])
      return bad("reverse gradient extent disagrees");
    for(size_t j=0;j<i;++j)if(overlap(outputs[i],outputBytes[i],outputs[j],outputBytes[j]))
      return bad("reverse outputs overlap");
    for(size_t j=0;j<inputCount;++j)if(overlap(outputs[i],outputBytes[i],inputs[j],inputBytes[j]))
      return bad("reverse input/output overlap");
  }
  for(size_t i=0;i<primals;++i)
    if(!ok(cuMemcpyHtoD(s.buffers[i==3?6:i],inputs[s.mapping[i]],s.bytes[i==3?6:i]),"upload reverse primal"))return 3;
  if(!ok(cuMemcpyHtoD(s.buffers[5],inputs[primals],s.bytes[5]),"upload cotangent"))return 3;
  std::array<int64_t,11> scalars{};
  for(size_t i=0;i<7;++i)scalars[i]=s.dims[i];
  bool broadcast=s.bias;
  for(size_t i=0;i<4 && s.bias;++i){
    scalars[7+i]=s.biasShape[i];
  }
  if(s.bias){
    const int64_t score[4]={s.dims[0],s.dims[1],s.dims[3],s.dims[4]};
    broadcast=false;for(size_t i=0;i<4;++i)broadcast|=score[i]!=s.biasShape[i];
  }
  const size_t scalarCount=broadcast?11:7;
  void *fa[17]{};size_t arg=0;
  for(size_t i=0;i<3;++i)fa[arg++]=&s.buffers[i];
  if(s.bias)fa[arg++]=&s.buffers[6];
  fa[arg++]=&s.buffers[3];fa[arg++]=&s.buffers[4];
  for(size_t i=0;i<scalarCount;++i)fa[arg++]=&scalars[i];
  void *ba[22]{};arg=0;
  const unsigned baseSlots[5]={5,0,1,2,3};
  for(auto i:baseSlots)ba[arg++]=&s.buffers[i];
  if(s.bias)ba[arg++]=&s.buffers[6];
  ba[arg++]=&s.buffers[4];
  for(size_t role=0;role<4;++role)if(s.bytes[7+role])ba[arg++]=&s.buffers[7+role];
  for(size_t i=0;i<scalarCount;++i)ba[arg++]=&scalars[i];
  // Synchronous host uploads use the default stream. Complete that stream
  // before the private nonblocking stream reads the new saved generation.
  if(!ok(cuStreamSynchronize(nullptr),"complete reverse uploads") ||
     !ok(cuEventRecord(s.events[0],s.stream),"start saved forward") ||
     !ok(cuLaunchKernel(s.forward,s.forwardGrid,1,1,128,1,1,0,s.stream,fa,nullptr),"launch saved forward") ||
     !ok(cuEventRecord(s.events[1],s.stream),"end saved forward") ||
     !ok(cuEventRecord(s.events[2],s.stream),"start backward") ||
     !ok(cuLaunchKernel(s.tangent,s.reverseGrid,1,1,s.reverseThreads,1,1,0,s.stream,ba,nullptr),"launch backward") ||
     !ok(cuEventRecord(s.events[3],s.stream),"end backward") ||
     !ok(cuStreamSynchronize(s.stream),"complete reverse product"))return 3;
  for(size_t i=0;i<outputCount;++i)
    if(!ok(cuMemcpyDtoH(outputs[i],s.buffers[7+s.roles[i]],outputBytes[i]),"download requested gradient"))return 3;
  if(deviceMilliseconds &&
     (!ok(cuEventElapsedTime(&deviceMilliseconds[0],s.events[0],s.events[1]),"saved forward elapsed") ||
      !ok(cuEventElapsedTime(&deviceMilliseconds[1],s.events[2],s.events[3]),"backward elapsed")))return 3;
  return 0;
}
extern "C" int tessera_nvidia_attention_vjp_close(uint64_t handle){
  ownerError.clear();
  if(getpid()!=ownerProcess)return bad("prepared attention cannot cross fork");
  std::lock_guard<std::mutex> lock(ownerMutex);
  auto found=owners.find(handle);
  if(found==owners.end() || !found->second->reverse)return bad("prepared reverse is closed or wrong product");
  owners.erase(found);return 0;
}
