// Additive native owner for compiler-derived source pullback programs.
// Uses the same completed-call arena and stream rules as the existing owners.
namespace {
std::map<uint64_t,std::unique_ptr<TensorProgramOwner>> sourceReversePrograms;
int runSourceReverse(TensorProgramOwner &owner,
    const TesseraNvidiaMatmulHostView *views,size_t count,
    const uint64_t *producerStreams,size_t producerCount,
    int repeats,bool captured,float *stageMs,float *programMs) {
  if (!views || count!=4 || (producerStreams ? producerCount!=3 : producerCount!=0))
    return bad("source reverse invocation frame mismatch");
  if (!tensorProgramContext(owner)) return 1;
  for (size_t i=0;i<4;++i)
    if (!tensorProgramView(views[i],owner.buffers[i<3?i:4]))
      return bad("source reverse shape/stride/storage mismatch");
  for (size_t i=0;i<3;++i) {
    auto out=reinterpret_cast<uintptr_t>(views[3].data), root=reinterpret_cast<uintptr_t>(views[i].data);
    if (out<root+views[i].bytes && root<out+views[3].bytes)
      return bad("source reverse output aliases an input");
  }
  std::array<CUdeviceptr,5> bindings{};
  ProducerEvents ordering;
  if (producerStreams) {
    for (size_t i=0;i<3;++i) {
      bindings[i]=reinterpret_cast<CUdeviceptr>(views[i].data);
      if (!residentAllocation(bindings[i],views[i].bytes,owner.context))
        return bad("source reverse resident allocation context/capacity mismatch");
      if (owner.arena->base && bindings[i]<owner.arena->base+owner.arena->capacity &&
          owner.arena->base<bindings[i]+views[i].bytes)
        return bad("source reverse root aliases private scratch");
    }
    if (!ordering.wait(owner.context,owner.stream,producerStreams,producerCount)) return 1;
  }
  if (!owner.arena->grow(owner.bytes) || !owner.arena->growHost(owner.bytes)) return 1;
  ++owner.arena->generation; owner.generation=0;
  auto *host=static_cast<unsigned char *>(owner.arena->host);
  for (size_t i=producerStreams?3:0;i<5;++i)
    bindings[i]=owner.arena->base+owner.offsets[i];
  bool status=true;
  if (!producerStreams) for (size_t i=0;i<3 && status;++i) {
    std::memcpy(host+owner.offsets[i],views[i].data,views[i].bytes);
    status=ok(cuMemcpyHtoDAsync(bindings[i],host+owner.offsets[i],views[i].bytes,owner.stream),
        "upload source reverse root");
  }
  if (status) status=submitTensorProgram(owner,1,nullptr,false,bindings.data());
  if (status && repeats) {
    status=ok(cuStreamSynchronize(owner.stream),"complete source reverse warmup");
    if (status) status=measureLaunches(owner.stream,repeats,captured,[&]() {
      return submitTensorProgram(owner,1,nullptr,false,bindings.data());
    },programMs);
    if (status) status=submitTensorProgram(owner,repeats,stageMs,captured,bindings.data());
  }
  if (status) status=ok(cuMemcpyDtoHAsync(host+owner.offsets[4],bindings[4],
      owner.buffers[4].bytes,owner.stream),"read source reverse gradient");
  const std::string saved=error;
  bool completed=ok(cuStreamSynchronize(owner.stream),"complete source reverse program");
  if (!status) {error=saved;return 1;}
  if (!completed) return 1;
  std::memcpy(views[3].data,host+owner.offsets[4],owner.buffers[4].bytes);
  return 0;
}
}
extern "C" int tessera_nvidia_source_reverse_prepare_v1(
    const TesseraNvidiaMatmulHostView *buffers,size_t count,const int64_t *roles,
    const TesseraNvidiaActiveSourceProgramStep *steps,size_t stageCount,uint64_t *handle) {
  error.clear();if(handle)*handle=0;
  if(getpid()!=process)return bad("source reverse program cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);
  if(!buffers || !roles || !steps || !handle || count!=5 || stageCount!=2 ||
      roles[0]<0 || roles[0]>1 || roles[1]!=1-roles[0] || roles[2]!=2)
    return bad("source reverse preparation frame mismatch");
  try {
    auto owner=std::make_unique<TensorProgramOwner>();
    if(!current(owner->context) || !ok(cuCtxGetId(owner->context,&owner->identity),
        "source reverse context identity")) return bad("source reverse requires live SM120 context");
    const auto &source=buffers[roles[0]], &rhs=buffers[roles[1]];
    int64_t m=source.shape[0],k=source.shape[1],n=rhs.shape[1];
    if((source.dtype!=2 && source.dtype!=3) || rhs.dtype!=source.dtype ||
        rhs.shape[0]!=k || m<=0 || m>65535LL*16 || k<=0 || k>INT32_MAX || n<=0 || n>INT32_MAX)
      return bad("source reverse logical envelope mismatch");
    for(size_t i=0;i<5;++i) {
      bool right=i==size_t(roles[1]),seed=i==2;
      int64_t rows=right?k:m,cols=right||seed?n:k;
      int dtype=seed?1:source.dtype;size_t width=seed?4:2;
      __int128 bytes=(__int128)rows*cols*width;
      if(bytes<=0 || bytes>INT64_MAX || buffers[i].data || buffers[i].rank!=2 ||
          buffers[i].dtype!=dtype || buffers[i].shape[0]!=rows || buffers[i].shape[1]!=cols ||
          buffers[i].bytes!=size_t(bytes) || buffers[i].strides[0]!=int64_t(cols*width) ||
          buffers[i].strides[1]!=int64_t(width) || owner->bytes>size_t(INT64_MAX)-255 ||
          size_t(bytes)>size_t(INT64_MAX)-owner->bytes-255)
        return bad("source reverse typed buffer capacity mismatch");
      owner->buffers.push_back(buffers[i]);owner->offsets.push_back(owner->bytes);
      owner->bytes=(owner->bytes+size_t(bytes)+255)&~size_t(255);
    }
    for(size_t i=0;i<2;++i) {
      const auto &s=steps[i];bool matmul=i==0;
      std::array<int64_t,3> dims=matmul?std::array<int64_t,3>{m,k,n}:std::array<int64_t,3>{m,k,0};
      if(!s.image || !s.image_bytes || !s.entry || s.schema!=1 || s.reserved ||
          s.kind!=(matmul?3:4) || s.storage!=source.dtype || s.input_count!=2 ||
          s.output_count!=1 || s.inputs[0]!=(matmul?2:roles[0]) ||
          s.inputs[1]!=(matmul?roles[1]:3) || s.outputs[0]!=int64_t(3+i) || s.outputs[1]!=-1 ||
          std::array<int64_t,3>{s.dimensions[0],s.dimensions[1],s.dimensions[2]}!=dims)
        return bad("source reverse SSA stage contract mismatch");
      const std::string entry=s.entry;
      if(matmul) {
        const std::string suffix="_scalar_fma_kernel";
        if(entry.size()<=suffix.size() || entry.compare(entry.size()-suffix.size(),suffix.size(),suffix)!=0 ||
            entry.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.$")!=std::string::npos)
          return bad("source reverse scalar entry mismatch");
      } else {
        const std::string prefix="tessera_native_row_";
        if(entry.size()!=prefix.size()+16 || entry.rfind(prefix,0)!=0 ||
            entry.find_first_not_of("0123456789abcdef",prefix.size())!=std::string::npos)
          return bad("source reverse row entry mismatch");
      }
      owner->stages.emplace_back();auto &stage=owner->stages.back();
      std::string image(static_cast<const char *>(s.image),s.image_bytes);
      if(!ok(cuModuleLoadData(&stage.module,image.c_str()),"load source reverse PTX") ||
          !ok(cuModuleGetFunction(&stage.function,stage.module,s.entry),"resolve source reverse entry"))return 1;
      size_t parameters=matmul?6:5;
      for(size_t p=0;p<parameters;++p) {
        size_t offset=0,bytes=0;
        if(!ok(cuFuncGetParamInfo(stage.function,p,&offset,&bytes),"inspect source reverse parameter") ||
            offset!=8*p || bytes!=8)return bad("source reverse parameter ABI mismatch");
      }
      size_t offset=0,bytes=0;
      if(cuFuncGetParamInfo(stage.function,parameters,&offset,&bytes)!=CUDA_ERROR_INVALID_VALUE)
        return bad("source reverse parameter count mismatch");
      stage.matmul=matmul;stage.rowPullback=!matmul;stage.inputs[0]=s.inputs[0];stage.inputs[1]=s.inputs[1];
      stage.output=3+i;stage.dims=dims;
      stage.gridX=unsigned(matmul?(k+7)/8:(m+127)/128);stage.gridY=unsigned(matmul?(m+15)/16:1);
    }
    if(!ok(cuStreamCreate(&owner->stream,CU_STREAM_NON_BLOCKING),"create source reverse stream"))return 1;
    owner->arena=scratch[owner->identity].lock();
    if(!owner->arena) {
      owner->arena=std::make_shared<Scratch>();owner->arena->context=owner->context;
      owner->arena->identity=owner->identity;scratch[owner->identity]=owner->arena;
    }
    uint64_t id=nextHandle++;sourceReversePrograms.emplace(id,std::move(owner));*handle=id;return 0;
  }catch(...){return bad("source reverse preparation failed");}
}
extern "C" int tessera_nvidia_source_reverse_invoke_v1(uint64_t handle,
    const TesseraNvidiaMatmulHostView *views,size_t count,const uint64_t *streams,size_t streamCount) {
  error.clear();if(getpid()!=process)return bad("source reverse program cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);auto found=sourceReversePrograms.find(handle);
  if(found==sourceReversePrograms.end())return bad("source reverse owner is closed");
  return runSourceReverse(*found->second,views,count,streams,streamCount,0,false,nullptr,nullptr);
}
extern "C" int tessera_nvidia_source_reverse_profile_v1(uint64_t handle,
    const TesseraNvidiaMatmulHostView *views,size_t count,const uint64_t *streams,size_t streamCount,
    int repeats,int captured,float *stageMs,size_t stageCount,float *programMs) {
  error.clear();if(getpid()!=process)return bad("source reverse program cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);auto found=sourceReversePrograms.find(handle);
  if(found==sourceReversePrograms.end() || !stageMs || stageCount!=2 || !programMs ||
      (captured!=0 && captured!=1) || repeats<=0 || repeats>(captured?4096:1000000))
    return bad("source reverse profile frame mismatch");
  return runSourceReverse(*found->second,views,count,streams,streamCount,repeats,bool(captured),stageMs,programMs);
}
extern "C" int tessera_nvidia_source_reverse_close_v1(uint64_t handle) {
  error.clear();if(getpid()!=process)return bad("source reverse program cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);auto found=sourceReversePrograms.find(handle);
  if(found==sourceReversePrograms.end())return bad("source reverse owner is closed");
  sourceReversePrograms.erase(found);return 0;
}
