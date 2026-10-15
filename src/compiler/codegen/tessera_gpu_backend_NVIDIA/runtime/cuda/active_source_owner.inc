// Kept separate for review; compiled in the same native owner translation unit.
extern "C" int tessera_nvidia_active_source_program_prepare(
    const TesseraNvidiaMatmulHostView *buffers,size_t count,const int64_t *roles,
    const TesseraNvidiaActiveSourceProgramStep *steps,size_t stageCount,uint64_t *handle) {
  error.clear(); if (handle) *handle=0;
  if (getpid()!=process) return bad("active source program cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);
  if (!buffers || !roles || !steps || !handle || count!=7 || stageCount!=3 ||
      roles[0]<0 || roles[0]>1 || roles[1]!=1-roles[0] || roles[2]!=2)
    return bad("invalid active source program frame");
  try {
    auto owner=std::make_unique<TensorProgramOwner>(); owner->sourceActive=true;
    if (!current(owner->context) || !ok(cuCtxGetId(owner->context,&owner->identity),"active source context identity"))
      return bad("active source program requires live SM120 context");
    const auto &source=buffers[roles[0]], &rhs=buffers[roles[1]];
    const int64_t m=source.shape[0], k=source.shape[1], n=rhs.shape[1];
    if ((source.dtype!=2 && source.dtype!=3) || rhs.dtype!=source.dtype || rhs.shape[0]!=k ||
        m<=0 || m>65535LL*16 || k<=0 || k>=(1LL<<31) || n<=0 || n>=(1LL<<31))
      return bad("active source program storage/shape mismatch");
    for (size_t i=0;i<count;++i) {
      const bool output=i>=5, right=i==size_t(roles[1]);
      const int64_t rows=right?k:m, cols=right?n:output?n:k;
      const int dtype=output?1:source.dtype; const size_t width=dtype==1?4:2;
      const __int128 bytes=(__int128)rows*cols*width;
      if (bytes<=0 || bytes>INT64_MAX || buffers[i].data || buffers[i].rank!=2 || buffers[i].dtype!=dtype ||
          buffers[i].shape[0]!=rows || buffers[i].shape[1]!=cols || buffers[i].bytes!=size_t(bytes) ||
          buffers[i].strides[0]!=int64_t(cols*width) || buffers[i].strides[1]!=int64_t(width) ||
          owner->bytes>size_t(INT64_MAX)-255 || size_t(bytes)>size_t(INT64_MAX)-owner->bytes-255)
        return bad("active source program capacity/layout mismatch");
      owner->buffers.push_back(buffers[i]); owner->offsets.push_back(owner->bytes);
      owner->bytes=(owner->bytes+size_t(bytes)+255)&~size_t(255);
    }
    for (size_t i=0;i<3;++i) {
      const auto &s=steps[i]; const bool matmul=i!=0;
      const int64_t a=matmul?int64_t(i+2):roles[0], b=matmul?roles[1]:roles[2];
      const int64_t y=matmul?int64_t(i+4):3, dy=matmul?-1:4;
      const std::array<int64_t,3> dims=matmul?std::array<int64_t,3>{m,n,k}:std::array<int64_t,3>{m,k,0};
      if (!s.image || !s.image_bytes || !s.entry || s.schema!=1 || s.reserved ||
          s.kind!=(matmul?1:2) || s.storage!=source.dtype || s.input_count!=2 || s.output_count!=(matmul?1:2) ||
          s.inputs[0]!=a || s.inputs[1]!=b || s.outputs[0]!=y || s.outputs[1]!=dy ||
          std::array<int64_t,3>{s.dimensions[0],s.dimensions[1],s.dimensions[2]}!=dims)
        return bad("active source program SSA member/dimension mismatch");
      const std::string entry=s.entry;
      if (matmul) {
        if (entry!="nvidia_sm120_scheduled_matmul_native_tensor_consumer_kernel")
          return bad("active source matmul entry contract mismatch");
      } else {
        const std::string prefix="tessera_native_row_";
        if (entry.size()!=prefix.size()+16 || entry.rfind(prefix,0)!=0 ||
            entry.find_first_not_of("0123456789abcdef",prefix.size())!=std::string::npos)
          return bad("active source row entry contract mismatch");
      }
      owner->stages.emplace_back(); auto &stage=owner->stages.back();
      for (size_t j=0;j<i;++j)
        if (s.image_bytes==steps[j].image_bytes && std::strcmp(s.entry,steps[j].entry)==0 &&
            std::memcmp(s.image,steps[j].image,s.image_bytes)==0) {
          stage.function=owner->stages[j].function; break;
        }
      if (!stage.function) {
        std::string image(static_cast<const char *>(s.image),s.image_bytes);
        if (!ok(cuModuleLoadData(&stage.module,image.c_str()),"load active source PTX") ||
            !ok(cuModuleGetFunction(&stage.function,stage.module,s.entry),"resolve active source entry")) return 1;
      }
      for (size_t param=0;param<6;++param) {
        size_t offset=0,bytes=0;
        if (!ok(cuFuncGetParamInfo(stage.function,param,&offset,&bytes),"inspect active source parameter") ||
            offset!=param*8 || bytes!=8) return bad("active source member parameter ABI mismatch");
      }
      size_t offset=0,bytes=0;
      if (cuFuncGetParamInfo(stage.function,6,&offset,&bytes)!=CUDA_ERROR_INVALID_VALUE)
        return bad("active source member parameter count mismatch");
      stage.matmul=matmul; stage.inputs[0]=a; stage.inputs[1]=b;
      stage.output=y; stage.secondOutput=dy; stage.dims=dims;
      stage.gridX=unsigned(matmul?(n+7)/8:(m+127)/128);
      stage.gridY=unsigned(matmul?(m+15)/16:1);
    }
    if (!ok(cuStreamCreate(&owner->stream,CU_STREAM_NON_BLOCKING),"create active source stream")) return 1;
    owner->arena=scratch[owner->identity].lock();
    if (!owner->arena) {
      owner->arena=std::make_shared<Scratch>();owner->arena->context=owner->context;
      owner->arena->identity=owner->identity;scratch[owner->identity]=owner->arena;
    }
    const uint64_t id=nextHandle++; tensorPrograms.emplace(id,std::move(owner)); *handle=id;
    return 0;
  } catch (...) { return bad("active source program preparation failed"); }
}
extern "C" int tessera_nvidia_active_source_program_invoke(uint64_t handle,
    const TesseraNvidiaMatmulHostView *views,size_t count) {
  error.clear();if (getpid()!=process) return bad("active source program cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);auto found=tensorPrograms.find(handle);
  if (found==tensorPrograms.end() || !found->second->sourceActive || !views || count!=5)
    return bad("active source host frame mismatch");
  return invokeTensorProgram(*found->second,views);
}
extern "C" int tessera_nvidia_active_source_program_profile(uint64_t handle,int repeats,int captured,
    float *stageMs,size_t count,float *programMs,const TesseraNvidiaMatmulHostView *views,size_t viewCount) {
  error.clear();if (getpid()!=process) return bad("active source program cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);auto found=tensorPrograms.find(handle);
  if (found==tensorPrograms.end() || !found->second->sourceActive) return bad("active source profile owner is closed");
  auto &owner=*found->second;
  if (!views || viewCount!=5 || !stageMs || !programMs || count!=3 ||
      (captured!=0 && captured!=1) || repeats<=0 || repeats>(captured?4096:1000000) ||
      (captured && count>65536/size_t(repeats))) return bad("active source profiling count mismatch");
  if (!owner.generation || owner.generation!=owner.arena->generation || owner.base!=owner.arena->base)
    return bad("active source profiling lease is stale");
  for (size_t i=0;i<3;++i)
    if (!tensorProgramView(views[i],owner.buffers[i]) || views[i].data!=owner.lastInputs[i].data)
      return bad("active source profiling roots differ from native snapshot");
  if (!tensorProgramOutputs(owner,views+3) || !tensorProgramFrameAliases(views) || !tensorProgramContext(owner)) return 1;
  bool status=measureLaunches(owner.stream,repeats,bool(captured),[&](){return submitTensorProgram(owner);},programMs);
  if (status) status=submitTensorProgram(owner,repeats,stageMs,bool(captured));
  if (status) status=readTensorProgram(owner,views+3);
  if (!status) {const std::string saved=error;cuStreamSynchronize(owner.stream);owner.generation=0;error=saved;return 1;}
  return 0;
}
extern "C" int tessera_nvidia_active_source_program_scratch_stats(uint64_t handle,size_t *capacity,size_t *allocations) {
  error.clear();if (getpid()!=process) return bad("active source program cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);auto found=tensorPrograms.find(handle);
  if (found==tensorPrograms.end() || !found->second->sourceActive || !capacity || !allocations)
    return bad("invalid active source scratch query");
  *capacity=found->second->arena->capacity;*allocations=found->second->arena->allocations;return 0;
}
extern "C" int tessera_nvidia_active_source_program_close(uint64_t handle) {
  error.clear();if (getpid()!=process) return bad("active source program cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);auto found=tensorPrograms.find(handle);
  if (found==tensorPrograms.end() || !found->second->sourceActive) return bad("active source owner is closed or unknown");
  tensorPrograms.erase(found);return 0;
}
