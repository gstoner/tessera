#include <cuda.h>
#include <cstdint>
#include <cstddef>
#include <climits>
#include <cstdio>
#include <cstring>
static int checked(CUresult r,const char*where){
  if(r==CUDA_SUCCESS)return 0;
  const char *msg=nullptr;cuGetErrorString(r,&msg);
  std::fprintf(stderr,"%s: %s\n",where,msg?msg:"CUDA error");return int(r);
}
#define TRY(call) if(int rc=checked((call),#call))return rc
extern "C" int benchmark(const char*ptx,const char*entry,const void*source,void*output,
 int64_t rows,int64_t cols,float*uncaptured,float*captured,int samples,int repeats) {
 if(!ptx||!entry||!source||!output||rows<=0||cols<=0||
    rows>=(1LL<<31)||cols>=(1LL<<31)||rows>(1LL<<31)/cols||
    samples<1||repeats<1)return -1;
 TRY(cuInit(0));CUdevice dev;TRY(cuDeviceGet(&dev,0));
 CUcontext ctx;TRY(cuDevicePrimaryCtxRetain(&ctx,dev));TRY(cuCtxSetCurrent(ctx));
 CUmodule module;TRY(cuModuleLoadData(&module,ptx));CUfunction function;
 TRY(cuModuleGetFunction(&function,module,entry));
 CUstream stream;TRY(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING));
 CUdeviceptr x,y;size_t bytes=size_t(rows)*size_t(cols)*2;
 TRY(cuMemAlloc(&x,bytes));TRY(cuMemAlloc(&y,bytes));
 TRY(cuMemcpyHtoD(x,source,bytes));
 void *args[]={&x,&y,&rows,&cols};
 unsigned grid=unsigned(((std::strstr(entry,"_flat_128_") ? rows*cols : rows)+127)/128);
 auto launch=[&](){return cuLaunchKernel(function,grid,1,1,128,1,1,0,stream,args,nullptr);};
 for(int i=0;i<30;++i)TRY(launch());
 TRY(cuStreamSynchronize(stream));
 TRY(cuMemcpyDtoH(output,y,bytes));
 CUgraph graph;CUgraphExec executable;
 TRY(cuStreamBeginCapture(stream,CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
 for(int i=0;i<repeats;++i)TRY(launch());
 TRY(cuStreamEndCapture(stream,&graph));
 TRY(cuGraphInstantiate(&executable,graph,0));
 TRY(cuGraphLaunch(executable,stream));TRY(cuStreamSynchronize(stream));
 CUevent start,end;TRY(cuEventCreate(&start,0));TRY(cuEventCreate(&end,0));
 // Alternate order to avoid assigning a drift trend to one timing mode.
 for(int s=0;s<samples;++s){
   for(int order=0;order<2;++order){
     bool capture=((s+order)&1)!=0;
     TRY(cuEventRecord(start,stream));
     if(capture){TRY(cuGraphLaunch(executable,stream));}
     else for(int i=0;i<repeats;++i)TRY(launch());
     TRY(cuEventRecord(end,stream));TRY(cuEventSynchronize(end));
     float ms;TRY(cuEventElapsedTime(&ms,start,end));
     (capture?captured:uncaptured)[s]=ms/repeats;
   }
 }
 TRY(cuMemcpyDtoH(output,y,bytes));
 TRY(cuEventDestroy(start));TRY(cuEventDestroy(end));
 TRY(cuGraphExecDestroy(executable));TRY(cuGraphDestroy(graph));
 TRY(cuMemFree(x));TRY(cuMemFree(y));TRY(cuModuleUnload(module));
 TRY(cuStreamDestroy(stream));TRY(cuDevicePrimaryCtxRelease(dev));return 0;
}
