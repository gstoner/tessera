"""Compile the native movement service against a controlled HIP ABI."""
from pathlib import Path
import shutil
import subprocess
import pytest


def test_native_movement_capacity_context_completion_and_failure_recovery(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires host C++ compiler")
    hip = tmp_path/"hip"
    hip.mkdir()
    (hip/"hip_runtime.h").write_text(r"""
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <map>
#include <vector>
using hipCtx_t=void*;using hipFunction_t=void*;
constexpr int hipSuccess=0,hipMemcpyHostToDevice=1,hipMemcpyDeviceToHost=2;
struct hipDeviceProp_t {char gcnArchName[256];};
inline thread_local int device=0;
inline thread_local uintptr_t context=1;
inline bool failAllocate=false,failCopy=false,failFree=false,failCompletion=false,failAfterLaunch=false;
inline std::map<void*,size_t> allocations;
inline int leaseCount=0,synchronizations=0,launches=0,copies=0;
inline bool failConsumer=false;
inline void (*launchHook)()=nullptr;
inline int hipInit(unsigned){return 0;}
inline int hipGetDevice(int* p){*p=device;return 0;}
inline int hipCtxGetCurrent(void** p){*p=reinterpret_cast<void*>(context);return 0;}
inline int hipGetDeviceProperties(hipDeviceProp_t* p,int d){
 std::strcpy(p->gcnArchName,d?"gfx1201":"gfx1151");return 0;
}
inline int hipDeviceSynchronize(){++synchronizations;return failCompletion?1:0;}
inline int hipMalloc(void** p,size_t n){
 if(failAllocate)return 1;
 *p=std::malloc(n);if(!*p)return 1;allocations[*p]=n;return 0;
}
inline int hipFree(void* p){
 if(failFree)return 1;
 if(!allocations.erase(p))return 1;std::free(p);return 0;
}
inline int hipMemcpy(void* dst,const void* src,size_t n,int kind){
 if(failCopy)return 1;
 auto found=allocations.find(kind==1?dst:const_cast<void*>(src));
 if(found==allocations.end()||found->second<n)return 1;
 ++copies;std::memcpy(dst,src,n);return 0;
}
using hipStream_t=void*;using hipEvent_t=void*;
using hipError_t=int;using hipGraph_t=void*;using hipGraphExec_t=void*;using hipGraphNode_t=void*;
using hipGraphNodeType=int;
constexpr int hipErrorInvalidValue=1,hipStreamCaptureModeThreadLocal=1,hipGraphNodeTypeKernel=0;
// This controlled harness tests direct ownership/failures, not graph execution.
// Graph APIs refuse explicitly; real graph proof belongs to owning-device tests.
inline int hipStreamBeginCapture(void*,int){return hipErrorInvalidValue;}
inline int hipStreamEndCapture(void*,void** p){*p=nullptr;return hipErrorInvalidValue;}
inline int hipGraphGetNodes(void*,void**,size_t*){return hipErrorInvalidValue;}
inline int hipGraphNodeGetType(void*,int*){return hipErrorInvalidValue;}
inline int hipGraphInstantiateWithFlags(void**,void*,uint64_t){return hipErrorInvalidValue;}
inline int hipGraphLaunch(void*,void*){return hipErrorInvalidValue;}
inline int hipGraphExecDestroy(void*){return hipErrorInvalidValue;}
inline int hipGraphDestroy(void*){return hipErrorInvalidValue;}
constexpr int hipStreamNonBlocking=1;
inline int hipStreamCreateWithFlags(void** p,unsigned){*p=new int(1);return 0;}
inline int hipStreamSynchronize(void*){return hipDeviceSynchronize();}
inline int hipStreamDestroy(void* p){delete static_cast<int*>(p);return 0;}
inline int hipEventCreate(void** p){*p=new int(1);return 0;}
inline int hipEventRecord(void*,void*){return 0;}
inline int hipEventDestroy(void* p){delete static_cast<int*>(p);return 0;}
inline int hipEventElapsedTime(float* p,void*,void*){*p=.001f;return 0;}
inline int hipMemcpyAsync(void* d,const void* s,size_t n,int kind,void*){return hipMemcpy(d,s,n,kind);}
using hipError_t=int;
using hipGraphNode_t=void*;using hipGraphNodeType=int;
constexpr int hipStreamCaptureModeThreadLocal=1,hipGraphNodeTypeKernel=0;
struct hipGraphRecord {
 struct Launch {void* fn;unsigned gx,bx;void* stream;void** args;};
 std::vector<Launch> nodes;
};
using hipGraph_t=hipGraphRecord*;using hipGraphExec_t=hipGraphRecord*;
inline thread_local hipGraphRecord* capturing=nullptr;
inline int hipModuleLaunchKernel(hipFunction_t fn,unsigned gx,unsigned,unsigned,
 unsigned bx,unsigned,unsigned,unsigned,void* stream,void** args,void**);
inline int hipStreamBeginCapture(void*,int){
 if(capturing)return 1;capturing=new hipGraphRecord;return 0;
}
inline int hipStreamEndCapture(void*,hipGraph_t* g){
 if(!capturing)return 1;*g=capturing;capturing=nullptr;return 0;
}
inline int hipGraphGetNodes(hipGraph_t g,hipGraphNode_t* nodes,size_t* count){
 if(nodes)for(size_t i=0;i<*count&&i<g->nodes.size();++i)nodes[i]=&g->nodes[i];
 *count=g->nodes.size();return 0;
}
inline int hipGraphNodeGetType(hipGraphNode_t,hipGraphNodeType* t){*t=hipGraphNodeTypeKernel;return 0;}
inline int hipGraphInstantiateWithFlags(hipGraphExec_t* e,hipGraph_t g,unsigned long long){
 *e=new hipGraphRecord(*g);return 0;
}
inline int hipGraphLaunch(hipGraphExec_t e,void*){
 for(auto& n:e->nodes){
  auto saved=capturing;capturing=nullptr;
  int rc=hipModuleLaunchKernel(n.fn,n.gx,1,1,n.bx,1,1,0,n.stream,n.args,nullptr);
  capturing=saved;if(rc)return rc;
 }
 return 0;
}
inline int hipGraphExecDestroy(hipGraphExec_t e){delete e;return 0;}
inline int hipGraphDestroy(hipGraph_t g){delete g;return 0;}
inline int hipModuleLaunchKernel(hipFunction_t fn,unsigned gx,unsigned,unsigned,
 unsigned bx,unsigned,unsigned,unsigned,void* stream,void** args,void**){
 if(capturing){capturing->nodes.push_back({fn,gx,bx,stream,args});return 0;}
 if(bx!=256||gx==0)return 1;
 if(launchHook)launchHook();
 if(std::strcmp(static_cast<const char*>(fn),"math_sqrt")==0 ||
    std::strcmp(static_cast<const char*>(fn),"math_add")==0){
  bool binary=std::strcmp(static_cast<const char*>(fn),"math_add")==0;
  auto x=static_cast<float*>(*static_cast<void**>(args[1]));
  auto y=static_cast<float*>(*static_cast<void**>(args[binary?11:6]));
  auto n=*static_cast<int64_t*>(args[binary?15:10]);
  auto b=binary?static_cast<float*>(*static_cast<void**>(args[6])):nullptr;
  for(int64_t i=0;i<n;++i)y[i]=binary?x[i]+b[i]:std::sqrt(x[i]);
  ++launches;if(failAfterLaunch)failCompletion=true;return 0;
 }
 if(std::strcmp(static_cast<const char*>(fn),"softmax")==0){
  if(failConsumer)return 1;
  auto x=static_cast<float*>(*static_cast<void**>(args[1]));
  auto y=static_cast<float*>(*static_cast<void**>(args[6]));
  auto rows=*static_cast<int64_t*>(args[10]),cols=*static_cast<int64_t*>(args[11]);
  for(int64_t r=0;r<rows;++r){
   float high=x[r*cols];for(int64_t c=1;c<cols;++c)high=std::max(high,x[r*cols+c]);
   double sum=0;for(int64_t c=0;c<cols;++c)sum+=std::exp(double(x[r*cols+c]-high));
   for(int64_t c=0;c<cols;++c)y[r*cols+c]=float(std::exp(double(x[r*cols+c]-high))/sum);
  }
  ++launches;return 0;
 }
 auto x=static_cast<float*>(*static_cast<void**>(args[1]));
 auto idx=static_cast<int32_t*>(*static_cast<void**>(args[6]));
 auto out=static_cast<float*>(*static_cast<void**>(args[11]));
 auto dim=[&](int i){return *static_cast<int64_t*>(args[15+i]);};
 if(std::strcmp(static_cast<const char*>(fn),"paged")==0){
  auto page=dim(2),width=dim(3)*dim(4),start=dim(5),tokens=dim(6);
  for(int64_t t=0;t<tokens;++t)for(int64_t c=0;c<width;++c)
   out[t*width+c]=x[(idx[(start+t)/page]*page+(start+t)%page)*width+c];
 }else{
  auto slots=dim(1),width=dim(2);
  for(int64_t t=0;t<slots;++t)for(int64_t c=0;c<width;++c)
   out[t*width+c]=x[idx[t]*width+c];
 }
 ++launches;if(failAfterLaunch)failCompletion=true;return 0;
}
""")
    runtime = Path(__file__).resolve().parents[2]/"src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_movement_runtime.cpp"
    source = tmp_path/"probe.cpp"
    source.write_text('#include "'+str(runtime)+'"\n'+r"""
#include <cassert>
#include <cmath>
#include <thread>
#include <sys/wait.h>
struct Lease {int device;uintptr_t context;};
extern "C" int tessera_rocm_image_acquire(const void*,size_t,const char* name,
 void** lease,void** module,void** function,int* hit){
 *lease=new Lease{device,context};*module=reinterpret_cast<void*>(1);
 *function=const_cast<char*>(name);*hit=1;++leaseCount;return 0;
}
extern "C" int tessera_rocm_image_release(void* p){
 auto lease=static_cast<Lease*>(p);
 if(lease->device!=device||lease->context!=context)return 2;
 delete lease;--leaseCount;return 0;
}
int main(){
 const char image[]={127,'E','L','F',0};
 float pages[24],out[45];for(int i=0;i<24;++i)pages[i]=float(i);
 int32_t table[]={2,0,3,1};int64_t dims[]={4,4,2,1,3,1,5};
 auto paged=[&](const char* arch="gfx1151",int reuse=1){
  return tessera_rocm_movement_launch(image,5,"paged",arch,0,
   pages,sizeof(pages),table,sizeof(table),out,60,dims,7,reuse);
 };
 assert(tessera_rocm_movement_launch(image,5,"paged","gfx1151",0,
  reinterpret_cast<const char*>(pages)+1,sizeof(pages),table,sizeof(table),
  out,60,dims,7,1)==1 && allocations.empty());
 assert(paged()==0&&allocations.size()==3&&leaseCount==0);
 for(int t=0;t<5;++t)for(int c=0;c<3;++c)
  assert(out[t*3+c]==pages[(table[(1+t)/2]*2+(1+t)%2)*3+c]);
 uint64_t a,f,r,l;
 assert(tessera_rocm_movement_stats(&a,&f,&r,&l)==0&&a==3&&r==0&&l==1);
 assert(paged()==0);
 assert(tessera_rocm_movement_stats(&a,&f,&r,&l)==0&&a==3&&r==3&&l==2);
 table[0]=4;int syncBefore=synchronizations;
 assert(paged()==1&&synchronizations==syncBefore);table[0]=2;
 assert(paged("gfx1201")==2&&leaseCount==0);
 float x[35];for(int i=0;i<35;++i)x[i]=float(i+10);
 int32_t tokens[]={6,0,5,2,1,3,4,6,0};int64_t moeDims[]={7,9,5};
 auto moe=[&](int reuse=1){
  return tessera_rocm_movement_launch(image,5,"moe","gfx1151",1,
   x,sizeof(x),tokens,sizeof(tokens),out,sizeof(out),moeDims,3,reuse);
 };
 assert(moe()==0&&allocations.size()==3);
 for(int t=0;t<9;++t)for(int c=0;c<5;++c)assert(out[t*5+c]==x[tokens[t]*5+c]);
 assert(tessera_rocm_movement_stats(&a,&f,&r,&l)==0&&a==6&&f==3);
 assert(moe(0)==0&&allocations.empty());
 failAllocate=true;assert(paged()==4&&leaseCount==0);failAllocate=false;
 assert(paged()==0);
 failCopy=true;assert(paged()==5&&leaseCount==0);failCopy=false;
 assert(paged()==0);
 failAfterLaunch=true;assert(paged()==7&&leaseCount==1&&allocations.size()==3);
 assert(tessera_rocm_movement_clear_current()==7&&leaseCount==1);
 failAfterLaunch=false;failCompletion=false;
 assert(paged()==10&&leaseCount==1);
 assert(tessera_rocm_movement_clear_current()==0&&leaseCount==0&&allocations.empty());
 assert(paged()==0);
 failFree=true;assert(moe()==9);failFree=false;
 assert(tessera_rocm_movement_clear_current()==0&&allocations.empty());
 device=1;context=2;
 assert(paged("gfx1201")==0&&allocations.size()==3);
 device=0;context=1;
 assert(paged()==0&&allocations.size()==6);
 assert(tessera_rocm_movement_clear_current()==0&&allocations.size()==3);
 device=1;context=2;
 assert(tessera_rocm_movement_clear_current()==0&&allocations.empty());
 device=0;context=1;
 assert(paged()==0);
 std::vector<std::thread> threads;
 for(int i=0;i<8;++i)threads.emplace_back([&]{assert(paged()==0);});
 for(auto& t:threads)t.join();
 assert(allocations.size()==3&&leaseCount==0);

 // Prepared calls seal an owned image/shape; caller mutations cannot alter them.
 char mutableImage[]={127,'E','L','F',0};
 uint64_t handle=0;
 assert(tessera_rocm_movement_prepare(mutableImage,5,"paged","gfx1151",0,
                                     dims,7,&handle)==0 && handle);
 auto view=[](void *data,size_t bytes,int dtype,std::initializer_list<int64_t> shape){
  TesseraMovementHostView v{};v.data=data;v.bytes=bytes;v.dtype=dtype;v.rank=shape.size();
  int i=0;for(auto d:shape)v.shape[i++]=d;
  int64_t stride=4;for(int j=v.rank-1;j>=0;--j){v.strides[j]=stride;stride*=v.shape[j];}
  return v;
 };
 TesseraMovementHostView views[]={view(pages,sizeof(pages),1,{4,2,1,3}),
                                  view(table,sizeof(table),2,{4}),
                                  view(out,60,1,{5,1,3})};
 dims[5]=0;mutableImage[0]=0;
 assert(tessera_rocm_movement_invoke(handle,views,3,1)==0);
 for(int t=0;t<5;++t)for(int c=0;c<3;++c)
  assert(out[t*3+c]==pages[(table[(1+t)/2]*2+(1+t)%2)*3+c]);
 auto beforeLaunches=launches;
 views[0].dtype=2;assert(tessera_rocm_movement_invoke(handle,views,3,1)==1);
 views[0].dtype=1;views[0].strides[0]+=4;
 assert(tessera_rocm_movement_invoke(handle,views,3,1)==1);
 views[0].strides[0]-=4;views[2].data=pages;
 assert(tessera_rocm_movement_invoke(handle,views,3,1)==1);views[2].data=out;
 views[1].bytes-=4;assert(tessera_rocm_movement_invoke(handle,views,3,1)==1);
 views[1].bytes+=4;assert(launches==beforeLaunches);
 context=5;assert(tessera_rocm_movement_invoke(handle,views,3,1)==2);context=1;
 table[0]=4;assert(tessera_rocm_movement_invoke(handle,views,3,1)==1);table[0]=2;
 auto forked=fork();
 if(forked==0){assert(tessera_rocm_movement_invoke(handle,views,3,1)==1);_exit(0);}
 int childStatus=0;waitpid(forked,&childStatus,0);assert(childStatus==0);
 assert(tessera_rocm_movement_close(handle)==0);
 assert(tessera_rocm_movement_invoke(handle,views,3,1)==1);
 assert(tessera_rocm_movement_close(handle)==1);
 assert(tessera_rocm_movement_invoke(UINT64_MAX,views,3,1)==1);
 dims[5]=1;mutableImage[0]=127;
 // Close after native invocation acquires ownership but before the kernel reads.
 static uint64_t closingHandle=0;
 assert(tessera_rocm_movement_prepare(mutableImage,5,"paged","gfx1151",0,
                                     dims,7,&closingHandle)==0);
 launchHook=+[](){assert(tessera_rocm_movement_close(closingHandle)==0);launchHook=nullptr;};
 assert(tessera_rocm_movement_invoke(closingHandle,views,3,1)==0);
 for(int t=0;t<5;++t)for(int c=0;c<3;++c)
  assert(out[t*3+c]==pages[(table[(1+t)/2]*2+(1+t)%2)*3+c]);
 assert(tessera_rocm_movement_invoke(closingHandle,views,3,1)==1&&leaseCount==0);


 // Resident owners keep image and private buffers independently of prepare handles.
 uint64_t resident=0,generation=0;float ms=0;
 assert(tessera_rocm_movement_prepare(mutableImage,5,"paged","gfx1151",0,dims,7,&handle)==0);
 assert(tessera_rocm_movement_resident_prepare(handle,&resident)==0&&resident);
 assert(leaseCount==1&&allocations.size()==6);
 assert(tessera_rocm_movement_close(handle)==0);
 assert(tessera_rocm_movement_resident_invoke(resident,&generation,&ms)==10);
 views[0].dtype=2;assert(tessera_rocm_movement_resident_upload(resident,views,2)==1);
 views[0].dtype=1;table[0]=4;
 assert(tessera_rocm_movement_resident_upload(resident,views,2)==1);table[0]=2;
 assert(tessera_rocm_movement_resident_upload(resident,views,2)==0);
 uint64_t refusedNodes=99;
 assert(tessera_rocm_movement_resident_capture(resident,&refusedNodes)==6&&refusedNodes==0);
 assert(tessera_rocm_movement_resident_invoke(resident,&generation,&ms)==0&&generation==1&&ms>0);
 assert(tessera_rocm_movement_resident_read(resident,generation,&views[2])==0);
 for(int t=0;t<5;++t)for(int c=0;c<3;++c)
  assert(out[t*3+c]==pages[(table[(1+t)/2]*2+(1+t)%2)*3+c]);
 uint64_t prior=generation;
 assert(tessera_rocm_movement_resident_invoke(resident,&generation,&ms)==0&&generation>prior);
 assert(tessera_rocm_movement_resident_read(resident,prior,&views[2])==10);
 context=5;assert(tessera_rocm_movement_resident_invoke(resident,&generation,&ms)==2);
 assert(tessera_rocm_movement_resident_close(resident)==2&&allocations.size()==6);context=1;
 failCompletion=true;
 assert(tessera_rocm_movement_resident_close(resident)==7&&leaseCount==1&&allocations.size()==6);
 failCompletion=false;
 assert(tessera_rocm_movement_resident_close(resident)==0&&leaseCount==0&&allocations.size()==3);
 assert(tessera_rocm_movement_resident_close(resident)==1);
 assert(tessera_rocm_movement_resident_read(resident,1,&views[2])==1);
 // Context teardown retires resident owners before image clearing.
 assert(tessera_rocm_movement_prepare(mutableImage,5,"paged","gfx1151",0,dims,7,&handle)==0);
 assert(tessera_rocm_movement_resident_prepare(handle,&resident)==0);
 assert(tessera_rocm_movement_clear_current()==0&&allocations.empty()&&leaseCount==0);
 assert(tessera_rocm_movement_resident_invoke(resident,&generation,&ms)==1);
 assert(tessera_rocm_movement_close(handle)==0);

 // Paged -> softmax retains the intermediate allocation and both image leases.
 uint64_t edge=0;float producerMs=0,consumerMs=0;
 assert(tessera_rocm_movement_prepare(mutableImage,5,"paged","gfx1151",0,dims,7,&handle)==0);
 assert(tessera_rocm_movement_resident_prepare_softmax(handle,image,5,"softmax",3,3,&edge)==1);
 assert(!edge&&allocations.empty()&&leaseCount==0);
 assert(tessera_rocm_movement_resident_prepare_softmax(handle,image,5,"softmax",5,3,&edge)==0);
 assert(allocations.size()==4&&leaseCount==2);
 assert(tessera_rocm_movement_close(handle)==0);
 assert(tessera_rocm_movement_resident_upload(edge,views,2)==0);
 int copiesBefore=copies,launchesBefore=launches;
 assert(tessera_rocm_movement_resident_invoke_softmax(edge,&generation,&producerMs,&consumerMs)==0);
 assert(launches==launchesBefore+2&&copies==copiesBefore);
 assert(producerMs>0&&consumerMs>0);
 assert(tessera_rocm_movement_resident_read(edge,generation,&views[2])==0);
 for(int t=0;t<5;++t){
  double sum=0;for(int c=0;c<3;++c)sum+=std::exp(double(pages[(table[(1+t)/2]*2+(1+t)%2)*3+c])-pages[(table[(1+t)/2]*2+(1+t)%2)*3+2]);
  for(int c=0;c<3;++c)assert(std::abs(out[t*3+c]-std::exp(double(pages[(table[(1+t)/2]*2+(1+t)%2)*3+c])-pages[(table[(1+t)/2]*2+(1+t)%2)*3+2])/sum)<1e-6);
 }
 prior=generation;failConsumer=true;
 assert(tessera_rocm_movement_resident_invoke_softmax(edge,&generation,&producerMs,&consumerMs)==6&&generation==0);
 assert(tessera_rocm_movement_resident_read(edge,prior,&views[2])==10);
 assert(allocations.size()==4&&leaseCount==2);failConsumer=false;
 assert(tessera_rocm_movement_resident_invoke_softmax(edge,&generation,&producerMs,&consumerMs)==0);
 assert(tessera_rocm_movement_clear_current()==0&&allocations.empty()&&leaseCount==0);
 assert(tessera_rocm_movement_resident_invoke_softmax(edge,&generation,&producerMs,&consumerMs)==1);
 // Math shares the same checked capacity arena without retaining input content.
 float mathA[17],mathB[17],mathOut[17];int64_t mathDims[]={17};
 for(int i=0;i<17;++i){mathA[i]=float(i+1);mathB[i]=float(i+2);}
 auto math=[&](int family=0,int reuse=1,const char* arch="gfx1151") {
  return tessera_rocm_math_launch(image,5,family?"math_add":"math_sqrt",arch,
   family,4,mathA,sizeof(mathA),family?mathB:nullptr,family?sizeof(mathB):0,
   mathOut,sizeof(mathOut),mathDims,1,reuse);
 };
 assert(math()==0&&allocations.size()==2&&leaseCount==0);
 for(int i=0;i<17;++i)assert(mathOut[i]==std::sqrt(mathA[i]));
 assert(tessera_rocm_movement_stats(&a,&f,&r,&l)==0);
 auto beforeAlloc=a,beforeReuse=r;
 for(int i=0;i<17;++i)mathA[i]*=4;
 assert(math()==0);
 assert(tessera_rocm_movement_stats(&a,&f,&r,&l)==0&&a==beforeAlloc&&r==beforeReuse+2);
 for(int i=0;i<17;++i)assert(mathOut[i]==std::sqrt(mathA[i]));
 assert(math(1)==0&&allocations.size()==3);
 for(int i=0;i<17;++i)assert(mathOut[i]==mathA[i]+mathB[i]);
 assert(math(1,0)==0&&allocations.empty());
 int priorSync=synchronizations;
 assert(tessera_rocm_math_launch(image,5,"math_sqrt","gfx1151",0,4,
  mathA,sizeof(mathA)-4,nullptr,0,mathOut,sizeof(mathOut),mathDims,1,1)==1);
 assert(tessera_rocm_math_launch(image,5,"math_sqrt","gfx1151",0,4,
  mathA,sizeof(mathA),nullptr,0,mathA,sizeof(mathA),mathDims,1,1)==1);
 assert(synchronizations==priorSync&&allocations.empty());
 assert(math(0,1,"gfx1201")==2&&allocations.empty());
 mathDims[0]=INT64_MAX;
 assert(math()==1&&allocations.empty());mathDims[0]=17;
 failAllocate=true;assert(math()==4&&leaseCount==0);failAllocate=false;
 assert(math()==0);
 failCopy=true;assert(math()==5&&leaseCount==0);failCopy=false;
 assert(math()==0);
 failAfterLaunch=true;
 assert(math()==7&&leaseCount==1);
 assert(tessera_rocm_movement_clear_current()==7&&leaseCount==1);
 failAfterLaunch=false;failCompletion=false;
 assert(math()==10&&leaseCount==1);
 assert(tessera_rocm_movement_clear_current()==0&&allocations.empty()&&leaseCount==0);
 assert(math(0,0)==0&&allocations.empty());
 // Interleaving math and movement reuses raw capacity, never old contents/types.
 assert(math()==0);
 assert(paged()==0);
 for(int t=0;t<5;++t)for(int c=0;c<3;++c)
  assert(out[t*3+c]==pages[(table[(1+t)/2]*2+(1+t)%2)*3+c]);
 assert(math(1)==0);
 for(int i=0;i<17;++i)assert(mathOut[i]==mathA[i]+mathB[i]);
 assert(tessera_rocm_movement_clear_current()==0&&allocations.empty()&&leaseCount==0);
 auto child=fork();
 if(child==0){assert(paged()==2);assert(math()==2);_exit(0);}
 int status=0;waitpid(child,&status,0);assert(status==0);
 assert(tessera_rocm_movement_clear_current()==0&&allocations.empty());
}
""")
    binary = tmp_path/"probe"
    subprocess.run([compiler,"-std=c++17","-pthread","-I",str(tmp_path),
                    str(source),"-o",str(binary)],check=True,capture_output=True,text=True)
    subprocess.run([str(binary)],check=True,capture_output=True,text=True)
