"""Host failure injection for native NVFP4 allocation/image/copy lifetime."""
from pathlib import Path
import shutil
import subprocess
import pytest

def test_native_nvfp4_completion_quarantine_and_cleanup(tmp_path):
    compiler=shutil.which("c++")
    if compiler is None:pytest.skip("host C++ compiler required")
    hip=tmp_path/"hip";hip.mkdir()
    (hip/"hip_runtime.h").write_text(r"""
#pragma once
#include <cstdlib>
#include <cstring>
#include <map>
#include <cstdint>
using hipCtx_t=void*;using hipFunction_t=void*;using hipStream_t=void*;using hipEvent_t=void*;
constexpr int hipSuccess=0,hipMemcpyHostToDevice=1,hipMemcpyDeviceToHost=2,hipStreamNonBlocking=1;
struct hipDeviceProp_t {char gcnArchName[256];};
inline uintptr_t fakeContext=1;
inline bool failCompletion=false,failCopy=false,failAfterLaunch=false,failEventDestroy=false;
inline int leases=0,launchCount=0,freeCountdown=-1;
inline std::map<void*,size_t> allocations;
inline std::map<void*,bool> eventObjects;
inline int hipInit(unsigned){return 0;}
inline int hipGetDevice(int *p){*p=0;return 0;}
inline int hipCtxGetCurrent(void **p){*p=reinterpret_cast<void*>(fakeContext);return 0;}
inline int hipGetDeviceProperties(hipDeviceProp_t *p,int){std::strcpy(p->gcnArchName,"gfx1201");return 0;}
inline int hipStreamCreateWithFlags(void **p,unsigned){*p=new int(1);return 0;}
inline int hipStreamSynchronize(void*){return failCompletion?1:0;}
inline int hipStreamDestroy(void *p){delete static_cast<int*>(p);return 0;}
inline int hipMalloc(void **p,size_t n){*p=std::calloc(1,n);allocations[*p]=n;return *p?0:1;}
inline int hipFree(void *p){if(freeCountdown==0)return 1;if(freeCountdown>0)--freeCountdown;if(!allocations.erase(p))return 1;std::free(p);return 0;}
inline int hipMemcpyAsync(void *d,const void *s,size_t n,int,void*) {
 if(failCopy)return 1;std::memcpy(d,s,n);return 0;
}
inline int hipEventCreate(void **p){*p=new int(1);eventObjects[*p]=true;return 0;}
inline int hipEventRecord(void*,void*){return 0;}
inline int hipEventSynchronize(void*){return failCompletion?1:0;}
inline int hipEventElapsedTime(float *p,void*,void*){*p=.03f;return 0;}
inline int hipEventDestroy(void *p) {
 if(failEventDestroy)return 1;
 if(!eventObjects.erase(p))return 1;delete static_cast<int*>(p);return 0;
}

struct MockGraph {size_t nodes;};
using hipGraph_t=MockGraph*;using hipGraphExec_t=MockGraph*;
enum hipStreamCaptureStatus {hipStreamCaptureStatusNone,hipStreamCaptureStatusActive};
constexpr int hipStreamCaptureModeThreadLocal=1;
inline bool capturing=false,failEndCapture=false,failGraphDestroy=false,wrongNodes=false;
inline size_t capturedNodes=0;
inline std::map<MockGraph*,bool> graphObjects,graphExecObjects;
inline int hipStreamIsCapturing(void*,hipStreamCaptureStatus *p) {
 *p=capturing?hipStreamCaptureStatusActive:hipStreamCaptureStatusNone;return 0;
}
inline int hipStreamBeginCapture(void*,int) {capturing=true;capturedNodes=0;return 0;}
inline int hipStreamEndCapture(void*,MockGraph **p) {
 capturing=false;
 if(failEndCapture){*p=nullptr;return 1;}
 *p=new MockGraph{capturedNodes};graphObjects[*p]=true;return 0;
}
inline int hipGraphGetNodes(MockGraph *p,void**,size_t *n) {*n=p->nodes-(wrongNodes?1:0);return 0;}
inline int hipGraphInstantiateWithFlags(MockGraph **out,MockGraph *g,unsigned long long) {
 *out=new MockGraph{g->nodes};graphExecObjects[*out]=true;return 0;
}
inline int hipGraphLaunch(MockGraph *g,void*) {
 launchCount+=g->nodes;if(failAfterLaunch)failCompletion=true;return 0;
}
inline int hipGraphExecDestroy(MockGraph *p) {
 if(failGraphDestroy)return 1;
 if(!graphExecObjects.erase(p))return 1;delete p;return 0;
}
inline int hipGraphDestroy(MockGraph *p) {
 if(failGraphDestroy)return 1;
 if(!graphObjects.erase(p))return 1;delete p;return 0;
}
inline int hipModuleLaunchKernel(void*,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,
 unsigned,void*,void**,void**) {if(capturing){++capturedNodes;return 0;}++launchCount;if(failAfterLaunch)failCompletion=true;return 0;}
""")
    runtime=Path(__file__).resolve().parents[2]/"src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_nvfp4_runtime.cpp"
    source=tmp_path/"probe.cpp"
    source.write_text('#include "'+str(runtime)+'"\n'+r"""
#include <cassert>
#include <sys/wait.h>
extern "C" int tessera_rocm_image_acquire(const void*,size_t,const char*,
 void **lease,void **module,void **function,int *hit) {
 *lease=new int(1);*module=*function=reinterpret_cast<void*>(1);*hit=0;++leases;return 0;
}
extern "C" int tessera_rocm_image_release(void *p){delete static_cast<int*>(p);--leases;return 0;}
int main() {
 const char blob[]={127,'E','L','F',0};
 const void *images[]={blob,blob,blob};size_t lengths[]={5,5,5};
 const char *entries[]={"convert","storage","matmul"};
 int64_t dims[]={128,32,64,2};unsigned geometry[18];
 for(auto &x:geometry)x=1;geometry[3]=geometry[9]=geometry[15]=256;
 unsigned char codes[1024]{},scales[128]{},a[8192]{};
 double globals[]={.5,2.};float sa[128];for(auto &x:sa)x=1;
 const void *inputs[]={codes,scales,globals,a,sa};
 size_t sizes[]={sizeof(codes),sizeof(scales),sizeof(globals),sizeof(a),sizeof(sa)};
 auto prepare=[&](uint64_t &h){return tessera_rocm_nvfp4_prepare(images,lengths,entries,dims,
   geometry,inputs,sizes,&h);};
 uint64_t handle=0,generation=0;float elapsed=0;
 sizes[3]--;assert(prepare(handle)==1&&handle==0&&allocations.empty());sizes[3]++;
 assert(prepare(handle)==0&&handle&&allocations.size()==11&&leases==3);
 assert(tessera_rocm_nvfp4_invoke(handle,2,1,&generation,&elapsed)==10);
 assert(tessera_rocm_nvfp4_invoke(handle,4,2,&generation,&elapsed)==0);
 assert(launchCount==6&&generation==1&&elapsed>0&&eventObjects.empty());
 unsigned char output[8192];std::memset(output,91,sizeof(output));
 assert(tessera_rocm_nvfp4_read(handle,10,generation+1,output,sizeof(output))==10);
 failCopy=true;
 assert(tessera_rocm_nvfp4_read(handle,10,generation,output,sizeof(output))==5);
 for(auto x:output)assert(x==91);
 assert(allocations.size()==11&&leases==3);
 failCopy=false;assert(tessera_rocm_nvfp4_close(handle)==0&&allocations.empty()&&leases==0);
 assert(tessera_rocm_nvfp4_close(handle)==1);
 assert(prepare(handle)==0);
 failAfterLaunch=true;
 assert(tessera_rocm_nvfp4_invoke(handle,4,1,&generation,&elapsed)==7);
 assert(tessera_rocm_nvfp4_close(handle)==7&&allocations.size()==11&&leases==3&&eventObjects.size()==2);
 failAfterLaunch=failCompletion=false;
 fakeContext=2;assert(tessera_rocm_nvfp4_close(handle)==2&&leases==3);
 fakeContext=1;
 assert(tessera_rocm_nvfp4_close(handle)==0&&allocations.empty()&&eventObjects.empty()&&leases==0);
 assert(prepare(handle)==0);
 failEventDestroy=true;
 assert(tessera_rocm_nvfp4_invoke(handle,4,1,&generation,&elapsed)==9);
 assert(tessera_rocm_nvfp4_close(handle)==9&&allocations.size()==11&&leases==3);
 failEventDestroy=false;
 assert(tessera_rocm_nvfp4_close(handle)==0&&allocations.empty()&&eventObjects.empty()&&leases==0);
 assert(prepare(handle)==0);
 failAfterLaunch=true;
 assert(tessera_rocm_nvfp4_invoke(handle,4,1,&generation,nullptr)==0);
 std::memset(output,91,sizeof(output));
 assert(tessera_rocm_nvfp4_read(handle,10,generation,output,sizeof(output))==7);
 for(auto x:output)assert(x==91);
 assert(tessera_rocm_nvfp4_close(handle)==7&&allocations.size()==11&&leases==3);
 failAfterLaunch=failCompletion=false;
 assert(tessera_rocm_nvfp4_close(handle)==0&&allocations.empty()&&leases==0);
 assert(prepare(handle)==0);
 freeCountdown=1;
 assert(tessera_rocm_nvfp4_close(handle)==9&&allocations.size()==10&&leases==3);
 int prior=launchCount;
 assert(tessera_rocm_nvfp4_invoke(handle,4,1,&generation,nullptr)==10&&launchCount==prior);
 freeCountdown=-1;
 assert(tessera_rocm_nvfp4_close(handle)==0&&allocations.empty()&&leases==0);
 assert(prepare(handle)==0);
 uint64_t nodes=0;
 assert(tessera_rocm_nvfp4_graph(handle,4,4,&generation,&nodes,&elapsed)==0&&nodes==12);
 assert(graphObjects.size()==1&&graphExecObjects.size()==1);
 int launchesBefore=launchCount;
 assert(tessera_rocm_nvfp4_graph(handle,4,4,&generation,&nodes,nullptr)==0&&launchCount==launchesBefore+12);
 for(int r=1;r<=10;++r)
  assert(tessera_rocm_nvfp4_graph(handle,4,r,&generation,&nodes,nullptr)==0&&nodes==uint64_t(3*r));
 assert(graphObjects.size()==8&&graphExecObjects.size()==8);
 failGraphDestroy=true;
 assert(tessera_rocm_nvfp4_close(handle)==9&&allocations.size()==11&&leases==3);
 assert(tessera_rocm_nvfp4_graph(handle,4,1,&generation,&nodes,nullptr)==10);
 failGraphDestroy=false;
 assert(tessera_rocm_nvfp4_close(handle)==0&&graphObjects.empty()&&graphExecObjects.empty());
 assert(prepare(handle)==0);
 failEndCapture=true;
 assert(tessera_rocm_nvfp4_graph(handle,4,1,&generation,&nodes,nullptr)==6);
 assert(!capturing&&allocations.size()==11&&leases==3);
 failEndCapture=false;
 assert(tessera_rocm_nvfp4_close(handle)==0);
 assert(prepare(handle)==0);
 wrongNodes=true;
 assert(tessera_rocm_nvfp4_graph(handle,4,1,&generation,&nodes,nullptr)==6);
 wrongNodes=false;
 assert(tessera_rocm_nvfp4_close(handle)==0&&graphObjects.empty()&&graphExecObjects.empty());

 int hit=0;
 auto cached=[&](uint64_t &h){return tessera_rocm_nvfp4_prepare_cached(
   images,lengths,entries,dims,geometry,inputs,sizes,&h,&hit);};
 assert(cached(handle)==0&&!hit);
 uint64_t stale=handle;
 assert(tessera_rocm_nvfp4_release_cached(handle)==0&&allocations.size()==11&&leases==3);
 assert(cached(handle)==0&&hit&&handle!=stale&&allocations.size()==11);
 assert(tessera_rocm_nvfp4_invoke(stale,4,1,&generation,nullptr)==1);
 assert(tessera_rocm_nvfp4_invoke(handle,2,1,&generation,nullptr)==10);
 assert(tessera_rocm_nvfp4_invoke(handle,4,1,&generation,nullptr)==0);
 globals[0]=-1;
 assert(tessera_rocm_nvfp4_update_inputs(handle,inputs,sizes)==1);
 globals[0]=.5;
 assert(tessera_rocm_nvfp4_read(handle,10,generation,output,sizeof(output))==0);
 failCopy=true;
 assert(tessera_rocm_nvfp4_update_inputs(handle,inputs,sizes)==5);
 assert(tessera_rocm_nvfp4_invoke(handle,4,1,&generation,nullptr)==10);
 failCompletion=true;
 assert(tessera_rocm_nvfp4_release_cached(handle)==7&&allocations.size()==11&&leases==3);
 failCopy=failCompletion=false;
 assert(tessera_rocm_nvfp4_release_cached(handle)==0&&allocations.empty()&&leases==0);
 uint64_t owners[6];
 for(auto &h:owners)assert(cached(h)==0&&!hit);
 for(auto h:owners)assert(tessera_rocm_nvfp4_release_cached(h)==0);
 assert(allocations.size()==44&&leases==12); // bounded four idle owners
 fakeContext=2;
 assert(tessera_rocm_nvfp4_cache_clear()==2&&allocations.size()==44);
 fakeContext=1;
 freeCountdown=1;
 assert(tessera_rocm_nvfp4_cache_clear()==9&&allocations.size()==43);
 freeCountdown=-1;
 assert(tessera_rocm_nvfp4_cache_clear()==0&&allocations.empty()&&leases==0);
 assert(cached(handle)==0&&!hit);
 assert(tessera_rocm_nvfp4_release_cached(handle)==0);
 const char alternative[]={127,'E','L','F',1};
 const void *different[]={blob,blob,alternative};
 assert(tessera_rocm_nvfp4_prepare_cached(different,lengths,entries,dims,
   geometry,inputs,sizes,&handle,&hit)==0&&!hit);
 assert(tessera_rocm_nvfp4_release_cached(handle)==0&&allocations.size()==22);
 assert(tessera_rocm_nvfp4_cache_clear()==0&&allocations.empty());

 // Bounded frames use actual spans while retaining their capacity allocation.
 size_t rowSizes[5];std::copy_n(sizes,5,rowSizes);
 rowSizes[3]=17*64;rowSizes[4]=17*sizeof(float);
 assert(tessera_rocm_nvfp4_prepare_rows(images,lengths,entries,dims,geometry,
   inputs,rowSizes,17,&handle)==0);
 int64_t capacity=0,active=0;uint64_t span=0,count=0;
 assert(tessera_rocm_nvfp4_frame_stats(handle,&capacity,&active,&span,&count)==0);
 assert(capacity==128&&active==17&&count==11);
 size_t realSpan=0;for(auto &allocation:allocations)realSpan+=allocation.second;
 assert(span==realSpan);
 assert(tessera_rocm_nvfp4_graph(handle,4,1,&generation,&nodes,nullptr)==0);
 assert(graphObjects.size()==1);
 // Invalid lengths/rows do not retire the previous completed graph or frame.
 assert(tessera_rocm_nvfp4_update_inputs_rows(handle,129,inputs,rowSizes)==1);
 assert(tessera_rocm_nvfp4_update_inputs_rows(handle,1,inputs,rowSizes)==1);
 assert(graphObjects.size()==1&&allocations.size()==11);
 rowSizes[3]=64;rowSizes[4]=sizeof(float);
 failGraphDestroy=true;
 assert(tessera_rocm_nvfp4_update_inputs_rows(handle,1,inputs,rowSizes)==9);
 assert(allocations.size()==11&&leases==3);
 assert(tessera_rocm_nvfp4_invoke(handle,4,1,&generation,nullptr)==10);
 failGraphDestroy=false;
 assert(tessera_rocm_nvfp4_close(handle)==0&&allocations.empty()&&graphObjects.empty());
 assert(tessera_rocm_nvfp4_prepare_cached_rows(images,lengths,entries,dims,geometry,
   inputs,rowSizes,1,&handle,&hit)==0&&!hit);
 assert(tessera_rocm_nvfp4_invoke(handle,4,1,&generation,nullptr)==0);
 unsigned char oneRow[64];
 assert(tessera_rocm_nvfp4_read(handle,10,generation,oneRow,sizeof(oneRow))==0);
 assert(tessera_rocm_nvfp4_read(handle,10,generation,output,sizeof(output))==1);
 assert(tessera_rocm_nvfp4_release_cached(handle)==0);
 // Static and bounded views may reuse capacity-compatible native owners,
 // but a static rebinding restores its full declared row frame.
 assert(cached(handle)==0&&hit);
 assert(tessera_rocm_nvfp4_frame_stats(handle,&capacity,&active,&span,&count)==0);
 assert(active==128&&count==11);
 assert(tessera_rocm_nvfp4_release_cached(handle)==0);
 assert(tessera_rocm_nvfp4_cache_clear()==0&&allocations.empty());

 // The byte budget can bind before the four-entry limit.
 int64_t largeDims[]={8192,32,1024,2};
 std::vector<unsigned char> largeCodes(16384),largeScales(2048),largeA(8388608);
 std::vector<float> largeAScale(8192,1);
 const void *largeInputs[]={largeCodes.data(),largeScales.data(),globals,largeA.data(),largeAScale.data()};
 size_t largeSizes[]={largeCodes.size(),largeScales.size(),sizeof(globals),largeA.size(),largeAScale.size()*sizeof(float)};
 uint64_t largeOwners[4];
 for(auto &h:largeOwners)
   assert(tessera_rocm_nvfp4_prepare_cached(images,lengths,entries,largeDims,
     geometry,largeInputs,largeSizes,&h,&hit)==0&&!hit);
 for(auto h:largeOwners)assert(tessera_rocm_nvfp4_release_cached(h)==0);
 assert(allocations.size()==33&&leases==9);
 assert(tessera_rocm_nvfp4_cache_clear()==0&&allocations.empty()&&leases==0);
 assert(prepare(handle)==0);
 mutex.lock();
 pid_t child=fork();
 if(child==0)_exit(tessera_rocm_nvfp4_invoke(handle,4,1,&generation,&elapsed)==2 &&
   tessera_rocm_nvfp4_graph(handle,4,1,&generation,&nodes,nullptr)==2?0:1);
 mutex.unlock();
 int status=0;waitpid(child,&status,0);assert(WIFEXITED(status)&&WEXITSTATUS(status)==0);
 assert(tessera_rocm_nvfp4_close(handle)==0);
}
""")
    binary=tmp_path/"probe"
    subprocess.run([compiler,"-std=c++17","-pthread","-I",str(tmp_path),str(source),"-o",str(binary)],
                   check=True,capture_output=True,text=True)
    subprocess.run([str(binary)],check=True,capture_output=True,text=True,timeout=30)
