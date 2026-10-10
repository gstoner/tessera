"""Compile the actual native lease service against a controlled HIP ABI."""
from pathlib import Path
import shutil
import subprocess
import pytest

def test_native_image_cache_lifetime_context_and_eviction(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires host C++ compiler")
    hip = tmp_path / "hip"
    hip.mkdir()
    (hip / "hip_runtime.h").write_text(r"""
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
using hipModule_t=void*; using hipFunction_t=void*; using hipCtx_t=void*;
constexpr int hipSuccess=0;
struct hipDeviceProp_t { char gcnArchName[256]; };
struct Module { int device; uintptr_t context; bool unloaded=false; };
inline thread_local int device=0;
inline thread_local uintptr_t context=1;
inline int loads=0,unloads=0,lookups=0;
inline bool failLoad=false,failIdentity=false;
inline std::vector<Module*> allocated;
inline int hipGetDevice(int* p) { *p=device; return failIdentity ? 1 : 0; }
inline int hipCtxGetCurrent(void** p) { *p=reinterpret_cast<void*>(context); return 0; }
inline int hipGetDeviceProperties(hipDeviceProp_t* p,int d) {
 std::strcpy(p->gcnArchName,d ? "gfx1201" : "gfx1151"); return 0;
}
inline int hipModuleLoadData(void** p,const void*) {
 if(failLoad) return 1;
 auto m=new Module{device,context};
 allocated.push_back(m); *p=m; ++loads; return 0;
}
inline int hipModuleGetFunction(void** p,void* v,const char* name) {
 auto m=static_cast<Module*>(v);
 if(m->unloaded || std::strcmp(name,"missing")==0) return 1;
 *p=v; ++lookups; return 0;
}
inline int hipModuleUnload(void* v) {
 auto m=static_cast<Module*>(v);
 if(m->unloaded || m->device!=device || m->context!=context) return 1;
 m->unloaded=true; ++unloads; return 0;
}
""")
    runtime = Path(__file__).resolve().parents[2] / "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_image_cache.cpp"
    source = tmp_path / "probe.cpp"
    source.write_text('#include "' + str(runtime) + '"\n' + r"""
#include <cassert>
#include <thread>
#include <sys/wait.h>
int main() {
 const char a[]={127,'E','L','F',0,'a'};
 const char b[]={127,'E','L','F',0,'b'};
 void *l1=nullptr,*l2=nullptr,*m=nullptr,*f=nullptr; int hit=-1;
 assert(tessera_rocm_image_acquire(a,6,"gemm",&l1,&m,&f,&hit)==0 && hit==0 && loads==1);
 assert(tessera_rocm_image_acquire(a,6,"gemm",&l2,&m,&f,&hit)==0 && hit==1 && loads==1 && lookups==1);
 assert(tessera_rocm_image_clear_current()==5 && unloads==0);
 assert(tessera_rocm_image_release(l1)==0);
 assert(tessera_rocm_image_clear_current()==5);
 assert(tessera_rocm_image_lookup(l2,"reduce",&f)==0 && lookups==2);
 assert(tessera_rocm_image_lookup(l2,"reduce",&f)==0 && lookups==2);
 assert(tessera_rocm_image_lookup(l2,"missing",&f)==4);
 context=2;
 assert(tessera_rocm_image_lookup(l2,"gemm",&f)==2);
 assert(tessera_rocm_image_release(l2)==2);
 context=1;
 assert(tessera_rocm_image_release(l2)==0 && unloads==0);
 assert(tessera_rocm_image_acquire(b,6,"gemm",&l1,&m,&f,&hit)==0 && hit==0 && loads==2);
 assert(tessera_rocm_image_release(l1)==0);
 context=2;
 assert(tessera_rocm_image_acquire(a,6,"gemm",&l1,&m,&f,&hit)==0 && hit==0 && loads==3);
 assert(tessera_rocm_image_release(l1)==0);
 assert(tessera_rocm_image_clear_current()==0 && unloads==1);
 context=1; device=1;
 assert(tessera_rocm_image_acquire(a,6,"gemm",&l1,&m,&f,&hit)==0 && hit==0 && loads==4);
 assert(tessera_rocm_image_release(l1)==0);
 assert(tessera_rocm_image_clear_current()==0 && unloads==2);
 device=0;
 assert(tessera_rocm_image_clear_current()==0 && unloads==4);
 failLoad=true;
 assert(tessera_rocm_image_acquire(a,6,"gemm",&l1,&m,&f,&hit)==3 && !l1);
 failLoad=false;
 assert(tessera_rocm_image_acquire(a,6,"missing",&l1,&m,&f,&hit)==4 && !l1);
 assert(unloads==5);
 failIdentity=true;
 assert(tessera_rocm_image_acquire(a,6,"gemm",&l1,&m,&f,&hit)==2 && !l1);
 failIdentity=false;
 // Repeated parallel acquisitions resolve one module and function.
 int before=loads;
 std::vector<std::thread> threads;
 for(int i=0;i<8;++i) threads.emplace_back([&] {
   void *lease=nullptr,*module=nullptr,*fn=nullptr; int found=-1;
   assert(tessera_rocm_image_acquire(a,6,"gemm",&lease,&module,&fn,&found)==0);
   assert(tessera_rocm_image_release(lease)==0);
 });
 for(auto &t:threads)t.join();
 assert(loads==before+1);
 assert(tessera_rocm_image_clear_current()==0);
 // Bound the cache; an active oldest lease is never evicted.
 std::vector<void*> leases;
 for(int i=0;i<18;++i) {
   char image[]={127,'E','L','F',char(i)};
   assert(tessera_rocm_image_acquire(image,5,"gemm",&l1,&m,&f,&hit)==0);
   leases.push_back(l1);
 }
 int heldUnloads=unloads;
 assert(tessera_rocm_image_clear_current()==5);
 for(void* l:leases) assert(tessera_rocm_image_release(l)==0);
 assert(unloads==heldUnloads+2); // two uncached overflow leases
 assert(tessera_rocm_image_clear_current()==0 && unloads==heldUnloads+18);
 // Inactive LRU entries can be evicted under their owning context.
 before=unloads;
 for(int i=0;i<18;++i) {
   char image[]={127,'E','L','F',char(i)};
   assert(tessera_rocm_image_acquire(image,5,"gemm",&l1,&m,&f,&hit)==0);
   assert(tessera_rocm_image_release(l1)==0);
 }
 assert(unloads==before+2);
 assert(tessera_rocm_image_clear_current()==0 && unloads==before+18);
 // A fork cannot reuse inherited HIP modules or a possibly held mutex.
 pid_t child=fork(); assert(child>=0);
 if(child==0) {
   int rc=tessera_rocm_image_acquire(a,6,"gemm",&l1,&m,&f,&hit);
   _exit(rc==2 ? 0 : 1);
 }
 int status=0; waitpid(child,&status,0);
 assert(WIFEXITED(status) && WEXITSTATUS(status)==0);
 uint64_t nload,nhit,nfn,nunload;
 assert(tessera_rocm_image_stats(&nload,&nhit,&nfn,&nunload)==0);
 assert(nload==uint64_t(loads) && nunload==uint64_t(unloads));
 assert(nhit>=8);
 for(auto *module:allocated) { assert(module->unloaded); delete module; }
}
""")
    binary = tmp_path / "probe"
    subprocess.run([compiler, "-std=c++17", "-pthread", "-I", str(tmp_path), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_cache_optout_does_not_hide_explicit_context_clear(monkeypatch):
    from tessera import runtime as rt
    calls = []
    class BoundLibrary:
        def tessera_rocm_image_clear_current(self):
            calls.append("clear")
            return 0
    monkeypatch.setenv("TESSERA_ROCM_NATIVE_IMAGE_CACHE", "0")
    monkeypatch.setattr(rt, "_rocm_native_image_runtime", BoundLibrary())
    assert rt._load_rocm_native_image_runtime() is None
    rt._clear_rocm_native_image_cache()
    assert calls == ["clear"]
