"""Compile the real attention runtime against a controlled multi-device HIP ABI."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_attention_cache_follows_current_device_and_architecture(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires host C++ compiler")
    hip = tmp_path / "hip"
    hip.mkdir()
    (hip / "hip_runtime.h").write_text(r'''
#pragma once
#include <string>
#include <vector>
#include <cstring>
#include <cstdint>
using hipModule_t = void*; using hipFunction_t = void*;
struct hipDeviceProp_t { char gcnArchName[256]; };
constexpr int hipSuccess=0, hipMemcpyHostToDevice=1, hipMemcpyDeviceToHost=2;
inline int current=0, serial=0; inline bool failDevice=false;
inline std::vector<std::string> sources, options;
inline int hipGetDevice(int* d) { *d=current; return failDevice ? 1 : 0; }
inline int hipGetDeviceProperties(hipDeviceProp_t* p,int d) {
 std::strcpy(p->gcnArchName,d==0 ? "gfx1151" : "gfx1201:sramecc-:xnack-"); return 0;
}
inline int hipModuleLoadData(hipModule_t* m,const void*) { *m=reinterpret_cast<void*>(uintptr_t(++serial)); return 0; }
inline int hipModuleGetFunction(hipFunction_t* f,hipModule_t m,const char*) { *f=m; return 0; }
inline int hipMalloc(void** p,size_t) { *p=nullptr; return 0; }
inline int hipFree(void*) { return 0; }
inline int hipMemcpy(void*,const void*,size_t,int) { return 0; }
inline int hipDeviceSynchronize() { return 0; }
inline int hipModuleLaunchKernel(hipFunction_t,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,void*,void**,void**) { return 0; }
''')
    (hip / "hiprtc.h").write_text(r'''
#pragma once
using hiprtcProgram=int; using hiprtcResult=int;
constexpr int HIPRTC_SUCCESS=0;
inline int hiprtcCreateProgram(int* p,const char* s,const char*,int,const char**,const char**) { sources.emplace_back(s); *p=1; return 0; }
inline int hiprtcCompileProgram(int,int,const char** opts) { options.emplace_back(opts[0]); return 0; }
inline int hiprtcGetCodeSize(int,size_t* n) { *n=1; return 0; }
inline int hiprtcGetCode(int,char* p) { *p=0; return 0; }
inline int hiprtcDestroyProgram(int*) { return 0; }
''')
    runtime = Path(__file__).resolve().parents[2] / "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/tessera_rocm_flash_attn.cpp"
    source = tmp_path / "probe.cpp"
    source.write_text('#include "' + str(runtime) + '"\n' + r'''
#include <cassert>
int main() {
 auto first=kernelFor(&g_f16,64); assert(first);
 assert(options.back()=="--offload-arch=gfx1151");
 assert(sources.back().find("ext_vector_type(16)")!=std::string::npos);
 current=1; auto second=kernelFor(&g_f16,64); assert(second && second!=first);
 assert(options.back()=="--offload-arch=gfx1201:sramecc-:xnack-");
 assert(sources.back().find("ext_vector_type(16)")==std::string::npos);
 assert(sources.back().find("_w32_gfx12(")!=std::string::npos);
 current=2; auto third=kernelFor(&g_f16,64); assert(third && third!=second);
 current=0; assert(kernelFor(&g_f16,64)==first); assert(serial==3);
 failDevice=true; assert(kernelFor(&g_f16,64)==nullptr); assert(serial==3);
}
''')
    binary = tmp_path / "probe"
    subprocess.run([compiler, "-std=c++17", "-pthread", "-I", str(tmp_path), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
