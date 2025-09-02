#pragma once
#include <cuda.h>
#include <string>
#include <vector>

namespace tessera { namespace gpu {

struct CudaModule {
    CUmodule mod = nullptr;
    ~CudaModule();
};

struct CudaContext {
    CUcontext ctx = nullptr;
    CUdevice  dev = 0;
    int sm = 0;
    CudaContext();
    ~CudaContext();
    bool ok() const { return ctx != nullptr; }
};

// Helpers
bool checkCu(CUresult r, const char* what);
std::string cuErrStr(CUresult r);

CUfunction getKernel(CudaModule& m, const char* name);

// Memory helpers
void*  deviceAlloc(size_t bytes);
void   deviceFree(void* p);
void   memcpyHtoD(void* dst, const void* src, size_t bytes);
void   memcpyDtoH(void* dst, const void* src, size_t bytes);

// Module loading
bool   loadPtx(CudaModule& m, const char* ptx, const char* name="ptx");
bool   loadCubin(CudaModule& m, const void* data, size_t size);

}} // namespace
