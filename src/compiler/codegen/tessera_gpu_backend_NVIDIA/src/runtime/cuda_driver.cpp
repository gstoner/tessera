#include "tessera/gpu/cuda_driver.h"
#include <cstdio>
#include <cstring>

using namespace tessera::gpu;

CudaModule::~CudaModule() {
    if (mod) cuModuleUnload(mod);
}

CudaContext::CudaContext() {
    cuInit(0);
    int ndev=0;
    if (cuDeviceGetCount(&ndev) != CUDA_SUCCESS || ndev==0) return;
    cuDeviceGet(&dev, 0);
    int major=0, minor=0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    sm = major*10 + minor;
    cuCtxCreate(&ctx, 0, dev);
}

CudaContext::~CudaContext() {
    if (ctx) cuCtxDestroy(ctx);
}

bool tessera::gpu::checkCu(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) {
        const char* s = nullptr;
        cuGetErrorString(r, &s);
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, s ? s : "unknown");
        return false;
    }
    return true;
}

std::string tessera::gpu::cuErrStr(CUresult r) {
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    return s ? std::string(s) : std::string("unknown");
}

CUfunction tessera::gpu::getKernel(CudaModule& m, const char* name) {
    CUfunction f=nullptr;
    checkCu(cuModuleGetFunction(f, m.mod, name), "cuModuleGetFunction");
    return f;
}

void* tessera::gpu::deviceAlloc(size_t bytes) {
    CUdeviceptr p=0; checkCu(cuMemAlloc(&p, bytes), "cuMemAlloc"); return (void*)p;
}
void tessera::gpu::deviceFree(void* p) {
    if (!p) return; cuMemFree((CUdeviceptr)p);
}
void tessera::gpu::memcpyHtoD(void* dst, const void* src, size_t bytes) {
    checkCu(cuMemcpyHtoD((CUdeviceptr)dst, src, bytes), "cuMemcpyHtoD");
}
void tessera::gpu::memcpyDtoH(void* dst, const void* src, size_t bytes) {
    checkCu(cuMemcpyDtoH(dst, (CUdeviceptr)src, bytes), "cuMemcpyDtoH");
}

bool tessera::gpu::loadPtx(CudaModule& m, const char* ptx, const char* name) {
    CUjit_option options[2];
    void*  values[2];
    unsigned int logSize = 16*1024;
    std::vector<char> log(logSize);
    options[0] = CU_JIT_ERROR_LOG_BUFFER; values[0] = log.data();
    options[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES; values[1] = (void*)(uintptr_t)logSize;

    CUresult r = cuModuleLoadDataEx(&m.mod, ptx, 2, options, values);
    if (r != CUDA_SUCCESS) {
        std::fprintf(stderr, "PTX load failed: %s\n%s\n", cuErrStr(r).c_str(), log.data());
        return false;
    }
    return true;
}

bool tessera::gpu::loadCubin(CudaModule& m, const void* data, size_t size) {
    (void)size;
    return checkCu(cuModuleLoadData(&m.mod, data), "cuModuleLoadData");
}
