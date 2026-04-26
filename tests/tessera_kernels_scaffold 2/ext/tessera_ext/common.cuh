#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __has_include
#  if __has_include(<nvToolsExt.h>)
#    include <nvToolsExt.h>
     inline nvtxEventAttributes_t nvtx_attrs(const char* name, uint32_t color){
       nvtxEventAttributes_t a{};
       a.version = NVTX_VERSION; a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
       a.colorType = NVTX_COLOR_ARGB; a.color = color;
       a.messageType = NVTX_MESSAGE_TYPE_ASCII; a.message.ascii = name;
       return a;
     }
#    define TESSERA_NVTX_RANGE(name) nvtxRangePushA(name); struct _nvtx_scope_t{~_nvtx_scope_t(){nvtxRangePop();}} _nvtx_scope_inst;
#    define TESSERA_NVTX_RANGE_COLORED(name,color) auto _attrs = nvtx_attrs(name,color); nvtxRangePushEx(&_attrs); struct _nvtx_scope_c{~_nvtx_scope_c(){nvtxRangePop();}} _nvtx_scope_c_inst;
#  else
#    define TESSERA_NVTX_RANGE(name) struct _dummy_nvtx_t{} _nvtx_dummy
#    define TESSERA_NVTX_RANGE_COLORED(name,color) struct _dummy_nvtx_c_t{} _nvtx_dummy_c
#  endif
#else
#  define TESSERA_NVTX_RANGE(name) struct _dummy_nvtx_t{} _nvtx_dummy
#  define TESSERA_NVTX_RANGE_COLORED(name,color) struct _dummy_nvtx_c_t{} _nvtx_dummy_c
#endif

#define CUDA_CHECK(stmt) do { cudaError_t e = (stmt); if (e != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
  asm volatile("trap;"); }} while(0)

__host__ __device__ inline int div_up(int a, int b) { return (a + b - 1) / b; }
