#include <cuda_runtime.h>
#include <stdint.h>

// This deliberately small probe establishes that the device executes the
// mbarrier instructions used by the TMA path.  TMA itself is only claimed by
// the production kernel when a validated CUtensorMap descriptor is supplied.
extern "C" __global__ void tessera_mbarrier_smoke_kernel(uint32_t* result) {
#if __CUDA_ARCH__ >= 900
  __shared__ __align__(8) unsigned long long barrier;
  unsigned long long address;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(address) : "l"(&barrier));
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "l"(address), "r"(1));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "l"(address));
  }
  __syncthreads();
  // The successful launch reaches a real init+arrive transition.  Waiting is
  // intentionally left to the TMA consumer, whose wait form depends on the
  // transaction phase and validated tensor-map descriptor.
  if (threadIdx.x == 0) *result = 1u;
#else
  if (threadIdx.x == 0) *result = 0u;
#endif
}

extern "C" cudaError_t tessera_mbarrier_smoke(uint32_t* host_result) {
  uint32_t* device_result = nullptr;
  cudaError_t status = cudaMalloc(&device_result, sizeof(*device_result));
  if (status != cudaSuccess) return status;
  tessera_mbarrier_smoke_kernel<<<1, 32>>>(device_result);
  status = cudaGetLastError();
  if (status == cudaSuccess)
    status = cudaMemcpy(host_result, device_result, sizeof(*host_result), cudaMemcpyDeviceToHost);
  cudaFree(device_result);
  return status;
}
