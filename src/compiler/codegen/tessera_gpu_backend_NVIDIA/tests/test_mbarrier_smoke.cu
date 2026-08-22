#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

extern "C" cudaError_t tessera_mbarrier_smoke(uint32_t* host_result);

int main() {
  uint32_t result = 0;
  cudaError_t status = tessera_mbarrier_smoke(&result);
  if (status != cudaSuccess || result != 1u) {
    std::fprintf(stderr, "mbarrier smoke failed: %s result=%u\n",
                 cudaGetErrorString(status), result);
    return 1;
  }
  return 0;
}
