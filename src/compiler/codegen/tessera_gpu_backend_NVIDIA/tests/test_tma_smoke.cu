#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

extern "C" cudaError_t tessera_tma_smoke(float*, std::uint32_t);
int main() {
  float output[32]{};
  const cudaError_t status = tessera_tma_smoke(output, 32);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "TMA smoke launch failed: %s\n", cudaGetErrorString(status));
    return 1;
  }
  for (std::uint32_t i = 0; i < 32; ++i)
    if (output[i] != float(i) + .25f) {
      std::fprintf(stderr, "TMA smoke mismatch at %u: %g\n", i, output[i]);
      return 1;
    }
  return 0;
}
