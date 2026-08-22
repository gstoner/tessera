#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/ptx>
#include <cstdint>
#include <cstdio>

// One 128-byte rank-1 TMA load.  The descriptor is created with
// cuTensorMapEncodeTiled on the host and passed by value, so the device sees
// the opaque, correctly aligned CUtensorMap rather than a fabricated pointer.
extern "C" __global__ void tessera_tma_smoke_kernel(
    CUtensorMap tensor_map, float* output) {
#if __CUDA_ARCH__ >= 900
  __shared__ __align__(16) float tile[32];
  __shared__ __align__(8) std::uint64_t barrier;
  if (threadIdx.x == 0) {
    cuda::ptx::mbarrier_init(&barrier, 1);
    std::uint32_t bytes = sizeof(tile);
    const auto state = cuda::ptx::mbarrier_arrive_expect_tx(
        cuda::ptx::sem_release, cuda::ptx::scope_cta,
        cuda::ptx::space_shared, &barrier, bytes);
    const std::int32_t coordinates[1] = {0};
    cuda::ptx::cp_async_bulk_tensor(
        cuda::ptx::space_shared, cuda::ptx::space_global, tile, &tensor_map,
        coordinates, &barrier);
    while (!cuda::ptx::mbarrier_try_wait(
        cuda::ptx::sem_acquire, cuda::ptx::scope_cta, &barrier, state)) {}
  }
  __syncthreads();
  if (threadIdx.x < 32) output[threadIdx.x] = tile[threadIdx.x];
#else
  (void)tensor_map; (void)output;
#endif
}

extern "C" cudaError_t tessera_tma_smoke(float* host_output, std::uint32_t count) {
  if (count != 32) return cudaErrorInvalidValue;
  if (cuInit(0) != CUDA_SUCCESS) return cudaErrorInitializationError;
  float *source = nullptr, *output = nullptr;
  cudaError_t status = cudaMalloc(&source, count * sizeof(float));
  if (status != cudaSuccess) return status;
  status = cudaMalloc(&output, count * sizeof(float));
  if (status != cudaSuccess) { cudaFree(source); return status; }
  float input[32];
  for (std::uint32_t index = 0; index < count; ++index) input[index] = float(index) + 0.25f;
  status = cudaMemcpy(source, input, sizeof(input), cudaMemcpyHostToDevice);
  CUtensorMap map{};
  const cuuint64_t dimensions[1] = {count};
  const cuuint32_t box[1] = {count};
  const cuuint32_t strides[1] = {1};
  if (status == cudaSuccess) {
    const CUresult driver = cuTensorMapEncodeTiled(
        &map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 1, source, dimensions, nullptr,
        box, strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (driver != CUDA_SUCCESS) {
      const char* text = nullptr;
      cuGetErrorString(driver, &text);
      std::fprintf(stderr, "cuTensorMapEncodeTiled: %s\n", text ? text : "unknown");
      status = cudaErrorInvalidValue;
    }
  }
  if (status == cudaSuccess) tessera_tma_smoke_kernel<<<1, 32>>>(map, output);
  if (status == cudaSuccess) status = cudaGetLastError();
  if (status == cudaSuccess)
    status = cudaMemcpy(host_output, output, sizeof(input), cudaMemcpyDeviceToHost);
  cudaFree(output);
  cudaFree(source);
  return status;
}
