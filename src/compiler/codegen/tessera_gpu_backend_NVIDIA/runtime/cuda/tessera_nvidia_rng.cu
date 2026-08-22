// Shipped NVIDIA stateless Philox4x32-10 runtime ABI.
//
// These entry points implement the same four modes accepted by the typed
// tessera_nvidia.philox compiler directive. Key and counter are call arguments;
// there is no process-global generator state and no hidden workspace.

#include <cuda_runtime.h>

#include <cstdint>

namespace {

__device__ __forceinline__ float philoxUniform(uint64_t seed,
                                                uint64_t counterBase,
                                                uint64_t wordIndex) {
  constexpr uint32_t m0 = 0xD2511F53u;
  constexpr uint32_t m1 = 0xCD9E8D57u;
  constexpr uint32_t w0 = 0x9E3779B9u;
  constexpr uint32_t w1 = 0xBB67AE85u;
  uint64_t counter = counterBase + wordIndex / 4;
  uint32_t c0 = static_cast<uint32_t>(counter);
  uint32_t c1 = static_cast<uint32_t>(counter >> 32);
  uint32_t c2 = 0;
  uint32_t c3 = 0;
  uint32_t k0 = static_cast<uint32_t>(seed);
  uint32_t k1 = static_cast<uint32_t>(seed >> 32);
  for (int round = 0; round < 10; ++round) {
    if (round != 0) {
      k0 += w0;
      k1 += w1;
    }
    uint64_t p0 = static_cast<uint64_t>(c0) * m0;
    uint64_t p1 = static_cast<uint64_t>(c2) * m1;
    uint32_t next0 = static_cast<uint32_t>(p1 >> 32) ^ c1 ^ k0;
    uint32_t next2 = static_cast<uint32_t>(p0 >> 32) ^ c3 ^ k1;
    c0 = next0;
    c1 = static_cast<uint32_t>(p1);
    c2 = next2;
    c3 = static_cast<uint32_t>(p0);
  }
  uint32_t words[4] = {c0, c1, c2, c3};
  return __fmul_rn(__uint2float_rn(words[wordIndex & 3]), 0x1p-32f);
}

__global__ void uniformCoreKernel(uint64_t seed, uint64_t counter, int64_t n,
                                  float *output) {
  uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < static_cast<uint64_t>(n))
    output[index] = philoxUniform(seed, counter, index);
}

__global__ void uniformRangeKernel(uint64_t seed, uint64_t counter, int64_t n,
                                   float low, float high, float *output) {
  uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < static_cast<uint64_t>(n)) {
    float u = philoxUniform(seed, counter, index);
    output[index] = __fadd_rn(low, __fmul_rn(__fsub_rn(high, low), u));
  }
}

__global__ void normalKernel(uint64_t seed, uint64_t counter, int64_t n,
                             float mean, float stddev, float *output) {
  uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<uint64_t>(n))
    return;
  uint64_t pair = index / 2;
  uint64_t pairs = (static_cast<uint64_t>(n) + 1) / 2;
  uint64_t secondCounter = counter + (pairs + 3) / 4 + 1;
  float u1 = fmaxf(philoxUniform(seed, counter, pair), 1.0e-7f);
  float u2 = philoxUniform(seed, secondCounter, pair);
  float radius = sqrtf(-2.0f * logf(u1));
  float theta = 6.283185307179586f * u2;
  float z = (index & 1) ? sinf(theta) : cosf(theta);
  output[index] = mean + stddev * radius * z;
}

__global__ void dropoutKernel(const float *input, uint64_t seed,
                              uint64_t counter, int64_t n, float probability,
                              float *output) {
  uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < static_cast<uint64_t>(n)) {
    float u = philoxUniform(seed, counter, index);
    float scale = probability >= 1.0f ? 0.0f : 1.0f / (1.0f - probability);
    output[index] = input[index] * (u >= probability ? scale : 0.0f);
  }
}

template <typename Launch>
int runOutputKernel(int64_t n, float *hostOutput, Launch launch) {
  if (n < 0 || (n != 0 && hostOutput == nullptr))
    return 1;
  if (n == 0)
    return 0;
  float *deviceOutput = nullptr;
  if (cudaMalloc(&deviceOutput, static_cast<size_t>(n) * sizeof(float)) !=
      cudaSuccess)
    return 2;
  launch(deviceOutput, static_cast<unsigned>((n + 255) / 256));
  cudaError_t status = cudaGetLastError();
  if (status == cudaSuccess)
    status = cudaDeviceSynchronize();
  if (status == cudaSuccess)
    status = cudaMemcpy(hostOutput, deviceOutput,
                        static_cast<size_t>(n) * sizeof(float),
                        cudaMemcpyDeviceToHost);
  cudaFree(deviceOutput);
  return status == cudaSuccess ? 0 : 3;
}

} // namespace

extern "C" int tessera_nvidia_philox_uniform_f32(
    uint64_t seed, uint64_t counter, int64_t n, float *output) {
  return runOutputKernel(n, output, [&](float *deviceOutput, unsigned blocks) {
    uniformCoreKernel<<<blocks, 256>>>(seed, counter, n, deviceOutput);
  });
}

extern "C" int tessera_nvidia_philox_uniform_range_f32(
    uint64_t seed, uint64_t counter, int64_t n, float low, float high,
    float *output) {
  return runOutputKernel(n, output, [&](float *deviceOutput, unsigned blocks) {
    uniformRangeKernel<<<blocks, 256>>>(seed, counter, n, low, high,
                                        deviceOutput);
  });
}

extern "C" int tessera_nvidia_philox_normal_f32(
    uint64_t seed, uint64_t counter, int64_t n, float mean, float stddev,
    float *output) {
  return runOutputKernel(n, output, [&](float *deviceOutput, unsigned blocks) {
    normalKernel<<<blocks, 256>>>(seed, counter, n, mean, stddev, deviceOutput);
  });
}

extern "C" int tessera_nvidia_philox_dropout_f32(
    const float *input, uint64_t seed, uint64_t counter, int64_t n,
    float probability, float *output) {
  if (n < 0 || (n != 0 && (input == nullptr || output == nullptr)))
    return 1;
  if (n == 0)
    return 0;
  float *deviceInput = nullptr;
  float *deviceOutput = nullptr;
  size_t bytes = static_cast<size_t>(n) * sizeof(float);
  if (cudaMalloc(&deviceInput, bytes) != cudaSuccess ||
      cudaMalloc(&deviceOutput, bytes) != cudaSuccess) {
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    return 2;
  }
  cudaError_t status = cudaMemcpy(deviceInput, input, bytes,
                                  cudaMemcpyHostToDevice);
  if (status == cudaSuccess) {
    dropoutKernel<<<static_cast<unsigned>((n + 255) / 256), 256>>>(
        deviceInput, seed, counter, n, probability, deviceOutput);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess)
    status = cudaDeviceSynchronize();
  if (status == cudaSuccess)
    status = cudaMemcpy(output, deviceOutput, bytes, cudaMemcpyDeviceToHost);
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  return status == cudaSuccess ? 0 : 3;
}
