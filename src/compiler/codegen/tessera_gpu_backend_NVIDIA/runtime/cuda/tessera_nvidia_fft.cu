#include "tessera_nvidia_fft.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstdint>
#include <new>

namespace {

enum class FFTKind { C2C, R2C, C2R };

struct FFTPlan {
  cufftHandle handle{};
  int64_t batch{};
  int64_t length{};
  size_t workspaceBytes{};
  FFTKind kind{FFTKind::C2C};
};

__global__ void normalizeInverse(cufftComplex *values, int64_t count,
                                 float scale) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    values[index].x *= scale;
    values[index].y *= scale;
  }
}

__global__ void normalizeRealInverse(float *values, int64_t count,
                                     float scale) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    values[index] *= scale;
}

int createPlan(int64_t batch, int64_t length, cufftType type, FFTKind kind,
               void **opaquePlan, size_t *workspaceBytes) {
  if (batch <= 0 || length <= 0 || batch > INT32_MAX || length > INT32_MAX ||
      opaquePlan == nullptr || workspaceBytes == nullptr)
    return 1;
  auto *plan = new (std::nothrow) FFTPlan;
  if (plan == nullptr)
    return 2;
  if (cufftCreate(&plan->handle) != CUFFT_SUCCESS ||
      cufftSetAutoAllocation(plan->handle, 0) != CUFFT_SUCCESS) {
    if (plan->handle)
      cufftDestroy(plan->handle);
    delete plan;
    return 2;
  }
  int n = static_cast<int>(length);
  int half = n / 2 + 1;
  int idist = kind == FFTKind::C2R ? half : n;
  int odist = kind == FFTKind::R2C ? half : n;
  size_t bytes = 0;
  if (cufftMakePlanMany(plan->handle, 1, &n, nullptr, 1, idist, nullptr, 1,
                        odist, type, static_cast<int>(batch), &bytes) !=
      CUFFT_SUCCESS) {
    cufftDestroy(plan->handle);
    delete plan;
    return 2;
  }
  plan->batch = batch;
  plan->length = length;
  plan->workspaceBytes = bytes;
  plan->kind = kind;
  *opaquePlan = plan;
  *workspaceBytes = bytes;
  return 0;
}

} // namespace

extern "C" const char *tessera_nvidia_fft_package_abi() {
  return "tessera.nvidia.cuda_fft_workspace.v2";
}

extern "C" int tessera_nvidia_fft_plan_create_c2c_f32(
    int64_t batch, int64_t length, void **opaquePlan, size_t *workspaceBytes) {
  return createPlan(batch, length, CUFFT_C2C, FFTKind::C2C, opaquePlan,
                    workspaceBytes);
}

extern "C" int tessera_nvidia_fft_plan_create_r2c_f32(
    int64_t batch, int64_t length, void **opaquePlan, size_t *workspaceBytes) {
  return createPlan(batch, length, CUFFT_R2C, FFTKind::R2C, opaquePlan,
                    workspaceBytes);
}

extern "C" int tessera_nvidia_fft_plan_create_c2r_f32(
    int64_t batch, int64_t length, void **opaquePlan, size_t *workspaceBytes) {
  return createPlan(batch, length, CUFFT_C2R, FFTKind::C2R, opaquePlan,
                    workspaceBytes);
}

extern "C" int tessera_nvidia_fft_plan_destroy(void *opaquePlan) {
  if (opaquePlan == nullptr)
    return 0;
  auto *plan = static_cast<FFTPlan *>(opaquePlan);
  cufftResult status = cufftDestroy(plan->handle);
  delete plan;
  return status == CUFFT_SUCCESS ? 0 : 3;
}

extern "C" int tessera_nvidia_fft_workspace_alloc(size_t bytes,
                                                   void **workspace) {
  if (workspace == nullptr)
    return 1;
  // CUDA permits no zero-byte allocation contract, while some small plans
  // report zero workspace. Retain a stable non-null identity with one byte.
  return cudaMalloc(workspace, bytes == 0 ? 1 : bytes) == cudaSuccess ? 0 : 2;
}

extern "C" int tessera_nvidia_fft_workspace_free(void *workspace) {
  return workspace == nullptr || cudaFree(workspace) == cudaSuccess ? 0 : 3;
}

extern "C" int tessera_nvidia_fft_execute_c2c_f32(
    void *opaquePlan, const float *input, float *output, void *workspace,
    size_t workspaceBytes, int inverse) {
  if (opaquePlan == nullptr || input == nullptr || output == nullptr ||
      workspace == nullptr || (inverse != 0 && inverse != 1))
    return 1;
  auto *plan = static_cast<FFTPlan *>(opaquePlan);
  if (plan->kind != FFTKind::C2C || workspaceBytes < plan->workspaceBytes)
    return 1;
  int64_t elements = plan->batch * plan->length;
  size_t bytes = static_cast<size_t>(elements) * sizeof(cufftComplex);
  cufftComplex *deviceData = nullptr;
  if (cudaMalloc(&deviceData, bytes) != cudaSuccess)
    return 2;
  cudaError_t cudaStatus = cudaMemcpy(deviceData, input, bytes,
                                      cudaMemcpyHostToDevice);
  cufftResult fftStatus = CUFFT_SUCCESS;
  if (cudaStatus == cudaSuccess)
    fftStatus = cufftSetWorkArea(plan->handle, workspace);
  if (fftStatus == CUFFT_SUCCESS)
    fftStatus = cufftExecC2C(plan->handle, deviceData, deviceData,
                             inverse ? CUFFT_INVERSE : CUFFT_FORWARD);
  if (fftStatus == CUFFT_SUCCESS && inverse) {
    normalizeInverse<<<static_cast<unsigned>((elements + 255) / 256), 256>>>(
        deviceData, elements, 1.0f / static_cast<float>(plan->length));
    cudaStatus = cudaGetLastError();
  }
  if (fftStatus == CUFFT_SUCCESS && cudaStatus == cudaSuccess)
    cudaStatus = cudaDeviceSynchronize();
  if (fftStatus == CUFFT_SUCCESS && cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(output, deviceData, bytes,
                            cudaMemcpyDeviceToHost);
  cudaFree(deviceData);
  return fftStatus == CUFFT_SUCCESS && cudaStatus == cudaSuccess ? 0 : 3;
}

extern "C" int tessera_nvidia_fft_execute_r2c_f32(
    void *opaquePlan, const float *input, float *output, void *workspace,
    size_t workspaceBytes) {
  if (opaquePlan == nullptr || input == nullptr || output == nullptr ||
      workspace == nullptr)
    return 1;
  auto *plan = static_cast<FFTPlan *>(opaquePlan);
  if (plan->kind != FFTKind::R2C || workspaceBytes < plan->workspaceBytes)
    return 1;
  int64_t realElements = plan->batch * plan->length;
  int64_t complexElements = plan->batch * (plan->length / 2 + 1);
  float *deviceInput = nullptr;
  cufftComplex *deviceOutput = nullptr;
  cudaError_t cudaStatus = cudaMalloc(&deviceInput, realElements * sizeof(float));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMalloc(&deviceOutput, complexElements * sizeof(cufftComplex));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(deviceInput, input, realElements * sizeof(float),
                            cudaMemcpyHostToDevice);
  cufftResult fftStatus = CUFFT_SUCCESS;
  if (cudaStatus == cudaSuccess)
    fftStatus = cufftSetWorkArea(plan->handle, workspace);
  if (fftStatus == CUFFT_SUCCESS && cudaStatus == cudaSuccess)
    fftStatus = cufftExecR2C(plan->handle, deviceInput, deviceOutput);
  if (fftStatus == CUFFT_SUCCESS && cudaStatus == cudaSuccess)
    cudaStatus = cudaDeviceSynchronize();
  if (fftStatus == CUFFT_SUCCESS && cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(output, deviceOutput,
                            complexElements * sizeof(cufftComplex),
                            cudaMemcpyDeviceToHost);
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  return fftStatus == CUFFT_SUCCESS && cudaStatus == cudaSuccess ? 0 : 3;
}

extern "C" int tessera_nvidia_fft_execute_c2r_f32(
    void *opaquePlan, const float *input, float *output, void *workspace,
    size_t workspaceBytes) {
  if (opaquePlan == nullptr || input == nullptr || output == nullptr ||
      workspace == nullptr)
    return 1;
  auto *plan = static_cast<FFTPlan *>(opaquePlan);
  if (plan->kind != FFTKind::C2R || workspaceBytes < plan->workspaceBytes)
    return 1;
  int64_t realElements = plan->batch * plan->length;
  int64_t complexElements = plan->batch * (plan->length / 2 + 1);
  cufftComplex *deviceInput = nullptr;
  float *deviceOutput = nullptr;
  cudaError_t cudaStatus =
      cudaMalloc(&deviceInput, complexElements * sizeof(cufftComplex));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMalloc(&deviceOutput, realElements * sizeof(float));
  if (cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(deviceInput, input,
                            complexElements * sizeof(cufftComplex),
                            cudaMemcpyHostToDevice);
  cufftResult fftStatus = CUFFT_SUCCESS;
  if (cudaStatus == cudaSuccess)
    fftStatus = cufftSetWorkArea(plan->handle, workspace);
  if (fftStatus == CUFFT_SUCCESS && cudaStatus == cudaSuccess)
    fftStatus = cufftExecC2R(plan->handle, deviceInput, deviceOutput);
  if (fftStatus == CUFFT_SUCCESS && cudaStatus == cudaSuccess) {
    normalizeRealInverse<<<static_cast<unsigned>((realElements + 255) / 256), 256>>>(
        deviceOutput, realElements, 1.0f / static_cast<float>(plan->length));
    cudaStatus = cudaGetLastError();
  }
  if (fftStatus == CUFFT_SUCCESS && cudaStatus == cudaSuccess)
    cudaStatus = cudaDeviceSynchronize();
  if (fftStatus == CUFFT_SUCCESS && cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(output, deviceOutput, realElements * sizeof(float),
                            cudaMemcpyDeviceToHost);
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  return fftStatus == CUFFT_SUCCESS && cudaStatus == cudaSuccess ? 0 : 3;
}
