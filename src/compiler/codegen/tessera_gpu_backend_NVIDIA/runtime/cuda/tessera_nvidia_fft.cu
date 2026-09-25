#include "tessera_nvidia_fft.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstdint>
#include <mutex>
#include <new>

namespace {

enum class FFTKind { C2C, R2C, C2R };

struct FFTPlan {
  cufftHandle handle{};
  int64_t batch{};
  int64_t length{};
  size_t workspaceBytes{};
  FFTKind kind{FFTKind::C2C};
  // The CUDA device current when the plan was created. A cuFFT plan (and the
  // caller's workspace for it) belongs to that device's context.
  int device{-1};
  // Host-pointer executes stage through these, allocated on first use and
  // reused until the plan is destroyed. Measured on the RTX 5070: allocating
  // them per call cost ~0.65 ms of a 2.8 ms 256x4096 C2C (kernel: 21 us).
  void *stageIn{};
  void *stageOut{};
  // Serializes executes on this plan: the staging buffers and the plan's
  // stream and work-area bindings are per-plan state.
  std::mutex lock;
};

// Allocates the plan's staging buffers on first use (caller holds plan->lock).
bool planStaging(FFTPlan *plan, size_t inBytes, size_t outBytes) {
  if (!plan->stageIn && cudaMalloc(&plan->stageIn, inBytes) != cudaSuccess)
    return false;
  if (outBytes && !plan->stageOut &&
      cudaMalloc(&plan->stageOut, outBytes) != cudaSuccess)
    return false;
  return true;
}

// A plan executes only on the device that created it; a caller that switched
// devices must create a plan there instead. Returns the execute status: 0 when
// the current device is the plan's; 3 when the device query itself fails -- a
// CUDA error, like any other call failing during execution; 4 only when the
// query succeeds and names a different device.
int checkPlanDevice(const FFTPlan *plan) {
  int current = -1;
  if (cudaGetDevice(&current) != cudaSuccess)
    return 3;
  return current == plan->device ? 0 : 4;
}

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
  if (cudaGetDevice(&plan->device) != cudaSuccess) {
    delete plan;
    return 2;
  }
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
  return "tessera.nvidia.cuda_fft_workspace.v4";
}

extern "C" int tessera_nvidia_fft_current_device(int *device) {
  if (device == nullptr)
    return 1;
  return cudaGetDevice(device) == cudaSuccess ? 0 : 2;
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
  if (plan->stageIn)
    cudaFree(plan->stageIn);
  if (plan->stageOut)
    cudaFree(plan->stageOut);
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
  if (int deviceStatus = checkPlanDevice(plan))
    return deviceStatus;
  int64_t elements = plan->batch * plan->length;
  size_t bytes = static_cast<size_t>(elements) * sizeof(cufftComplex);
  std::lock_guard<std::mutex> guard(plan->lock);
  if (!planStaging(plan, bytes, 0))
    return 2;
  auto *deviceData = static_cast<cufftComplex *>(plan->stageIn);
  cudaError_t cudaStatus = cudaMemcpy(deviceData, input, bytes,
                                      cudaMemcpyHostToDevice);
  cufftResult fftStatus = CUFFT_SUCCESS;
  if (cudaStatus == cudaSuccess)
    fftStatus = cufftSetStream(plan->handle, nullptr);
  if (fftStatus == CUFFT_SUCCESS && cudaStatus == cudaSuccess)
    fftStatus = cufftSetWorkArea(plan->handle, workspace);
  if (fftStatus == CUFFT_SUCCESS)
    fftStatus = cufftExecC2C(plan->handle, deviceData, deviceData,
                             inverse ? CUFFT_INVERSE : CUFFT_FORWARD);
  if (fftStatus == CUFFT_SUCCESS && inverse) {
    normalizeInverse<<<static_cast<unsigned>((elements + 255) / 256), 256>>>(
        deviceData, elements, 1.0f / static_cast<float>(plan->length));
    cudaStatus = cudaGetLastError();
  }
  // The synchronous copy back orders after the transform and reports its
  // errors; no separate device synchronize.
  if (fftStatus == CUFFT_SUCCESS && cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(output, deviceData, bytes,
                            cudaMemcpyDeviceToHost);
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
  if (int deviceStatus = checkPlanDevice(plan))
    return deviceStatus;
  int64_t realElements = plan->batch * plan->length;
  int64_t complexElements = plan->batch * (plan->length / 2 + 1);
  std::lock_guard<std::mutex> guard(plan->lock);
  if (!planStaging(plan, realElements * sizeof(float),
                   complexElements * sizeof(cufftComplex)))
    return 2;
  auto *deviceInput = static_cast<float *>(plan->stageIn);
  auto *deviceOutput = static_cast<cufftComplex *>(plan->stageOut);
  cudaError_t cudaStatus = cudaMemcpy(deviceInput, input,
                                      realElements * sizeof(float),
                                      cudaMemcpyHostToDevice);
  cufftResult fftStatus = CUFFT_SUCCESS;
  if (cudaStatus == cudaSuccess)
    fftStatus = cufftSetStream(plan->handle, nullptr);
  if (fftStatus == CUFFT_SUCCESS && cudaStatus == cudaSuccess)
    fftStatus = cufftSetWorkArea(plan->handle, workspace);
  if (fftStatus == CUFFT_SUCCESS && cudaStatus == cudaSuccess)
    fftStatus = cufftExecR2C(plan->handle, deviceInput, deviceOutput);
  if (fftStatus == CUFFT_SUCCESS && cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(output, deviceOutput,
                            complexElements * sizeof(cufftComplex),
                            cudaMemcpyDeviceToHost);
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
  if (int deviceStatus = checkPlanDevice(plan))
    return deviceStatus;
  int64_t realElements = plan->batch * plan->length;
  int64_t complexElements = plan->batch * (plan->length / 2 + 1);
  std::lock_guard<std::mutex> guard(plan->lock);
  // C2R overwrites its input: it consumes the staging copy, never the caller's.
  if (!planStaging(plan, complexElements * sizeof(cufftComplex),
                   realElements * sizeof(float)))
    return 2;
  auto *deviceInput = static_cast<cufftComplex *>(plan->stageIn);
  auto *deviceOutput = static_cast<float *>(plan->stageOut);
  cudaError_t cudaStatus = cudaMemcpy(deviceInput, input,
                                      complexElements * sizeof(cufftComplex),
                                      cudaMemcpyHostToDevice);
  cufftResult fftStatus = CUFFT_SUCCESS;
  if (cudaStatus == cudaSuccess)
    fftStatus = cufftSetStream(plan->handle, nullptr);
  if (fftStatus == CUFFT_SUCCESS && cudaStatus == cudaSuccess)
    fftStatus = cufftSetWorkArea(plan->handle, workspace);
  if (fftStatus == CUFFT_SUCCESS && cudaStatus == cudaSuccess)
    fftStatus = cufftExecC2R(plan->handle, deviceInput, deviceOutput);
  if (fftStatus == CUFFT_SUCCESS && cudaStatus == cudaSuccess) {
    normalizeRealInverse<<<static_cast<unsigned>((realElements + 255) / 256), 256>>>(
        deviceOutput, realElements, 1.0f / static_cast<float>(plan->length));
    cudaStatus = cudaGetLastError();
  }
  if (fftStatus == CUFFT_SUCCESS && cudaStatus == cudaSuccess)
    cudaStatus = cudaMemcpy(output, deviceOutput, realElements * sizeof(float),
                            cudaMemcpyDeviceToHost);
  return fftStatus == CUFFT_SUCCESS && cudaStatus == cudaSuccess ? 0 : 3;
}

// ---------------------------------------------------------------------------
// Device-pointer execution (ROCm parity: ts_fft_plan_execute_*_device_batch).
// Input and output are device buffers on the plan's device; the transform and
// the inverse normalization are enqueued on `stream` (nullptr: the legacy
// default stream) and NOT synchronized -- the caller orders its own reads.
// No staging and no host copies, so data that stays on the GPU pays neither.
// Statuses as for the host-pointer entry points; 3 includes a failed launch.
namespace {

cufftResult bindStream(FFTPlan *plan, void *workspace, void *stream) {
  cufftResult status =
      cufftSetStream(plan->handle, static_cast<cudaStream_t>(stream));
  return status == CUFFT_SUCCESS ? cufftSetWorkArea(plan->handle, workspace)
                                 : status;
}

} // namespace

extern "C" int tessera_nvidia_fft_execute_c2c_device_f32(
    void *opaquePlan, const void *input, void *output, void *workspace,
    size_t workspaceBytes, int inverse, void *stream) {
  if (opaquePlan == nullptr || input == nullptr || output == nullptr ||
      workspace == nullptr || (inverse != 0 && inverse != 1))
    return 1;
  auto *plan = static_cast<FFTPlan *>(opaquePlan);
  if (plan->kind != FFTKind::C2C || workspaceBytes < plan->workspaceBytes)
    return 1;
  if (int deviceStatus = checkPlanDevice(plan))
    return deviceStatus;
  std::lock_guard<std::mutex> guard(plan->lock);
  int64_t elements = plan->batch * plan->length;
  auto *out = static_cast<cufftComplex *>(output);
  cufftResult fftStatus = bindStream(plan, workspace, stream);
  if (fftStatus == CUFFT_SUCCESS)
    fftStatus = cufftExecC2C(
        plan->handle, static_cast<cufftComplex *>(const_cast<void *>(input)),
        out, inverse ? CUFFT_INVERSE : CUFFT_FORWARD);
  if (fftStatus != CUFFT_SUCCESS)
    return 3;
  if (inverse) {
    normalizeInverse<<<static_cast<unsigned>((elements + 255) / 256), 256, 0,
                       static_cast<cudaStream_t>(stream)>>>(
        out, elements, 1.0f / static_cast<float>(plan->length));
    if (cudaGetLastError() != cudaSuccess)
      return 3;
  }
  return 0;
}

extern "C" int tessera_nvidia_fft_execute_r2c_device_f32(
    void *opaquePlan, const void *input, void *output, void *workspace,
    size_t workspaceBytes, void *stream) {
  if (opaquePlan == nullptr || input == nullptr || output == nullptr ||
      workspace == nullptr)
    return 1;
  auto *plan = static_cast<FFTPlan *>(opaquePlan);
  if (plan->kind != FFTKind::R2C || workspaceBytes < plan->workspaceBytes)
    return 1;
  if (int deviceStatus = checkPlanDevice(plan))
    return deviceStatus;
  std::lock_guard<std::mutex> guard(plan->lock);
  cufftResult fftStatus = bindStream(plan, workspace, stream);
  if (fftStatus == CUFFT_SUCCESS)
    fftStatus = cufftExecR2C(
        plan->handle, static_cast<float *>(const_cast<void *>(input)),
        static_cast<cufftComplex *>(output));
  return fftStatus == CUFFT_SUCCESS ? 0 : 3;
}

// C2R overwrites `input` (cuFFT's out-of-place C2R contract); pass a copy if
// the spectrum is needed afterwards.
extern "C" int tessera_nvidia_fft_execute_c2r_device_f32(
    void *opaquePlan, void *input, void *output, void *workspace,
    size_t workspaceBytes, void *stream) {
  if (opaquePlan == nullptr || input == nullptr || output == nullptr ||
      workspace == nullptr)
    return 1;
  auto *plan = static_cast<FFTPlan *>(opaquePlan);
  if (plan->kind != FFTKind::C2R || workspaceBytes < plan->workspaceBytes)
    return 1;
  if (int deviceStatus = checkPlanDevice(plan))
    return deviceStatus;
  std::lock_guard<std::mutex> guard(plan->lock);
  int64_t realElements = plan->batch * plan->length;
  auto *out = static_cast<float *>(output);
  cufftResult fftStatus = bindStream(plan, workspace, stream);
  if (fftStatus == CUFFT_SUCCESS)
    fftStatus = cufftExecC2R(plan->handle, static_cast<cufftComplex *>(input),
                             out);
  if (fftStatus != CUFFT_SUCCESS)
    return 3;
  normalizeRealInverse<<<static_cast<unsigned>((realElements + 255) / 256), 256,
                         0, static_cast<cudaStream_t>(stream)>>>(
      out, realElements, 1.0f / static_cast<float>(plan->length));
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
