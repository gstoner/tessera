#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

// Versioned canonical CUDA FFT package contract.
const char *tessera_nvidia_fft_package_abi();

// Creates one reusable batched contiguous complex64 C2C plan. cuFFT automatic
// workspace allocation is disabled. The returned byte count is the minimum
// device workspace required by tessera_nvidia_fft_execute_c2c_f32.
int tessera_nvidia_fft_plan_create_c2c_f32(int64_t batch, int64_t length,
                                           void **plan,
                                           size_t *workspace_bytes);
int tessera_nvidia_fft_plan_create_r2c_f32(int64_t batch, int64_t length,
                                           void **plan,
                                           size_t *workspace_bytes);
int tessera_nvidia_fft_plan_create_c2r_f32(int64_t batch, int64_t length,
                                           void **plan,
                                           size_t *workspace_bytes);
int tessera_nvidia_fft_plan_destroy(void *plan);

// Explicit caller-owned device workspace lifecycle. These helpers are narrow
// CUDA allocation shims; workspace identity remains visible to the caller.
int tessera_nvidia_fft_workspace_alloc(size_t bytes, void **workspace);
int tessera_nvidia_fft_workspace_free(void *workspace);

// Host-staging execution. input/output are interleaved real/imag f32 arrays of
// length 2*batch*length. inverse=0 is forward; inverse=1 applies canonical 1/N
// normalization on-device. Workspace must match the plan's reported bound.
int tessera_nvidia_fft_execute_c2c_f32(void *plan, const float *input,
                                      float *output, void *workspace,
                                      size_t workspace_bytes, int inverse);

// Native real transforms. R2C writes batch*(length/2+1) interleaved complex
// values. C2R consumes that Hermitian half-spectrum and applies canonical 1/N
// normalization on-device.
int tessera_nvidia_fft_execute_r2c_f32(void *plan, const float *input,
                                      float *output, void *workspace,
                                      size_t workspace_bytes);
int tessera_nvidia_fft_execute_c2r_f32(void *plan, const float *input,
                                      float *output, void *workspace,
                                      size_t workspace_bytes);

}
