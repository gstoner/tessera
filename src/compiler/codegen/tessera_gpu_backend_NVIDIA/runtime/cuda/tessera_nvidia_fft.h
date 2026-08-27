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

// Target-owned SM120 STFT/ISTFT policy package. Shape and stride descriptors
// are in elements, not bytes. The host boundary owns only arbitrary-layout
// packing and result staging; framing, transforms, scaling, and overlap-add
// execute on the CUDA device. pad_mode is 0=constant or 1=reflect.
const char *tessera_nvidia_spectral_package_abi();
int tessera_nvidia_spectral_arch();
int tessera_nvidia_dct_policy_layout_f32(
    const char *digest, const float *input, float *output, int rank,
    const int64_t *shape, const int64_t *strides, int axis, int dct_type,
    float output_scale);
int tessera_nvidia_dct_policy_layout_storage(
    const char *digest, const void *input, void *output, int rank,
    const int64_t *shape, const int64_t *strides, int axis, int dct_type,
    int storage, float output_scale);
int tessera_nvidia_stft_policy_broadcast_layout_f32(
    const char *digest, const float *input, const float *window, float *output,
    int rank, const int64_t *shape, const int64_t *strides, int axis,
    int window_rank, const int64_t *window_shape,
    const int64_t *window_strides, int n_fft, int hop, int frames,
    float output_scale, int center, int pad_mode, int onesided);
int tessera_nvidia_stft_policy_broadcast_layout_storage(
    const char *digest, const void *input, const void *window, float *output,
    int rank, const int64_t *shape, const int64_t *strides, int axis,
    int window_rank, const int64_t *window_shape,
    const int64_t *window_strides, int n_fft, int hop, int frames,
    int storage, float output_scale, int center, int pad_mode, int onesided);
int tessera_nvidia_stft_jvp_broadcast_layout_f32(
    const char *digest, const float *input, const float *window,
    const float *dinput, const float *dwindow, float *primal,
    float *tangent, int rank, const int64_t *shape,
    const int64_t *strides, int axis, int window_rank,
    const int64_t *window_shape, const int64_t *window_strides, int n_fft,
    int hop, int frames, float output_scale, int center, int pad_mode,
    int onesided);
int tessera_nvidia_stft_jvp_broadcast_layout_storage(
    const char *digest, const void *input, const void *window,
    const void *dinput, const void *dwindow, float *primal,
    float *tangent, int rank, const int64_t *shape,
    const int64_t *strides, int axis, int window_rank,
    const int64_t *window_shape, const int64_t *window_strides, int n_fft,
    int hop, int frames, int storage, float output_scale, int center,
    int pad_mode, int onesided);
int tessera_nvidia_istft_policy_broadcast_layout_f32(
    const char *digest, const float *input, const float *window, float *output,
    int rank, const int64_t *shape, const int64_t *strides, int axis,
    int window_rank, const int64_t *window_shape,
    const int64_t *window_strides, int n_fft, int hop, float output_scale,
    int center, int output_samples, int onesided);
int tessera_nvidia_istft_policy_broadcast_layout_storage(
    const char *digest, const float *input, const void *window, void *output,
    int rank, const int64_t *shape, const int64_t *strides, int axis,
    int window_rank, const int64_t *window_shape,
    const int64_t *window_strides, int n_fft, int hop, int storage,
    float output_scale, int center, int output_samples, int onesided);
int tessera_nvidia_istft_jvp_broadcast_layout_f32(
    const char *digest, const float *input, const float *window,
    const float *dinput, const float *dwindow, float *primal,
    float *tangent, int rank, const int64_t *shape,
    const int64_t *strides, int axis, int window_rank,
    const int64_t *window_shape, const int64_t *window_strides, int n_fft,
    int hop, float output_scale, int center, int output_samples, int onesided);
int tessera_nvidia_istft_jvp_broadcast_layout_storage(
    const char *digest, const float *input, const void *window,
    const float *dinput, const void *dwindow, void *primal, void *tangent,
    int rank, const int64_t *shape, const int64_t *strides, int axis,
    int window_rank, const int64_t *window_shape,
    const int64_t *window_strides, int n_fft, int hop, int storage,
    float output_scale, int center, int output_samples, int onesided);
int tessera_nvidia_stft_backward_broadcast_layout_f32(
    const char *digest, const float *dy, const float *input,
    const float *window, float *dx, float *dwindow, int x_rank,
    const int64_t *x_shape, const int64_t *x_strides, int axis, int dy_rank,
    const int64_t *dy_shape, const int64_t *dy_strides, int window_rank,
    const int64_t *window_shape, const int64_t *window_strides, int n_fft,
    int hop, float forward_scale, int center, int pad_mode, int onesided);
int tessera_nvidia_stft_backward_broadcast_layout_storage(
    const char *digest, const float *dy, const void *input,
    const void *window, void *dx, void *dwindow, int x_rank,
    const int64_t *x_shape, const int64_t *x_strides, int axis, int dy_rank,
    const int64_t *dy_shape, const int64_t *dy_strides, int window_rank,
    const int64_t *window_shape, const int64_t *window_strides, int n_fft,
    int hop, int storage, float forward_scale, int center, int pad_mode,
    int onesided);
int tessera_nvidia_istft_backward_broadcast_layout_f32(
    const char *digest, const float *dy, const float *spectrum,
    const float *window, float *dspectrum, float *dwindow, int dy_rank,
    const int64_t *dy_shape, const int64_t *dy_strides, int output_axis,
    int spectrum_rank, const int64_t *spectrum_shape,
    const int64_t *spectrum_strides, int frame_axis, int bin_axis,
    int window_rank, const int64_t *window_shape,
    const int64_t *window_strides, int n_fft, int hop, float inverse_scale,
    int center, int onesided);
int tessera_nvidia_istft_backward_broadcast_layout_storage(
    const char *digest, const void *dy, const float *spectrum,
    const void *window, float *dspectrum, void *dwindow, int dy_rank,
    const int64_t *dy_shape, const int64_t *dy_strides, int output_axis,
    int spectrum_rank, const int64_t *spectrum_shape,
    const int64_t *spectrum_strides, int frame_axis, int bin_axis,
    int window_rank, const int64_t *window_shape,
    const int64_t *window_strides, int n_fft, int hop, int storage,
    float inverse_scale, int center, int onesided);
int tessera_nvidia_streaming_stft_broadcast_layout_f32(
    const char *digest, const float *input, const float *tail,
    const float *window, float *output, float *next_tail, int rank,
    const int64_t *shape, const int64_t *strides, int axis, int tail_samples,
    int window_rank, const int64_t *window_shape,
    const int64_t *window_strides, int n_fft, int hop, int frames,
    float output_scale, int onesided);

}
