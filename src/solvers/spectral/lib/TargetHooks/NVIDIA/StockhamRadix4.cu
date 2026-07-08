//===- StockhamRadix4.cu (NVIDIA CUDA target hook) ------------*- CUDA -*-===//
//
// Complete mixed-radix Stockham *autosort* FFT for the NVIDIA backend.
//
// Direct mirror of the AMD/ROCm hook (TargetHooks/AMD/StockhamRadix4.hip)
// and the CPU reference: one kernel launch per stage, ping-ponging between
// two device buffers, radix-4 stages then a radix-2 tail.  The butterfly +
// twiddle indexing is identical across all three backends so the CPU
// correctness sentinel certifies the exact math the GPU kernels run.
//
// Precision: complex64 storage (float2), twiddles in fp32 (__sincosf).
// Sign convention matches numpy.fft (forward W_N = exp(-2pi i/N)).
//
// C ABI symbols (what LowerSpectralToTargetIRPass emits for backend "nvidia"):
//   ts_stockham_r4_nvidia / ts_stockham_r2_nvidia — device stage kernels
//   ts_fft_stockham_nvidia(in,out,scratch,N,sign) — host driver
//
// Build: nvcc -arch=sm_90 -c StockhamRadix4.cu   (sm_80+/sm_120 all fine)
//
//===----------------------------------------------------------------------===//

#include <cuda_runtime.h>
#include <math_constants.h>

__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
  return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
__device__ __forceinline__ float2 crot(float2 a, int sign) {
  return sign < 0 ? make_float2(a.y, -a.x) : make_float2(-a.y, a.x);
}

extern "C" __global__ void ts_stockham_r4_nvidia(const float2 *__restrict__ in,
                                                 float2 *__restrict__ out,
                                                 int N, int L, int sign) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int nbf = N / 4;
  if (t >= nbf) return;
  int m = N / (4 * L);
  int j = t / m;
  int k = t % m;

  float ang = sign * 2.0f * CUDART_PI_F * float(j) / (4.0f * float(L));
  float s, c;
  __sincosf(ang, &s, &c);
  float2 w1 = make_float2(c, s);
  float2 w2 = cmul(w1, w1);
  float2 w3 = cmul(w2, w1);

  int i = k * L + j;
  float2 c0 = in[i + 0 * L * m];
  float2 c1 = cmul(in[i + 1 * L * m], w1);
  float2 c2 = cmul(in[i + 2 * L * m], w2);
  float2 c3 = cmul(in[i + 3 * L * m], w3);
  float2 t0 = make_float2(c0.x + c2.x, c0.y + c2.y);
  float2 t1 = make_float2(c0.x - c2.x, c0.y - c2.y);
  float2 t2 = make_float2(c1.x + c3.x, c1.y + c3.y);
  float2 t3 = crot(make_float2(c1.x - c3.x, c1.y - c3.y), sign);
  int o = k * (4 * L) + j;
  out[o + 0 * L] = make_float2(t0.x + t2.x, t0.y + t2.y);
  out[o + 1 * L] = make_float2(t1.x + t3.x, t1.y + t3.y);
  out[o + 2 * L] = make_float2(t0.x - t2.x, t0.y - t2.y);
  out[o + 3 * L] = make_float2(t1.x - t3.x, t1.y - t3.y);
}

extern "C" __global__ void ts_stockham_r2_nvidia(const float2 *__restrict__ in,
                                                 float2 *__restrict__ out,
                                                 int N, int L, int sign) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int nbf = N / 2;
  if (t >= nbf) return;
  int m = N / (2 * L);
  int j = t / m;
  int k = t % m;

  float ang = sign * 2.0f * CUDART_PI_F * float(j) / (2.0f * float(L));
  float s, c;
  __sincosf(ang, &s, &c);
  float2 w1 = make_float2(c, s);

  int i = k * L + j;
  float2 c0 = in[i + 0 * L * m];
  float2 c1 = cmul(in[i + 1 * L * m], w1);
  int o = k * (2 * L) + j;
  out[o + 0 * L] = make_float2(c0.x + c1.x, c0.y + c1.y);
  out[o + 1 * L] = make_float2(c0.x - c1.x, c0.y - c1.y);
}

extern "C" __global__ void ts_fft_scale_nvidia(float2 *__restrict__ x, int N,
                                               float inv) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N) return;
  x[t].x *= inv;
  x[t].y *= inv;
}

extern "C" void ts_fft_stockham_nvidia(const float2 *d_in, float2 *d_out,
                                       float2 *d_scratch, int N, int sign,
                                       cudaStream_t stream) {
  const int TPB = 256;
  // Ping-pong strictly between d_scratch and d_out; copy to d_out at the end
  // only if the result didn't already land there.  See the AMD hook for why
  // we avoid precomputing a stage-count parity (the radix-2 tail is easy to
  // miscount, which corrupts the ping-pong).
  float2 *cur = d_scratch;
  float2 *other = d_out;
  cudaMemcpyAsync(cur, d_in, size_t(N) * sizeof(float2),
                  cudaMemcpyDeviceToDevice, stream);

  int L = 1, n = N;
  auto launch = [&](int radix) {
    int nbf = N / radix;
    int blocks = (nbf + TPB - 1) / TPB;
    if (radix == 4)
      ts_stockham_r4_nvidia<<<blocks, TPB, 0, stream>>>(cur, other, N, L, sign);
    else
      ts_stockham_r2_nvidia<<<blocks, TPB, 0, stream>>>(cur, other, N, L, sign);
    L *= radix;
    float2 *tmp = cur; cur = other; other = tmp;
  };
  while (n % 4 == 0) { launch(4); n /= 4; }
  while (n % 2 == 0) { launch(2); n /= 2; }
  if (cur != d_out)
    cudaMemcpyAsync(d_out, cur, size_t(N) * sizeof(float2),
                    cudaMemcpyDeviceToDevice, stream);

  if (sign > 0) {
    int blocks = (N + TPB - 1) / TPB;
    ts_fft_scale_nvidia<<<blocks, TPB, 0, stream>>>(d_out, N, 1.0f / float(N));
  }
}
