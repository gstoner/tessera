//===- StockhamRadix4.hip (AMD ROCm target hook) --------------*- HIP -*-===//
//
// Complete mixed-radix Stockham *autosort* FFT for the AMD ROCm backend.
//
// One kernel launch per stage, ping-ponging between two device buffers
// (autosort => no bit-reversal).  Radix-4 stages drain all factors of 4,
// then a single radix-2 stage handles the residual factor of 2, so every
// power-of-two N is covered.  The butterfly + twiddle indexing is bit-for-bit
// identical to the CPU reference (TargetHooks/CPU/StockhamRadix4.cpp) and the
// NVIDIA hook; that shared math is what the correctness harness certifies
// against a naive fp64 DFT on gfx1151 (RDNA 3.5, Strix Halo).
//
// Precision: complex64 storage (float2), twiddles computed in fp32
// (__sincosf).  Sign convention matches numpy.fft (forward W_N=exp(-2pi i/N)).
//
// C ABI symbols (what LowerSpectralToTargetIRPass emits for backend "amd"):
//   ts_stockham_r4_amd / ts_stockham_r2_amd  — device stage kernels
//   ts_fft_stockham_amd(in,out,N,sign)       — host driver (device pointers)
//
// Build: hipcc --offload-arch=gfx1151 -c StockhamRadix4.hip
//
//===----------------------------------------------------------------------===//

#include <hip/hip_runtime.h>

__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
  return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
// rotate by -i (forward, sign<0) or +i (inverse, sign>0)
__device__ __forceinline__ float2 crot(float2 a, int sign) {
  return sign < 0 ? make_float2(a.y, -a.x) : make_float2(-a.y, a.x);
}

// One radix-4 Stockham stage.  Thread t owns one butterfly, mapped to
// (j = t / m, k = t % m) with m = N/(4L) groups.  L = sub-DFT size done.
extern "C" __global__ void ts_stockham_r4_amd(const float2 *__restrict__ in,
                                              float2 *__restrict__ out, int N,
                                              int L, int sign) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int nbf = N / 4; // total butterflies this stage
  if (t >= nbf) return;
  int m = N / (4 * L);
  int j = t / m;
  int k = t % m;

  float ang = sign * 2.0f * float(M_PI) * float(j) / (4.0f * float(L));
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

// One radix-2 Stockham stage (residual factor of two).
extern "C" __global__ void ts_stockham_r2_amd(const float2 *__restrict__ in,
                                              float2 *__restrict__ out, int N,
                                              int L, int sign) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int nbf = N / 2;
  if (t >= nbf) return;
  int m = N / (2 * L);
  int j = t / m;
  int k = t % m;

  float ang = sign * 2.0f * float(M_PI) * float(j) / (2.0f * float(L));
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

// Simple element-wise 1/N scale for the inverse transform.
extern "C" __global__ void ts_fft_scale_amd(float2 *__restrict__ x, int N,
                                            float inv) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N) return;
  x[t].x *= inv;
  x[t].y *= inv;
}

// Host driver.  `d_in` / `d_out` are DEVICE pointers to N float2 elements;
// `d_scratch` is a device scratch buffer of N float2 (may alias neither).
// Loops radix-4 stages then a radix-2 tail, ping-ponging; the final result
// is left in `d_out`.  sign=-1 forward, sign=+1 inverse (1/N scaled).
extern "C" void ts_fft_stockham_amd(const float2 *d_in, float2 *d_out,
                                    float2 *d_scratch, int N, int sign,
                                    hipStream_t stream) {
  const int TPB = 256;
  // Ping-pong strictly between d_scratch and d_out.  Seed the input into
  // d_scratch (so the caller's d_in is never clobbered), then each stage
  // reads `cur` and writes `other`.  We do NOT precompute a parity — the
  // final result lands in whichever buffer the last stage wrote, and we copy
  // it to d_out only if it isn't already there.  (Precomputing the stage
  // count is error-prone: the residual radix-2 tail is easy to miscount.)
  float2 *cur = d_scratch;
  float2 *other = d_out;
  hipMemcpyAsync(cur, d_in, size_t(N) * sizeof(float2),
                 hipMemcpyDeviceToDevice, stream);

  int L = 1, n = N;
  auto launch = [&](int radix) {
    int nbf = N / radix;
    int blocks = (nbf + TPB - 1) / TPB;
    if (radix == 4)
      hipLaunchKernelGGL(ts_stockham_r4_amd, dim3(blocks), dim3(TPB), 0, stream,
                         cur, other, N, L, sign);
    else
      hipLaunchKernelGGL(ts_stockham_r2_amd, dim3(blocks), dim3(TPB), 0, stream,
                         cur, other, N, L, sign);
    L *= radix;
    float2 *tmp = cur; cur = other; other = tmp;
  };
  while (n % 4 == 0) { launch(4); n /= 4; }
  while (n % 2 == 0) { launch(2); n /= 2; }
  // `cur` now holds the result.  Land it in d_out if it isn't already.
  if (cur != d_out)
    hipMemcpyAsync(d_out, cur, size_t(N) * sizeof(float2),
                   hipMemcpyDeviceToDevice, stream);

  if (sign > 0) {
    int blocks = (N + TPB - 1) / TPB;
    hipLaunchKernelGGL(ts_fft_scale_amd, dim3(blocks), dim3(TPB), 0, stream,
                       d_out, N, 1.0f / float(N));
  }
}

// Host-pointer convenience entry for the runtime / D1 arbiter: allocates the
// device buffers, copies host<->device, runs the transform, and returns 0 on
// success (non-zero on any HIP error).  Lets a ctypes caller run the real GPU
// kernel with plain host complex64 arrays.
extern "C" int ts_fft_stockham_amd_hostptr(const float2 *in_host,
                                           float2 *out_host, int N, int sign) {
  float2 *d_in = nullptr, *d_out = nullptr, *d_scr = nullptr;
  int rc = 0;
  size_t bytes = size_t(N) * sizeof(float2);
  if (hipMalloc(&d_in, bytes) != hipSuccess ||
      hipMalloc(&d_out, bytes) != hipSuccess ||
      hipMalloc(&d_scr, bytes) != hipSuccess) {
    rc = 1;
    goto cleanup;
  }
  if (hipMemcpy(d_in, in_host, bytes, hipMemcpyHostToDevice) != hipSuccess) {
    rc = 2;
    goto cleanup;
  }
  ts_fft_stockham_amd(d_in, d_out, d_scr, N, sign, /*stream=*/0);
  if (hipDeviceSynchronize() != hipSuccess) {
    rc = 3;
    goto cleanup;
  }
  if (hipMemcpy(out_host, d_out, bytes, hipMemcpyDeviceToHost) != hipSuccess)
    rc = 4;
cleanup:
  if (d_in) hipFree(d_in);
  if (d_out) hipFree(d_out);
  if (d_scr) hipFree(d_scr);
  return rc;
}
