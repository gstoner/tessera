//===- StockhamRadix4.hip (AMD ROCm target hook) --------------*- HIP -*-===//
//
// Mixed-radix Stockham *autosort* FFT + Bluestein, for the AMD ROCm backend.
//
// Planning (which radices, and whether N factors at all) is shared with the
// CPU and NVIDIA hooks via TargetHooks/Common/FFTPlan.h.  This file is the
// AMD *execution*: kernels, launches, device buffers.
//
// One kernel launch per stage, ping-ponging between two device buffers
// (autosort => no bit-reversal).  Radix-4 stages drain all factors of 4, then
// a radix-2 stage handles the residual factor of 2; a generic radix-r kernel
// covers the odd small primes; Bluestein covers everything else.
//
// This driver previously carried its own copy of `while (n%4)... while (n%2)`
// and, like the CPU and NVIDIA copies, silently returned a half-finished
// transform for any other N.  Planning now happens once, in one place.  The butterfly + twiddle indexing is bit-for-bit
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

#include "../Common/FFTPlan.h"

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

// One generic radix-r Stockham stage for the odd small primes (3..13).
// Thread t owns one butterfly.  The r-point DFT is evaluated inline; at these
// radices the O(r^2) inner loop is cheaper than the extra global traffic a
// table would cost, which is the opposite of the CPU trade-off and the reason
// the two are not the same code.
extern "C" __global__ void ts_stockham_rn_amd(const float2 *__restrict__ in,
                                              float2 *__restrict__ out, int N,
                                              int L, int r, int sign) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int nbf = N / r;
  if (t >= nbf) return;
  int m = N / (r * L);
  int j = t / m;
  int k = t % m;
  int i = k * L + j;
  int o = k * (r * L) + j;

  float2 c[16];  // kMaxRadix is 13; 16 keeps the array a power of two
  for (int q = 0; q < r; ++q) {
    float ang = sign * 2.0f * float(M_PI) * float(j) * float(q) /
                (float(r) * float(L));
    float s, cc;
    __sincosf(ang, &s, &cc);
    float2 w = make_float2(cc, s);
    float2 v = in[i + (size_t)q * L * m];
    c[q] = q == 0 ? v : cmul(v, w);
  }
  for (int p = 0; p < r; ++p) {
    float2 acc = make_float2(0.0f, 0.0f);
    for (int q = 0; q < r; ++q) {
      float ang = sign * 2.0f * float(M_PI) * float((p * q) % r) / float(r);
      float s, cc;
      __sincosf(ang, &s, &cc);
      float2 prod = cmul(c[q], make_float2(cc, s));
      acc.x += prod.x;
      acc.y += prod.y;
    }
    out[o + (size_t)p * L] = acc;
  }
}

// Bluestein helpers.  `chirp[j] = exp(-sign*pi*i*j^2/N)`; the (j*j) mod 2N
// reduction matches FFTPlan.h's `fft_chirp_angle` and exists for the same
// reason -- j^2 leaves a float's exact range quickly, and the error shows up
// as a phase drift that still looks like a spectrum.
extern "C" __global__ void ts_bluestein_pre_amd(const float2 *__restrict__ x,
                                                float2 *__restrict__ a,
                                                float2 *__restrict__ b, int N,
                                                int M, int sign) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= M) return;
  if (j >= N) { a[j] = make_float2(0.f, 0.f); }
  if (j < N) {
    long long jj = ((long long)j * (long long)j) % (2LL * (long long)N);
    double ang = -(double)sign * 3.14159265358979323846 * (double)jj / (double)N;
    float2 ch = make_float2((float)cos(ang), (float)sin(ang));
    a[j] = cmul(x[j], make_float2(ch.x, -ch.y));
    b[j] = ch;
    if (j) b[M - j] = ch;
  } else if (j > M - N) {
    // filled by the mirrored write above
  } else {
    b[j] = make_float2(0.f, 0.f);
  }
}

extern "C" __global__ void ts_bluestein_post_amd(const float2 *__restrict__ conv,
                                                 float2 *__restrict__ out,
                                                 int N, int sign) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= N) return;
  long long kk = ((long long)k * (long long)k) % (2LL * (long long)N);
  double ang = -(double)sign * 3.14159265358979323846 * (double)kk / (double)N;
  float2 ch = make_float2((float)cos(ang), (float)sin(ang));
  out[k] = cmul(conv[k], make_float2(ch.x, -ch.y));
}

extern "C" __global__ void ts_cmul_inplace_amd(float2 *__restrict__ a,
                                               const float2 *__restrict__ b,
                                               int M) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  a[i] = cmul(a[i], b[i]);
}

extern "C" __global__ void ts_conj_scale_amd(float2 *__restrict__ a, int M,
                                             float s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  a[i] = make_float2(a[i].x * s, -a[i].y * s);
}

// Last error from a void-returning driver call on this thread.
//
// `ts_fft_stockham_amd` is a stream-ordered launch helper and returns void, so
// an allocation failure inside the Bluestein path had nowhere to go. Rather
// than change a published C ABI signature, the status is parked here and the
// host-pointer wrapper -- which DOES return a status -- reads it.
static int &fft_last_error_slot() {
  static thread_local int last = 0;
  return last;
}

// C-callable accessors.  A reference return is not C-compatible, so the slot
// stays internal and the ABI exposes plain ints.
extern "C" int ts_fft_last_error_amd(void) { return fft_last_error_slot(); }
extern "C" void ts_fft_clear_error_amd(void) { fft_last_error_slot() = 0; }

extern "C" void ts_fft_stockham_amd(const float2 *d_in, float2 *d_out,
                                    float2 *d_scratch, int N, int sign,
                                    hipStream_t stream);

// Bluestein on device.  Allocates its own M-sized buffers: the caller's
// scratch is sized for N, and M can be nearly 4N, so reusing it would be a
// silent overrun.
static int bluestein_amd(const float2 *d_in, float2 *d_out, int N, int M,
                         int sign, hipStream_t stream) {
  const int TPB = 256;
  float2 *a = nullptr, *b = nullptr, *fa = nullptr, *fb = nullptr,
         *scr = nullptr;
  size_t mb = size_t(M) * sizeof(float2);
  int rc = 0;
  if (hipMalloc(&a, mb) != hipSuccess || hipMalloc(&b, mb) != hipSuccess ||
      hipMalloc(&fa, mb) != hipSuccess || hipMalloc(&fb, mb) != hipSuccess ||
      hipMalloc(&scr, mb) != hipSuccess) {
    rc = 1;
    goto done;
  }
  // Errors are surfaced through the driver's own synchronise/return path; the
  // casts silence -Wunused-value on hipError_t without pretending to handle it
  // here, where there is nothing useful to do but continue to cleanup.
  (void)hipMemsetAsync(a, 0, mb, stream);
  (void)hipMemsetAsync(b, 0, mb, stream);
  hipLaunchKernelGGL(ts_bluestein_pre_amd, dim3((M + TPB - 1) / TPB), dim3(TPB),
                     0, stream, d_in, a, b, N, M, sign);
  ts_fft_stockham_amd(a, fa, scr, M, -1, stream);
  ts_fft_stockham_amd(b, fb, scr, M, -1, stream);
  hipLaunchKernelGGL(ts_cmul_inplace_amd, dim3((M + TPB - 1) / TPB), dim3(TPB),
                     0, stream, fa, fb, M);
  // ifft(y) = conj(fft(conj(y)))/M -- using the driver's inverse would apply
  // the 1/M scale twice.
  hipLaunchKernelGGL(ts_conj_scale_amd, dim3((M + TPB - 1) / TPB), dim3(TPB), 0,
                     stream, fa, M, 1.0f);
  ts_fft_stockham_amd(fa, fb, scr, M, -1, stream);
  hipLaunchKernelGGL(ts_conj_scale_amd, dim3((M + TPB - 1) / TPB), dim3(TPB), 0,
                     stream, fb, M, 1.0f / float(M));
  hipLaunchKernelGGL(ts_bluestein_post_amd, dim3((N + TPB - 1) / TPB), dim3(TPB),
                     0, stream, fb, d_out, N, sign);
done:
  if (a) (void)hipFree(a);
  if (b) (void)hipFree(b);
  if (fa) (void)hipFree(fa);
  if (fb) (void)hipFree(fb);
  if (scr) (void)hipFree(scr);
  return rc;
}

// Host driver.  `d_in` / `d_out` are DEVICE pointers to N float2 elements;
// `d_scratch` is a device scratch buffer of N float2 (may alias neither).
// Loops radix-4 stages then a radix-2 tail, ping-ponging; the final result
// is left in `d_out`.  sign=-1 forward, sign=+1 inverse (1/N scaled).
extern "C" void ts_fft_stockham_amd(const float2 *d_in, float2 *d_out,
                                    float2 *d_scratch, int N, int sign,
                                    hipStream_t stream) {
  const int TPB = 256;
  if (N <= 1) {
    if (N == 1)
      (void)hipMemcpyAsync(d_out, d_in, sizeof(float2),
                           hipMemcpyDeviceToDevice, stream);
    return;
  }
  const tessera::spectral::FFTPlan plan = tessera::spectral::fft_plan(N);
  if (plan.kind == tessera::spectral::kFFTBluestein) {
    // Bluestein allocates five M-sized buffers and M can be nearly 4N, so this
    // fails in practice under memory pressure or at a large prime length.
    // Discarding the status let the host wrapper synchronise, copy an
    // UNINITIALISED d_out, and return 0 -- the Python candidate would then
    // label garbage `rocm_stockham`, which is the exact wrong-answer-under-a-
    // kernel-name failure this PR set out to remove. Record it so the wrapper
    // can report it; the driver's own signature is void (it is a stream-
    // ordered launch helper), so the status travels in a thread-local.
    if (bluestein_amd(d_in, d_out, N, plan.bluestein_m, sign, stream) != 0) {
      fft_last_error_slot() = 5;
      return;
    }
    if (sign > 0) {
      int blocks = (N + TPB - 1) / TPB;
      hipLaunchKernelGGL(ts_fft_scale_amd, dim3(blocks), dim3(TPB), 0, stream,
                         d_out, N, 1.0f / float(N));
    }
    return;
  }
  // Ping-pong strictly between d_scratch and d_out.  Seed the input into
  // d_scratch (so the caller's d_in is never clobbered), then each stage
  // reads `cur` and writes `other`.  We do NOT precompute a parity — the
  // final result lands in whichever buffer the last stage wrote, and we copy
  // it to d_out only if it isn't already there.  (Precomputing the stage
  // count is error-prone: the residual radix-2 tail is easy to miscount.)
  float2 *cur = d_scratch;
  float2 *other = d_out;
  (void)hipMemcpyAsync(cur, d_in, size_t(N) * sizeof(float2),
                       hipMemcpyDeviceToDevice, stream);

  int L = 1;
  auto launch = [&](int radix) {
    int nbf = N / radix;
    int blocks = (nbf + TPB - 1) / TPB;
    if (radix == 4)
      hipLaunchKernelGGL(ts_stockham_r4_amd, dim3(blocks), dim3(TPB), 0, stream,
                         cur, other, N, L, sign);
    else if (radix == 2)
      hipLaunchKernelGGL(ts_stockham_r2_amd, dim3(blocks), dim3(TPB), 0, stream,
                         cur, other, N, L, sign);
    else
      hipLaunchKernelGGL(ts_stockham_rn_amd, dim3(blocks), dim3(TPB), 0, stream,
                         cur, other, N, L, radix, sign);
    L *= radix;
    float2 *tmp = cur; cur = other; other = tmp;
  };
  for (int s = 0; s < plan.stage_count; ++s)
    launch(plan.stages[s]);
  // `cur` now holds the result.  Land it in d_out if it isn't already.
  if (cur != d_out)
    (void)hipMemcpyAsync(d_out, cur, size_t(N) * sizeof(float2),
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
  ts_fft_clear_error_amd();
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
  if (ts_fft_last_error_amd() != 0) {
    // The transform never ran; d_out holds uninitialised device memory and
    // must NOT be copied back as a result.
    rc = ts_fft_last_error_amd();
    goto cleanup;
  }
  if (hipDeviceSynchronize() != hipSuccess) {
    rc = 3;
    goto cleanup;
  }
  if (hipMemcpy(out_host, d_out, bytes, hipMemcpyDeviceToHost) != hipSuccess)
    rc = 4;
cleanup:
  if (d_in) (void)hipFree(d_in);
  if (d_out) (void)hipFree(d_out);
  if (d_scr) (void)hipFree(d_scr);
  return rc;
}
