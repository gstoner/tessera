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

#include <cmath>
#include <cstring>
#include <mutex>
#include <new>

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

extern "C" __global__ void ts_rfft_post_amd(
    const float2 *__restrict__ packed, float2 *__restrict__ spectrum, int M,
    int N, int batch) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * (M + 1);
  if (t >= total) return;
  int row = t / (M + 1), k = t % (M + 1);
  const float2 *z = packed + size_t(row) * M;
  int ai = k == M ? 0 : k;
  int bi = (M - k) % M;
  float2 a = z[ai], b = make_float2(z[bi].x, -z[bi].y);
  float2 diff = make_float2(a.x - b.x, a.y - b.y);
  float s, c;
  __sincosf(-2.0f * float(M_PI) * float(k) / float(N), &s, &c);
  float2 q = cmul(make_float2(c, s), diff);
  spectrum[t] = make_float2(0.5f * (a.x + b.x + q.y),
                            (k == 0 || k == M)
                                ? 0.0f
                                : 0.5f * (a.y + b.y - q.x));
}

extern "C" __global__ void ts_irfft_pre_amd(
    const float2 *__restrict__ spectrum, float2 *__restrict__ packed, int M,
    int N, int batch) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * M;
  if (t >= total) return;
  int row = t / M, k = t % M;
  const float2 *x = spectrum + size_t(row) * (M + 1);
  float2 a = x[k], b = make_float2(x[M - k].x, -x[M - k].y);
  float2 diff = make_float2(a.x - b.x, a.y - b.y);
  float s, c;
  __sincosf(2.0f * float(M_PI) * float(k) / float(N), &s, &c);
  float2 q = cmul(make_float2(c, s), diff);
  packed[t] = make_float2(0.5f * (a.x + b.x - q.y),
                          0.5f * (a.y + b.y + q.x));
}

// One generic radix-r Stockham stage for the odd small primes (3..17).
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

  float2 c[32];  // kMaxRadix is 17; retain a power-of-two private array
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

// Small power-of-two residency selected by the content-addressed gfx1151 v4
// FFT contract. All stages ping-pong between two LDS slabs and touch global
// memory only at entry/exit; one workgroup owns each transform in the batch.
extern "C" __global__ void
ts_fft_stockham_fused_lds_amd(const float2 *__restrict__ in,
                             float2 *__restrict__ out, int N, int sign) {
  extern __shared__ float2 storage[];
  float2 *cur = storage;
  float2 *other = storage + N;
  const size_t rowOffset = size_t(blockIdx.x) * size_t(N);
  const float2 *rowIn = in + rowOffset;
  float2 *rowOut = out + rowOffset;
  for (int i = threadIdx.x; i < N; i += blockDim.x)
    cur[i] = rowIn[i];
  __syncthreads();

  int L = 1;
  int rest = N;
  while (rest % 4 == 0) {
    const int m = N / (4 * L);
    for (int t = threadIdx.x; t < N / 4; t += blockDim.x) {
      const int j = t / m;
      const int k = t % m;
      const float ang = sign * 2.0f * float(M_PI) * float(j) /
                        (4.0f * float(L));
      float s, c;
      __sincosf(ang, &s, &c);
      const float2 w1 = make_float2(c, s);
      const float2 w2 = cmul(w1, w1);
      const float2 w3 = cmul(w2, w1);
      const int i = k * L + j;
      const float2 c0 = cur[i];
      const float2 c1 = cmul(cur[i + L * m], w1);
      const float2 c2 = cmul(cur[i + 2 * L * m], w2);
      const float2 c3 = cmul(cur[i + 3 * L * m], w3);
      const float2 t0 = make_float2(c0.x + c2.x, c0.y + c2.y);
      const float2 t1 = make_float2(c0.x - c2.x, c0.y - c2.y);
      const float2 t2 = make_float2(c1.x + c3.x, c1.y + c3.y);
      const float2 t3 = crot(make_float2(c1.x - c3.x, c1.y - c3.y), sign);
      const int o = k * (4 * L) + j;
      other[o] = make_float2(t0.x + t2.x, t0.y + t2.y);
      other[o + L] = make_float2(t1.x + t3.x, t1.y + t3.y);
      other[o + 2 * L] = make_float2(t0.x - t2.x, t0.y - t2.y);
      other[o + 3 * L] = make_float2(t1.x - t3.x, t1.y - t3.y);
    }
    __syncthreads();
    float2 *tmp = cur; cur = other; other = tmp;
    L *= 4;
    rest /= 4;
  }
  while (rest % 2 == 0) {
    const int m = N / (2 * L);
    for (int t = threadIdx.x; t < N / 2; t += blockDim.x) {
      const int j = t / m;
      const int k = t % m;
      const float ang = sign * 2.0f * float(M_PI) * float(j) /
                        (2.0f * float(L));
      float s, c;
      __sincosf(ang, &s, &c);
      const float2 w = make_float2(c, s);
      const int i = k * L + j;
      const float2 c0 = cur[i];
      const float2 c1 = cmul(cur[i + L * m], w);
      const int o = k * (2 * L) + j;
      other[o] = make_float2(c0.x + c1.x, c0.y + c1.y);
      other[o + L] = make_float2(c0.x - c1.x, c0.y - c1.y);
    }
    __syncthreads();
    float2 *tmp = cur; cur = other; other = tmp;
    L *= 2;
    rest /= 2;
  }

  const float scale = sign > 0 ? 1.0f / float(N) : 1.0f;
  for (int i = threadIdx.x; i < N; i += blockDim.x)
    rowOut[i] = make_float2(cur[i].x * scale, cur[i].y * scale);
}

extern "C" int ts_fft_stockham_fused_lds_launch_amd(
    const float2 *d_in, float2 *d_out, int N, int sign, hipStream_t stream) {
  if (N < 2 || N > 1024 || (N & (N - 1)) != 0)
    return 7;
  hipLaunchKernelGGL(ts_fft_stockham_fused_lds_amd, dim3(1), dim3(256),
                     size_t(2 * N) * sizeof(float2), stream, d_in, d_out, N,
                     sign);
  return hipGetLastError() == hipSuccess ? 0 : 8;
}

static int launch_fused_lds_batch(const float2 *d_in, float2 *d_out, int batch,
                                  int N, int sign, hipStream_t stream) {
  if (batch <= 0 || N < 2 || N > 1024 || (N & (N - 1)) != 0)
    return 7;
  hipLaunchKernelGGL(ts_fft_stockham_fused_lds_amd, dim3(batch), dim3(256),
                     size_t(2 * N) * sizeof(float2), stream, d_in, d_out, N,
                     sign);
  return hipGetLastError() == hipSuccess ? 0 : 8;
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

// Persistent-plan variants split the input chirp from the immutable
// convolution kernel.  The kernel FFT is built once at plan creation; warm
// execution only prepares `a`.
extern "C" __global__ void
ts_bluestein_input_amd(const float2 *__restrict__ x, float2 *__restrict__ a,
                       int N, int M, int sign) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= M) return;
  if (j >= N) {
    a[j] = make_float2(0.f, 0.f);
    return;
  }
  long long jj = ((long long)j * (long long)j) % (2LL * (long long)N);
  double ang = -(double)sign * 3.14159265358979323846 * (double)jj / (double)N;
  float2 chirp = make_float2((float)cos(ang), (float)sin(ang));
  a[j] = cmul(x[j], make_float2(chirp.x, -chirp.y));
}

extern "C" __global__ void ts_bluestein_kernel_amd(float2 *__restrict__ b,
                                                    int N, int M, int sign) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= N) return;
  long long jj = ((long long)j * (long long)j) % (2LL * (long long)N);
  double ang = -(double)sign * 3.14159265358979323846 * (double)jj / (double)N;
  float2 chirp = make_float2((float)cos(ang), (float)sin(ang));
  b[j] = chirp;
  if (j) b[M - j] = chirp;
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

namespace {

struct TSFFTPlanAMD {
  int n = 0;
  int sign = -1;
  int bluesteinM = 0;
  tessera::spectral::FFTPlan arithmetic{};
  size_t batchCapacity = 0;
  size_t realSpectrumCapacity = 0;
  float2 *input = nullptr;
  float2 *output = nullptr;
  float2 *realSpectrum = nullptr;
  float2 *nScratch = nullptr;
  // Bluestein: four persistent M-sized buffers. `kernelFFT` is immutable after
  // creation; a/fa/mScratch are serialized mutable execution workspace.
  float2 *a = nullptr;
  float2 *fa = nullptr;
  float2 *kernelFFT = nullptr;
  float2 *mScratch = nullptr;
  // Empty only for the legacy diagnostic entry point.  Canonical runtime
  // plans carry the exact Schedule -> Tile content identity that selected
  // this native package.
  char artifactDigest[65] = {};
  std::mutex mutex;
};

void destroy_plan_storage(TSFFTPlanAMD *plan) {
  if (!plan) return;
  if (plan->input) (void)hipFree(plan->input);
  if (plan->output) (void)hipFree(plan->output);
  if (plan->realSpectrum) (void)hipFree(plan->realSpectrum);
  if (plan->nScratch) (void)hipFree(plan->nScratch);
  if (plan->a) (void)hipFree(plan->a);
  if (plan->fa) (void)hipFree(plan->fa);
  if (plan->kernelFFT) (void)hipFree(plan->kernelFFT);
  if (plan->mScratch) (void)hipFree(plan->mScratch);
}

int ensure_batch_capacity(TSFFTPlanAMD *plan, size_t batch) {
  if (batch <= plan->batchCapacity) return 0;
  float2 *newInput = nullptr, *newOutput = nullptr;
  const size_t bytes = batch * size_t(plan->n) * sizeof(float2);
  if (hipMalloc(&newInput, bytes) != hipSuccess ||
      hipMalloc(&newOutput, bytes) != hipSuccess) {
    if (newInput) (void)hipFree(newInput);
    if (newOutput) (void)hipFree(newOutput);
    return 1;
  }
  if (plan->input) (void)hipFree(plan->input);
  if (plan->output) (void)hipFree(plan->output);
  plan->input = newInput;
  plan->output = newOutput;
  plan->batchCapacity = batch;
  return 0;
}

int ensure_real_spectrum_capacity(TSFFTPlanAMD *plan, size_t batch) {
  if (batch <= plan->realSpectrumCapacity) return 0;
  float2 *replacement = nullptr;
  if (hipMalloc(&replacement,
                batch * size_t(plan->n + 1) * sizeof(float2)) != hipSuccess)
    return 1;
  if (plan->realSpectrum) (void)hipFree(plan->realSpectrum);
  plan->realSpectrum = replacement;
  plan->realSpectrumCapacity = batch;
  return 0;
}

void execute_cached_bluestein(TSFFTPlanAMD *plan, const float2 *input,
                              float2 *output, hipStream_t stream) {
  constexpr int threads = 256;
  const int mBlocks = (plan->bluesteinM + threads - 1) / threads;
  const int nBlocks = (plan->n + threads - 1) / threads;
  hipLaunchKernelGGL(ts_bluestein_input_amd, dim3(mBlocks), dim3(threads), 0,
                     stream, input, plan->a, plan->n, plan->bluesteinM,
                     plan->sign);
  ts_fft_stockham_amd(plan->a, plan->fa, plan->mScratch, plan->bluesteinM,
                      -1, stream);
  hipLaunchKernelGGL(ts_cmul_inplace_amd, dim3(mBlocks), dim3(threads), 0,
                     stream, plan->fa, plan->kernelFFT, plan->bluesteinM);
  hipLaunchKernelGGL(ts_conj_scale_amd, dim3(mBlocks), dim3(threads), 0, stream,
                     plan->fa, plan->bluesteinM, 1.0f);
  // Reuse `a` as the convolution output, preserving immutable kernelFFT.
  ts_fft_stockham_amd(plan->fa, plan->a, plan->mScratch, plan->bluesteinM,
                      -1, stream);
  hipLaunchKernelGGL(ts_conj_scale_amd, dim3(mBlocks), dim3(threads), 0, stream,
                     plan->a, plan->bluesteinM,
                     1.0f / float(plan->bluesteinM));
  hipLaunchKernelGGL(ts_bluestein_post_amd, dim3(nBlocks), dim3(threads), 0,
                     stream, plan->a, output, plan->n, plan->sign);
  if (plan->sign > 0)
    hipLaunchKernelGGL(ts_fft_scale_amd, dim3(nBlocks), dim3(threads), 0, stream,
                       output, plan->n, 1.0f / float(plan->n));
}

} // namespace

// Opaque persistent-plan ABI used by the prebuilt ROCm spectral image.
static int create_fft_plan_amd(int N, int sign, const char *artifactDigest,
                               void **handle) {
  if (!handle || N <= 0 || (sign != -1 && sign != 1)) return 10;
  *handle = nullptr;
  TSFFTPlanAMD *plan = new (std::nothrow) TSFFTPlanAMD();
  if (!plan) return 11;
  plan->n = N;
  plan->sign = sign;
  if (artifactDigest)
    std::memcpy(plan->artifactDigest, artifactDigest, 65);
  plan->arithmetic = tessera::spectral::fft_plan(N);
  if (plan->arithmetic.kind == tessera::spectral::kFFTBluestein) {
    plan->bluesteinM = plan->arithmetic.bluestein_m;
    const size_t bytes = size_t(plan->bluesteinM) * sizeof(float2);
    float2 *kernel = nullptr;
    if (hipMalloc(&plan->a, bytes) != hipSuccess ||
        hipMalloc(&plan->fa, bytes) != hipSuccess ||
        hipMalloc(&plan->kernelFFT, bytes) != hipSuccess ||
        hipMalloc(&plan->mScratch, bytes) != hipSuccess ||
        hipMalloc(&kernel, bytes) != hipSuccess) {
      if (kernel) (void)hipFree(kernel);
      destroy_plan_storage(plan);
      delete plan;
      return 13;
    }
    (void)hipMemset(kernel, 0, bytes);
    constexpr int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    hipLaunchKernelGGL(ts_bluestein_kernel_amd, dim3(blocks), dim3(threads), 0,
                       0, kernel, N, plan->bluesteinM, sign);
    ts_fft_stockham_amd(kernel, plan->kernelFFT, plan->mScratch,
                        plan->bluesteinM, -1, 0);
    hipError_t sync = hipDeviceSynchronize();
    (void)hipFree(kernel);
    if (sync != hipSuccess) {
      destroy_plan_storage(plan);
      delete plan;
      return 14;
    }
  } else if (hipMalloc(&plan->nScratch, size_t(N) * sizeof(float2)) !=
             hipSuccess) {
    destroy_plan_storage(plan);
    delete plan;
    return 12;
  }
  *handle = plan;
  return 0;
}

extern "C" int ts_fft_plan_create_amd(int N, int sign, void **handle) {
  return create_fft_plan_amd(N, sign, nullptr, handle);
}

extern "C" int ts_fft_plan_create_for_artifact_amd(
    int N, int sign, const char *artifactDigest, void **handle) {
  if (!artifactDigest || std::strlen(artifactDigest) != 64) return 15;
  for (int i = 0; i < 64; ++i) {
    const char c = artifactDigest[i];
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return 15;
  }
  return create_fft_plan_amd(N, sign, artifactDigest, handle);
}

extern "C" const char *ts_fft_plan_artifact_digest_amd(void *handle) {
  auto *plan = static_cast<TSFFTPlanAMD *>(handle);
  return plan ? plan->artifactDigest : nullptr;
}

extern "C" const char *ts_fft_package_abi_amd() {
  return "tessera.rocm.fft.plan.v1";
}

static int execute_device_batch_locked(TSFFTPlanAMD *plan,
                                       const float2 *inputDevice,
                                       float2 *outputDevice, int batch,
                                       hipStream_t stream) {
  ts_fft_clear_error_amd();
  // gfx1151 power-of-two transforms up to 1024 fit two complex slabs in LDS.
  // One workgroup owns one row, so a composite STFT launches the entire
  // [batch*frames, N] slab once instead of issuing stages for every row.
  if (plan->arithmetic.kind != tessera::spectral::kFFTBluestein &&
      plan->n >= 2 && plan->n <= 1024 && (plan->n & (plan->n - 1)) == 0)
    return launch_fused_lds_batch(inputDevice, outputDevice, batch, plan->n,
                                  plan->sign, stream);
  for (int row = 0; row < batch; ++row) {
    const size_t offset = size_t(row) * size_t(plan->n);
    if (plan->arithmetic.kind == tessera::spectral::kFFTBluestein)
      execute_cached_bluestein(plan, inputDevice + offset,
                               outputDevice + offset, stream);
    else
      ts_fft_stockham_amd(inputDevice + offset, outputDevice + offset,
                          plan->nScratch, plan->n, plan->sign, stream);
    if (ts_fft_last_error_amd() != 0) return 23;
  }
  return 0;
}

// Device-resident entry used by compound TSOL packages. The opaque plan owns
// its mutable FFT workspace and serializes callers; input/output storage stays
// owned by the compound package, avoiding intermediate host round trips.
extern "C" int ts_fft_plan_execute_device_batch_amd(
    void *handle, const float2 *inputDevice, float2 *outputDevice, int batch,
    hipStream_t stream) {
  auto *plan = static_cast<TSFFTPlanAMD *>(handle);
  if (!plan || !inputDevice || !outputDevice || batch <= 0) return 20;
  std::lock_guard<std::mutex> guard(plan->mutex);
  return execute_device_batch_locked(plan, inputDevice, outputDevice, batch,
                                     stream);
}

// Device-pointer packed-real ABI for compound packages. The caller owns the
// real and Hermitian slabs; the child plan owns only the N/2 complex scratch.
// This keeps RFFT -> consumer -> IRFFT intermediates device resident.
extern "C" int ts_fft_plan_execute_r2c_packed_device_batch_amd(
    void *handle, const float *inputDevice, float2 *outputDevice, int batch,
    int logicalN, hipStream_t stream) {
  auto *plan = static_cast<TSFFTPlanAMD *>(handle);
  if (!plan || !inputDevice || !outputDevice || batch <= 0 || logicalN < 2 ||
      (logicalN & 1) || plan->n != logicalN / 2 || plan->sign != -1)
    return 30;
  const int m = logicalN / 2;
  std::lock_guard<std::mutex> guard(plan->mutex);
  if (ensure_batch_capacity(plan, static_cast<size_t>(batch)) != 0) return 21;
  int rc = execute_device_batch_locked(
      plan, reinterpret_cast<const float2 *>(inputDevice), plan->output, batch,
      stream);
  if (rc != 0) return rc;
  constexpr int threads = 256;
  int total = batch * (m + 1);
  hipLaunchKernelGGL(ts_rfft_post_amd,
                     dim3((total + threads - 1) / threads), dim3(threads), 0,
                     stream, plan->output, outputDevice, m, logicalN, batch);
  return hipGetLastError() == hipSuccess ? 0 : 31;
}

extern "C" int ts_fft_plan_execute_c2r_packed_device_batch_amd(
    void *handle, const float2 *inputDevice, float *outputDevice, int batch,
    int logicalN, hipStream_t stream) {
  auto *plan = static_cast<TSFFTPlanAMD *>(handle);
  if (!plan || !inputDevice || !outputDevice || batch <= 0 || logicalN < 2 ||
      (logicalN & 1) || plan->n != logicalN / 2 || plan->sign != 1)
    return 30;
  const int m = logicalN / 2;
  std::lock_guard<std::mutex> guard(plan->mutex);
  if (ensure_batch_capacity(plan, static_cast<size_t>(batch)) != 0) return 21;
  constexpr int threads = 256;
  int total = batch * m;
  hipLaunchKernelGGL(ts_irfft_pre_amd,
                     dim3((total + threads - 1) / threads), dim3(threads), 0,
                     stream, inputDevice, plan->input, m, logicalN, batch);
  int rc = execute_device_batch_locked(plan, plan->input, plan->output, batch,
                                       stream);
  if (rc != 0) return rc;
  return hipMemcpyAsync(outputDevice, plan->output,
                        size_t(batch) * logicalN * sizeof(float),
                        hipMemcpyDeviceToDevice, stream) == hipSuccess
             ? 0
             : 32;
}

extern "C" int ts_fft_plan_execute_hostptr_batch_amd(
    void *handle, const float2 *inputHost, float2 *outputHost, int batch) {
  auto *plan = static_cast<TSFFTPlanAMD *>(handle);
  if (!plan || !inputHost || !outputHost || batch <= 0) return 20;
  std::lock_guard<std::mutex> guard(plan->mutex);
  if (ensure_batch_capacity(plan, static_cast<size_t>(batch)) != 0) return 21;
  const size_t elements = size_t(batch) * size_t(plan->n);
  const size_t bytes = elements * sizeof(float2);
  if (hipMemcpy(plan->input, inputHost, bytes, hipMemcpyHostToDevice) !=
      hipSuccess)
    return 22;
  int execute = execute_device_batch_locked(plan, plan->input, plan->output,
                                             batch, 0);
  if (execute != 0) return execute;
  if (hipDeviceSynchronize() != hipSuccess) return 24;
  if (hipMemcpy(outputHost, plan->output, bytes, hipMemcpyDeviceToHost) !=
      hipSuccess)
    return 25;
  return 0;
}

// Native package-owned even-length real transforms.  The opaque plan must be
// an N/2 complex plan; host pointers are staged once by the package and the
// GPU therefore executes half the complex arithmetic/storage of the legacy
// N-point wrapper.  Device-resident pre/post kernels are the fusion follow-up.
extern "C" int ts_fft_plan_execute_r2c_packed_hostptr_amd(
    void *handle, const float *inputHost, float2 *outputHost, int batch,
    int logicalN) {
  auto *plan = static_cast<TSFFTPlanAMD *>(handle);
  if (!plan || !inputHost || !outputHost || batch <= 0 || logicalN < 2 ||
      (logicalN & 1) || plan->n != logicalN / 2 || plan->sign != -1)
    return 30;
  const int m = logicalN / 2;
  std::lock_guard<std::mutex> guard(plan->mutex);
  if (ensure_batch_capacity(plan, static_cast<size_t>(batch)) != 0 ||
      ensure_real_spectrum_capacity(plan, static_cast<size_t>(batch)) != 0)
    return 21;
  if (hipMemcpy(plan->input, inputHost,
                size_t(batch) * logicalN * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess)
    return 22;
  int rc = execute_device_batch_locked(plan, plan->input, plan->output, batch, 0);
  if (rc != 0) return rc;
  constexpr int threads = 256;
  int total = batch * (m + 1);
  hipLaunchKernelGGL(ts_rfft_post_amd, dim3((total + threads - 1) / threads),
                     dim3(threads), 0, 0, plan->output, plan->realSpectrum, m,
                     logicalN, batch);
  if (hipDeviceSynchronize() != hipSuccess) return 24;
  if (hipMemcpy(outputHost, plan->realSpectrum,
                size_t(total) * sizeof(float2), hipMemcpyDeviceToHost) !=
      hipSuccess)
    return 25;
  return 0;
}

extern "C" int ts_fft_plan_execute_c2r_packed_hostptr_amd(
    void *handle, const float2 *inputHost, float *outputHost, int batch,
    int logicalN) {
  auto *plan = static_cast<TSFFTPlanAMD *>(handle);
  if (!plan || !inputHost || !outputHost || batch <= 0 || logicalN < 2 ||
      (logicalN & 1) || plan->n != logicalN / 2 || plan->sign != 1)
    return 30;
  const int m = logicalN / 2;
  std::lock_guard<std::mutex> guard(plan->mutex);
  if (ensure_batch_capacity(plan, static_cast<size_t>(batch)) != 0 ||
      ensure_real_spectrum_capacity(plan, static_cast<size_t>(batch)) != 0)
    return 21;
  int totalSpectrum = batch * (m + 1);
  if (hipMemcpy(plan->realSpectrum, inputHost,
                size_t(totalSpectrum) * sizeof(float2),
                hipMemcpyHostToDevice) != hipSuccess)
    return 22;
  constexpr int threads = 256;
  int totalPacked = batch * m;
  hipLaunchKernelGGL(ts_irfft_pre_amd,
                     dim3((totalPacked + threads - 1) / threads),
                     dim3(threads), 0, 0, plan->realSpectrum, plan->input, m,
                     logicalN, batch);
  int rc = execute_device_batch_locked(plan, plan->input, plan->output, batch, 0);
  if (rc != 0) return rc;
  if (hipDeviceSynchronize() != hipSuccess) return 24;
  if (hipMemcpy(outputHost, plan->output,
                size_t(batch) * logicalN * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess)
    return 25;
  return 0;
}

extern "C" long long ts_fft_plan_workspace_elems_amd(void *handle) {
  auto *plan = static_cast<TSFFTPlanAMD *>(handle);
  if (!plan) return -1;
  return plan->arithmetic.kind == tessera::spectral::kFFTBluestein
             ? 4LL * plan->bluesteinM
             : plan->n;
}

extern "C" int ts_fft_plan_length_amd(void *handle) {
  auto *plan = static_cast<TSFFTPlanAMD *>(handle);
  return plan ? plan->n : -1;
}

extern "C" void ts_fft_plan_destroy_amd(void *handle) {
  auto *plan = static_cast<TSFFTPlanAMD *>(handle);
  if (!plan) return;
  std::unique_lock<std::mutex> guard(plan->mutex);
  (void)hipDeviceSynchronize();
  destroy_plan_storage(plan);
  guard.unlock();
  delete plan;
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

// Batched host-pointer entry used by the canonical ROCm FFT runtime package.
// Allocate and transfer the full [batch, N] slab once, then apply the exact
// same shipping driver to each row.  This removes the per-row hipMalloc/free
// overhead without inventing a second butterfly implementation.  Rows remain
// independent and execute in a deterministic order on the default stream.
extern "C" int ts_fft_stockham_amd_hostptr_batch(const float2 *in_host,
                                                 float2 *out_host, int batch,
                                                 int N, int sign) {
  if (batch <= 0 || N <= 0)
    return 6;
  float2 *d_in = nullptr, *d_out = nullptr, *d_scr = nullptr;
  int rc = 0;
  const size_t elems = size_t(batch) * size_t(N);
  const size_t bytes = elems * sizeof(float2);
  ts_fft_clear_error_amd();
  if (hipMalloc(&d_in, bytes) != hipSuccess ||
      hipMalloc(&d_out, bytes) != hipSuccess ||
      hipMalloc(&d_scr, size_t(N) * sizeof(float2)) != hipSuccess) {
    rc = 1;
    goto cleanup;
  }
  if (hipMemcpy(d_in, in_host, bytes, hipMemcpyHostToDevice) != hipSuccess) {
    rc = 2;
    goto cleanup;
  }
  for (int row = 0; row < batch; ++row) {
    const size_t offset = size_t(row) * size_t(N);
    ts_fft_stockham_amd(d_in + offset, d_out + offset, d_scr, N, sign,
                        /*stream=*/0);
    if (ts_fft_last_error_amd() != 0) {
      rc = ts_fft_last_error_amd();
      goto cleanup;
    }
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
