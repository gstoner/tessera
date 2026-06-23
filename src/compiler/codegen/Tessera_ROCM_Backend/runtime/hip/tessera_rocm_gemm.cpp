// tessera_rocm_gemm.cpp — shipped ROCm WMMA GEMM runtime symbols.
//
// Exposes stable C-ABI entry points, ``tessera_rocm_wmma_gemm_f16`` and
// ``tessera_rocm_wmma_gemm_bf16``, that run a real RDNA 3.5 WMMA matrix-core
// GEMM on the AMD GPU. These are the *shipped* runtime symbols the
// backend_manifest ``hardware_verified`` contract requires (the numerical proof
// lives in a checked-in execute_compare_fixture that calls them). They are the
// production counterpart of the Stage C/D test-harness launcher, and the lane
// that ``runtime.launch()`` dispatches ``target="rocm"`` matmul artifacts to.
//
// The device kernel is compiled at load time with **HIPRTC** for whatever arch
// the device enumerates (gfx1151 on the Strix Halo box; early-bring-up WSL
// transiently reported gfx1100 — same WMMA family either way) — so this object
// is built by the ordinary host C++ compiler and only needs the HIP runtime +
// HIPRTC at link time, no hipcc-as-compiler.
//
// Kernel: D[MxN] = A[MxK] @ B[KxN], row-major, f16/bf16 in / f32 accumulate.
// General tiled/K-looped GEMM. Each 32-lane wave computes an MTxNT grid of
// 16x16 WMMA output tiles (output-tile *register blocking*): a loaded A
// fragment is reused across NT B-tiles and a loaded B fragment across MT
// A-tiles, so global-load traffic per output element drops by ~the reuse
// factor. This is the GEMM perf-ladder lever the AMD Gluon tutorial calls out
// (register-budget-driven output-tile sizing); see ROCM_PATTERNS_FROM_AMD_
// ECOSYSTEM.md §B1 and STRIX_HALO_EXECUTION_PLAN.md. Ragged M/N/K are
// zero-padded on load and bounds-checked on store. Fragment/accumulator layout
// per the RDNA 3.5 ISA (matches python/tessera/compiler/rocdl_emit.py): A frag
// row = lane&15, B frag col = lane&15, output row = 2*e + lane>>4, col = lane&15.
//
// Rung 0 of the ladder is MT=NT=1 (one tile per wave, no reuse); the shipped
// symbols use the measured-best production tiling below. The bench entry point
// (tessera_rocm_wmma_gemm_f16_bench) times any (MT,NT) on-device so rungs are
// proven, not asserted.

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace {

// Production output-tile blocking factor for the shipped symbols. Measured-best
// on gfx1151 (Strix Halo, RDNA 3.5): 2x4 wins at the compute-bound sizes — ~2.3x
// over the 1x1 naive baseline at 1024^3/2048^3 — and notably 2x2 *regressed*
// below naive (the AMD Gluon "the obvious tiling can lose; shape is the lever"
// lesson). See the perf ladder in STRIX_HALO_EXECUTION_PLAN.md.
constexpr int kProdMT = 2;
constexpr int kProdNT = 4;

// One HIPRTC source template per (storage dtype, MTxNT). ``%TYPE%`` / ``%WMMA%``
// / ``%NAME%`` / ``%MT%`` / ``%NT%`` are substituted at load. MT/NT are literal
// ints so the tile loops fully unroll and the accumulators live in registers.
const char* kKernelTemplate = R"HIPSRC(
typedef %TYPE% wtype16 __attribute__((ext_vector_type(16)));
typedef float  float8  __attribute__((ext_vector_type(8)));
extern "C" __global__ void %NAME%(
    const %TYPE%* A, const %TYPE%* B, float* D, int M, int N, int K) {
  const int MT = %MT%, NT = %NT%;
  int l = threadIdx.x, lane = l & 15;
  int baseRow = blockIdx.y * 16 * MT;   // this wave's macro-tile origin
  int baseCol = blockIdx.x * 16 * NT;
  float8 c[MT][NT];
  for (int mi = 0; mi < MT; ++mi)
    for (int ni = 0; ni < NT; ++ni) c[mi][ni] = (float8){0,0,0,0,0,0,0,0};
  for (int k0 = 0; k0 < K; k0 += 16) {
    wtype16 a[MT], b[NT];
    for (int mi = 0; mi < MT; ++mi)
      for (int i = 0; i < 16; ++i) {
        int ar = baseRow + mi * 16 + lane, ak = k0 + i;
        a[mi][i] = (ar < M && ak < K) ? A[ar * K + ak] : (%TYPE%)0;
      }
    for (int ni = 0; ni < NT; ++ni)
      for (int i = 0; i < 16; ++i) {
        int bk = k0 + i, bc = baseCol + ni * 16 + lane;
        b[ni][i] = (bk < K && bc < N) ? B[bk * N + bc] : (%TYPE%)0;
      }
    for (int mi = 0; mi < MT; ++mi)
      for (int ni = 0; ni < NT; ++ni)
        c[mi][ni] = %WMMA%(a[mi], b[ni], c[mi][ni]);
  }
  for (int mi = 0; mi < MT; ++mi)
    for (int ni = 0; ni < NT; ++ni)
      for (int e = 0; e < 8; ++e) {
        int r = baseRow + mi * 16 + e * 2 + (l >> 4);
        int col = baseCol + ni * 16 + lane;
        if (r < M && col < N) D[r * N + col] = c[mi][ni][e];
      }
}
)HIPSRC";

// Rung 2 — LDS-staged, multi-wave workgroup. A workgroup of WM×WN waves
// cooperatively stages the A and B K-panels (16-wide) for its
// (WM*MT*16)×(WN*NT*16) macro-tile into LDS once per K-step, then every wave
// reads its WMMA fragments from LDS (not global) and does MT×NT register-blocked
// WMMA. Global-load traffic per output element drops by the number of waves
// sharing each staged panel. ROCM_PATTERNS §B2 ("pipelining + LDS layout"); the
// staging here is the prerequisite single-buffer step before software
// pipelining. WM/WN/MT/NT are literal ints → LDS sizes are constexpr.
const char* kKernelTemplateLDS = R"HIPSRC(
typedef %TYPE% wtype16 __attribute__((ext_vector_type(16)));
typedef float  float8  __attribute__((ext_vector_type(8)));
extern "C" __global__ void %NAME%(
    const %TYPE%* A, const %TYPE%* B, float* D, int M, int N, int K) {
  constexpr int WM = %WM%, WN = %WN%, MT = %MT%, NT = %NT%;
  constexpr int WG_M = WM * MT * 16;     // workgroup macro-tile rows
  constexpr int WG_N = WN * NT * 16;     // workgroup macro-tile cols
  __shared__ %TYPE% smemA[WG_M * 16];    // [WG_M][16]  (row, k)
  __shared__ %TYPE% smemB[16 * WG_N];    // [16][WG_N]  (k, col)
  int tid = threadIdx.x;
  int nthreads = WM * WN * 32;
  int waveId = tid >> 5, lane = tid & 31, l15 = lane & 15;
  int rowOff = (waveId / WN) * MT * 16;  // this wave's tile origin in the macro
  int colOff = (waveId % WN) * NT * 16;
  int blockRow = blockIdx.y * WG_M, blockCol = blockIdx.x * WG_N;
  float8 c[MT][NT];
  for (int mi = 0; mi < MT; ++mi)
    for (int ni = 0; ni < NT; ++ni) c[mi][ni] = (float8){0,0,0,0,0,0,0,0};
  for (int k0 = 0; k0 < K; k0 += 16) {
    for (int e = tid; e < WG_M * 16; e += nthreads) {
      int row = e >> 4, kk = e & 15, gr = blockRow + row, gk = k0 + kk;
      smemA[e] = (gr < M && gk < K) ? A[gr * K + gk] : (%TYPE%)0;
    }
    for (int e = tid; e < 16 * WG_N; e += nthreads) {
      int kk = e / WG_N, col = e % WG_N, gk = k0 + kk, gc = blockCol + col;
      smemB[e] = (gk < K && gc < N) ? B[gk * N + gc] : (%TYPE%)0;
    }
    __syncthreads();
    wtype16 bf[NT];
    for (int ni = 0; ni < NT; ++ni) {
      int bcol = colOff + ni * 16 + l15;
      for (int i = 0; i < 16; ++i) bf[ni][i] = smemB[i * WG_N + bcol];
    }
    for (int mi = 0; mi < MT; ++mi) {
      wtype16 a;
      int arow = rowOff + mi * 16 + l15;
      for (int i = 0; i < 16; ++i) a[i] = smemA[arow * 16 + i];
      for (int ni = 0; ni < NT; ++ni)
        c[mi][ni] = %WMMA%(a, bf[ni], c[mi][ni]);
    }
    __syncthreads();
  }
  for (int mi = 0; mi < MT; ++mi)
    for (int ni = 0; ni < NT; ++ni)
      for (int e = 0; e < 8; ++e) {
        int r = blockRow + rowOff + mi * 16 + e * 2 + (lane >> 4);
        int col = blockCol + colOff + ni * 16 + l15;
        if (r < M && col < N) D[r * N + col] = c[mi][ni][e];
      }
}
)HIPSRC";

std::string substitute(const std::string& tmpl, const std::string& from,
                       const std::string& to) {
  std::string out = tmpl;
  for (size_t p = out.find(from); p != std::string::npos;
       p = out.find(from, p + to.size()))
    out.replace(p, from.size(), to);
  return out;
}

// HIPRTC-compile a fully-substituted kernel source for device 0's arch and hand
// back the module + named function (caller owns the module). Shared by the
// register-blocked (rung 1) and LDS-staged (rung 2) variants.
bool compileSrc(const std::string& src, const std::string& name,
                hipModule_t* outMod, hipFunction_t* outFn) {
  hipDeviceProp_t props;
  if (hipGetDeviceProperties(&props, 0) != hipSuccess) return false;

  hiprtcProgram prog;
  if (hiprtcCreateProgram(&prog, src.c_str(), "tessera_rocm_wmma_gemm.hip",
                          0, nullptr, nullptr) != HIPRTC_SUCCESS)
    return false;
  std::string archOpt = std::string("--offload-arch=") + props.gcnArchName;
  const char* opts[] = {archOpt.c_str()};
  hiprtcResult cr = hiprtcCompileProgram(prog, 1, opts);
  if (cr != HIPRTC_SUCCESS) { hiprtcDestroyProgram(&prog); return false; }

  size_t codeSize = 0;
  if (hiprtcGetCodeSize(prog, &codeSize) != HIPRTC_SUCCESS || codeSize == 0) {
    hiprtcDestroyProgram(&prog); return false;
  }
  std::vector<char> code(codeSize);
  hiprtcGetCode(prog, code.data());
  hiprtcDestroyProgram(&prog);

  if (hipModuleLoadData(outMod, code.data()) != hipSuccess) return false;
  if (hipModuleGetFunction(outFn, *outMod, name.c_str()) != hipSuccess)
    return false;
  return true;
}

// Compile a rung-1 register-blocked variant for (type, wmma builtin, MT, NT).
bool compileVariant(const char* type, const char* wmma, int mt, int nt,
                    const std::string& name, hipModule_t* outMod,
                    hipFunction_t* outFn) {
  std::string src = substitute(kKernelTemplate, "%TYPE%", type);
  src = substitute(src, "%WMMA%", wmma);
  src = substitute(src, "%NAME%", name);
  src = substitute(src, "%MT%", std::to_string(mt));
  src = substitute(src, "%NT%", std::to_string(nt));
  return compileSrc(src, name, outMod, outFn);
}

// Compile a rung-2 LDS-staged variant for (type, wmma, WM, WN waves, MT, NT
// register tiles/wave).
bool compileVariantLDS(const char* type, const char* wmma, int wm, int wn,
                       int mt, int nt, const std::string& name,
                       hipModule_t* outMod, hipFunction_t* outFn) {
  std::string src = substitute(kKernelTemplateLDS, "%TYPE%", type);
  src = substitute(src, "%WMMA%", wmma);
  src = substitute(src, "%NAME%", name);
  src = substitute(src, "%WM%", std::to_string(wm));
  src = substitute(src, "%WN%", std::to_string(wn));
  src = substitute(src, "%MT%", std::to_string(mt));
  src = substitute(src, "%NT%", std::to_string(nt));
  return compileSrc(src, name, outMod, outFn);
}

struct Kernel {
  std::once_flag once;
  hipModule_t module = nullptr;
  hipFunction_t fn = nullptr;
  bool ready = false;
  const char* name;
  const char* type;   // device element type
  const char* wmma;   // WMMA builtin
};

Kernel g_f16{{}, nullptr, nullptr, false,
             "tessera_rocm_wmma_gemm_f16_kernel", "__fp16",
             "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32"};
Kernel g_bf16{{}, nullptr, nullptr, false,
              "tessera_rocm_wmma_gemm_bf16_kernel", "__bf16",
              "__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32"};

void compileProd(Kernel* k) {
  k->ready = compileVariant(k->type, k->wmma, kProdMT, kProdNT, k->name,
                            &k->module, &k->fn);
}

// Launch grid for an (MT,NT)-blocked kernel over an MxN output.
void gridFor(int M, int N, int mt, int nt, unsigned* gx, unsigned* gy) {
  *gx = (unsigned)((N + 16 * nt - 1) / (16 * nt));
  *gy = (unsigned)((M + 16 * mt - 1) / (16 * mt));
}

// Launch `fn` once over an MxN problem (grid gx*gy, `threads`/block): H2D, one
// launch, sync, D2H. elemBytes = storage element size. 0/3 = ok/device-failed.
int runDeviceGemm(hipFunction_t fn, unsigned gx, unsigned gy, int threads,
                  const void* A, const void* B, void* D,
                  int M, int N, int K, size_t elemBytes) {
  const size_t aBytes = (size_t)M * K * elemBytes;
  const size_t bBytes = (size_t)K * N * elemBytes;
  const size_t dBytes = (size_t)M * N * sizeof(float);
  void *dA = nullptr, *dB = nullptr, *dD = nullptr;
  int rc = 3;
  do {
    if (hipMalloc(&dA, aBytes) != hipSuccess) break;
    if (hipMalloc(&dB, bBytes) != hipSuccess) break;
    if (hipMalloc(&dD, dBytes) != hipSuccess) break;
    if (hipMemcpy(dA, A, aBytes, hipMemcpyHostToDevice) != hipSuccess) break;
    if (hipMemcpy(dB, B, bBytes, hipMemcpyHostToDevice) != hipSuccess) break;
    void* args[] = {&dA, &dB, &dD, &M, &N, &K};
    if (hipModuleLaunchKernel(fn, gx, gy, 1, threads, 1, 1, 0, nullptr,
                              args, nullptr) != hipSuccess) break;
    if (hipDeviceSynchronize() != hipSuccess) break;
    if (hipMemcpy(D, dD, dBytes, hipMemcpyDeviceToHost) != hipSuccess) break;
    rc = 0;
  } while (0);
  if (dA) hipFree(dA);
  if (dB) hipFree(dB);
  if (dD) hipFree(dD);
  return rc;
}

// Zero-copy launch for the Strix Halo APU: host and device share the same
// physical LPDDR5x, so the explicit H2D/D2H copies in runDeviceGemm are
// redundant. Instead, page-lock + device-map the caller's host buffers
// (hipHostRegister + hipHostGetDevicePointer) and launch the kernel directly
// against them — no hipMalloc, no hipMemcpy. After hipDeviceSynchronize the
// caller's D buffer holds the result. Returns 0 / 3 (device op failed) /
// 4 (host-register unsupported here — caller should fall back to the copy path;
// e.g. some WSL/driver configs, unaligned/overlapping ranges).
int runDeviceGemmZeroCopy(hipFunction_t fn, unsigned gx, unsigned gy, int threads,
                          const void* A, const void* B, void* D,
                          int M, int N, int K, size_t elemBytes) {
  const size_t aBytes = (size_t)M * K * elemBytes;
  const size_t bBytes = (size_t)K * N * elemBytes;
  const size_t dBytes = (size_t)M * N * sizeof(float);
  void *hA = const_cast<void*>(A), *hB = const_cast<void*>(B);
  bool rA = false, rB = false, rD = false;
  void *dA = nullptr, *dB = nullptr, *dD = nullptr;
  int rc = 4;  // default: treat any registration failure as "unsupported"
  do {
    if (hipHostRegister(hA, aBytes, hipHostRegisterMapped) != hipSuccess) break;
    rA = true;
    if (hipHostRegister(hB, bBytes, hipHostRegisterMapped) != hipSuccess) break;
    rB = true;
    if (hipHostRegister(D, dBytes, hipHostRegisterMapped) != hipSuccess) break;
    rD = true;
    if (hipHostGetDevicePointer(&dA, hA, 0) != hipSuccess) break;
    if (hipHostGetDevicePointer(&dB, hB, 0) != hipSuccess) break;
    if (hipHostGetDevicePointer(&dD, D, 0) != hipSuccess) break;
    rc = 3;  // past setup: a failure now is a real device error, not "retry"
    void* args[] = {&dA, &dB, &dD, &M, &N, &K};
    if (hipModuleLaunchKernel(fn, gx, gy, 1, threads, 1, 1, 0, nullptr,
                              args, nullptr) != hipSuccess) break;
    if (hipDeviceSynchronize() != hipSuccess) break;
    rc = 0;
  } while (0);
  if (rA) hipHostUnregister(hA);
  if (rB) hipHostUnregister(hB);
  if (rD) hipHostUnregister(D);
  return rc;
}

// Whether the shipped symbols should use the zero-copy path. Opt-in via
// TESSERA_ROCM_ZEROCOPY={1,true,on,yes} (cached once). Off by default — the
// copy path is the portable correctness baseline; zero-copy is the APU tuning.
bool zeroCopyEnabled() {
  static int cached = -1;
  if (cached < 0) {
    const char* v = std::getenv("TESSERA_ROCM_ZEROCOPY");
    cached = (v && (*v == '1' || *v == 't' || *v == 'T' || *v == 'y'
                    || *v == 'Y' || *v == 'o' || *v == 'O')) ? 1 : 0;
  }
  return cached == 1;
}

// Run a general tiled GEMM through the cached production (rung-1) kernel.
// Returns 0 / 1 (bad shape) / 2 (no device / compile) / 3 (device op failed).
int runGemm(Kernel* k, const void* A, const void* B, void* D,
            int M, int N, int K, size_t elemBytes) {
  if (M <= 0 || N <= 0 || K <= 0) return 1;
  std::call_once(k->once, compileProd, k);
  if (!k->ready) return 2;
  unsigned gx, gy;
  gridFor(M, N, kProdMT, kProdNT, &gx, &gy);
  if (zeroCopyEnabled()) {
    int rc = runDeviceGemmZeroCopy(k->fn, gx, gy, 32, A, B, D, M, N, K, elemBytes);
    if (rc != 4) return rc;        // 0 = ok, 3 = real device error — honor it
    // rc == 4: host-register unsupported here → fall back to the copy path.
  }
  return runDeviceGemm(k->fn, gx, gy, 32, A, B, D, M, N, K, elemBytes);
}

// Device-time `iters` kernel-only launches of `fn` (buffers allocated + zeroed
// once, warmup, hipEvent timing — no H2D/D2H in the timed loop) so the measure
// is GEMM compute, not transfer. avg_ms <- mean per-launch ms. The rung-prover.
int timedKernelLaunches(hipFunction_t fn, unsigned gx, unsigned gy, int threads,
                        int M, int N, int K, int iters, double* avg_ms) {
  const size_t elemBytes = sizeof(unsigned short);
  void *dA = nullptr, *dB = nullptr, *dD = nullptr;
  hipEvent_t start = nullptr, stop = nullptr;
  int rc = 3;
  do {
    if (hipMalloc(&dA, (size_t)M * K * elemBytes) != hipSuccess) break;
    if (hipMalloc(&dB, (size_t)K * N * elemBytes) != hipSuccess) break;
    if (hipMalloc(&dD, (size_t)M * N * sizeof(float)) != hipSuccess) break;
    hipMemset(dA, 0, (size_t)M * K * elemBytes);
    hipMemset(dB, 0, (size_t)K * N * elemBytes);
    void* args[] = {&dA, &dB, &dD, &M, &N, &K};
    for (int w = 0; w < 5; ++w)
      if (hipModuleLaunchKernel(fn, gx, gy, 1, threads, 1, 1, 0, nullptr,
                                args, nullptr) != hipSuccess) { rc = 3; break; }
    if (hipDeviceSynchronize() != hipSuccess) break;
    if (hipEventCreate(&start) != hipSuccess) break;
    if (hipEventCreate(&stop) != hipSuccess) break;
    if (hipEventRecord(start, nullptr) != hipSuccess) break;
    for (int it = 0; it < iters; ++it)
      if (hipModuleLaunchKernel(fn, gx, gy, 1, threads, 1, 1, 0, nullptr,
                                args, nullptr) != hipSuccess) { rc = 3; break; }
    if (hipEventRecord(stop, nullptr) != hipSuccess) break;
    if (hipEventSynchronize(stop) != hipSuccess) break;
    float ms = 0.0f;
    if (hipEventElapsedTime(&ms, start, stop) != hipSuccess) break;
    *avg_ms = (double)ms / (double)iters;
    rc = 0;
  } while (0);
  if (start) hipEventDestroy(start);
  if (stop) hipEventDestroy(stop);
  if (dA) hipFree(dA);
  if (dB) hipFree(dB);
  if (dD) hipFree(dD);
  return rc;
}

// rung-1 (register-blocked) benchmark. 0/1/2/3 as above.
int benchVariant(const char* type, const char* wmma, int M, int N, int K,
                 int iters, int mt, int nt, double* avg_ms) {
  if (M <= 0 || N <= 0 || K <= 0 || iters <= 0 || mt <= 0 || nt <= 0)
    return 1;
  hipModule_t mod = nullptr;
  hipFunction_t fn = nullptr;
  std::string name = std::string("bench") + type + "_" + std::to_string(mt)
                     + "x" + std::to_string(nt);
  if (!compileVariant(type, wmma, mt, nt, name, &mod, &fn)) return 2;
  unsigned gx, gy;
  gridFor(M, N, mt, nt, &gx, &gy);
  int rc = timedKernelLaunches(fn, gx, gy, 32, M, N, K, iters, avg_ms);
  if (mod) hipModuleUnload(mod);
  return rc;
}

// rung-2 (LDS-staged, WM×WN waves) benchmark. 0/1/2/3 as above.
int benchVariantLDS(const char* type, const char* wmma, int M, int N, int K,
                    int iters, int wm, int wn, int mt, int nt, double* avg_ms) {
  if (M <= 0 || N <= 0 || K <= 0 || iters <= 0 || wm <= 0 || wn <= 0
      || mt <= 0 || nt <= 0)
    return 1;
  hipModule_t mod = nullptr;
  hipFunction_t fn = nullptr;
  std::string name = std::string("benchlds") + type + "_" + std::to_string(wm)
                     + "_" + std::to_string(wn) + "_" + std::to_string(mt)
                     + "x" + std::to_string(nt);
  if (!compileVariantLDS(type, wmma, wm, wn, mt, nt, name, &mod, &fn)) return 2;
  int wgM = wm * mt * 16, wgN = wn * nt * 16;
  unsigned gx = (unsigned)((N + wgN - 1) / wgN);
  unsigned gy = (unsigned)((M + wgM - 1) / wgM);
  int rc = timedKernelLaunches(fn, gx, gy, wm * wn * 32, M, N, K, iters, avg_ms);
  if (mod) hipModuleUnload(mod);
  return rc;
}

// Run an LDS-staged GEMM end-to-end (correctness path for the rung-2 kernel).
int runGemmLDS(const char* type, const char* wmma, const void* A,
               const void* B, void* D, int M, int N, int K, int wm, int wn,
               int mt, int nt, size_t elemBytes) {
  if (M <= 0 || N <= 0 || K <= 0) return 1;
  hipModule_t mod = nullptr;
  hipFunction_t fn = nullptr;
  std::string name = std::string("runlds") + type;
  if (!compileVariantLDS(type, wmma, wm, wn, mt, nt, name, &mod, &fn)) return 2;
  int wgM = wm * mt * 16, wgN = wn * nt * 16;
  unsigned gx = (unsigned)((N + wgN - 1) / wgN);
  unsigned gy = (unsigned)((M + wgM - 1) / wgM);
  int rc = runDeviceGemm(fn, gx, gy, wm * wn * 32, A, B, D, M, N, K, elemBytes);
  if (mod) hipModuleUnload(mod);
  return rc;
}

// END-TO-END CPU-timed benchmark: times the full launch()-equivalent path per
// call — memory setup + transfer + launch + sync (+ teardown) — over real host
// buffers, for ``zerocopy`` 0 (hipMalloc + H2D/D2H copy) vs 1 (hipHostRegister
// device-mapped, no copy). This is the path real callers pay (unlike the
// kernel-only *_bench), so it's where the APU zero-copy win shows up. CPU wall
// clock (std::chrono) because hipHostRegister/hipMalloc are host-side, off the
// timed stream. avg_ms <- mean per-call ms. 0/1/2/3 as elsewhere.
int benchE2E(const char* type, const char* wmma, int M, int N, int K, int iters,
             int mt, int nt, int zerocopy, double* avg_ms) {
  if (M <= 0 || N <= 0 || K <= 0 || iters <= 0 || mt <= 0 || nt <= 0)
    return 1;
  hipModule_t mod = nullptr;
  hipFunction_t fn = nullptr;
  std::string name = std::string("benche2e") + type + "_" + std::to_string(mt)
                     + "x" + std::to_string(nt);
  if (!compileVariant(type, wmma, mt, nt, name, &mod, &fn)) return 2;

  const size_t elemBytes = sizeof(unsigned short);
  std::vector<unsigned short> A((size_t)M * K, 0), B((size_t)K * N, 0);
  std::vector<float> D((size_t)M * N, 0.0f);
  unsigned gx, gy;
  gridFor(M, N, mt, nt, &gx, &gy);
  auto once = [&]() -> int {
    return zerocopy
      ? runDeviceGemmZeroCopy(fn, gx, gy, 32, A.data(), B.data(), D.data(),
                              M, N, K, elemBytes)
      : runDeviceGemm(fn, gx, gy, 32, A.data(), B.data(), D.data(),
                      M, N, K, elemBytes);
  };
  int rc = 3;
  bool failed = false;
  for (int w = 0; w < 5 && !failed; ++w) if (once() != 0) failed = true;
  if (!failed) {
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters && !failed; ++it) if (once() != 0) failed = true;
    auto t1 = std::chrono::steady_clock::now();
    if (!failed) {
      *avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count()
                / (double)iters;
      rc = 0;
    }
  }
  if (mod) hipModuleUnload(mod);
  return rc;
}

}  // namespace

// D[MxN] = A[MxK] @ B[KxN], row-major. A/B: f16 host buffers; D: f32 host
// buffer. General tiled/K-looped GEMM (any positive M/N/K; ragged edges padded).
// Returns 0 on success; 1 = bad shape; 2 = no usable HIP device / HIPRTC;
// 3 = a device memory / launch / copy operation failed.
extern "C" int tessera_rocm_wmma_gemm_f16(const void* A, const void* B, void* D,
                                          int M, int N, int K) {
  return runGemm(&g_f16, A, B, D, M, N, K, sizeof(unsigned short));
}

// As above, bf16 storage / f32 accumulate.
extern "C" int tessera_rocm_wmma_gemm_bf16(const void* A, const void* B,
                                           void* D, int M, int N, int K) {
  return runGemm(&g_bf16, A, B, D, M, N, K, sizeof(unsigned short));
}

// Device-timed kernel benchmark (f16). Times an (MT,NT)-blocked GEMM kernel,
// buffers reused, kernel-only (no H2D/D2H in the timed loop). avg_ms <- average
// per-launch milliseconds. Pass mt=nt=1 for the rung-0 naive baseline.
extern "C" int tessera_rocm_wmma_gemm_f16_bench(int M, int N, int K, int iters,
                                                int mt, int nt, double* avg_ms) {
  return benchVariant("__fp16", "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32",
                      M, N, K, iters, mt, nt, avg_ms);
}

// Device-timed benchmark for the rung-2 LDS-staged kernel (f16): WM×WN waves per
// workgroup, MT×NT register tiles per wave. avg_ms <- mean per-launch ms.
extern "C" int tessera_rocm_wmma_gemm_f16_bench_lds(int M, int N, int K,
                                                    int iters, int wm, int wn,
                                                    int mt, int nt,
                                                    double* avg_ms) {
  return benchVariantLDS("__fp16",
                         "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32",
                         M, N, K, iters, wm, wn, mt, nt, avg_ms);
}

// Run the rung-2 LDS-staged GEMM end-to-end (f16, f32 accumulate) — the
// correctness path for the LDS kernel. WM×WN waves, MT×NT register tiles/wave.
extern "C" int tessera_rocm_wmma_gemm_f16_lds(const void* A, const void* B,
                                              void* D, int M, int N, int K,
                                              int wm, int wn, int mt, int nt) {
  return runGemmLDS("__fp16", "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32",
                    A, B, D, M, N, K, wm, wn, mt, nt, sizeof(unsigned short));
}

// End-to-end CPU-timed benchmark of the full launch path. zerocopy: 0 = copy
// (hipMalloc + H2D/D2H), 1 = zero-copy (hipHostRegister device-mapped). avg_ms
// <- mean per-call ms. The APU win (if any) shows here, not in the kernel-only
// *_bench. mt=nt aren't the production constants — pass them explicitly (use
// 2,4 to match the shipped tiling).
extern "C" int tessera_rocm_wmma_gemm_f16_e2e_bench(int M, int N, int K,
                                                    int iters, int mt, int nt,
                                                    int zerocopy,
                                                    double* avg_ms) {
  return benchE2E("__fp16", "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32",
                  M, N, K, iters, mt, nt, zerocopy, avg_ms);
}
