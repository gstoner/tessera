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
// the device enumerates (gfx1100 under WSL today, gfx1151 after AMD's WSL
// enablement) — so this object is built by the ordinary host C++ compiler and
// only needs the HIP runtime + HIPRTC at link time, no hipcc-as-compiler.
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

#include <mutex>
#include <string>
#include <vector>

namespace {

// Production output-tile blocking factor for the shipped symbols. Measured-best
// on gfx1100/gfx1151 (Strix Halo): 2x4 wins at the compute-bound sizes — ~2.3x
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

std::string substitute(const std::string& tmpl, const std::string& from,
                       const std::string& to) {
  std::string out = tmpl;
  for (size_t p = out.find(from); p != std::string::npos;
       p = out.find(from, p + to.size()))
    out.replace(p, from.size(), to);
  return out;
}

// Compile a kernel variant for (type, wmma builtin, MT, NT) on device 0's arch.
// Returns true on success, handing back the module + function (caller owns the
// module lifetime). Used both by the cached production kernels and the bench.
bool compileVariant(const char* type, const char* wmma, int mt, int nt,
                    const std::string& name, hipModule_t* outMod,
                    hipFunction_t* outFn) {
  hipDeviceProp_t props;
  if (hipGetDeviceProperties(&props, 0) != hipSuccess) return false;

  std::string src = substitute(kKernelTemplate, "%TYPE%", type);
  src = substitute(src, "%WMMA%", wmma);
  src = substitute(src, "%NAME%", name);
  src = substitute(src, "%MT%", std::to_string(mt));
  src = substitute(src, "%NT%", std::to_string(nt));

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

// Run a general tiled GEMM through the cached production kernel.
// Returns 0 / 1 (bad shape) / 2 (no device / compile) / 3 (device op failed).
int runGemm(Kernel* k, const void* A, const void* B, void* D,
            int M, int N, int K, size_t elemBytes) {
  if (M <= 0 || N <= 0 || K <= 0) return 1;

  std::call_once(k->once, compileProd, k);
  if (!k->ready) return 2;

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
    unsigned gx, gy;
    gridFor(M, N, kProdMT, kProdNT, &gx, &gy);
    if (hipModuleLaunchKernel(k->fn, gx, gy, 1, 32, 1, 1, 0, nullptr,
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

// Device-timed kernel benchmark for a single (type, MT, NT) variant. Allocates
// the device buffers once, warms up, then hipEvent-times ``iters`` kernel-only
// launches (no H2D/D2H in the loop) so the measurement is GEMM compute, not
// transfer. Writes the average per-launch ms to *avg_ms. This is the rung-prover
// for the perf ladder. Returns 0 / 1 (bad args) / 2 (no device / compile) /
// 3 (device op failed).
int benchVariant(const char* type, const char* wmma, int M, int N, int K,
                 int iters, int mt, int nt, double* avg_ms) {
  if (M <= 0 || N <= 0 || K <= 0 || iters <= 0 || mt <= 0 || nt <= 0)
    return 1;

  hipModule_t mod = nullptr;
  hipFunction_t fn = nullptr;
  // ``type`` is "__fp16"/"__bf16" — underscores are legal in the kernel name.
  std::string name = std::string("bench") + type + "_" + std::to_string(mt)
                     + "x" + std::to_string(nt);
  if (!compileVariant(type, wmma, mt, nt, name, &mod, &fn)) return 2;

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
    unsigned gx, gy;
    gridFor(M, N, mt, nt, &gx, &gy);
    // Warmup.
    for (int w = 0; w < 5; ++w)
      if (hipModuleLaunchKernel(fn, gx, gy, 1, 32, 1, 1, 0, nullptr,
                                args, nullptr) != hipSuccess) { rc = 3; break; }
    if (hipDeviceSynchronize() != hipSuccess) break;
    if (hipEventCreate(&start) != hipSuccess) break;
    if (hipEventCreate(&stop) != hipSuccess) break;
    if (hipEventRecord(start, nullptr) != hipSuccess) break;
    for (int it = 0; it < iters; ++it)
      if (hipModuleLaunchKernel(fn, gx, gy, 1, 32, 1, 1, 0, nullptr,
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
