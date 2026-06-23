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
// General tiled/K-looped GEMM: each 32-lane wave (one block) computes one
// 16x16 output tile, looping over K in 16-wide chunks and accumulating in the
// WMMA fragment; ragged edges (M/N/K not a multiple of 16) are zero-padded on
// load and bounds-checked on store. Operand/accumulator fragment layout per the
// RDNA 3.5 ISA (matches python/tessera/compiler/rocdl_emit.py): A frag row =
// lane&15, B frag col = lane&15, output row = 2*e + lane>>4, col = lane&15.

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <mutex>
#include <string>
#include <vector>

namespace {

// One HIPRTC source per storage dtype. The body is identical except for the
// element type + the WMMA builtin; ``%TYPE%`` / ``%WMMA%`` are substituted at
// load. Tiled over (M, N) in 16x16 blocks with a K-loop, so any M/N/K works.
const char* kKernelTemplate = R"HIPSRC(
typedef %TYPE% wtype16 __attribute__((ext_vector_type(16)));
typedef float  float8  __attribute__((ext_vector_type(8)));
extern "C" __global__ void %NAME%(
    const %TYPE%* A, const %TYPE%* B, float* D, int M, int N, int K) {
  int l = threadIdx.x, lane = l & 15;
  int baseRow = blockIdx.y * 16;       // this wave's output tile origin
  int baseCol = blockIdx.x * 16;
  float8 c = {0,0,0,0,0,0,0,0};
  for (int k0 = 0; k0 < K; k0 += 16) {
    wtype16 a, b;
    for (int i = 0; i < 16; ++i) {
      int ar = baseRow + lane, ak = k0 + i;
      a[i] = (ar < M && ak < K) ? A[ar * K + ak] : (%TYPE%)0;
      int bk = k0 + i, bc = baseCol + lane;
      b[i] = (bk < K && bc < N) ? B[bk * N + bc] : (%TYPE%)0;
    }
    c = %WMMA%(a, b, c);
  }
  for (int e = 0; e < 8; ++e) {
    int r = baseRow + e * 2 + (l >> 4);
    int col = baseCol + lane;
    if (r < M && col < N) D[r * N + col] = c[e];
  }
}
)HIPSRC";

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

std::string substitute(const std::string& tmpl, const std::string& from,
                       const std::string& to) {
  std::string out = tmpl;
  for (size_t p = out.find(from); p != std::string::npos;
       p = out.find(from, p + to.size()))
    out.replace(p, from.size(), to);
  return out;
}

void compileKernel(Kernel* k) {
  hipDeviceProp_t props;
  if (hipGetDeviceProperties(&props, 0) != hipSuccess) return;

  std::string src = substitute(kKernelTemplate, "%TYPE%", k->type);
  src = substitute(src, "%WMMA%", k->wmma);
  src = substitute(src, "%NAME%", k->name);

  hiprtcProgram prog;
  if (hiprtcCreateProgram(&prog, src.c_str(), "tessera_rocm_wmma_gemm.hip",
                          0, nullptr, nullptr) != HIPRTC_SUCCESS)
    return;
  std::string archOpt = std::string("--offload-arch=") + props.gcnArchName;
  const char* opts[] = {archOpt.c_str()};
  hiprtcResult cr = hiprtcCompileProgram(prog, 1, opts);
  if (cr != HIPRTC_SUCCESS) { hiprtcDestroyProgram(&prog); return; }

  size_t codeSize = 0;
  if (hiprtcGetCodeSize(prog, &codeSize) != HIPRTC_SUCCESS || codeSize == 0) {
    hiprtcDestroyProgram(&prog); return;
  }
  std::vector<char> code(codeSize);
  hiprtcGetCode(prog, code.data());
  hiprtcDestroyProgram(&prog);

  if (hipModuleLoadData(&k->module, code.data()) != hipSuccess) return;
  if (hipModuleGetFunction(&k->fn, k->module, k->name) != hipSuccess) return;
  k->ready = true;
}

// Run a general tiled GEMM through the given (compiled-on-demand) kernel.
//   elemBytes = sizeof(storage element) — 2 for f16/bf16.
// Returns 0 / 1 (bad shape) / 2 (no device / compile) / 3 (device op failed).
int runGemm(Kernel* k, const void* A, const void* B, void* D,
            int M, int N, int K, size_t elemBytes) {
  if (M <= 0 || N <= 0 || K <= 0) return 1;

  std::call_once(k->once, compileKernel, k);
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
    unsigned gridX = (unsigned)((N + 15) / 16);
    unsigned gridY = (unsigned)((M + 15) / 16);
    if (hipModuleLaunchKernel(k->fn, gridX, gridY, 1, 32, 1, 1, 0, nullptr,
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
