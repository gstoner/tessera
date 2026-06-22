// tessera_rocm_gemm.cpp — shipped ROCm WMMA GEMM runtime symbol.
//
// Exposes a stable C-ABI entry point, ``tessera_rocm_wmma_gemm_f16``, that runs
// a real RDNA 3.5 WMMA matrix-core GEMM on the AMD GPU. This is the *shipped*
// runtime symbol the backend_manifest ``hardware_verified`` contract requires
// (the numerical proof lives in a checked-in execute_compare_fixture that calls
// this symbol). It is the production counterpart of the Stage C/D test-harness
// launcher.
//
// The device kernel is compiled at load time with **HIPRTC** for whatever arch
// the device enumerates (gfx1100 under WSL today, gfx1151 after AMD's WSL
// enablement) — so this object is built by the ordinary host C++ compiler and
// only needs the HIP runtime + HIPRTC at link time, no hipcc-as-compiler.
//
// Kernel: D[16x16] = A[16x16] @ B[16x16], row-major, f16 in / f32 accumulate,
// one 32-lane wave. Operand/accumulator fragment layout per the RDNA 3.5 ISA
// (matches python/tessera/compiler/rocdl_emit.py): A row r = lane&15, B col
// c = lane&15, output row = 2*e + lane>>4, col = lane&15.

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <mutex>
#include <string>
#include <vector>

namespace {

const char* kKernelSrc = R"HIPSRC(
typedef __fp16 half16 __attribute__((ext_vector_type(16)));
typedef float  float8 __attribute__((ext_vector_type(8)));
extern "C" __global__ void tessera_rocm_wmma_gemm_f16_kernel(
    const __fp16* A, const __fp16* B, float* D) {
  int l = threadIdx.x, lane = l & 15;
  half16 a, b; float8 c = {0,0,0,0,0,0,0,0};
  for (int i = 0; i < 16; ++i) { a[i] = A[lane*16 + i]; b[i] = B[i*16 + lane]; }
  c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
  for (int e = 0; e < 8; ++e) { int r = e*2 + (l >> 4); D[r*16 + lane] = c[e]; }
}
)HIPSRC";

// Compile-once cache for the HIPRTC module + kernel function.
std::once_flag g_once;
hipModule_t g_module = nullptr;
hipFunction_t g_kernel = nullptr;
bool g_ready = false;

void compileKernel() {
  hipDeviceProp_t props;
  if (hipGetDeviceProperties(&props, 0) != hipSuccess) return;

  hiprtcProgram prog;
  if (hiprtcCreateProgram(&prog, kKernelSrc, "tessera_rocm_wmma_gemm.hip",
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

  if (hipModuleLoadData(&g_module, code.data()) != hipSuccess) return;
  if (hipModuleGetFunction(&g_kernel, g_module,
                           "tessera_rocm_wmma_gemm_f16_kernel") != hipSuccess)
    return;
  g_ready = true;
}

}  // namespace

// Returns 0 on success; nonzero on error:
//   1 = unsupported shape (this kernel is a single 16x16x16 tile)
//   2 = kernel compile/load failed (no usable HIP device / HIPRTC)
//   3 = a device memory / launch / copy operation failed
//
// A, B: row-major 16x16 f16 host buffers; D: row-major 16x16 f32 host buffer.
extern "C" int tessera_rocm_wmma_gemm_f16(const void* A, const void* B, void* D,
                                          int M, int N, int K) {
  if (M != 16 || N != 16 || K != 16) return 1;

  std::call_once(g_once, compileKernel);
  if (!g_ready) return 2;

  const size_t halfBytes = 16 * 16 * sizeof(unsigned short);  // f16 = 2 bytes
  const size_t floatBytes = 16 * 16 * sizeof(float);
  void *dA = nullptr, *dB = nullptr, *dD = nullptr;
  int rc = 3;
  do {
    if (hipMalloc(&dA, halfBytes) != hipSuccess) break;
    if (hipMalloc(&dB, halfBytes) != hipSuccess) break;
    if (hipMalloc(&dD, floatBytes) != hipSuccess) break;
    if (hipMemcpy(dA, A, halfBytes, hipMemcpyHostToDevice) != hipSuccess) break;
    if (hipMemcpy(dB, B, halfBytes, hipMemcpyHostToDevice) != hipSuccess) break;
    void* args[] = {&dA, &dB, &dD};
    if (hipModuleLaunchKernel(g_kernel, 1, 1, 1, 32, 1, 1, 0, nullptr,
                              args, nullptr) != hipSuccess) break;
    if (hipDeviceSynchronize() != hipSuccess) break;
    if (hipMemcpy(D, dD, floatBytes, hipMemcpyDeviceToHost) != hipSuccess) break;
    rc = 0;
  } while (0);
  if (dA) hipFree(dA);
  if (dB) hipFree(dB);
  if (dD) hipFree(dD);
  return rc;
}
