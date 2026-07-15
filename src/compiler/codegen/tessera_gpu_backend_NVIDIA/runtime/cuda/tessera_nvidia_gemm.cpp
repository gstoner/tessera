// Shipped NVIDIA mma.sync GEMM runtime symbols (consumer Blackwell, CC 12.0+).
//
// The NVIDIA analog of src/.../Tessera_ROCM_Backend/runtime/hip/tessera_rocm_gemm.cpp.
// Exports C-ABI symbols tessera_nvidia_mma_gemm_{bf16,f16,tf32,e4m3,e5m2} that run
// a general tiled/K-looped warp-level mma.sync GEMM on the GPU:
//
//     D[M,N] f32 = A[M,K] @ B[K,N]   (row-major; ragged M/N/K zero-padded)
//
// The kernel is NVRTC-compiled for the live device arch at first call (compute_XX
// from cuDeviceGetAttribute) — the driver JIT path, so the .o needs only the host
// compiler + the CUDA driver (libcuda) + NVRTC at link time, no nvcc device pass.
// This is the shipped symbol the backend_manifest `hardware_verified` contract
// requires; the execute_compare_fixture (tests/unit/test_nvidia_mma_runtime_symbol.py)
// dlopens it and numerically validates each dtype against a numpy reference.
//
// Proven on-silicon 2026-06-25 (RTX 5070 Ti). Each dtype uses its documented MMA
// shape: 16-bit (bf16/f16) m16n8k16, tf32 m16n8k8, fp8 (e4m3/e5m2) m16n8k32.
// Return codes: 0 ok, 1 bad shape, 2 no usable GPU / NVRTC, 3 device op failed.

#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <cuda.h>
#include <nvrtc.h>

namespace {

// ── kernel templates (TYPE substituted per dtype) ────────────────────────────
// 16-bit operands (bf16/f16): m16n8k16, each .b32 packs 2 contiguous elems.
const char* kSrc16 = R"NVRTC(
extern "C" __global__ void gemm(const unsigned short* A, const unsigned short* B,
                                float* D, int M, int N, int K) {
  int mt=blockIdx.x*16, nt=blockIdx.y*8, lane=threadIdx.x, gid=lane>>2, tig=lane&3;
  float d0=0,d1=0,d2=0,d3=0;
  for (int k0=0;k0<K;k0+=16){
    auto la=[&](int r,int c)->unsigned{int rr=mt+r,cc=k0+c;
      unsigned lo=(rr<M&&cc<K)?A[rr*K+cc]:0u, hi=(rr<M&&cc+1<K)?A[rr*K+cc+1]:0u; return (hi<<16)|lo;};
    auto lb=[&](int r,int c)->unsigned{int rr=k0+r,cc=nt+c;
      unsigned lo=(rr<K&&cc<N)?B[rr*N+cc]:0u, hi=(rr+1<K&&cc<N)?B[(rr+1)*N+cc]:0u; return (hi<<16)|lo;};
    unsigned a0=la(gid,2*tig),a1=la(gid+8,2*tig),a2=la(gid,2*tig+8),a3=la(gid+8,2*tig+8);
    unsigned b0=lb(2*tig,gid),b1=lb(2*tig+8,gid);
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.TYPE.TYPE.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
  }
  auto st=[&](int r,int c,float v){int rr=mt+r,cc=nt+c;if(rr<M&&cc<N)D[rr*N+cc]=v;};
  st(gid,2*tig,d0);st(gid,2*tig+1,d1);st(gid+8,2*tig,d2);st(gid+8,2*tig+1,d3);
}
)NVRTC";

// tf32: m16n8k8, A holds 4 tf32 (one f32-bit pattern per .b32), B holds 2.
const char* kSrcTf32 = R"NVRTC(
extern "C" __global__ void gemm(const float* A, const float* B,
                                float* D, int M, int N, int K) {
  int mt=blockIdx.x*16, nt=blockIdx.y*8, lane=threadIdx.x, gid=lane>>2, tig=lane&3;
  float d0=0,d1=0,d2=0,d3=0;
  for (int k0=0;k0<K;k0+=8){
    auto la=[&](int r,int c)->unsigned{int rr=mt+r,cc=k0+c; float v=(rr<M&&cc<K)?A[rr*K+cc]:0.f; unsigned u; memcpy(&u,&v,4); return u;};
    auto lb=[&](int r,int c)->unsigned{int rr=k0+r,cc=nt+c; float v=(rr<K&&cc<N)?B[rr*N+cc]:0.f; unsigned u; memcpy(&u,&v,4); return u;};
    unsigned a0=la(gid,tig),a1=la(gid+8,tig),a2=la(gid,tig+4),a3=la(gid+8,tig+4);
    unsigned b0=lb(tig,gid),b1=lb(tig+4,gid);
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
  }
  auto st=[&](int r,int c,float v){int rr=mt+r,cc=nt+c;if(rr<M&&cc<N)D[rr*N+cc]=v;};
  st(gid,2*tig,d0);st(gid,2*tig+1,d1);st(gid+8,2*tig,d2);st(gid+8,2*tig+1,d3);
}
)NVRTC";

// fp8 (e4m3/e5m2): m16n8k32, each .b32 packs 4 fp8 bytes.
const char* kSrcF8 = R"NVRTC(
extern "C" __global__ void gemm(const unsigned char* A, const unsigned char* B,
                                float* D, int M, int N, int K) {
  int mt=blockIdx.x*16, nt=blockIdx.y*8, lane=threadIdx.x, gid=lane>>2, tig=lane&3;
  float d0=0,d1=0,d2=0,d3=0;
  for (int k0=0;k0<K;k0+=32){
    auto la=[&](int r,int c)->unsigned{int rr=mt+r; unsigned w=0;
      for(int j=0;j<4;j++){int cc=k0+c+j; unsigned b=(rr<M&&cc<K)?A[rr*K+cc]:0u; w|=b<<(8*j);} return w;};
    auto lb=[&](int r,int c)->unsigned{int cc=nt+c; unsigned w=0;
      for(int j=0;j<4;j++){int rr=k0+r+j; unsigned b=(rr<K&&cc<N)?B[rr*N+cc]:0u; w|=b<<(8*j);} return w;};
    unsigned a0=la(gid,4*tig),a1=la(gid+8,4*tig),a2=la(gid,4*tig+16),a3=la(gid+8,4*tig+16);
    unsigned b0=lb(4*tig,gid),b1=lb(4*tig+16,gid);
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.TYPE.TYPE.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
  }
  auto st=[&](int r,int c,float v){int rr=mt+r,cc=nt+c;if(rr<M&&cc<N)D[rr*N+cc]=v;};
  st(gid,2*tig,d0);st(gid,2*tig+1,d1);st(gid+8,2*tig,d2);st(gid+8,2*tig+1,d3);
}
)NVRTC";

std::once_flag g_ctx_once;
bool g_ctx_ok = false;
CUdevice g_dev = 0;

void initCtxOnce() {
  if (cuInit(0) != CUDA_SUCCESS) return;
  int n = 0;
  if (cuDeviceGetCount(&n) != CUDA_SUCCESS || n < 1) return;
  if (cuDeviceGet(&g_dev, 0) != CUDA_SUCCESS) return;
  CUcontext ctx;
  if (cuDevicePrimaryCtxRetain(&ctx, g_dev) != CUDA_SUCCESS) return;
  if (cuCtxSetCurrent(ctx) != CUDA_SUCCESS) return;
  g_ctx_ok = true;
}

// NVRTC-compile one kernel template (TYPE substituted) for the live device arch.
bool compileKernel(const char* src_tmpl, const char* type, CUfunction* out) {
  std::string src(src_tmpl);
  if (type) {
    for (size_t p; (p = src.find("TYPE")) != std::string::npos;)
      src.replace(p, 4, type);
  }
  nvrtcProgram prog;
  if (nvrtcCreateProgram(&prog, src.c_str(), "tessera_nvidia_gemm.cu", 0, nullptr,
                         nullptr) != NVRTC_SUCCESS)
    return false;
  int maj = 0, min = 0;
  cuDeviceGetAttribute(&maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, g_dev);
  cuDeviceGetAttribute(&min, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, g_dev);
  char arch[40];
  std::snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", maj, min);
  const char* opts[] = {arch};
  nvrtcResult cr = nvrtcCompileProgram(prog, 1, opts);
  if (cr != NVRTC_SUCCESS) { nvrtcDestroyProgram(&prog); return false; }
  size_t psz = 0;
  if (nvrtcGetPTXSize(prog, &psz) != NVRTC_SUCCESS) { nvrtcDestroyProgram(&prog); return false; }
  std::vector<char> ptx(psz);
  nvrtcGetPTX(prog, ptx.data());
  nvrtcDestroyProgram(&prog);
  CUmodule mod;
  if (cuModuleLoadData(&mod, ptx.data()) != CUDA_SUCCESS) return false;
  return cuModuleGetFunction(out, mod, "gemm") == CUDA_SUCCESS;
}

// Per-dtype kernel cache (compiled once).
struct Kernel { std::once_flag once; bool ok = false; CUfunction fn = nullptr; };
Kernel g_k16bf, g_k16f, g_ktf32, g_ke4, g_ke5;

CUfunction getKernel(Kernel* k, const char* tmpl, const char* type) {
  std::call_once(k->once, [&] { k->ok = compileKernel(tmpl, type, &k->fn); });
  return k->ok ? k->fn : nullptr;
}

// elemSize: bytes per A/B element (2 bf16/f16, 4 tf32, 1 fp8).
int runGemm(CUfunction fn, const void* A, const void* B, void* D,
            int M, int N, int K, int elemSize) {
  if (M <= 0 || N <= 0 || K <= 0) return 1;
  if (fn == nullptr) return 2;
  size_t sA = (size_t)M * K * elemSize, sB = (size_t)K * N * elemSize, sD = (size_t)M * N * 4;
  CUdeviceptr dA = 0, dB = 0, dD = 0;
  if (cuMemAlloc(&dA, sA) != CUDA_SUCCESS) return 3;
  if (cuMemAlloc(&dB, sB) != CUDA_SUCCESS) { cuMemFree(dA); return 3; }
  if (cuMemAlloc(&dD, sD) != CUDA_SUCCESS) { cuMemFree(dA); cuMemFree(dB); return 3; }
  int rc = 0;
  do {
    if (cuMemcpyHtoD(dA, A, sA) != CUDA_SUCCESS) { rc = 3; break; }
    if (cuMemcpyHtoD(dB, B, sB) != CUDA_SUCCESS) { rc = 3; break; }
    int mt = (M + 15) / 16, nt = (N + 7) / 8;
    void* args[] = {&dA, &dB, &dD, &M, &N, &K};
    if (cuLaunchKernel(fn, mt, nt, 1, 32, 1, 1, 0, 0, args, 0) != CUDA_SUCCESS) { rc = 3; break; }
    if (cuCtxSynchronize() != CUDA_SUCCESS) { rc = 3; break; }
    if (cuMemcpyDtoH(D, dD, sD) != CUDA_SUCCESS) { rc = 3; break; }
  } while (0);
  cuMemFree(dA); cuMemFree(dB); cuMemFree(dD);
  return rc;
}

int runGemmDevice(CUfunction fn, const void* A, const void* B, void* D,
                  int M, int N, int K, void* stream) {
  if (M <= 0 || N <= 0 || K <= 0 || A == nullptr || B == nullptr || D == nullptr)
    return 1;
  if (fn == nullptr) return 2;
  CUdeviceptr dA = reinterpret_cast<CUdeviceptr>(A);
  CUdeviceptr dB = reinterpret_cast<CUdeviceptr>(B);
  CUdeviceptr dD = reinterpret_cast<CUdeviceptr>(D);
  CUstream st = reinterpret_cast<CUstream>(stream);
  int mt = (M + 15) / 16, nt = (N + 7) / 8;
  void* args[] = {&dA, &dB, &dD, &M, &N, &K};
  return cuLaunchKernel(fn, mt, nt, 1, 32, 1, 1, 0, st, args, nullptr)
             == CUDA_SUCCESS ? 0 : 3;
}

int dispatch(Kernel* k, const char* tmpl, const char* type,
             const void* A, const void* B, void* D, int M, int N, int K, int esz) {
  std::call_once(g_ctx_once, initCtxOnce);
  if (!g_ctx_ok) return 2;
  return runGemm(getKernel(k, tmpl, type), A, B, D, M, N, K, esz);
}

int dispatchDevice(Kernel* k, const char* tmpl, const char* type,
                   const void* A, const void* B, void* D,
                   int M, int N, int K, void* stream) {
  std::call_once(g_ctx_once, initCtxOnce);
  if (!g_ctx_ok) return 2;
  return runGemmDevice(
      getKernel(k, tmpl, type), A, B, D, M, N, K, stream);
}

}  // namespace

extern "C" {

int tessera_nvidia_mma_gemm_bf16(const void* A, const void* B, void* D, int M, int N, int K) {
  return dispatch(&g_k16bf, kSrc16, "bf16", A, B, D, M, N, K, 2);
}
int tessera_nvidia_mma_gemm_f16(const void* A, const void* B, void* D, int M, int N, int K) {
  return dispatch(&g_k16f, kSrc16, "f16", A, B, D, M, N, K, 2);
}
int tessera_nvidia_mma_gemm_tf32(const void* A, const void* B, void* D, int M, int N, int K) {
  return dispatch(&g_ktf32, kSrcTf32, nullptr, A, B, D, M, N, K, 4);
}
int tessera_nvidia_mma_gemm_e4m3(const void* A, const void* B, void* D, int M, int N, int K) {
  return dispatch(&g_ke4, kSrcF8, "e4m3", A, B, D, M, N, K, 1);
}
int tessera_nvidia_mma_gemm_e5m2(const void* A, const void* B, void* D, int M, int N, int K) {
  return dispatch(&g_ke5, kSrcF8, "e5m2", A, B, D, M, N, K, 1);
}

#define TESSERA_DEVICE_GEMM(name, kernel, source, type)                         \
int name(const void* A, const void* B, void* D, int M, int N, int K,           \
         void* stream) {                                                        \
  return dispatchDevice(kernel, source, type, A, B, D, M, N, K, stream);       \
}

TESSERA_DEVICE_GEMM(tessera_nvidia_mma_gemm_bf16_device,
                    &g_k16bf, kSrc16, "bf16")
TESSERA_DEVICE_GEMM(tessera_nvidia_mma_gemm_f16_device,
                    &g_k16f, kSrc16, "f16")
TESSERA_DEVICE_GEMM(tessera_nvidia_mma_gemm_tf32_device,
                    &g_ktf32, kSrcTf32, nullptr)
TESSERA_DEVICE_GEMM(tessera_nvidia_mma_gemm_e4m3_device,
                    &g_ke4, kSrcF8, "e4m3")
TESSERA_DEVICE_GEMM(tessera_nvidia_mma_gemm_e5m2_device,
                    &g_ke5, kSrcF8, "e5m2")

#undef TESSERA_DEVICE_GEMM

int tessera_nvidia_device_alloc(void** out, size_t bytes) {
  if (out == nullptr || bytes == 0) return 1;
  std::call_once(g_ctx_once, initCtxOnce);
  if (!g_ctx_ok) return 2;
  CUdeviceptr ptr = 0;
  if (cuMemAlloc(&ptr, bytes) != CUDA_SUCCESS) return 3;
  *out = reinterpret_cast<void*>(ptr);
  return 0;
}

int tessera_nvidia_device_free(void* ptr) {
  return ptr && cuMemFree(reinterpret_cast<CUdeviceptr>(ptr)) == CUDA_SUCCESS
             ? 0 : 1;
}

int tessera_nvidia_stream_create(void** out) {
  if (out == nullptr) return 1;
  std::call_once(g_ctx_once, initCtxOnce);
  if (!g_ctx_ok) return 2;
  CUstream stream = nullptr;
  if (cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) return 3;
  *out = reinterpret_cast<void*>(stream);
  return 0;
}

int tessera_nvidia_stream_destroy(void* stream) {
  return stream && cuStreamDestroy(reinterpret_cast<CUstream>(stream)) == CUDA_SUCCESS
             ? 0 : 1;
}

int tessera_nvidia_stream_synchronize(void* stream) {
  return cuStreamSynchronize(reinterpret_cast<CUstream>(stream)) == CUDA_SUCCESS
             ? 0 : 3;
}

int tessera_nvidia_device_upload(void* dst, const void* src, size_t bytes,
                                 void* stream) {
  if (!dst || !src || bytes == 0) return 1;
  return cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(dst), src, bytes,
                           reinterpret_cast<CUstream>(stream)) == CUDA_SUCCESS
             ? 0 : 3;
}

int tessera_nvidia_device_download(void* dst, const void* src, size_t bytes,
                                   void* stream) {
  if (!dst || !src || bytes == 0) return 1;
  return cuMemcpyDtoHAsync(dst, reinterpret_cast<CUdeviceptr>(src), bytes,
                           reinterpret_cast<CUstream>(stream)) == CUDA_SUCCESS
             ? 0 : 3;
}

int tessera_nvidia_event_create(void** out) {
  if (!out) return 1;
  std::call_once(g_ctx_once, initCtxOnce);
  if (!g_ctx_ok) return 2;
  CUevent event = nullptr;
  if (cuEventCreate(&event, CU_EVENT_DEFAULT) != CUDA_SUCCESS) return 3;
  *out = reinterpret_cast<void*>(event);
  return 0;
}

int tessera_nvidia_event_destroy(void* event) {
  return event && cuEventDestroy(reinterpret_cast<CUevent>(event)) == CUDA_SUCCESS
             ? 0 : 1;
}

int tessera_nvidia_event_record(void* event, void* stream) {
  return cuEventRecord(reinterpret_cast<CUevent>(event),
                       reinterpret_cast<CUstream>(stream)) == CUDA_SUCCESS ? 0 : 3;
}

int tessera_nvidia_event_synchronize(void* event) {
  return cuEventSynchronize(reinterpret_cast<CUevent>(event)) == CUDA_SUCCESS
             ? 0 : 3;
}

int tessera_nvidia_event_elapsed_ms(void* start, void* stop, float* ms) {
  if (!start || !stop || !ms) return 1;
  return cuEventElapsedTime(ms, reinterpret_cast<CUevent>(start),
                            reinterpret_cast<CUevent>(stop)) == CUDA_SUCCESS ? 0 : 3;
}

}  // extern "C"
