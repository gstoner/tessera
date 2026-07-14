// Shipped NVIDIA PTX launch bridge (COMPILER_REFACTOR_PLAN C2 tail).
//
// Promotes the throwaway inline launcher in
// tests/unit/test_conformance_execute_compare_nvidia.py into a shipped, cached
// runtime component — the NVIDIA counterpart to Apple's apple_gpu_runtime.mm.
// It driver-JITs Tessera's *emitted* PTX (ptx_emit.py) and launches it on the
// GPU; nothing else launches the emit-path PTX today (the shipped
// libtessera_nvidia_gemm.so is a separate NVRTC'd CUDA-C GEMM).
//
// Design mirrors the shipped GEMM lib: host compiler + CUDA driver (libcuda)
// only, no nvcc device pass — the PTX is JIT'd by the driver at first launch.
// The per-kernel ABI (buffer sizes / directions / launch config) is keyed by
// kernel name, exactly as the Apple launcher maps a name to its native symbol;
// the table is seeded with the one on-silicon-proven kernel
// (tessera_mma_m16n8k16_bf16) and is the extension point for the next.

#include "tessera_nvidia_ptx_launch.h"

#include <cstring>
#include <map>
#include <mutex>
#include <string>

#include <cuda.h>

// The core-runtime seam (tsrGpuLauncherFn / tsrRegisterGpuLauncher / TsrStatus /
// tsrGpuLaunchParams). Only tessera_nvidia_register_ptx_launcher references the
// tsrRegisterGpuLauncher symbol, so the direct register/invoke path links (and
// dlopens) without the core runtime present.
#include "tessera/tessera_runtime.h"

// Weakly reference the core-runtime seam so the shipped .so dlopens standalone
// (the direct register/invoke path — Python/ctypes, live tests) even when
// libtessera_runtime is not in the load. When a hosting binary provides the
// symbol it binds normally; when it does not, tessera_nvidia_register_ptx_launcher
// reports that cleanly instead of failing the whole dlopen (Decision #21).
extern "C" __attribute__((weak)) TsrStatus
tsrRegisterGpuLauncher(tsrGpuLauncherFn fn, void* user);

namespace {

// ── per-kernel ABIs, keyed by entry name (extension point per Apple pattern) ───
// The single on-silicon-proven m16n8k16 tile ...
constexpr const char* kMmaEntry = "tessera_mma_m16n8k16_bf16";
constexpr int kMmaM = 16, kMmaN = 8, kMmaK = 16;
// ... and the general aligned mma.sync GEMM (K-loop + grid-tiled), bf16/f16.
// Same 16-bit operand ABI (2-byte A/B, f32 D), grid derived from M/N.
constexpr const char* kGemmBf16 = "tessera_mma_gemm_bf16";
constexpr const char* kGemmF16 = "tessera_mma_gemm_f16";
constexpr const char* kTileDirectF16 = "tessera_tile_matmul_direct_f16";
constexpr const char* kTileDirectBf16 = "tessera_tile_matmul_direct_bf16";
constexpr const char* kTileSharedF16 = "tessera_tile_matmul_shared_f16";
constexpr const char* kTileSharedBf16 = "tessera_tile_matmul_shared_bf16";

std::mutex g_mu;
std::map<std::string, std::string> g_ptx;           // kernel name -> PTX text
std::map<std::string, CUmodule>    g_modules;        // kernel name -> JIT'd module
std::map<std::string, CUfunction>  g_funcs;          // kernel name -> entry fn
bool g_ctx_ready = false;

// Ensure a current CUDA context on device 0 (lazy, once). Returns false when no
// usable GPU is present — the caller maps that to rc 2 (skip-clean upstream).
bool ensureContext() {
    if (g_ctx_ready) return true;
    if (cuInit(0) != CUDA_SUCCESS) return false;
    int n = 0;
    if (cuDeviceGetCount(&n) != CUDA_SUCCESS || n < 1) return false;
    CUdevice dev;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return false;
    CUcontext ctx;
    if (cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS) return false;
    if (cuCtxSetCurrent(ctx) != CUDA_SUCCESS) return false;
    g_ctx_ready = true;
    return true;
}

// Get-or-JIT the module + entry function for kernel_name (cached). Caller holds
// g_mu. Returns nullptr if no PTX is registered or the JIT/lookup fails.
CUfunction getFunctionLocked(const std::string& name) {
    auto cached = g_funcs.find(name);
    if (cached != g_funcs.end()) return cached->second;
    auto ptx = g_ptx.find(name);
    if (ptx == g_ptx.end()) return nullptr;

    char log[8192];
    log[0] = 0;
    CUjit_option opt[2] = {CU_JIT_ERROR_LOG_BUFFER,
                           CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void* optv[2] = {(void*)log, (void*)(size_t)sizeof(log)};
    CUmodule mod = nullptr;
    if (cuModuleLoadDataEx(&mod, ptx->second.c_str(), 2, opt, optv) != CUDA_SUCCESS)
        return nullptr;
    CUfunction fn = nullptr;
    if (cuModuleGetFunction(&fn, mod, name.c_str()) != CUDA_SUCCESS) {
        cuModuleUnload(mod);
        return nullptr;
    }
    g_modules[name] = mod;
    g_funcs[name] = fn;
    return fn;
}

// Launch the mma.sync m16n8k16 bf16 kernel: buffers {A bf16, B bf16, D f32},
// dims {M,N,K} == {16,8,16}; one warp (grid 1, block 32). Returns the C-ABI rc.
int invokeMma(CUfunction fn, void** buffers, size_t nbuf,
              const int64_t* dims, size_t ndim) {
    if (nbuf != 3 || ndim != 3) return 5;
    if (dims[0] != kMmaM || dims[1] != kMmaN || dims[2] != kMmaK) return 5;
    const void* A = buffers[0];
    const void* B = buffers[1];
    void* D = buffers[2];
    const size_t sA = (size_t)kMmaM * kMmaK * 2;   // bf16
    const size_t sB = (size_t)kMmaK * kMmaN * 2;   // bf16
    const size_t sD = (size_t)kMmaM * kMmaN * 4;   // f32
    CUdeviceptr dA = 0, dB = 0, dD = 0;
    if (cuMemAlloc(&dA, sA) != CUDA_SUCCESS) return 3;
    if (cuMemAlloc(&dB, sB) != CUDA_SUCCESS) { cuMemFree(dA); return 3; }
    if (cuMemAlloc(&dD, sD) != CUDA_SUCCESS) { cuMemFree(dA); cuMemFree(dB); return 3; }
    int rc = 0;
    do {
        if (cuMemcpyHtoD(dA, A, sA) != CUDA_SUCCESS) { rc = 3; break; }
        if (cuMemcpyHtoD(dB, B, sB) != CUDA_SUCCESS) { rc = 3; break; }
        void* args[] = {&dA, &dB, &dD};
        if (cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, 0, args, 0) != CUDA_SUCCESS) {
            rc = 3; break;
        }
        if (cuCtxSynchronize() != CUDA_SUCCESS) { rc = 3; break; }
        if (cuMemcpyDtoH(D, dD, sD) != CUDA_SUCCESS) { rc = 3; break; }
    } while (0);
    cuMemFree(dA); cuMemFree(dB); cuMemFree(dD);
    return rc;
}

// Launch the general aligned mma.sync 16-bit GEMM: buffers {A 16b, B 16b, D f32},
// dims {M,N,K} (M%16,N%8,K%16); grid (M/16, N/8), block 32 (one warp per 16x8
// tile); runtime M/N/K params. bf16 and f16 share this ABI (only the JIT'd PTX
// differs). Returns the C-ABI rc.
int invokeMmaGemm16(CUfunction fn, void** buffers, size_t nbuf,
                    const int64_t* dims, size_t ndim, int tileM = 16,
                    int tileN = 8, int threads = 32, bool ragged = false,
                    bool columnMajorGrid = false, bool dimensions64 = false) {
    if (nbuf != 3 || ndim != 3) return 5;
    const long long M64 = dims[0], N64 = dims[1], K64 = dims[2];
    if (M64 <= 0 || N64 <= 0 || K64 <= 0) return 5;
    if (!ragged && (M64 % 16 || N64 % 8 || K64 % 16)) return 5;
    // The emitted PTX addresses elements with 32-bit signed indices, so an
    // operand's LARGEST index (element count - 1) must fit INT32_MAX. Reject only
    // shapes whose element count EXCEEDS 2^31 (a count of exactly 2^31 has max
    // index 2^31-1 == INT32_MAX, which is representable) — Decision #21: honest
    // invalid-args, never silent corruption. 64-bit index math in the emitter is
    // the follow-on that lifts this cap.
    const long long kMaxElems = 1LL << 31;           // max operand element count
    // Each dim < 2^31 keeps the int (int32) cast below well-defined and the int64
    // products overflow-free (no valid shape reaches a dim of 2^31 anyway).
    if (M64 >= kMaxElems || N64 >= kMaxElems || K64 >= kMaxElems) return 5;
    if (M64 * K64 > kMaxElems || K64 * N64 > kMaxElems || M64 * N64 > kMaxElems) return 5;
    int M = (int)M64, N = (int)N64, K = (int)K64;
    const void* A = buffers[0];
    const void* B = buffers[1];
    void* D = buffers[2];
    const size_t sA = (size_t)M * K * 2;   // 16-bit A
    const size_t sB = (size_t)K * N * 2;   // 16-bit B (col-major)
    const size_t sD = (size_t)M * N * 4;   // f32 D
    CUdeviceptr dA = 0, dB = 0, dD = 0;
    if (cuMemAlloc(&dA, sA) != CUDA_SUCCESS) return 3;
    if (cuMemAlloc(&dB, sB) != CUDA_SUCCESS) { cuMemFree(dA); return 3; }
    if (cuMemAlloc(&dD, sD) != CUDA_SUCCESS) { cuMemFree(dA); cuMemFree(dB); return 3; }
    int rc = 0;
    do {
        if (cuMemcpyHtoD(dA, A, sA) != CUDA_SUCCESS) { rc = 3; break; }
        if (cuMemcpyHtoD(dB, B, sB) != CUDA_SUCCESS) { rc = 3; break; }
        long long MArg = M64, NArg = N64, KArg = K64;
        void* args32[] = {&dA, &dB, &dD, &M, &N, &K};
        void* args64[] = {&dA, &dB, &dD, &MArg, &NArg, &KArg};
        void** args = dimensions64 ? args64 : args32;
        unsigned gx = columnMajorGrid
            ? (unsigned)((N + tileN - 1) / tileN)
            : (unsigned)((M + tileM - 1) / tileM);
        unsigned gy = columnMajorGrid
            ? (unsigned)((M + tileM - 1) / tileM)
            : (unsigned)((N + tileN - 1) / tileN);
        if (cuLaunchKernel(fn, gx, gy, 1, (unsigned)threads, 1, 1,
                           0, 0, args, 0) != CUDA_SUCCESS) {
            rc = 3; break;
        }
        if (cuCtxSynchronize() != CUDA_SUCCESS) { rc = 3; break; }
        if (cuMemcpyDtoH(D, dD, sD) != CUDA_SUCCESS) { rc = 3; break; }
    } while (0);
    cuMemFree(dA); cuMemFree(dB); cuMemFree(dD);
    return rc;
}

bool tileLaunchConfig(const char* name, int& tileM, int& tileN, int& threads) {
    if (std::strcmp(name, kTileDirectF16) == 0 ||
        std::strcmp(name, kTileDirectBf16) == 0) {
        tileM = 16; tileN = 8; threads = 32;
        return true;
    }
    if (std::strcmp(name, kTileSharedF16) == 0 ||
        std::strcmp(name, kTileSharedBf16) == 0) {
        tileM = 32; tileN = 32; threads = 128;
        return true;
    }
    return false;
}

int benchmarkTileGemm16(CUfunction fn, const char* name, void** buffers,
                        size_t nbuf, const int64_t* dims, size_t ndim,
                        int warmup, int repetitions, float* latencyMs) {
    if (nbuf != 3 || ndim != 3 || !latencyMs || warmup < 0 || repetitions <= 0)
        return 5;
    const long long M = dims[0], N = dims[1], K = dims[2];
    if (M <= 0 || N <= 0 || K <= 0 || M >= (1LL << 31) ||
        N >= (1LL << 31) || K >= (1LL << 31)) return 5;
    if (M * K > (1LL << 31) || K * N > (1LL << 31) ||
        M * N > (1LL << 31)) return 5;
    int tileM = 0, tileN = 0, threads = 0;
    if (!tileLaunchConfig(name, tileM, tileN, threads)) return 5;

    const size_t sA = (size_t)M * K * 2;
    const size_t sB = (size_t)K * N * 2;
    const size_t sD = (size_t)M * N * 4;
    CUdeviceptr dA = 0, dB = 0, dD = 0;
    CUevent start = nullptr, stop = nullptr;
    int rc = 0;
    if (cuMemAlloc(&dA, sA) != CUDA_SUCCESS) return 3;
    if (cuMemAlloc(&dB, sB) != CUDA_SUCCESS) { cuMemFree(dA); return 3; }
    if (cuMemAlloc(&dD, sD) != CUDA_SUCCESS) {
        cuMemFree(dA); cuMemFree(dB); return 3;
    }
    do {
        if (cuMemcpyHtoD(dA, buffers[0], sA) != CUDA_SUCCESS ||
            cuMemcpyHtoD(dB, buffers[1], sB) != CUDA_SUCCESS) { rc = 3; break; }
        long long MArg = M, NArg = N, KArg = K;
        void* args[] = {&dA, &dB, &dD, &MArg, &NArg, &KArg};
        unsigned gx = (unsigned)((N + tileN - 1) / tileN);
        unsigned gy = (unsigned)((M + tileM - 1) / tileM);
        auto launch = [&]() {
            return cuLaunchKernel(fn, gx, gy, 1, (unsigned)threads, 1, 1,
                                  0, 0, args, 0);
        };
        for (int i = 0; i < warmup; ++i)
            if (launch() != CUDA_SUCCESS) { rc = 3; break; }
        if (rc) break;
        if (cuCtxSynchronize() != CUDA_SUCCESS ||
            cuEventCreate(&start, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
            cuEventCreate(&stop, CU_EVENT_DEFAULT) != CUDA_SUCCESS) {
            rc = 3; break;
        }
        if (cuEventRecord(start, 0) != CUDA_SUCCESS) { rc = 3; break; }
        for (int i = 0; i < repetitions; ++i)
            if (launch() != CUDA_SUCCESS) { rc = 3; break; }
        if (rc) break;
        if (cuEventRecord(stop, 0) != CUDA_SUCCESS ||
            cuEventSynchronize(stop) != CUDA_SUCCESS) { rc = 3; break; }
        float totalMs = 0.0f;
        if (cuEventElapsedTime(&totalMs, start, stop) != CUDA_SUCCESS) {
            rc = 3; break;
        }
        *latencyMs = totalMs / (float)repetitions;
    } while (0);
    if (start) cuEventDestroy(start);
    if (stop) cuEventDestroy(stop);
    cuMemFree(dA); cuMemFree(dB); cuMemFree(dD);
    return rc;
}

// Shared launch body behind both the direct C-ABI and the tsrGpuLauncherFn.
int invokeImpl(const char* kernel_name, void** buffers, size_t nbuf,
               const int64_t* dims, size_t ndim) {
    if (!kernel_name || !buffers || !dims) return 5;
    if (!ensureContext()) return 2;
    std::lock_guard<std::mutex> lock(g_mu);
    CUfunction fn = getFunctionLocked(kernel_name);
    if (fn == nullptr)
        return g_ptx.count(kernel_name) ? 3 : 4;   // JIT failure vs no PTX
    if (std::strcmp(kernel_name, kMmaEntry) == 0)
        return invokeMma(fn, buffers, nbuf, dims, ndim);
    if (std::strcmp(kernel_name, kGemmBf16) == 0 ||
        std::strcmp(kernel_name, kGemmF16) == 0)
        return invokeMmaGemm16(fn, buffers, nbuf, dims, ndim);
    if (std::strcmp(kernel_name, kTileDirectF16) == 0 ||
        std::strcmp(kernel_name, kTileDirectBf16) == 0)
        return invokeMmaGemm16(fn, buffers, nbuf, dims, ndim,
                               16, 8, 32, true, true, true);
    if (std::strcmp(kernel_name, kTileSharedF16) == 0 ||
        std::strcmp(kernel_name, kTileSharedBf16) == 0)
        return invokeMmaGemm16(fn, buffers, nbuf, dims, ndim,
                               32, 32, 128, true, true, true);
    return 5;                                        // unknown kernel ABI
}

// tsrGpuLauncherFn: the backend-agnostic seam. Maps a launch rc to a TsrStatus,
// declining (NOT_FOUND) for non-nvidia targets or unbridged kernels so the core
// runtime still reports honestly (Decision #21).
TsrStatus gpuLauncher(const char* target, const char* kernel_name,
                      const tsrGpuLaunchParams* p, void* /*user*/) {
    if (!target || std::strncmp(target, "nvidia", 6) != 0) return TSR_STATUS_NOT_FOUND;
    if (!kernel_name || !p) return TSR_STATUS_INVALID_ARGUMENT;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        if (!g_ptx.count(kernel_name)) return TSR_STATUS_NOT_FOUND;  // unbridged
    }
    int rc = invokeImpl(kernel_name, p->buffers, p->num_buffers,
                        p->dims, p->num_dims);
    switch (rc) {
        case 0: return TSR_STATUS_SUCCESS;
        case 4: return TSR_STATUS_NOT_FOUND;         // no PTX (race) — unbridged
        case 5: return TSR_STATUS_INVALID_ARGUMENT;  // bad shape / unknown ABI
        default: return TSR_STATUS_INTERNAL;         // 2 (no GPU) / 3 (device op)
    }
}

}  // namespace

extern "C" {

int tessera_nvidia_ptx_register(const char* kernel_name, const char* ptx) {
    if (!kernel_name || !ptx) return 1;
    std::lock_guard<std::mutex> lock(g_mu);
    g_ptx[kernel_name] = ptx;
    // Invalidate any cached module so a re-register recompiles the new PTX.
    auto m = g_modules.find(kernel_name);
    if (m != g_modules.end()) {
        cuModuleUnload(m->second);
        g_modules.erase(m);
        g_funcs.erase(kernel_name);
    }
    return 0;
}

int tessera_nvidia_ptx_invoke(const char* kernel_name, void** buffers,
                              size_t num_buffers, const int64_t* dims,
                              size_t num_dims) {
    return invokeImpl(kernel_name, buffers, num_buffers, dims, num_dims);
}

int tessera_nvidia_ptx_benchmark(const char* kernel_name, void** buffers,
                                 size_t num_buffers, const int64_t* dims,
                                 size_t num_dims, int warmup, int repetitions,
                                 float* latency_ms) {
    if (!kernel_name || !buffers || !dims || !latency_ms) return 5;
    if (!ensureContext()) return 2;
    std::lock_guard<std::mutex> lock(g_mu);
    CUfunction fn = getFunctionLocked(kernel_name);
    if (!fn) return g_ptx.count(kernel_name) ? 3 : 4;
    return benchmarkTileGemm16(fn, kernel_name, buffers, num_buffers, dims,
                               num_dims, warmup, repetitions, latency_ms);
}

int tessera_nvidia_register_ptx_launcher(void) {
    if (tsrRegisterGpuLauncher == nullptr) return 2;  // core runtime not in the load
    return tsrRegisterGpuLauncher(gpuLauncher, nullptr) == TSR_STATUS_SUCCESS ? 0 : 1;
}

}  // extern "C"
