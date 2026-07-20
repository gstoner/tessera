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
#include <initializer_list>
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
constexpr const char* kTileDirectTf32 = "tessera_tile_matmul_direct_tf32";
constexpr const char* kTileDirectE4m3 = "tessera_tile_matmul_direct_e4m3";
constexpr const char* kTileDirectE5m2 = "tessera_tile_matmul_direct_e5m2";
constexpr const char* kTileDirectS8 = "tessera_tile_matmul_direct_s8";
constexpr const char* kTileDirectF64 = "tessera_tile_matmul_direct_f64";
constexpr const char* kTileNvfp4 = "tessera_tile_matmul_nvfp4";
constexpr const char* kTileMxE2m3 = "tessera_tile_matmul_mx_e2m3";
constexpr const char* kTileMxE3m2 = "tessera_tile_matmul_mx_e3m2";
constexpr const char* kTileMxFp4 = "tessera_tile_matmul_mx_fp4_e2m1";
constexpr const char* kTileSoftmaxF16 = "tessera_tile_softmax_f16";
constexpr const char* kTileSoftmaxF32 = "tessera_tile_softmax_f32";
constexpr const char* kTileReduceSumF16 = "tessera_tile_reduce_sum_f16";
constexpr const char* kTileReduceMeanF16 = "tessera_tile_reduce_mean_f16";
constexpr const char* kTileReduceMaxF16 = "tessera_tile_reduce_max_f16";
constexpr const char* kTileReduceSumF32 = "tessera_tile_reduce_sum_f32";
constexpr const char* kTileReduceMeanF32 = "tessera_tile_reduce_mean_f32";
constexpr const char* kTileReduceMaxF32 = "tessera_tile_reduce_max_f32";
constexpr const char* kTileAttentionPrefix = "tessera_tile_attention_";
constexpr const char* kTileAttentionBackwardPrefix =
    "tessera_tile_attention_backward_";
constexpr const char* kTilePagedKV = "tessera_tile_paged_kv_read_f32_direct";
constexpr const char* kTilePagedAttentionPrefix = "tessera_tile_paged_attention_f32_fused_";
constexpr const char* kTileMoEDispatch = "tessera_tile_moe_dispatch_";
constexpr const char* kTileMoECombine = "tessera_tile_moe_combine_";
constexpr const char* kTileGroupedGemm = "tessera_tile_grouped_gemm_";

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
                    bool columnMajorGrid = false, bool dimensions64 = false,
                    size_t elementBytes = 2, size_t outputBytes = 4) {
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
    const size_t sA = (size_t)M * K * elementBytes;
    const size_t sB = (size_t)K * N * elementBytes;  // B is col-major
    const size_t sD = (size_t)M * N * outputBytes;
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

// Compiler-owned launch-level NVFP4 ABI: packed E2M1 A[M,ceil(K/2)] and
// B[ceil(K/2),N], logical UE4M3 scale views SFa[M,ceil(K/16)] and
// SFb[ceil(K/16),N], f32 D[M,N], and runtime i64 M/N/K.
int invokeNvfp4(CUfunction fn, void** buffers, size_t nbuf,
                const int64_t* dims, size_t ndim) {
    if (nbuf != 5 || ndim != 3) return 5;
    const long long M = dims[0], N = dims[1], K = dims[2];
    if (M <= 0 || N <= 0 || K <= 0 || M >= (1LL << 31) ||
        N >= (1LL << 31) || K >= (1LL << 31)) return 5;
    const size_t packedK = ((size_t)K + 1) / 2;
    const size_t scaleK = ((size_t)K + 15) / 16;
    if ((size_t)M > SIZE_MAX / packedK || packedK > SIZE_MAX / (size_t)N ||
        (size_t)M > SIZE_MAX / scaleK || scaleK > SIZE_MAX / (size_t)N ||
        (size_t)M > SIZE_MAX / (size_t)N / sizeof(float)) return 5;
    const size_t sizes[] = {
        (size_t)M * packedK, packedK * (size_t)N,
        (size_t)M * scaleK, scaleK * (size_t)N,
        (size_t)M * (size_t)N * sizeof(float),
    };
    CUdeviceptr device[5] = {};
    int rc = 0;
    for (int i = 0; i < 5; ++i) {
        if (cuMemAlloc(&device[i], sizes[i]) != CUDA_SUCCESS) {
            rc = 3;
            break;
        }
    }
    if (!rc) {
        for (int i = 0; i < 4; ++i)
            if (cuMemcpyHtoD(device[i], buffers[i], sizes[i]) != CUDA_SUCCESS) {
                rc = 3;
                break;
            }
    }
    if (!rc) {
        long long MArg = M, NArg = N, KArg = K;
        void* args[] = {&device[0], &device[1], &device[2], &device[3],
                        &device[4], &MArg, &NArg, &KArg};
        unsigned gx = (unsigned)((N + 7) / 8);
        unsigned gy = (unsigned)((M + 15) / 16);
        if (cuLaunchKernel(fn, gx, gy, 1, 32, 1, 1, 0, 0, args, 0) !=
                CUDA_SUCCESS || cuCtxSynchronize() != CUDA_SUCCESS ||
            cuMemcpyDtoH(buffers[4], device[4], sizes[4]) != CUDA_SUCCESS)
            rc = 3;
    }
    for (CUdeviceptr ptr : device)
        if (ptr) cuMemFree(ptr);
    return rc;
}

// OCP MX block-scaled ABI. FP6 uses one byte per logical value; MXFP4 packs
// two E2M1 nibbles per byte. Both use one UE8M0 scale per 32 logical values.
int invokeMx(CUfunction fn, void** buffers, size_t nbuf,
             const int64_t* dims, size_t ndim, bool packedFp4) {
    if (nbuf != 5 || ndim != 3) return 5;
    const long long M = dims[0], N = dims[1], K = dims[2];
    if (M <= 0 || N <= 0 || K <= 0 || M >= (1LL << 31) ||
        N >= (1LL << 31) || K >= (1LL << 31)) return 5;
    const size_t physicalK = packedFp4 ? ((size_t)K + 1) / 2 : (size_t)K;
    const size_t scaleK = ((size_t)K + 31) / 32;
    if ((size_t)M > SIZE_MAX / physicalK ||
        physicalK > SIZE_MAX / (size_t)N ||
        (size_t)M > SIZE_MAX / scaleK ||
        scaleK > SIZE_MAX / (size_t)N ||
        (size_t)M > SIZE_MAX / (size_t)N / sizeof(float)) return 5;
    const size_t sizes[] = {
        (size_t)M * physicalK, physicalK * (size_t)N,
        (size_t)M * scaleK, scaleK * (size_t)N,
        (size_t)M * (size_t)N * sizeof(float),
    };
    CUdeviceptr device[5] = {};
    int rc = 0;
    for (int i = 0; i < 5; ++i) {
        if (cuMemAlloc(&device[i], sizes[i]) != CUDA_SUCCESS) {
            rc = 3;
            break;
        }
    }
    if (!rc) {
        for (int i = 0; i < 4; ++i)
            if (cuMemcpyHtoD(device[i], buffers[i], sizes[i]) != CUDA_SUCCESS) {
                rc = 3;
                break;
            }
    }
    if (!rc) {
        long long MArg = M, NArg = N, KArg = K;
        void* args[] = {&device[0], &device[1], &device[2], &device[3],
                        &device[4], &MArg, &NArg, &KArg};
        const unsigned gx = (unsigned)((N + 7) / 8);
        const unsigned gy = (unsigned)((M + 15) / 16);
        if (cuLaunchKernel(fn, gx, gy, 1, 32, 1, 1, 0, 0, args, 0) !=
                CUDA_SUCCESS || cuCtxSynchronize() != CUDA_SUCCESS ||
            cuMemcpyDtoH(buffers[4], device[4], sizes[4]) != CUDA_SUCCESS)
            rc = 3;
    }
    for (CUdeviceptr ptr : device)
        if (ptr) cuMemFree(ptr);
    return rc;
}

int invokeFusedMatmul16(CUfunction fn, const char* name, void** buffers,
                        size_t nbuf, const int64_t* dims, size_t ndim) {
    if (ndim != 3) return 5;
    const bool hasBias = std::strstr(name, "_b1_r") != nullptr;
    const bool hasResidual = std::strstr(name, "_r1") != nullptr;
    const size_t expected = 3 + (hasBias ? 1 : 0) + (hasResidual ? 1 : 0);
    if (nbuf != expected) return 5;
    const long long M = dims[0], N = dims[1], K = dims[2];
    if (M <= 0 || N <= 0 || K <= 0 || M >= (1LL << 31) ||
        N >= (1LL << 31) || K >= (1LL << 31) ||
        M * K > (1LL << 31) || K * N > (1LL << 31) ||
        M * N > (1LL << 31)) return 5;
    const bool direct = std::strstr(name, "_tf32_") != nullptr ||
                        std::strstr(name, "_e4m3_") != nullptr ||
                        std::strstr(name, "_e5m2_") != nullptr;
    const size_t inputBytes = std::strstr(name, "_tf32_") ? 4 :
                              (direct ? 1 : 2);
    const size_t sizes[] = {
        (size_t)M * (size_t)K * inputBytes,
        (size_t)K * (size_t)N * inputBytes,
        hasBias ? (size_t)N * 4 : (hasResidual ? (size_t)M * (size_t)N * 4
                                               : (size_t)M * (size_t)N * 4),
        hasBias && hasResidual ? (size_t)M * (size_t)N * 4
                               : (size_t)M * (size_t)N * 4,
        (size_t)M * (size_t)N * 4,
    };
    CUdeviceptr device[5] = {};
    int rc = 0;
    for (size_t i = 0; i < nbuf; ++i) {
        size_t bytes = sizes[i];
        if (cuMemAlloc(&device[i], bytes) != CUDA_SUCCESS) { rc = 3; break; }
    }
    const size_t outputIndex = nbuf - 1;
    if (!rc) {
        for (size_t i = 0; i < outputIndex; ++i)
            if (cuMemcpyHtoD(device[i], buffers[i], sizes[i]) != CUDA_SUCCESS) {
                rc = 3;
                break;
            }
    }
    if (!rc) {
        long long MArg = M, NArg = N, KArg = K;
        void* args[8] = {};
        size_t arg = 0;
        for (size_t i = 0; i < nbuf; ++i) args[arg++] = &device[i];
        args[arg++] = &MArg; args[arg++] = &NArg; args[arg++] = &KArg;
        if (cuLaunchKernel(fn, (unsigned)((N + (direct ? 7 : 31)) / (direct ? 8 : 32)),
                           (unsigned)((M + (direct ? 15 : 31)) / (direct ? 16 : 32)), 1,
                           direct ? 32 : 128, 1, 1,
                           0, 0, args, 0) != CUDA_SUCCESS ||
            cuCtxSynchronize() != CUDA_SUCCESS ||
            cuMemcpyDtoH(buffers[outputIndex], device[outputIndex],
                         sizes[outputIndex]) != CUDA_SUCCESS)
            rc = 3;
    }
    for (CUdeviceptr ptr : device)
        if (ptr) cuMemFree(ptr);
    return rc;
}

// Compiler-owned stable row-softmax ABI: host f16/f32 X/O and flattened
// {rows, K}. The kernel maps 128 independent rows per CTA; each thread owns a
// complete row, matching the typed Tile schedule recorded in the descriptor.
int invokeSoftmax(CUfunction fn, void** buffers, size_t nbuf,
                  const int64_t* dims, size_t ndim, size_t elementBytes) {
    if (nbuf != 2 || ndim != 2) return 5;
    const long long rows = dims[0], K = dims[1];
    if (rows <= 0 || K <= 0 || rows >= (1LL << 31) || K >= (1LL << 31) ||
        rows > (1LL << 31) / K) return 5;
    const size_t elements = (size_t)rows * (size_t)K;
    if (elementBytes == 0 || elements > SIZE_MAX / elementBytes) return 5;
    const size_t bytes = elements * elementBytes;
    CUdeviceptr dx = 0, dout = 0;
    if (cuMemAlloc(&dx, bytes) != CUDA_SUCCESS) return 3;
    if (cuMemAlloc(&dout, bytes) != CUDA_SUCCESS) {
        cuMemFree(dx);
        return 3;
    }
    int rc = 0;
    do {
        if (cuMemcpyHtoD(dx, buffers[0], bytes) != CUDA_SUCCESS) {
            rc = 3;
            break;
        }
        long long rowsArg = rows, kArg = K;
        void* args[] = {&dx, &dout, &rowsArg, &kArg};
        unsigned grid = (unsigned)((rows + 127) / 128);
        if (cuLaunchKernel(fn, grid, 1, 1, 128, 1, 1, 0, 0, args, 0) !=
                CUDA_SUCCESS || cuCtxSynchronize() != CUDA_SUCCESS ||
            cuMemcpyDtoH(buffers[1], dout, bytes) != CUDA_SUCCESS)
            rc = 3;
    } while (0);
    cuMemFree(dx);
    cuMemFree(dout);
    return rc;
}

int invokeReduce(CUfunction fn, void** buffers, size_t nbuf,
                 const int64_t* dims, size_t ndim, size_t elementBytes,
                 bool cooperative) {
    if (nbuf != 2 || ndim != 3) return 5;
    const long long outer=dims[0], axis=dims[1], inner=dims[2];
    if (outer<=0||axis<=0||inner<=0||outer>=(1LL<<31)||axis>=(1LL<<31)||
        inner>=(1LL<<31)||outer>(1LL<<31)/axis||outer*axis>(1LL<<31)/inner) return 5;
    const size_t outputs=(size_t)outer*(size_t)inner;
    const size_t inputBytes=outputs*(size_t)axis*elementBytes;
    const size_t outputBytes=outputs*sizeof(float);
    CUdeviceptr dx = 0, dout = 0;
    if (cuMemAlloc(&dx, inputBytes) != CUDA_SUCCESS) return 3;
    if (cuMemAlloc(&dout, outputBytes) != CUDA_SUCCESS) {
        cuMemFree(dx);
        return 3;
    }
    int rc = 0;
    do {
        if (cuMemcpyHtoD(dx, buffers[0], inputBytes) != CUDA_SUCCESS) {
            rc = 3;
            break;
        }
        long long outerArg=outer,axisArg=axis,innerArg=inner;
        void* args[] = {&dx,&dout,&outerArg,&axisArg,&innerArg};
        unsigned grid=cooperative?(unsigned)outputs:(unsigned)((outputs+127)/128);
        if (cuLaunchKernel(fn, grid, 1, 1, 128, 1, 1, 0, 0, args, 0) !=
                CUDA_SUCCESS || cuCtxSynchronize() != CUDA_SUCCESS ||
            cuMemcpyDtoH(buffers[1], dout, outputBytes) != CUDA_SUCCESS)
            rc = 3;
    } while (0);
    cuMemFree(dx);
    cuMemFree(dout);
    return rc;
}

int invokeMoe(CUfunction fn, const char* name, void** buffers, size_t nbuf,
              const int64_t* dims, size_t ndim) {
    const bool dispatch = std::strncmp(name,kTileMoEDispatch,std::strlen(kTileMoEDispatch)) == 0;
    const bool combine = std::strncmp(name,kTileMoECombine,std::strlen(kTileMoECombine)) == 0;
    const bool grouped = std::strncmp(name,kTileGroupedGemm,std::strlen(kTileGroupedGemm)) == 0;
    const size_t elementBytes = (std::strstr(name,"_f16") ||
                                 std::strstr(name,"_bf16")) ? 2 : 4;
    if ((!dispatch && !combine && !grouped) ||
        (grouped ? (nbuf != 4 || ndim != 4)
                 : (nbuf != (dispatch ? 3u : 4u) || ndim != 3))) return 5;
    for (size_t i = 0; i < ndim; ++i)
        if (dims[i] <= 0 || dims[i] >= (1LL << 31)) return 5;
    size_t sizes[4] = {};
    unsigned threads = 256, grid = 0;
    if (dispatch) {
        const size_t T=dims[0], S=dims[1], H=dims[2];
        if (T > SIZE_MAX/H/elementBytes || S > SIZE_MAX/H/elementBytes) return 5;
        sizes[0]=T*H*elementBytes; sizes[1]=S*4; sizes[2]=S*H*elementBytes;
        grid=(unsigned)((S*H+threads-1)/threads);
    } else if (combine) {
        const size_t T=dims[0], S=dims[1], H=dims[2];
        if (T > SIZE_MAX/H/elementBytes || S > SIZE_MAX/H/elementBytes) return 5;
        sizes[0]=S*H*elementBytes; sizes[1]=S*4; sizes[2]=S*4; sizes[3]=T*H*elementBytes;
        grid=(unsigned)((T*H+threads-1)/threads);
    } else {
        const size_t T=dims[0], K=dims[1], N=dims[2], E=dims[3];
        if (T > SIZE_MAX/K/elementBytes || E > SIZE_MAX/K ||
            E*K > SIZE_MAX/N/elementBytes || T > SIZE_MAX/N/elementBytes || E == SIZE_MAX) return 5;
        sizes[0]=T*K*elementBytes; sizes[1]=E*K*N*elementBytes;
        sizes[2]=(E+1)*4; sizes[3]=T*N*elementBytes;
        grid=(unsigned)((T*N+threads-1)/threads);
    }
    CUdeviceptr device[4]={}; int rc=0;
    for(size_t i=0;i<nbuf;++i)
        if(cuMemAlloc(&device[i],sizes[i])!=CUDA_SUCCESS){rc=3;break;}
    const size_t outputIndex=nbuf-1;
    for(size_t i=0;!rc&&i<outputIndex;++i)
        if(cuMemcpyHtoD(device[i],buffers[i],sizes[i])!=CUDA_SUCCESS) rc=3;
    long long args64[4]={};
    for(size_t i=0;i<ndim;++i) args64[i]=dims[i];
    void* args[8]={}; size_t arg=0;
    for(size_t i=0;i<nbuf;++i) args[arg++]=&device[i];
    for(size_t i=0;i<ndim;++i) args[arg++]=&args64[i];
    if(!rc&&(cuLaunchKernel(fn,grid,1,1,threads,1,1,0,0,args,0)!=CUDA_SUCCESS||
             cuCtxSynchronize()!=CUDA_SUCCESS||
             cuMemcpyDtoH(buffers[outputIndex],device[outputIndex],sizes[outputIndex])!=CUDA_SUCCESS)) rc=3;
    for(CUdeviceptr ptr:device) if(ptr) cuMemFree(ptr);
    return rc;
}

int invokeAttention(CUfunction fn, const char* name, void** buffers,
                    size_t nbuf, const int64_t* dims, size_t ndim) {
    if ((nbuf != 4 && nbuf != 5) || ndim != 7 || !name) return 5;
    const bool hasBias = nbuf == 5;
    const size_t outputIndex = hasBias ? 4 : 3;
    for (size_t i = 0; i < ndim; ++i)
        if (dims[i] <= 0 || dims[i] >= (1LL << 31)) return 5;
    const size_t B = (size_t)dims[0], Hq = (size_t)dims[1];
    const size_t Hkv = (size_t)dims[2], Sq = (size_t)dims[3];
    const size_t Sk = (size_t)dims[4], D = (size_t)dims[5], Dv = (size_t)dims[6];
    if (Hq % Hkv) return 5;
    auto product = [](std::initializer_list<size_t> values, size_t& out) {
        out = 1;
        for (size_t value : values) {
            if (value && out > SIZE_MAX / value) return false;
            out *= value;
        }
        return true;
    };
    size_t qElements, kElements, vElements, oElements, biasElements = 0;
    if (!product({B, Hq, Sq, D}, qElements) ||
        !product({B, Hkv, Sk, D}, kElements) ||
        !product({B, Hkv, Sk, Dv}, vElements) ||
        !product({B, Hq, Sq, Dv}, oElements) ||
        (hasBias && !product({B, Hq, Sq, Sk}, biasElements))) return 5;
    const bool f16 = std::strncmp(name, "tessera_tile_attention_f16_", 27) == 0;
    const size_t elementBytes = f16 ? 2 : 4;
    if (qElements > SIZE_MAX / elementBytes || kElements > SIZE_MAX / elementBytes ||
        vElements > SIZE_MAX / elementBytes || oElements > SIZE_MAX / sizeof(float) ||
        (hasBias && biasElements > SIZE_MAX / sizeof(float))) return 5;
    size_t sizes[5] = {qElements * elementBytes, kElements * elementBytes,
                       vElements * elementBytes, 0, 0};
    if (hasBias) sizes[3] = biasElements * sizeof(float);
    sizes[outputIndex] = oElements * sizeof(float);
    CUdeviceptr device[5] = {};
    int rc = 0;
    for (size_t i = 0; i < nbuf; ++i)
        if (cuMemAlloc(&device[i], sizes[i]) != CUDA_SUCCESS) { rc = 3; break; }
    if (!rc)
        for (size_t i = 0; i < outputIndex; ++i)
            if (cuMemcpyHtoD(device[i], buffers[i], sizes[i]) != CUDA_SUCCESS) { rc = 3; break; }
    if (!rc) {
        long long args64[7];
        for (int i = 0; i < 7; ++i) args64[i] = dims[i];
        void* args[12] = {};
        size_t arg = 0;
        for (size_t i = 0; i < nbuf; ++i) args[arg++] = &device[i];
        for (int i = 0; i < 7; ++i) args[arg++] = &args64[i];
        unsigned grid = (unsigned)((oElements + 127) / 128);
        if (grid == 0 || grid > 0x7fffffffU ||
            cuLaunchKernel(fn, grid, 1, 1, 128, 1, 1, 0, 0, args, 0) != CUDA_SUCCESS ||
            cuCtxSynchronize() != CUDA_SUCCESS ||
            cuMemcpyDtoH(buffers[outputIndex], device[outputIndex],
                         sizes[outputIndex]) != CUDA_SUCCESS)
            rc = 3;
    }
    for (CUdeviceptr ptr : device) if (ptr) cuMemFree(ptr);
    return rc;
}

int invokePagedKV(CUfunction fn, void** buffers, size_t nbuf,
                  const int64_t* dims, size_t ndim) {
    if (nbuf != 3 || ndim != 7) return 5;
    const long long P=dims[0], LP=dims[1], PS=dims[2], H=dims[3], D=dims[4];
    const long long start=dims[5], tokens=dims[6];
    if (P<=0 || LP<=0 || PS<=0 || H<=0 || D<=0 || start<0 || tokens<=0 ||
        start+tokens > LP*PS) return 5;
    size_t pages=(size_t)P*PS*H*D*4, table=(size_t)LP*4;
    size_t output=(size_t)tokens*H*D*4;
    CUdeviceptr device[3] = {};
    size_t sizes[3] = {pages, table, output};
    int rc = 0;
    for (int i=0;i<3;++i) if (cuMemAlloc(&device[i], sizes[i]) != CUDA_SUCCESS) { rc=3; break; }
    if (!rc && (cuMemcpyHtoD(device[0], buffers[0], pages) != CUDA_SUCCESS ||
                cuMemcpyHtoD(device[1], buffers[1], table) != CUDA_SUCCESS)) rc=3;
    if (!rc) {
        long long args64[7]; for(int i=0;i<7;++i) args64[i]=dims[i];
        void* args[] = {&device[0],&device[1],&device[2],&args64[0],&args64[1],
                        &args64[2],&args64[3],&args64[4],&args64[5],&args64[6]};
        size_t count=(size_t)tokens*H*D;
        if (cuLaunchKernel(fn,(unsigned)((count+255)/256),1,1,256,1,1,0,0,args,0)!=CUDA_SUCCESS ||
            cuCtxSynchronize()!=CUDA_SUCCESS ||
            cuMemcpyDtoH(buffers[2],device[2],output)!=CUDA_SUCCESS) rc=3;
    }
    for(CUdeviceptr ptr:device) if(ptr) cuMemFree(ptr);
    return rc;
}

int invokePagedAttention(CUfunction fn, void** buffers, size_t nbuf,
                         const int64_t* dims, size_t ndim) {
    if (nbuf != 6 || ndim != 8) return 5;
    const long long P=dims[0], LP=dims[1], PS=dims[2], H=dims[3];
    const long long Q=dims[4], T=dims[5], D=dims[6], offset=dims[7];
    if (P<=0 || LP<=0 || PS<=0 || H<=0 || Q<=0 || T<=0 || D<=0 ||
        offset<0 || T>LP*PS || offset+Q>T) return 5;
    const int* table = static_cast<const int*>(buffers[3]);
    const long long* indices = static_cast<const long long*>(buffers[4]);
    for (long long i=0;i<LP;++i) if (table[i]<0 || table[i]>=P) return 5;
    for (long long i=0;i<T;++i) if (indices[i]<0 || indices[i]>=LP*PS) return 5;
    auto checkedBytes=[](std::initializer_list<size_t> extents,size_t width,size_t& out) {
        out=width; for(size_t extent:extents){if(extent && out>SIZE_MAX/extent)return false;out*=extent;} return true;
    };
    size_t qBytes=0,pageBytes=0,tableBytes=0,indexBytes=0,outBytes=0;
    if(!checkedBytes({(size_t)H,(size_t)Q,(size_t)D},4,qBytes) ||
       !checkedBytes({(size_t)P,(size_t)PS,(size_t)H,(size_t)D},4,pageBytes) ||
       !checkedBytes({(size_t)LP},4,tableBytes) ||
       !checkedBytes({(size_t)T},8,indexBytes) ||
       !checkedBytes({(size_t)H,(size_t)Q,(size_t)D},4,outBytes)) return 5;
    size_t sizes[6]={qBytes,pageBytes,pageBytes,tableBytes,indexBytes,outBytes};
    CUdeviceptr device[6]={}; int rc=0;
    for(int i=0;i<6;++i) if(cuMemAlloc(&device[i],sizes[i])!=CUDA_SUCCESS){rc=3;break;}
    if(!rc) for(int i=0;i<5;++i)
        if(cuMemcpyHtoD(device[i],buffers[i],sizes[i])!=CUDA_SUCCESS){rc=3;break;}
    if(!rc){
        long long args64[8]; for(int i=0;i<8;++i)args64[i]=dims[i];
        void* args[14]; size_t arg=0;
        for(int i=0;i<6;++i)args[arg++]=&device[i];
        for(int i=0;i<8;++i)args[arg++]=&args64[i];
        size_t count=(size_t)H*Q*D;
        if(count==0 || count>(size_t)0x7fffffffU*128 ||
           cuLaunchKernel(fn,(unsigned)((count+127)/128),1,1,128,1,1,0,0,args,0)!=CUDA_SUCCESS ||
           cuCtxSynchronize()!=CUDA_SUCCESS ||
           cuMemcpyDtoH(buffers[5],device[5],outBytes)!=CUDA_SUCCESS) rc=3;
    }
    for(CUdeviceptr ptr:device)if(ptr)cuMemFree(ptr);
    return rc;
}

int invokeAttentionBackward(CUfunction fn, const char* kernelName,
                            void** buffers, size_t nbuf,
                            const int64_t* dims, size_t ndim) {
    if ((nbuf != 7 && nbuf != 8) || ndim != 7) return 5;
    const bool hasBias = nbuf == 8;
    const size_t outputBase = hasBias ? 5 : 4;
    const long long B=dims[0], Hq=dims[1], Hkv=dims[2], Sq=dims[3];
    const long long Sk=dims[4], D=dims[5], Dv=dims[6];
    if (B<=0 || Hq<=0 || Hkv<=0 || Sq<=0 || Sk<=0 || D<=0 || Dv<=0 ||
        Hq%Hkv) return 5;
    const bool f16 = std::strstr(kernelName, "attention_backward_f16_") != nullptr;
    const size_t elementBytes = f16 ? 2 : 4;
    auto bytes = [](std::initializer_list<size_t> values, size_t width, size_t& out) {
        out = width;
        for (size_t value : values) {
            if (value && out > SIZE_MAX / value) return false;
            out *= value;
        }
        return true;
    };
    size_t doBytes=0, qBytes=0, kBytes=0, vBytes=0, biasBytes=0;
    if (!bytes({(size_t)B,(size_t)Hq,(size_t)Sq,(size_t)Dv},elementBytes,doBytes) ||
        !bytes({(size_t)B,(size_t)Hq,(size_t)Sq,(size_t)D},elementBytes,qBytes) ||
        !bytes({(size_t)B,(size_t)Hkv,(size_t)Sk,(size_t)D},elementBytes,kBytes) ||
        !bytes({(size_t)B,(size_t)Hkv,(size_t)Sk,(size_t)Dv},elementBytes,vBytes) ||
        (hasBias && !bytes({(size_t)B,(size_t)Hq,(size_t)Sq,(size_t)Sk},4,biasBytes)))
        return 5;
    size_t sizes[8] = {doBytes,qBytes,kBytes,vBytes,0,0,0,0};
    if (hasBias) sizes[4] = biasBytes;
    sizes[outputBase] = qBytes;
    sizes[outputBase+1] = kBytes;
    sizes[outputBase+2] = vBytes;
    CUdeviceptr device[8] = {};
    int rc = 0;
    for (size_t i=0;i<nbuf;++i)
        if (cuMemAlloc(&device[i], sizes[i]) != CUDA_SUCCESS) { rc=3; break; }
    if (!rc)
        for (size_t i=0;i<outputBase;++i)
            if (cuMemcpyHtoD(device[i],buffers[i],sizes[i]) != CUDA_SUCCESS) { rc=3; break; }
    if (!rc) {
        long long args64[7]; for(int i=0;i<7;++i) args64[i]=dims[i];
        void* args[15] = {};
        size_t arg=0;
        for(size_t i=0;i<nbuf;++i) args[arg++]=&device[i];
        for(int i=0;i<7;++i) args[arg++]=&args64[i];
        size_t elements=qBytes/elementBytes+kBytes/elementBytes+vBytes/elementBytes;
        if (elements==0 || elements > (size_t)0x7fffffffU*128 ||
            cuLaunchKernel(fn,(unsigned)((elements+127)/128),1,1,128,1,1,0,0,args,0)!=CUDA_SUCCESS ||
            cuCtxSynchronize()!=CUDA_SUCCESS) rc=3;
        for(size_t i=outputBase;!rc && i<nbuf;++i)
            if(cuMemcpyDtoH(buffers[i],device[i],sizes[i])!=CUDA_SUCCESS) rc=3;
    }
    for(CUdeviceptr ptr:device) if(ptr) cuMemFree(ptr);
    return rc;
}

bool tileLaunchConfig(const char* name, int& tileM, int& tileN, int& threads) {
    if (std::strcmp(name, kTileDirectF16) == 0 ||
        std::strcmp(name, kTileDirectBf16) == 0 ||
        std::strcmp(name, kTileDirectTf32) == 0 ||
        std::strcmp(name, kTileDirectE4m3) == 0 ||
        std::strcmp(name, kTileDirectE5m2) == 0 ||
        std::strcmp(name, kTileDirectS8) == 0 ||
        std::strcmp(name, kTileDirectF64) == 0) {
        tileM = std::strcmp(name, kTileDirectF64) == 0 ? 8 : 16;
        tileN = 8; threads = 32;
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

    size_t elementBytes = 2;
    if (std::strcmp(name, kTileDirectTf32) == 0) elementBytes = 4;
    if (std::strcmp(name, kTileDirectF64) == 0) elementBytes = 8;
    if (std::strcmp(name, kTileDirectE4m3) == 0 ||
        std::strcmp(name, kTileDirectE5m2) == 0 ||
        std::strcmp(name, kTileDirectS8) == 0) elementBytes = 1;
    const size_t sA = (size_t)M * K * elementBytes;
    const size_t sB = (size_t)K * N * elementBytes;
    const size_t outputBytes =
        std::strcmp(name, kTileDirectF64) == 0 ? 8 : 4;
    const size_t sD = (size_t)M * N * outputBytes;
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

int benchmarkMx(CUfunction fn, const char* name, void** buffers,
                size_t nbuf, const int64_t* dims, size_t ndim,
                int warmup, int repetitions, float* latencyMs) {
    if (nbuf != 5 || ndim != 3 || !latencyMs || warmup < 0 || repetitions <= 0)
        return 5;
    const long long M = dims[0], N = dims[1], K = dims[2];
    if (M <= 0 || N <= 0 || K <= 0 || M >= (1LL << 31) ||
        N >= (1LL << 31) || K >= (1LL << 31)) return 5;
    const bool packedFp4 = std::strcmp(name, kTileMxFp4) == 0;
    if (!packedFp4 && std::strcmp(name, kTileMxE2m3) != 0 &&
        std::strcmp(name, kTileMxE3m2) != 0) return 5;
    const size_t physicalK = packedFp4 ? ((size_t)K + 1) / 2 : (size_t)K;
    const size_t scaleK = ((size_t)K + 31) / 32;
    if ((size_t)M > SIZE_MAX / physicalK ||
        physicalK > SIZE_MAX / (size_t)N ||
        (size_t)M > SIZE_MAX / scaleK ||
        scaleK > SIZE_MAX / (size_t)N ||
        (size_t)M > SIZE_MAX / (size_t)N / sizeof(float)) return 5;
    const size_t sizes[] = {
        (size_t)M * physicalK, physicalK * (size_t)N,
        (size_t)M * scaleK, scaleK * (size_t)N,
        (size_t)M * (size_t)N * sizeof(float),
    };
    CUdeviceptr device[5] = {};
    CUevent start = nullptr, stop = nullptr;
    int rc = 0;
    for (int i = 0; i < 5; ++i) {
        if (cuMemAlloc(&device[i], sizes[i]) != CUDA_SUCCESS) {
            rc = 3;
            break;
        }
    }
    if (!rc) {
        for (int i = 0; i < 4; ++i)
            if (cuMemcpyHtoD(device[i], buffers[i], sizes[i]) != CUDA_SUCCESS) {
                rc = 3;
                break;
            }
    }
    if (!rc) {
        long long MArg = M, NArg = N, KArg = K;
        void* args[] = {&device[0], &device[1], &device[2], &device[3],
                        &device[4], &MArg, &NArg, &KArg};
        const unsigned gx = (unsigned)((N + 7) / 8);
        const unsigned gy = (unsigned)((M + 15) / 16);
        auto launch = [&]() {
            return cuLaunchKernel(fn, gx, gy, 1, 32, 1, 1, 0, 0, args, 0);
        };
        for (int i = 0; i < warmup; ++i)
            if (launch() != CUDA_SUCCESS) { rc = 3; break; }
        if (!rc && (cuCtxSynchronize() != CUDA_SUCCESS ||
                    cuEventCreate(&start, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
                    cuEventCreate(&stop, CU_EVENT_DEFAULT) != CUDA_SUCCESS))
            rc = 3;
        if (!rc && cuEventRecord(start, 0) != CUDA_SUCCESS) rc = 3;
        for (int i = 0; !rc && i < repetitions; ++i)
            if (launch() != CUDA_SUCCESS) rc = 3;
        if (!rc && (cuEventRecord(stop, 0) != CUDA_SUCCESS ||
                    cuEventSynchronize(stop) != CUDA_SUCCESS)) rc = 3;
        float totalMs = 0.0f;
        if (!rc && cuEventElapsedTime(&totalMs, start, stop) != CUDA_SUCCESS)
            rc = 3;
        if (!rc) *latencyMs = totalMs / (float)repetitions;
    }
    if (start) cuEventDestroy(start);
    if (stop) cuEventDestroy(stop);
    for (CUdeviceptr ptr : device)
        if (ptr) cuMemFree(ptr);
    return rc;
}

int benchmarkUnary(CUfunction fn, const char* name, void** buffers,
                   size_t nbuf, const int64_t* dims, size_t ndim,
                   int warmup, int repetitions, float* latencyMs) {
    if (nbuf != 2 || !latencyMs || warmup < 0 || repetitions <= 0)
        return 5;
    const bool softmax = std::strcmp(name, kTileSoftmaxF16) == 0 ||
        std::strcmp(name, kTileSoftmaxF32) == 0;
    if ((softmax && ndim != 2) || (!softmax && ndim != 3)) return 5;
    const bool f16 = std::strcmp(name, kTileSoftmaxF16) == 0 ||
        std::strstr(name, "_f16_") != nullptr;
    long long outer=dims[0],axis=dims[1],inner=softmax?1:dims[2];
    if(outer<=0||axis<=0||inner<=0||outer>=(1LL<<31)||axis>=(1LL<<31)||
       inner>=(1LL<<31)||outer>(1LL<<31)/axis||outer*axis>(1LL<<31)/inner) return 5;
    const size_t outputs=(size_t)outer*(size_t)inner;
    const size_t inputBytes=outputs*(size_t)axis*(f16?2:4);
    const size_t outputBytes=softmax?inputBytes:outputs*4;
    CUdeviceptr dx = 0, dout = 0;
    CUevent start = nullptr, stop = nullptr;
    int rc = 0;
    if (cuMemAlloc(&dx, inputBytes) != CUDA_SUCCESS) return 3;
    if (cuMemAlloc(&dout, outputBytes) != CUDA_SUCCESS) {
        cuMemFree(dx);
        return 3;
    }
    do {
        if (cuMemcpyHtoD(dx, buffers[0], inputBytes) != CUDA_SUCCESS) {
            rc = 3;
            break;
        }
        long long outerArg=outer,axisArg=axis,innerArg=inner;
        void* softmaxArgs[]={&dx,&dout,&outerArg,&axisArg};
        void* reduceArgs[]={&dx,&dout,&outerArg,&axisArg,&innerArg};
        void** args=softmax?softmaxArgs:reduceArgs;
        const bool cooperative=!softmax&&std::strstr(name,"_cooperative_128")!=nullptr;
        const unsigned grid=cooperative?(unsigned)outputs:(unsigned)((outputs+127)/128);
        auto launch = [&]() {
            return cuLaunchKernel(fn, grid, 1, 1, 128, 1, 1, 0, 0, args, 0);
        };
        for (int i = 0; i < warmup; ++i)
            if (launch() != CUDA_SUCCESS) { rc = 3; break; }
        if (rc) break;
        if (cuCtxSynchronize() != CUDA_SUCCESS ||
            cuEventCreate(&start, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
            cuEventCreate(&stop, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
            cuEventRecord(start, 0) != CUDA_SUCCESS) { rc = 3; break; }
        for (int i = 0; i < repetitions; ++i)
            if (launch() != CUDA_SUCCESS) { rc = 3; break; }
        if (rc || cuEventRecord(stop, 0) != CUDA_SUCCESS ||
            cuEventSynchronize(stop) != CUDA_SUCCESS) { rc = 3; break; }
        float totalMs = 0.0f;
        if (cuEventElapsedTime(&totalMs, start, stop) != CUDA_SUCCESS) {
            rc = 3;
            break;
        }
        *latencyMs = totalMs / (float)repetitions;
    } while (0);
    if (start) cuEventDestroy(start);
    if (stop) cuEventDestroy(stop);
    cuMemFree(dx);
    cuMemFree(dout);
    return rc;
}

int benchmarkAttention(CUfunction fn, const char* name, void** buffers,
                       size_t nbuf, const int64_t* dims, size_t ndim,
                       int warmup, int repetitions, float* latencyMs) {
    if ((nbuf != 4 && nbuf != 5) || ndim != 7 || !name || !latencyMs ||
        warmup < 0 || repetitions <= 0)
        return 5;
    const bool hasBias = nbuf == 5;
    const size_t outputIndex = hasBias ? 4 : 3;
    for (size_t i = 0; i < ndim; ++i)
        if (dims[i] <= 0 || dims[i] >= (1LL << 31)) return 5;
    const size_t B = (size_t)dims[0], Hq = (size_t)dims[1], Hkv = (size_t)dims[2];
    const size_t Sq = (size_t)dims[3], Sk = (size_t)dims[4];
    const size_t D = (size_t)dims[5], Dv = (size_t)dims[6];
    if (Hq % Hkv) return 5;
    auto product = [](std::initializer_list<size_t> values, size_t& out) {
        out = 1;
        for (size_t value : values) {
            if (value && out > SIZE_MAX / value) return false;
            out *= value;
        }
        return true;
    };
    size_t counts[5] = {};
    if (!product({B, Hq, Sq, D}, counts[0]) ||
        !product({B, Hkv, Sk, D}, counts[1]) ||
        !product({B, Hkv, Sk, Dv}, counts[2]) ||
        !product({B, Hq, Sq, Dv}, counts[outputIndex]) ||
        (hasBias && !product({B, Hq, Sq, Sk}, counts[3]))) return 5;
    const bool f16 = std::strncmp(name, "tessera_tile_attention_f16_", 27) == 0;
    const size_t elementBytes = f16 ? 2 : 4;
    size_t sizes[5] = {};
    for (int i = 0; i < 3; ++i) {
        if (counts[i] > SIZE_MAX / elementBytes) return 5;
        sizes[i] = counts[i] * elementBytes;
    }
    if (hasBias && counts[3] > SIZE_MAX / sizeof(float)) return 5;
    if (hasBias) sizes[3] = counts[3] * sizeof(float);
    if (counts[outputIndex] > SIZE_MAX / sizeof(float)) return 5;
    sizes[outputIndex] = counts[outputIndex] * sizeof(float);
    CUdeviceptr device[5] = {};
    CUevent start = nullptr, stop = nullptr;
    int rc = 0;
    for (size_t i = 0; i < nbuf; ++i)
        if (cuMemAlloc(&device[i], sizes[i]) != CUDA_SUCCESS) { rc = 3; break; }
    if (!rc)
        for (size_t i = 0; i < outputIndex; ++i)
            if (cuMemcpyHtoD(device[i], buffers[i], sizes[i]) != CUDA_SUCCESS) { rc = 3; break; }
    if (!rc) {
        long long args64[7];
        for (int i = 0; i < 7; ++i) args64[i] = dims[i];
        void* args[12] = {};
        size_t arg = 0;
        for (size_t i = 0; i < nbuf; ++i) args[arg++] = &device[i];
        for (int i = 0; i < 7; ++i) args[arg++] = &args64[i];
        unsigned grid = (unsigned)((counts[outputIndex] + 127) / 128);
        auto launch = [&]() {
            return cuLaunchKernel(fn, grid, 1, 1, 128, 1, 1, 0, 0, args, 0);
        };
        for (int i = 0; i < warmup; ++i)
            if (launch() != CUDA_SUCCESS) { rc = 3; break; }
        if (!rc && (cuCtxSynchronize() != CUDA_SUCCESS ||
                    cuEventCreate(&start, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
                    cuEventCreate(&stop, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
                    cuEventRecord(start, 0) != CUDA_SUCCESS)) rc = 3;
        for (int i = 0; !rc && i < repetitions; ++i)
            if (launch() != CUDA_SUCCESS) rc = 3;
        if (!rc && (cuEventRecord(stop, 0) != CUDA_SUCCESS ||
                    cuEventSynchronize(stop) != CUDA_SUCCESS)) rc = 3;
        float totalMs = 0.0f;
        if (!rc && cuEventElapsedTime(&totalMs, start, stop) != CUDA_SUCCESS) rc = 3;
        if (!rc) *latencyMs = totalMs / (float)repetitions;
    }
    if (start) cuEventDestroy(start);
    if (stop) cuEventDestroy(stop);
    for (CUdeviceptr ptr : device) if (ptr) cuMemFree(ptr);
    return rc;
}

int benchmarkFusedMatmul16(CUfunction fn, const char* name, void** buffers,
                           size_t nbuf, const int64_t* dims, size_t ndim,
                           int warmup, int repetitions, float* latencyMs) {
    if (ndim != 3 || !latencyMs || warmup < 0 || repetitions <= 0) return 5;
    const bool hasBias = std::strstr(name, "_b1_r") != nullptr;
    const bool hasResidual = std::strstr(name, "_r1") != nullptr;
    const size_t expected = 3 + (hasBias ? 1 : 0) + (hasResidual ? 1 : 0);
    if (nbuf != expected) return 5;
    const long long M = dims[0], N = dims[1], K = dims[2];
    if (M <= 0 || N <= 0 || K <= 0 || M >= (1LL << 31) ||
        N >= (1LL << 31) || K >= (1LL << 31)) return 5;
    const bool direct = std::strstr(name, "_tf32_") != nullptr ||
                        std::strstr(name, "_e4m3_") != nullptr ||
                        std::strstr(name, "_e5m2_") != nullptr;
    const size_t inputBytes = std::strstr(name, "_tf32_") ? 4 :
                              (direct ? 1 : 2);
    size_t sizes[5] = {(size_t)M * (size_t)K * inputBytes,
                       (size_t)K * (size_t)N * inputBytes, 0, 0, 0};
    size_t index = 2;
    if (hasBias) sizes[index++] = (size_t)N * 4;
    if (hasResidual) sizes[index++] = (size_t)M * (size_t)N * 4;
    sizes[index++] = (size_t)M * (size_t)N * 4;
    CUdeviceptr device[5] = {};
    CUevent start = nullptr, stop = nullptr;
    int rc = 0;
    for (size_t i = 0; i < nbuf; ++i)
        if (cuMemAlloc(&device[i], sizes[i]) != CUDA_SUCCESS) { rc = 3; break; }
    const size_t outputIndex = nbuf - 1;
    if (!rc)
        for (size_t i = 0; i < outputIndex; ++i)
            if (cuMemcpyHtoD(device[i], buffers[i], sizes[i]) != CUDA_SUCCESS) {
                rc = 3;
                break;
            }
    if (!rc) {
        long long MArg = M, NArg = N, KArg = K;
        void* args[8] = {};
        size_t arg = 0;
        for (size_t i = 0; i < nbuf; ++i) args[arg++] = &device[i];
        args[arg++] = &MArg; args[arg++] = &NArg; args[arg++] = &KArg;
        auto launch = [&]() {
            return cuLaunchKernel(fn,
                                  (unsigned)((N + (direct ? 7 : 31)) / (direct ? 8 : 32)),
                                  (unsigned)((M + (direct ? 15 : 31)) / (direct ? 16 : 32)),
                                  1, direct ? 32 : 128, 1, 1,
                                  0, 0, args, 0);
        };
        for (int i = 0; i < warmup; ++i)
            if (launch() != CUDA_SUCCESS) { rc = 3; break; }
        if (!rc && (cuCtxSynchronize() != CUDA_SUCCESS ||
                    cuEventCreate(&start, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
                    cuEventCreate(&stop, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
                    cuEventRecord(start, 0) != CUDA_SUCCESS)) rc = 3;
        for (int i = 0; !rc && i < repetitions; ++i)
            if (launch() != CUDA_SUCCESS) rc = 3;
        if (!rc && (cuEventRecord(stop, 0) != CUDA_SUCCESS ||
                    cuEventSynchronize(stop) != CUDA_SUCCESS)) rc = 3;
        float totalMs = 0.0f;
        if (!rc && cuEventElapsedTime(&totalMs, start, stop) != CUDA_SUCCESS)
            rc = 3;
        if (!rc) *latencyMs = totalMs / (float)repetitions;
    }
    if (start) cuEventDestroy(start);
    if (stop) cuEventDestroy(stop);
    for (CUdeviceptr ptr : device)
        if (ptr) cuMemFree(ptr);
    return rc;
}

int benchmarkPagedKV(CUfunction fn, void** buffers, size_t nbuf,
                     const int64_t* dims, size_t ndim, int warmup,
                     int repetitions, float* latencyMs) {
    if (nbuf!=3 || ndim!=7 || !latencyMs || warmup<0 || repetitions<=0) return 5;
    const long long P=dims[0],LP=dims[1],PS=dims[2],H=dims[3],D=dims[4];
    const long long startToken=dims[5],tokens=dims[6];
    if(P<=0||LP<=0||PS<=0||H<=0||D<=0||startToken<0||tokens<=0||
       startToken+tokens>LP*PS) return 5;
    size_t pages=(size_t)P*PS*H*D*4,table=(size_t)LP*4,output=(size_t)tokens*H*D*4;
    CUdeviceptr device[3]={}; int rc=0; CUevent start=nullptr,stop=nullptr;
    size_t sizes[3]={pages,table,output};
    for(int i=0;i<3;++i) if(cuMemAlloc(&device[i],sizes[i])!=CUDA_SUCCESS){rc=3;break;}
    if(!rc&&(cuMemcpyHtoD(device[0],buffers[0],pages)!=CUDA_SUCCESS||
             cuMemcpyHtoD(device[1],buffers[1],table)!=CUDA_SUCCESS)) rc=3;
    long long args64[7]; for(int i=0;i<7;++i) args64[i]=dims[i];
    void* args[]={&device[0],&device[1],&device[2],&args64[0],&args64[1],
                  &args64[2],&args64[3],&args64[4],&args64[5],&args64[6]};
    size_t count=(size_t)tokens*H*D; unsigned grid=(unsigned)((count+255)/256);
    for(int i=0;!rc&&i<warmup;++i)
        if(cuLaunchKernel(fn,grid,1,1,256,1,1,0,0,args,0)!=CUDA_SUCCESS) rc=3;
    if(!rc&&(cuCtxSynchronize()!=CUDA_SUCCESS||cuEventCreate(&start,0)!=CUDA_SUCCESS||
             cuEventCreate(&stop,0)!=CUDA_SUCCESS)) rc=3;
    if(!rc&&cuEventRecord(start,0)!=CUDA_SUCCESS) rc=3;
    for(int i=0;!rc&&i<repetitions;++i)
        if(cuLaunchKernel(fn,grid,1,1,256,1,1,0,0,args,0)!=CUDA_SUCCESS) rc=3;
    if(!rc&&(cuEventRecord(stop,0)!=CUDA_SUCCESS||cuEventSynchronize(stop)!=CUDA_SUCCESS)) rc=3;
    float total=0.0f;
    if(!rc&&(cuEventElapsedTime(&total,start,stop)!=CUDA_SUCCESS||
             cuMemcpyDtoH(buffers[2],device[2],output)!=CUDA_SUCCESS)) rc=3;
    if(!rc)*latencyMs=total/(float)repetitions;
    if(start)cuEventDestroy(start); if(stop)cuEventDestroy(stop);
    for(CUdeviceptr ptr:device)if(ptr)cuMemFree(ptr);
    return rc;
}

int benchmarkMoe(CUfunction fn, const char* name, void** buffers, size_t nbuf,
                 const int64_t* dims, size_t ndim, int warmup,
                 int repetitions, float* latencyMs) {
    const bool dispatch = std::strncmp(name,kTileMoEDispatch,std::strlen(kTileMoEDispatch)) == 0;
    const bool combine = std::strncmp(name,kTileMoECombine,std::strlen(kTileMoECombine)) == 0;
    const bool grouped = std::strncmp(name,kTileGroupedGemm,std::strlen(kTileGroupedGemm)) == 0;
    const size_t elementBytes = (std::strstr(name,"_f16") ||
                                 std::strstr(name,"_bf16")) ? 2 : 4;
    if (!latencyMs || warmup < 0 || repetitions <= 0 ||
        (!dispatch && !combine && !grouped) ||
        (grouped ? (nbuf != 4 || ndim != 4)
                 : (nbuf != (dispatch ? 3u : 4u) || ndim != 3))) return 5;
    for (size_t i=0;i<ndim;++i)
        if (dims[i] <= 0 || dims[i] >= (1LL << 31)) return 5;
    size_t sizes[4]={}; unsigned threads=256,grid=0;
    if (dispatch) {
        const size_t T=dims[0],S=dims[1],H=dims[2];
        if(T>SIZE_MAX/H/elementBytes||S>SIZE_MAX/H/elementBytes)return 5;
        sizes[0]=T*H*elementBytes;sizes[1]=S*4;sizes[2]=S*H*elementBytes;
        grid=(unsigned)((S*H+threads-1)/threads);
    } else if (combine) {
        const size_t T=dims[0],S=dims[1],H=dims[2];
        if(T>SIZE_MAX/H/elementBytes||S>SIZE_MAX/H/elementBytes)return 5;
        sizes[0]=S*H*elementBytes;sizes[1]=S*4;sizes[2]=S*4;sizes[3]=T*H*elementBytes;
        grid=(unsigned)((T*H+threads-1)/threads);
    } else {
        const size_t T=dims[0],K=dims[1],N=dims[2],E=dims[3];
        if(T>SIZE_MAX/K/elementBytes||E>SIZE_MAX/K||
           E*K>SIZE_MAX/N/elementBytes||T>SIZE_MAX/N/elementBytes||E==SIZE_MAX)return 5;
        sizes[0]=T*K*elementBytes;sizes[1]=E*K*N*elementBytes;
        sizes[2]=(E+1)*4;sizes[3]=T*N*elementBytes;
        grid=(unsigned)((T*N+threads-1)/threads);
    }
    CUdeviceptr device[4]={}; CUevent start=nullptr,stop=nullptr; int rc=0;
    for(size_t i=0;i<nbuf;++i)
        if(cuMemAlloc(&device[i],sizes[i])!=CUDA_SUCCESS){rc=3;break;}
    const size_t outputIndex=nbuf-1;
    for(size_t i=0;!rc&&i<outputIndex;++i)
        if(cuMemcpyHtoD(device[i],buffers[i],sizes[i])!=CUDA_SUCCESS)rc=3;
    long long args64[4]={};for(size_t i=0;i<ndim;++i)args64[i]=dims[i];
    void* args[8]={};size_t arg=0;
    for(size_t i=0;i<nbuf;++i)args[arg++]=&device[i];
    for(size_t i=0;i<ndim;++i)args[arg++]=&args64[i];
    for(int i=0;!rc&&i<warmup;++i)
        if(cuLaunchKernel(fn,grid,1,1,threads,1,1,0,0,args,0)!=CUDA_SUCCESS)rc=3;
    if(!rc&&(cuCtxSynchronize()!=CUDA_SUCCESS||cuEventCreate(&start,0)!=CUDA_SUCCESS||
             cuEventCreate(&stop,0)!=CUDA_SUCCESS))rc=3;
    if(!rc&&cuEventRecord(start,0)!=CUDA_SUCCESS)rc=3;
    for(int i=0;!rc&&i<repetitions;++i)
        if(cuLaunchKernel(fn,grid,1,1,threads,1,1,0,0,args,0)!=CUDA_SUCCESS)rc=3;
    if(!rc&&(cuEventRecord(stop,0)!=CUDA_SUCCESS||cuEventSynchronize(stop)!=CUDA_SUCCESS))rc=3;
    float total=0.0f;
    if(!rc&&(cuEventElapsedTime(&total,start,stop)!=CUDA_SUCCESS||
             cuMemcpyDtoH(buffers[outputIndex],device[outputIndex],sizes[outputIndex])!=CUDA_SUCCESS))rc=3;
    if(!rc)*latencyMs=total/(float)repetitions;
    if(start)cuEventDestroy(start);if(stop)cuEventDestroy(stop);
    for(CUdeviceptr ptr:device)if(ptr)cuMemFree(ptr);
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
    if (std::strcmp(kernel_name, kTileDirectTf32) == 0)
        return invokeMmaGemm16(fn, buffers, nbuf, dims, ndim,
                               16, 8, 32, true, true, true, 4);
    if (std::strcmp(kernel_name, kTileDirectF64) == 0)
        return invokeMmaGemm16(fn, buffers, nbuf, dims, ndim,
                               8, 8, 32, true, true, true, 8, 8);
    if (std::strcmp(kernel_name, kTileDirectE4m3) == 0 ||
        std::strcmp(kernel_name, kTileDirectE5m2) == 0 ||
        std::strcmp(kernel_name, kTileDirectS8) == 0)
        return invokeMmaGemm16(fn, buffers, nbuf, dims, ndim,
                               16, 8, 32, true, true, true, 1);
    if (std::strcmp(kernel_name, kTileSharedF16) == 0 ||
        std::strcmp(kernel_name, kTileSharedBf16) == 0)
        return invokeMmaGemm16(fn, buffers, nbuf, dims, ndim,
                               32, 32, 128, true, true, true);
    if (std::strncmp(kernel_name, "tessera_tile_matmul_fused_", 26) == 0)
        return invokeFusedMatmul16(fn, kernel_name, buffers, nbuf, dims, ndim);
    if (std::strcmp(kernel_name, kTileNvfp4) == 0)
        return invokeNvfp4(fn, buffers, nbuf, dims, ndim);
    if (std::strcmp(kernel_name, kTileMxE2m3) == 0 ||
        std::strcmp(kernel_name, kTileMxE3m2) == 0)
        return invokeMx(fn, buffers, nbuf, dims, ndim, false);
    if (std::strcmp(kernel_name, kTileMxFp4) == 0)
        return invokeMx(fn, buffers, nbuf, dims, ndim, true);
    if (std::strcmp(kernel_name, kTileSoftmaxF16) == 0)
        return invokeSoftmax(fn, buffers, nbuf, dims, ndim, 2);
    if (std::strcmp(kernel_name, kTileSoftmaxF32) == 0)
        return invokeSoftmax(fn, buffers, nbuf, dims, ndim, 4);
    if (std::strncmp(kernel_name, kTileReduceSumF16, std::strlen(kTileReduceSumF16)) == 0 ||
        std::strncmp(kernel_name, kTileReduceMeanF16, std::strlen(kTileReduceMeanF16)) == 0 ||
        std::strncmp(kernel_name, kTileReduceMaxF16, std::strlen(kTileReduceMaxF16)) == 0)
        return invokeReduce(fn,buffers,nbuf,dims,ndim,2,std::strstr(kernel_name,"_cooperative_128")!=nullptr);
    if (std::strncmp(kernel_name, kTileReduceSumF32, std::strlen(kTileReduceSumF32)) == 0 ||
        std::strncmp(kernel_name, kTileReduceMeanF32, std::strlen(kTileReduceMeanF32)) == 0 ||
        std::strncmp(kernel_name, kTileReduceMaxF32, std::strlen(kTileReduceMaxF32)) == 0)
        return invokeReduce(fn,buffers,nbuf,dims,ndim,4,std::strstr(kernel_name,"_cooperative_128")!=nullptr);
    if (std::strncmp(kernel_name, kTileAttentionBackwardPrefix,
                     std::strlen(kTileAttentionBackwardPrefix)) == 0)
        return invokeAttentionBackward(fn, kernel_name, buffers, nbuf, dims, ndim);
    if (std::strncmp(kernel_name, kTileAttentionPrefix,
                     std::strlen(kTileAttentionPrefix)) == 0)
        return invokeAttention(fn, kernel_name, buffers, nbuf, dims, ndim);
    if (std::strcmp(kernel_name, kTilePagedKV) == 0)
        return invokePagedKV(fn, buffers, nbuf, dims, ndim);
    if (std::strncmp(kernel_name, kTilePagedAttentionPrefix,
                     std::strlen(kTilePagedAttentionPrefix)) == 0)
        return invokePagedAttention(fn, buffers, nbuf, dims, ndim);
    if (std::strncmp(kernel_name,kTileMoEDispatch,std::strlen(kTileMoEDispatch)) == 0 ||
        std::strncmp(kernel_name,kTileMoECombine,std::strlen(kTileMoECombine)) == 0 ||
        std::strncmp(kernel_name,kTileGroupedGemm,std::strlen(kTileGroupedGemm)) == 0)
        return invokeMoe(fn, kernel_name, buffers, nbuf, dims, ndim);
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
    auto existing = g_ptx.find(kernel_name);
    if (existing != g_ptx.end() && existing->second == ptx)
        return 0;  // Preserve the driver-JIT module/function cache on warm hits.
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
    if (std::strcmp(kernel_name, kTileMxE2m3) == 0 ||
        std::strcmp(kernel_name, kTileMxE3m2) == 0 ||
        std::strcmp(kernel_name, kTileMxFp4) == 0)
        return benchmarkMx(fn, kernel_name, buffers, num_buffers, dims,
                           num_dims, warmup, repetitions, latency_ms);
    if (std::strncmp(kernel_name, "tessera_tile_softmax_", 21) == 0 ||
        std::strncmp(kernel_name, "tessera_tile_reduce_", 20) == 0)
        return benchmarkUnary(fn, kernel_name, buffers, num_buffers, dims,
                              num_dims, warmup, repetitions, latency_ms);
    if (std::strncmp(kernel_name, kTileAttentionPrefix,
                     std::strlen(kTileAttentionPrefix)) == 0)
        return benchmarkAttention(fn, kernel_name, buffers, num_buffers, dims,
                                  num_dims, warmup, repetitions, latency_ms);
    if (std::strncmp(kernel_name, "tessera_tile_matmul_fused_", 26) == 0)
        return benchmarkFusedMatmul16(fn, kernel_name, buffers, num_buffers,
                                     dims, num_dims, warmup, repetitions,
                                     latency_ms);
    if (std::strcmp(kernel_name, kTilePagedKV) == 0)
        return benchmarkPagedKV(fn, buffers, num_buffers, dims, num_dims,
                                warmup, repetitions, latency_ms);
    if (std::strncmp(kernel_name,kTileMoEDispatch,std::strlen(kTileMoEDispatch)) == 0 ||
        std::strncmp(kernel_name,kTileMoECombine,std::strlen(kTileMoECombine)) == 0 ||
        std::strncmp(kernel_name,kTileGroupedGemm,std::strlen(kTileGroupedGemm)) == 0)
        return benchmarkMoe(fn, kernel_name, buffers, num_buffers, dims,
                            num_dims, warmup, repetitions, latency_ms);
    return benchmarkTileGemm16(fn, kernel_name, buffers, num_buffers, dims,
                               num_dims, warmup, repetitions, latency_ms);
}

int tessera_nvidia_register_ptx_launcher(void) {
    if (tsrRegisterGpuLauncher == nullptr) return 2;  // core runtime not in the load
    return tsrRegisterGpuLauncher(gpuLauncher, nullptr) == TSR_STATUS_SUCCESS ? 0 : 1;
}

}  // extern "C"
