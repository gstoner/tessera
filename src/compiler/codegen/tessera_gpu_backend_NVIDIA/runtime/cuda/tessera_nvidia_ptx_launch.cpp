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

#include <climits>
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
// Canonical Schedule->Tile SM120 kernels retain their content-addressed
// compiler-owned symbol, but use the established f16 A/B/D/M/N/K launch ABI.
constexpr const char* kScheduledSm120MatmulPrefix =
    "nvidia_sm120_scheduled_matmul_";
constexpr const char* kTileDirectTf32 = "tessera_tile_matmul_direct_tf32";
constexpr const char* kTileDirectE4m3 = "tessera_tile_matmul_direct_e4m3";
constexpr const char* kTileDirectE5m2 = "tessera_tile_matmul_direct_e5m2";
constexpr const char* kTileDirectS8 = "tessera_tile_matmul_direct_s8";
constexpr const char* kTileInt4 = "tessera_tile_matmul_int4";
constexpr const char* kTileCudaIntrinsic = "tessera_tile_cuda_intrinsic_";
constexpr const char* kTilePackedDecode = "tessera_tile_packed_decode_";
constexpr const char* kTileDirectF64 = "tessera_tile_matmul_direct_f64";
constexpr const char* kTileNvfp4 = "tessera_tile_matmul_nvfp4";
constexpr const char* kTileMxE2m3 = "tessera_tile_matmul_mx_e2m3";
constexpr const char* kTileMxE3m2 = "tessera_tile_matmul_mx_e3m2";
constexpr const char* kTileMxFp4 = "tessera_tile_matmul_mx_fp4_e2m1";
constexpr const char* kTileSoftmaxF16 = "tessera_tile_softmax_f16";
constexpr const char* kTileSoftmaxBf16 = "tessera_tile_softmax_bf16";
constexpr const char* kTileSoftmaxF32 = "tessera_tile_softmax_f32";
constexpr const char* kTileReducePrefix = "tessera_tile_reduce_";
constexpr const char* kTileNormPrefix = "tessera_tile_norm_";
constexpr const char* kTileAttentionPrefix = "tessera_tile_attention_";
constexpr const char* kTileAttentionBackwardPrefix =
    "tessera_tile_attention_backward_";
constexpr const char* kTilePagedKV = "tessera_tile_paged_kv_read_f32_direct";
constexpr const char* kTilePagedAttentionPrefix = "tessera_tile_paged_attention_f32_fused_";
constexpr const char* kTileMoEDispatch = "tessera_tile_moe_dispatch_";
constexpr const char* kTileMoECombine = "tessera_tile_moe_combine_";
constexpr const char* kTileGroupedGemm = "tessera_tile_grouped_gemm_";
constexpr const char* kTrainingPrefix = "tessera_cuda_training_";
constexpr const char* kDynamicSharedPrefix =
    "tessera_cuda_dynamic_smem_";

std::mutex g_mu;
std::map<std::string, std::string> g_ptx;           // kernel name -> PTX text
std::map<std::string, CUmodule>    g_modules;        // kernel name -> JIT'd module
std::map<std::string, CUfunction>  g_funcs;          // kernel name -> entry fn
bool g_ctx_ready = false;

// The bridge retains device 0's primary context and serializes every invoke
// under g_mu through cuCtxSynchronize. A single grow-only staging region is
// therefore safe for the synchronous host-buffer GEMM ABIs below. This is not
// a CUDA async mempool: streams and concurrent leases need a different
// ownership contract before they can share this storage.
constexpr size_t kStagingAlignment = 256;
struct StagingArena {
    CUdeviceptr base = 0;
    size_t capacity = 0;
};
StagingArena g_staging;

// Assign aligned slices of the retained staging region. Caller holds g_mu and
// has made the bridge primary context current. Prefer retaining the old arena
// until growth succeeds. If that transient double-residency is the only reason
// an allocation reports OOM, release the idle old arena and retry once.
bool stagingPointersLocked(const size_t* sizes, size_t count,
                           CUdeviceptr* pointers) {
    if (!sizes || !pointers || count == 0) return false;
    size_t total = 0;
    for (size_t i = 0; i < count; ++i) {
        if (sizes[i] == 0 || total > SIZE_MAX - (kStagingAlignment - 1))
            return false;
        const size_t offset =
            (total + kStagingAlignment - 1) & ~(kStagingAlignment - 1);
        if (sizes[i] > SIZE_MAX - offset) return false;
        total = offset + sizes[i];
    }
    if (!g_staging.base || g_staging.capacity < total) {
        CUdeviceptr replacement = 0;
        const CUdeviceptr previous = g_staging.base;
        CUresult allocation = cuMemAlloc(&replacement, total);
        if (allocation == CUDA_ERROR_OUT_OF_MEMORY && previous) {
            if (cuMemFree(previous) != CUDA_SUCCESS) return false;
            g_staging = {};
            allocation = cuMemAlloc(&replacement, total);
        }
        if (allocation != CUDA_SUCCESS) return false;
        if (previous && g_staging.base &&
            cuMemFree(previous) != CUDA_SUCCESS) {
            cuMemFree(replacement);
            return false;
        }
        g_staging.base = replacement;
        g_staging.capacity = total;
    }
    total = 0;
    for (size_t i = 0; i < count; ++i) {
        total = (total + kStagingAlignment - 1) & ~(kStagingAlignment - 1);
        pointers[i] = g_staging.base + total;
        total += sizes[i];
    }
    return true;
}

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
    CUdeviceptr device[3] = {};
    const size_t sizes[] = {sA, sB, sD};
    if (!stagingPointersLocked(sizes, 3, device)) return 3;
    CUdeviceptr dA = device[0], dB = device[1], dD = device[2];
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
    return rc;
}

// Launch the general aligned mma.sync 16-bit GEMM: buffers {A 16b, B 16b, D f32},
// dims {M,N,K} (M%16,N%8,K%16); grid (M/16, N/8), block 32 (one warp per 16x8
// tile); runtime M/N/K params. bf16 and f16 share this ABI (only the JIT'd PTX
// differs). Returns the C-ABI rc.
int copyLogicalRowsDtoH(void* host, CUdeviceptr device, long long rows,
                       long long columns, long long leadingDimension,
                       size_t elementBytes) {
    if (!host || !device || rows <= 0 || columns <= 0 ||
        leadingDimension < columns || elementBytes == 0)
        return 5;
    auto* hostBytes = static_cast<unsigned char*>(host);
    const size_t rowBytes = static_cast<size_t>(columns) * elementBytes;
    if (leadingDimension == columns)
        return cuMemcpyDtoH(host, device,
                           static_cast<size_t>(rows) * rowBytes) == CUDA_SUCCESS
            ? 0 : 3;
    const size_t pitchBytes = static_cast<size_t>(leadingDimension) * elementBytes;
    for (long long row = 0; row < rows; ++row) {
        if (cuMemcpyDtoH(hostBytes + static_cast<size_t>(row) * pitchBytes,
                         device + static_cast<CUdeviceptr>(row) * pitchBytes,
                         rowBytes) != CUDA_SUCCESS)
            return 3;
    }
    return 0;
}

int invokeMmaGemm16(CUfunction fn, void** buffers, size_t nbuf,
                    const int64_t* dims, size_t ndim, int tileM = 16,
                    int tileN = 8, int threads = 32, bool ragged = false,
                    bool columnMajorGrid = false, bool dimensions64 = false,
                    size_t elementBytes = 2, size_t outputBytes = 4,
                    bool requiresEvenK = false) {
    if (nbuf != 3 || (ndim != 3 && ndim != 6)) return 5;
    const long long M64 = dims[0], N64 = dims[1], K64 = dims[2];
    const long long LDA64 = ndim == 6 ? dims[3] : K64;
    const long long LDB64 = ndim == 6 ? dims[4] : K64;
    const long long LDD64 = ndim == 6 ? dims[5] : N64;
    if (M64 <= 0 || N64 <= 0 || K64 <= 0) return 5;
    if (LDA64 < K64 || LDB64 < K64 || LDD64 < N64) return 5;
    if (!ragged && (M64 % 16 || N64 % 8 || K64 % 16)) return 5;
    // `requiresEvenK` belongs to the ptx_emit fragment layout, NOT to ragged
    // shapes in general: `ld.global.b32` needs a 4-byte-aligned address and
    // the fragments address 2-byte elements, so `row*K + k` must be even. With
    // K odd every odd row starts misaligned and the load faults
    // (CUDA_ERROR_MISALIGNED_ADDRESS, measured on sm_120) -- in the main loop,
    // not just the tail. Rejecting here keeps that a diagnosable rc rather
    // than a device fault that poisons the context for the rest of the
    // process. The Tile kernels mask their own boundaries and have no such
    // constraint, so they must not inherit it.
    if (requiresEvenK && (K64 % 2)) return 5;
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
    const __int128 aSpan = (__int128)(M64 - 1) * LDA64 + K64;
    const __int128 bSpan = (__int128)(N64 - 1) * LDB64 + K64;
    const __int128 dSpan = (__int128)(M64 - 1) * LDD64 + N64;
    if (aSpan > kMaxElems || bSpan > kMaxElems || dSpan > kMaxElems) return 5;
    const long long aElems = (long long)aSpan;
    const long long bElems = (long long)bSpan;
    const long long dElems = (long long)dSpan;
    int M = (int)M64, N = (int)N64, K = (int)K64;
    const void* A = buffers[0];
    const void* B = buffers[1];
    void* D = buffers[2];
    const size_t sA = (size_t)aElems * elementBytes;
    const size_t sB = (size_t)bElems * elementBytes;  // B is col-major
    const size_t sD = (size_t)dElems * outputBytes;
    CUdeviceptr device[3] = {};
    const size_t sizes[] = {sA, sB, sD};
    if (!stagingPointersLocked(sizes, 3, device)) return 3;
    CUdeviceptr dA = device[0], dB = device[1], dD = device[2];
    int rc = 0;
    do {
        if (cuMemcpyHtoD(dA, A, sA) != CUDA_SUCCESS) { rc = 3; break; }
        if (cuMemcpyHtoD(dB, B, sB) != CUDA_SUCCESS) { rc = 3; break; }
        long long MArg = M64, NArg = N64, KArg = K64;
        long long LDAArg = LDA64, LDBArg = LDB64, LDDArg = LDD64;
        void* args32[] = {&dA, &dB, &dD, &M, &N, &K};
        void* args64[] = {&dA, &dB, &dD, &MArg, &NArg, &KArg};
        void* argsStrided64[] = {&dA, &dB, &dD, &MArg, &NArg, &KArg,
                                 &LDAArg, &LDBArg, &LDDArg};
        void** args = ndim == 6 ? argsStrided64
                                : (dimensions64 ? args64 : args32);
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
        rc = copyLogicalRowsDtoH(D, dD, M64, N64, LDD64, outputBytes);
        if (rc) break;
    } while (0);
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

int invokeInt4(CUfunction fn, void** buffers, size_t nbuf,
               const int64_t* dims, size_t ndim) {
    if (nbuf != 3 || ndim != 3) return 5;
    const long long M = dims[0], N = dims[1], K = dims[2];
    if (M <= 0 || N <= 0 || K <= 0 || M >= (1LL << 31) ||
        N >= (1LL << 31) || K >= (1LL << 31)) return 5;
    const size_t packedK = ((size_t)K + 1) / 2;
    if ((size_t)M > SIZE_MAX / packedK ||
        packedK > SIZE_MAX / (size_t)N ||
        (size_t)M > SIZE_MAX / (size_t)N / sizeof(int32_t)) return 5;
    const size_t sizes[] = {
        (size_t)M * packedK,
        packedK * (size_t)N,
        (size_t)M * (size_t)N * sizeof(int32_t),
    };
    CUdeviceptr device[3] = {};
    int rc = 0;
    for (int i = 0; i < 3; ++i)
        if (cuMemAlloc(&device[i], sizes[i]) != CUDA_SUCCESS) {
            rc = 3; break;
        }
    if (!rc &&
        (cuMemcpyHtoD(device[0], buffers[0], sizes[0]) != CUDA_SUCCESS ||
         cuMemcpyHtoD(device[1], buffers[1], sizes[1]) != CUDA_SUCCESS))
        rc = 3;
    if (!rc) {
        long long MArg = M, NArg = N, KArg = K;
        void* args[] = {
            &device[0], &device[1], &device[2], &MArg, &NArg, &KArg,
        };
        const unsigned gx = (unsigned)((N + 31) / 32);
        const unsigned gy = (unsigned)((M + 7) / 8);
        if (cuLaunchKernel(fn, gx, gy, 1, 32, 8, 1, 0, 0, args, 0) !=
                CUDA_SUCCESS ||
            cuCtxSynchronize() != CUDA_SUCCESS ||
            cuMemcpyDtoH(buffers[2], device[2], sizes[2]) != CUDA_SUCCESS)
            rc = 3;
    }
    for (CUdeviceptr ptr : device)
        if (ptr) cuMemFree(ptr);
    return rc;
}

int invokeCudaIntrinsic(CUfunction fn, void** buffers, size_t nbuf,
                        const int64_t* dims, size_t ndim) {
    if (nbuf != 4 || ndim != 1 || dims[0] <= 0) return 5;
    const size_t n = (size_t)dims[0];
    if (n > SIZE_MAX / sizeof(int32_t)) return 5;
    const size_t bytes = n * sizeof(int32_t);
    CUdeviceptr device[4] = {};
    int rc = 0;
    for (int i = 0; i < 4; ++i)
        if (cuMemAlloc(&device[i], bytes) != CUDA_SUCCESS) {
            rc = 3; break;
        }
    if (!rc)
        for (int i = 0; i < 3; ++i)
            if (cuMemcpyHtoD(device[i], buffers[i], bytes) != CUDA_SUCCESS) {
                rc = 3; break;
            }
    if (!rc) {
        long long nArg = (long long)n;
        void* args[] = {
            &device[0], &device[1], &device[2], &device[3], &nArg,
        };
        const unsigned grid = (unsigned)((n + 127) / 128);
        if (cuLaunchKernel(fn, grid, 1, 1, 128, 1, 1, 0, 0, args, 0) !=
                CUDA_SUCCESS ||
            cuCtxSynchronize() != CUDA_SUCCESS ||
            cuMemcpyDtoH(buffers[3], device[3], bytes) != CUDA_SUCCESS)
            rc = 3;
    }
    for (CUdeviceptr ptr : device)
        if (ptr) cuMemFree(ptr);
    return rc;
}

int invokePackedDecode(CUfunction fn, void** buffers, size_t nbuf,
                       const int64_t* dims, size_t ndim) {
    if (nbuf != 3 || ndim != 6) return 5;
    const long long row = dims[0], col = dims[1];
    const long long rows = dims[2], columns = dims[3];
    const long long sourceBytes = dims[4], scaleBytes = dims[5];
    if (row < 0 || col < 0 || rows <= 0 || columns <= 0 ||
        sourceBytes <= 0 || scaleBytes <= 0 ||
        (size_t)rows > SIZE_MAX / (size_t)columns ||
        (size_t)rows * (size_t)columns > SIZE_MAX / sizeof(float))
        return 5;
    const size_t outputBytes =
        (size_t)rows * (size_t)columns * sizeof(float);
    CUdeviceptr device[3] = {};
    const size_t sizes[] = {
        (size_t)sourceBytes, (size_t)scaleBytes, outputBytes,
    };
    int rc = 0;
    for (int i = 0; i < 3; ++i)
        if (cuMemAlloc(&device[i], sizes[i]) != CUDA_SUCCESS) {
            rc = 3; break;
        }
    if (!rc &&
        (cuMemcpyHtoD(device[0], buffers[0], sizes[0]) != CUDA_SUCCESS ||
         cuMemcpyHtoD(device[1], buffers[1], sizes[1]) != CUDA_SUCCESS))
        rc = 3;
    if (!rc) {
        long long rowArg = row, colArg = col;
        long long rowsArg = rows, columnsArg = columns;
        long long sourceBytesArg = sourceBytes, scaleBytesArg = scaleBytes;
        void* args[] = {
            &device[0], &device[1], &device[2],
            &rowArg, &colArg, &rowsArg, &columnsArg,
            &sourceBytesArg, &scaleBytesArg,
        };
        const unsigned grid =
            (unsigned)(((size_t)rows * (size_t)columns + 127) / 128);
        if (cuLaunchKernel(fn, grid, 1, 1, 128, 1, 1, 0, 0, args, 0) !=
                CUDA_SUCCESS ||
            cuCtxSynchronize() != CUDA_SUCCESS ||
            cuMemcpyDtoH(buffers[2], device[2], outputBytes) != CUDA_SUCCESS)
            rc = 3;
    }
    for (CUdeviceptr ptr : device)
        if (ptr) cuMemFree(ptr);
    return rc;
}

int benchmarkPackedDecode(CUfunction fn, void** buffers, size_t nbuf,
                          const int64_t* dims, size_t ndim, int warmup,
                          int repetitions, float* latencyMs) {
    if (nbuf != 3 || ndim != 6 || !latencyMs || warmup < 0 ||
        repetitions <= 0)
        return 5;
    const long long row = dims[0], col = dims[1];
    const long long rows = dims[2], columns = dims[3];
    const long long sourceBytes = dims[4], scaleBytes = dims[5];
    if (row < 0 || col < 0 || rows <= 0 || columns <= 0 ||
        sourceBytes <= 0 || scaleBytes <= 0 ||
        (size_t)rows > SIZE_MAX / (size_t)columns ||
        (size_t)rows * (size_t)columns > SIZE_MAX / sizeof(float))
        return 5;
    const size_t outputBytes =
        (size_t)rows * (size_t)columns * sizeof(float);
    const size_t sizes[] = {
        (size_t)sourceBytes, (size_t)scaleBytes, outputBytes,
    };
    CUdeviceptr device[3] = {};
    CUevent start = nullptr, stop = nullptr;
    int rc = 0;
    for (int i = 0; i < 3; ++i)
        if (cuMemAlloc(&device[i], sizes[i]) != CUDA_SUCCESS) {
            rc = 3; break;
        }
    if (!rc &&
        (cuMemcpyHtoD(device[0], buffers[0], sizes[0]) != CUDA_SUCCESS ||
         cuMemcpyHtoD(device[1], buffers[1], sizes[1]) != CUDA_SUCCESS))
        rc = 3;
    if (!rc) {
        long long rowArg = row, colArg = col;
        long long rowsArg = rows, columnsArg = columns;
        long long sourceBytesArg = sourceBytes, scaleBytesArg = scaleBytes;
        void* args[] = {
            &device[0], &device[1], &device[2],
            &rowArg, &colArg, &rowsArg, &columnsArg,
            &sourceBytesArg, &scaleBytesArg,
        };
        const unsigned grid =
            (unsigned)(((size_t)rows * (size_t)columns + 127) / 128);
        auto launch = [&]() {
            return cuLaunchKernel(fn, grid, 1, 1, 128, 1, 1, 0, 0, args, 0);
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
                    cuEventSynchronize(stop) != CUDA_SUCCESS))
            rc = 3;
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
    if (ndim != 3 && ndim != 6) return 5;
    const bool hasBias = std::strstr(name, "_b1_r") != nullptr;
    const bool hasResidual = std::strstr(name, "_r1") != nullptr;
    const size_t expected = 3 + (hasBias ? 1 : 0) + (hasResidual ? 1 : 0);
    if (nbuf != expected) return 5;
    const long long M = dims[0], N = dims[1], K = dims[2];
    const long long LDA = ndim == 6 ? dims[3] : K;
    const long long LDB = ndim == 6 ? dims[4] : K;
    const long long LDD = ndim == 6 ? dims[5] : N;
    if (M <= 0 || N <= 0 || K <= 0 || M >= (1LL << 31) ||
        N >= (1LL << 31) || K >= (1LL << 31) ||
        LDA < K || LDB < K || LDD < N) return 5;
    const __int128 aSpan = (__int128)(M - 1) * LDA + K;
    const __int128 bSpan = (__int128)(N - 1) * LDB + K;
    const __int128 dSpan = (__int128)(M - 1) * LDD + N;
    if (aSpan > (1LL << 31) || bSpan > (1LL << 31) ||
        dSpan > (1LL << 31)) return 5;
    const bool direct = std::strstr(name, "_tf32_") != nullptr ||
                        std::strstr(name, "_e4m3_") != nullptr ||
                        std::strstr(name, "_e5m2_") != nullptr;
    const bool scheduled =
        std::strncmp(name, kScheduledSm120MatmulPrefix,
                     std::strlen(kScheduledSm120MatmulPrefix)) == 0;
    const bool scheduledMacro =
        scheduled && std::strstr(name, "_macro_kernel") != nullptr;
    const size_t inputBytes = std::strstr(name, "_tf32_") ? 4 :
                              (direct ? 1 : 2);
    const size_t outputBytes = std::strstr(name, "_outf16") ? 2 : 4;
    size_t sizes[5] = {};
    size_t sizeIndex = 0;
    sizes[sizeIndex++] = static_cast<size_t>(aSpan) * inputBytes;
    sizes[sizeIndex++] = static_cast<size_t>(bSpan) * inputBytes;
    if (hasBias) sizes[sizeIndex++] = (size_t)N * 4;
    if (hasResidual) sizes[sizeIndex++] = static_cast<size_t>(dSpan) * 4;
    sizes[sizeIndex++] = static_cast<size_t>(dSpan) * outputBytes;
    if (sizeIndex != nbuf) return 5;
    CUdeviceptr device[5] = {};
    int rc = stagingPointersLocked(sizes, nbuf, device) ? 0 : 3;
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
        long long LDAArg = LDA, LDBArg = LDB, LDDArg = LDD;
        void* args[11] = {};
        size_t arg = 0;
        for (size_t i = 0; i < nbuf; ++i) args[arg++] = &device[i];
        args[arg++] = &MArg; args[arg++] = &NArg; args[arg++] = &KArg;
        if (ndim == 6) {
            args[arg++] = &LDAArg;
            args[arg++] = &LDBArg;
            args[arg++] = &LDDArg;
        }
        const unsigned tileN = direct || (scheduled && !scheduledMacro) ? 8 : 32;
        const unsigned tileM = direct || (scheduled && !scheduledMacro) ? 16 : 32;
        const unsigned threads = direct || (scheduled && !scheduledMacro) ? 32 : 128;
        if (cuLaunchKernel(fn, (unsigned)((N + tileN - 1) / tileN),
                           (unsigned)((M + tileM - 1) / tileM), 1,
                           threads, 1, 1,
                           0, 0, args, 0) != CUDA_SUCCESS ||
            cuCtxSynchronize() != CUDA_SUCCESS)
            rc = 3;
        if (!rc)
            rc = copyLogicalRowsDtoH(
                buffers[outputIndex], device[outputIndex], M, N, LDD,
                outputBytes);
    }
    return rc;
}

// Compiler-owned stable row-softmax ABI: host f16/bf16/f32 X/O and flattened
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
    if (ndim != 7 || !name) return 5;
    const bool hasSavedLse = std::strstr(name, "_lse_") != nullptr;
    if ((!hasSavedLse && nbuf != 4 && nbuf != 5) ||
        (hasSavedLse && nbuf != 5 && nbuf != 6)) return 5;
    const bool hasBias = hasSavedLse ? nbuf == 6 : nbuf == 5;
    const size_t outputIndex = hasBias ? 4 : 3;
    const size_t lseIndex = outputIndex + 1;
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
    size_t qElements, kElements, vElements, oElements, rowElements = 0, biasElements = 0;
    if (!product({B, Hq, Sq, D}, qElements) ||
        !product({B, Hkv, Sk, D}, kElements) ||
        !product({B, Hkv, Sk, Dv}, vElements) ||
        !product({B, Hq, Sq, Dv}, oElements) ||
        (hasSavedLse && !product({B, Hq, Sq}, rowElements)) ||
        (hasBias && !product({B, Hq, Sq, Sk}, biasElements))) return 5;
    const bool narrow =
        std::strncmp(name, "tessera_tile_attention_f16_", 27) == 0 ||
        std::strncmp(name, "tessera_tile_attention_bf16_", 28) == 0;
    const size_t elementBytes = narrow ? 2 : 4;
    if (qElements > SIZE_MAX / elementBytes || kElements > SIZE_MAX / elementBytes ||
        vElements > SIZE_MAX / elementBytes || oElements > SIZE_MAX / sizeof(float) ||
        (hasSavedLse && rowElements > SIZE_MAX / sizeof(float)) ||
        (hasBias && biasElements > SIZE_MAX / sizeof(float))) return 5;
    size_t sizes[6] = {qElements * elementBytes, kElements * elementBytes,
                       vElements * elementBytes, 0, 0, 0};
    if (hasBias) sizes[3] = biasElements * sizeof(float);
    sizes[outputIndex] = oElements * sizeof(float);
    if (hasSavedLse) sizes[lseIndex] = rowElements * sizeof(float);
    CUdeviceptr device[6] = {};
    int rc = 0;
    for (size_t i = 0; i < nbuf; ++i)
        if (cuMemAlloc(&device[i], sizes[i]) != CUDA_SUCCESS) { rc = 3; break; }
    if (!rc)
        for (size_t i = 0; i < outputIndex; ++i)
            if (cuMemcpyHtoD(device[i], buffers[i], sizes[i]) != CUDA_SUCCESS) { rc = 3; break; }
    if (!rc) {
        long long args64[7];
        for (int i = 0; i < 7; ++i) args64[i] = dims[i];
        void* args[13] = {};
        size_t arg = 0;
        for (size_t i = 0; i < nbuf; ++i) args[arg++] = &device[i];
        for (int i = 0; i < 7; ++i) args[arg++] = &args64[i];
        unsigned grid = (unsigned)((oElements + 127) / 128);
        if (grid == 0 || grid > 0x7fffffffU ||
            cuLaunchKernel(fn, grid, 1, 1, 128, 1, 1, 0, 0, args, 0) != CUDA_SUCCESS ||
            cuCtxSynchronize() != CUDA_SUCCESS ||
            cuMemcpyDtoH(buffers[outputIndex], device[outputIndex],
                         sizes[outputIndex]) != CUDA_SUCCESS ||
            (hasSavedLse && cuMemcpyDtoH(buffers[lseIndex], device[lseIndex],
                                         sizes[lseIndex]) != CUDA_SUCCESS))
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
    if (ndim != 7 || !kernelName) return 5;
    const bool hasSavedLse = std::strstr(kernelName, "_lse_") != nullptr;
    if ((!hasSavedLse && nbuf != 7 && nbuf != 8) ||
        (hasSavedLse && nbuf != 8 && nbuf != 9)) return 5;
    const bool hasBias = hasSavedLse ? nbuf == 9 : nbuf == 8;
    const size_t lseIndex = 4 + size_t(hasBias);
    const size_t outputBase = lseIndex + size_t(hasSavedLse);
    const long long B=dims[0], Hq=dims[1], Hkv=dims[2], Sq=dims[3];
    const long long Sk=dims[4], D=dims[5], Dv=dims[6];
    if (B<=0 || Hq<=0 || Hkv<=0 || Sq<=0 || Sk<=0 || D<=0 || Dv<=0 ||
        Hq%Hkv) return 5;
    const bool narrow =
        std::strstr(kernelName, "attention_backward_f16_") != nullptr ||
        std::strstr(kernelName, "attention_backward_bf16_") != nullptr;
    const size_t elementBytes = narrow ? 2 : 4;
    auto bytes = [](std::initializer_list<size_t> values, size_t width, size_t& out) {
        out = width;
        for (size_t value : values) {
            if (value && out > SIZE_MAX / value) return false;
            out *= value;
        }
        return true;
    };
    size_t doBytes=0, qBytes=0, kBytes=0, vBytes=0, lseBytes=0, biasBytes=0;
    if (!bytes({(size_t)B,(size_t)Hq,(size_t)Sq,(size_t)Dv},elementBytes,doBytes) ||
        !bytes({(size_t)B,(size_t)Hq,(size_t)Sq,(size_t)D},elementBytes,qBytes) ||
        !bytes({(size_t)B,(size_t)Hkv,(size_t)Sk,(size_t)D},elementBytes,kBytes) ||
        !bytes({(size_t)B,(size_t)Hkv,(size_t)Sk,(size_t)Dv},elementBytes,vBytes) ||
        (hasSavedLse && !bytes({(size_t)B,(size_t)Hq,(size_t)Sq},4,lseBytes)) ||
        (hasBias && !bytes({(size_t)B,(size_t)Hq,(size_t)Sq,(size_t)Sk},4,biasBytes)))
        return 5;
    size_t sizes[9] = {doBytes,qBytes,kBytes,vBytes,0,0,0,0,0};
    if (hasBias) sizes[4] = biasBytes;
    if (hasSavedLse) sizes[lseIndex] = lseBytes;
    sizes[outputBase] = qBytes;
    sizes[outputBase+1] = kBytes;
    sizes[outputBase+2] = vBytes;
    CUdeviceptr device[9] = {};
    int rc = 0;
    for (size_t i=0;i<nbuf;++i)
        if (cuMemAlloc(&device[i], sizes[i]) != CUDA_SUCCESS) { rc=3; break; }
    if (!rc)
        for (size_t i=0;i<outputBase;++i)
            if (cuMemcpyHtoD(device[i],buffers[i],sizes[i]) != CUDA_SUCCESS) { rc=3; break; }
    if (!rc) {
        long long args64[7]; for(int i=0;i<7;++i) args64[i]=dims[i];
        void* args[16] = {};
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

// `columnMajorGrid` mirrors the parameter of the same name on
// `invokeMmaGemm16`: true maps blockIdx.x to N (the Tile lowering's
// convention, `NVIDIALowering.cpp`: mt = blockY*16, nt = blockX*8), false maps
// blockIdx.x to M (what `ptx_emit` and the shipped AOT kernel do:
// mt = ctaid.x*16, nt = ctaid.y*8).
//
// The benchmark path previously hardcoded the column-major mapping, so it
// could only time kernels that used it. That is why `tessera_mma_gemm_f16`
// returned rc=5 and the emitted GEMM had no device latency at all: registering
// its geometry without this flag would have launched a transposed grid --
// at 512x512, rows to 1024 and columns only to 256, half the output unwritten,
// with a plausible-looking number.
bool tileLaunchConfig(const char* name, int& tileM, int& tileN, int& threads,
                      bool& columnMajorGrid, bool* ragged, bool* requiresEvenK) {
    columnMajorGrid = true;
    // Ragged M/N is the NORM here, not the exception. Every entry the invoke
    // dispatch routes through `invokeMmaGemm16` passes `ragged=true` -- the
    // Tile-direct, Tile-shared and scheduled-SM120 kernels have masked their
    // out-of-bounds loads and stores in `NVIDIALowering.cpp` since they were
    // written. Defaulting to false here (as this did on first landing) told
    // the benchmark path they were aligned-only, so `rc=5` came back for every
    // ragged device measurement and BOTH live Tile candidates returned None --
    // an incomplete field, which is the exact autotune bias
    // `_record_raced_the_live_field` exists to refuse. Review finding on #675.
    if (ragged) *ragged = true;
    if (requiresEvenK) *requiresEvenK = false;
    if (std::strcmp(name, kGemmF16) == 0 || std::strcmp(name, kGemmBf16) == 0) {
        tileM = 16; tileN = 8; threads = 32;
        columnMajorGrid = false;   // ptx_emit maps blockIdx.x to M
        // These two -- and only these two -- carry the b32 fragment-alignment
        // constraint: `ld.global.b32` needs a 4-byte-aligned address and the
        // fragments address 2-byte elements, so `row*K + k` must be even.
        // It is a property of THIS emitter's fragment layout, not of ragged
        // shapes in general, which is why it is a separate flag: folding it
        // into `ragged` would have imposed an odd-K refusal on Tile kernels
        // that have no such limit.
        if (requiresEvenK) *requiresEvenK = true;
        return true;
    }
    if (std::strncmp(name, kScheduledSm120MatmulPrefix,
                     std::strlen(kScheduledSm120MatmulPrefix)) == 0) {
        const bool macro = std::strstr(name, "_macro_kernel") != nullptr;
        tileM = macro ? 32 : 16;
        tileN = macro ? 32 : 8;
        threads = macro ? 128 : 32;
        return true;
    }
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
    if (std::strncmp(name, kTileSharedF16, std::strlen(kTileSharedF16)) == 0 ||
        std::strncmp(name, kTileSharedBf16, std::strlen(kTileSharedBf16)) == 0) {
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
    bool columnMajorGrid = true;
    bool ragged = true, requiresEvenK = false;
    if (!tileLaunchConfig(name, tileM, tileN, threads, columnMajorGrid, &ragged,
                          &requiresEvenK))
        return 5;
    // The same shape contract the invoke path enforces, so a measurement can
    // never fault a device the dispatch would have served -- and, just as
    // importantly, can never REFUSE a shape the dispatch would have served,
    // which silently shrinks the field an autotune race sees.
    if (!ragged && (M % 16 || N % 8 || K % 16)) return 5;
    if (requiresEvenK && (K % 2)) return 5;

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
        unsigned gx = columnMajorGrid
            ? (unsigned)((N + tileN - 1) / tileN)
            : (unsigned)((M + tileM - 1) / tileM);
        unsigned gy = columnMajorGrid
            ? (unsigned)((M + tileM - 1) / tileM)
            : (unsigned)((N + tileN - 1) / tileN);
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
        std::strcmp(name, kTileSoftmaxBf16) == 0 ||
        std::strcmp(name, kTileSoftmaxF32) == 0;
    const bool norm =
        std::strncmp(name, kTileNormPrefix, std::strlen(kTileNormPrefix)) == 0;
    const bool rowwise = softmax || norm;
    if ((rowwise && ndim != 2) || (!rowwise && ndim != 3)) return 5;
    const bool narrow = std::strcmp(name, kTileSoftmaxF16) == 0 ||
        std::strcmp(name, kTileSoftmaxBf16) == 0 ||
        std::strstr(name, "_f16_") != nullptr ||
        std::strstr(name, "_bf16_") != nullptr;
    long long outer=dims[0],axis=dims[1],inner=rowwise?1:dims[2];
    if(outer<=0||axis<=0||inner<=0||outer>=(1LL<<31)||axis>=(1LL<<31)||
       inner>=(1LL<<31)||outer>(1LL<<31)/axis||outer*axis>(1LL<<31)/inner) return 5;
    const size_t outputs=(size_t)outer*(size_t)inner;
    const size_t inputBytes=outputs*(size_t)axis*(narrow?2:4);
    const size_t outputBytes=rowwise?inputBytes:outputs*4;
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
        void** args=rowwise?softmaxArgs:reduceArgs;
        const bool cooperative=!rowwise&&std::strstr(name,"_cooperative_128")!=nullptr;
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
    if (ndim != 7 || !name || !latencyMs || warmup < 0 || repetitions <= 0)
        return 5;
    const bool hasSavedLse = std::strstr(name, "_lse_") != nullptr;
    if ((!hasSavedLse && nbuf != 4 && nbuf != 5) ||
        (hasSavedLse && nbuf != 5 && nbuf != 6)) return 5;
    const bool hasBias = hasSavedLse ? nbuf == 6 : nbuf == 5;
    const size_t outputIndex = hasBias ? 4 : 3;
    const size_t lseIndex = outputIndex + 1;
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
    size_t counts[6] = {};
    if (!product({B, Hq, Sq, D}, counts[0]) ||
        !product({B, Hkv, Sk, D}, counts[1]) ||
        !product({B, Hkv, Sk, Dv}, counts[2]) ||
        !product({B, Hq, Sq, Dv}, counts[outputIndex]) ||
        (hasSavedLse && !product({B, Hq, Sq}, counts[lseIndex])) ||
        (hasBias && !product({B, Hq, Sq, Sk}, counts[3]))) return 5;
    const bool narrow =
        std::strncmp(name, "tessera_tile_attention_f16_", 27) == 0 ||
        std::strncmp(name, "tessera_tile_attention_bf16_", 28) == 0;
    const size_t elementBytes = narrow ? 2 : 4;
    size_t sizes[6] = {};
    for (int i = 0; i < 3; ++i) {
        if (counts[i] > SIZE_MAX / elementBytes) return 5;
        sizes[i] = counts[i] * elementBytes;
    }
    if (hasBias && counts[3] > SIZE_MAX / sizeof(float)) return 5;
    if (hasBias) sizes[3] = counts[3] * sizeof(float);
    if (counts[outputIndex] > SIZE_MAX / sizeof(float)) return 5;
    sizes[outputIndex] = counts[outputIndex] * sizeof(float);
    if (hasSavedLse) {
        if (counts[lseIndex] > SIZE_MAX / sizeof(float)) return 5;
        sizes[lseIndex] = counts[lseIndex] * sizeof(float);
    }
    CUdeviceptr device[6] = {};
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
        void* args[13] = {};
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

int benchmarkAttentionBackward(CUfunction fn, const char* name, void** buffers,
                               size_t nbuf, const int64_t* dims, size_t ndim,
                               int warmup, int repetitions, float* latencyMs) {
    if (ndim != 7 || !name || !latencyMs || warmup < 0 || repetitions <= 0)
        return 5;
    const bool hasSavedLse = std::strstr(name, "_lse_") != nullptr;
    if ((!hasSavedLse && nbuf != 7 && nbuf != 8) ||
        (hasSavedLse && nbuf != 8 && nbuf != 9)) return 5;
    const bool hasBias = hasSavedLse ? nbuf == 9 : nbuf == 8;
    const size_t lseIndex = 4 + size_t(hasBias);
    const size_t outputBase = lseIndex + size_t(hasSavedLse);
    const long long B=dims[0], Hq=dims[1], Hkv=dims[2], Sq=dims[3];
    const long long Sk=dims[4], D=dims[5], Dv=dims[6];
    if (B<=0 || Hq<=0 || Hkv<=0 || Sq<=0 || Sk<=0 || D<=0 || Dv<=0 || Hq%Hkv)
        return 5;
    const bool narrow = std::strstr(name, "attention_backward_f16_") != nullptr ||
                        std::strstr(name, "attention_backward_bf16_") != nullptr;
    const size_t elementBytes = narrow ? 2 : 4;
    auto bytes = [](std::initializer_list<size_t> values, size_t width, size_t& out) {
        out = width;
        for (size_t value : values) { if (value && out > SIZE_MAX / value) return false; out *= value; }
        return true;
    };
    size_t doBytes=0, qBytes=0, kBytes=0, vBytes=0, lseBytes=0, biasBytes=0;
    if (!bytes({(size_t)B,(size_t)Hq,(size_t)Sq,(size_t)Dv},elementBytes,doBytes) ||
        !bytes({(size_t)B,(size_t)Hq,(size_t)Sq,(size_t)D},elementBytes,qBytes) ||
        !bytes({(size_t)B,(size_t)Hkv,(size_t)Sk,(size_t)D},elementBytes,kBytes) ||
        !bytes({(size_t)B,(size_t)Hkv,(size_t)Sk,(size_t)Dv},elementBytes,vBytes) ||
        (hasSavedLse && !bytes({(size_t)B,(size_t)Hq,(size_t)Sq},4,lseBytes)) ||
        (hasBias && !bytes({(size_t)B,(size_t)Hq,(size_t)Sq,(size_t)Sk},4,biasBytes))) return 5;
    size_t sizes[9] = {doBytes,qBytes,kBytes,vBytes,0,0,0,0,0};
    if (hasBias) sizes[4] = biasBytes;
    if (hasSavedLse) sizes[lseIndex] = lseBytes;
    sizes[outputBase] = qBytes; sizes[outputBase+1] = kBytes; sizes[outputBase+2] = vBytes;
    CUdeviceptr device[9] = {}; CUevent start=nullptr, stop=nullptr; int rc=0;
    for (size_t i=0; i<nbuf; ++i)
        if (cuMemAlloc(&device[i], sizes[i]) != CUDA_SUCCESS) { rc=3; break; }
    if (!rc) for (size_t i=0; i<outputBase; ++i)
        if (cuMemcpyHtoD(device[i], buffers[i], sizes[i]) != CUDA_SUCCESS) { rc=3; break; }
    if (!rc) {
        long long args64[7]; for (int i=0;i<7;++i) args64[i]=dims[i];
        void* args[16] = {}; size_t arg=0;
        for (size_t i=0;i<nbuf;++i) args[arg++]=&device[i];
        for (int i=0;i<7;++i) args[arg++]=&args64[i];
        size_t elements=qBytes/elementBytes+kBytes/elementBytes+vBytes/elementBytes;
        unsigned grid=(unsigned)((elements+127)/128);
        auto launch = [&]() { return cuLaunchKernel(fn, grid,1,1,128,1,1,0,0,args,0); };
        for (int i=0;i<warmup;++i) if (launch()!=CUDA_SUCCESS) { rc=3; break; }
        if (!rc && (cuCtxSynchronize()!=CUDA_SUCCESS || cuEventCreate(&start, CU_EVENT_DEFAULT)!=CUDA_SUCCESS ||
                    cuEventCreate(&stop, CU_EVENT_DEFAULT)!=CUDA_SUCCESS || cuEventRecord(start,0)!=CUDA_SUCCESS)) rc=3;
        for (int i=0;!rc && i<repetitions;++i) if (launch()!=CUDA_SUCCESS) rc=3;
        if (!rc && (cuEventRecord(stop,0)!=CUDA_SUCCESS || cuEventSynchronize(stop)!=CUDA_SUCCESS)) rc=3;
        float totalMs=0.0f;
        if (!rc && cuEventElapsedTime(&totalMs,start,stop)!=CUDA_SUCCESS) rc=3;
        if (!rc) *latencyMs=totalMs/(float)repetitions;
    }
    if (start) cuEventDestroy(start); if (stop) cuEventDestroy(stop);
    for (CUdeviceptr ptr:device) if (ptr) cuMemFree(ptr);
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

struct TrainingLayout {
    size_t sizes[16] = {};
    bool outputs[16] = {};
    unsigned grid = 0;
};

bool checkedBytes(long long count, size_t elementBytes, size_t& result) {
    if (count <= 0 || static_cast<unsigned long long>(count) >
                          SIZE_MAX / elementBytes)
        return false;
    result = static_cast<size_t>(count) * elementBytes;
    return true;
}

size_t trainingStorageBytes(const char* name) {
    if (std::strstr(name, "_bf16") || std::strstr(name, "_f16"))
        return sizeof(unsigned short);
    return sizeof(float);
}

bool trainingLayout(const char* name, size_t nbuf, const int64_t* dims,
                    size_t ndim, TrainingLayout& layout) {
    if (std::strncmp(name, kTrainingPrefix, std::strlen(kTrainingPrefix)) != 0)
        return false;
    long long work = 0;
    bool oneBlockPerWork = false;
    if (std::strncmp(name, "tessera_cuda_training_norm_", 27) == 0) {
        if (nbuf != 6 || ndim != 2 || dims[0] <= 0 || dims[1] <= 0 ||
            dims[0] > LLONG_MAX / dims[1])
            return false;
        const long long rows = dims[0], columns = dims[1];
        size_t matrixBytes = 0, vectorBytes = 0;
        const size_t storageBytes = trainingStorageBytes(name);
        if (!checkedBytes(rows * columns, storageBytes, matrixBytes) ||
            !checkedBytes(columns, storageBytes, vectorBytes))
            return false;
        layout.sizes[0] = matrixBytes;
        layout.sizes[1] = vectorBytes;
        layout.sizes[2] = matrixBytes;
        layout.sizes[3] = matrixBytes;
        if (!checkedBytes(columns, sizeof(float), layout.sizes[4]) ||
            !checkedBytes(columns, sizeof(float), layout.sizes[5]))
            return false;
        layout.outputs[3] = layout.outputs[4] = layout.outputs[5] = true;
        work = rows > columns ? rows : columns;
    } else if (std::strncmp(name, "tessera_cuda_training_loss_", 27) == 0) {
        if (nbuf != 5 || ndim != 1) return false;
        size_t bytes = 0;
        if (!checkedBytes(dims[0], trainingStorageBytes(name), bytes))
            return false;
        for (size_t i = 0; i < nbuf; ++i) layout.sizes[i] = bytes;
        if (std::strstr(name, "_sum") || std::strstr(name, "_mean"))
            layout.sizes[2] = trainingStorageBytes(name);
        layout.outputs[3] = layout.outputs[4] = true;
        work = dims[0];
    } else if (std::strncmp(name, "tessera_cuda_training_class_", 28) == 0) {
        if (nbuf != 4 || ndim != 2 || dims[0] <= 0 || dims[1] <= 0 ||
            dims[0] > LLONG_MAX / dims[1])
            return false;
        size_t matrixBytes = 0, rowBytes = 0, labelBytes = 0;
        const size_t storageBytes = trainingStorageBytes(name);
        if (!checkedBytes(dims[0] * dims[1], storageBytes, matrixBytes) ||
            !checkedBytes(dims[0], storageBytes, rowBytes) ||
            !checkedBytes(dims[0], sizeof(int64_t), labelBytes))
            return false;
        layout.sizes[0] = matrixBytes;
        layout.sizes[1] = labelBytes;
        layout.sizes[2] = rowBytes;
        if (std::strstr(name, "_sum_") || std::strstr(name, "_mean_"))
            layout.sizes[2] = storageBytes;
        layout.sizes[3] = matrixBytes;
        layout.outputs[3] = true;
        work = dims[0];
    } else if (std::strncmp(
                   name, "tessera_cuda_training_broadcast_reduce_", 39) == 0) {
        if (nbuf != 2 || ndim != 2) return false;
        if (!checkedBytes(
                dims[0], trainingStorageBytes(name), layout.sizes[0]) ||
            !checkedBytes(
                dims[1], trainingStorageBytes(name), layout.sizes[1]))
            return false;
        layout.outputs[1] = true;
        work = dims[1];
    } else if (std::strncmp(
                   name, "tessera_cuda_training_optvjp_",
                   std::strlen("tessera_cuda_training_optvjp_")) == 0) {
        if (ndim != 1) return false;
        size_t bytes = 0;
        if (!checkedBytes(dims[0], sizeof(float), bytes)) return false;
        if (std::strstr(name, "optvjp_sgd_")) {
            if (nbuf != 5) return false;
            for (size_t i = 0; i < nbuf; ++i) layout.sizes[i] = bytes;
            layout.outputs[3] = layout.outputs[4] = true;
        } else if (std::strstr(name, "optvjp_momentum_") ||
                   std::strstr(name, "optvjp_nesterov_")) {
            if (nbuf != 8) return false;
            for (size_t i = 0; i < nbuf; ++i) layout.sizes[i] = bytes;
            layout.outputs[5] = layout.outputs[6] = layout.outputs[7] = true;
        } else if (std::strstr(name, "optvjp_adam_") ||
                   std::strstr(name, "optvjp_adamw_")) {
            if (nbuf != 11) return false;
            for (size_t i = 0; i < nbuf; ++i) layout.sizes[i] = bytes;
            for (size_t i = 7; i < 11; ++i) layout.outputs[i] = true;
        } else {
            return false;
        }
        work = dims[0];
    } else if (std::strncmp(
                   name, "tessera_cuda_training_adafactorvjp_",
                   std::strlen("tessera_cuda_training_adafactorvjp_")) == 0) {
        if (std::strstr(name, "adafactorvjp_full_")) {
            if (nbuf != 7 || ndim != 1) return false;
            size_t bytes = 0;
            if (!checkedBytes(dims[0], sizeof(float), bytes)) return false;
            for (size_t i = 0; i < nbuf; ++i) layout.sizes[i] = bytes;
            layout.outputs[4] = layout.outputs[5] = layout.outputs[6] = true;
            work = dims[0];
        } else if (std::strstr(name, "adafactorvjp_factored_")) {
            if (nbuf != 9 || ndim != 2 || dims[0] <= 0 || dims[1] <= 0 ||
                dims[0] > LLONG_MAX / dims[1]) return false;
            size_t matrix = 0, rows = 0, columns = 0;
            if (!checkedBytes(dims[0] * dims[1], sizeof(float), matrix) ||
                !checkedBytes(dims[0], sizeof(float), rows) ||
                !checkedBytes(dims[1], sizeof(float), columns)) return false;
            layout.sizes[0] = layout.sizes[1] = layout.sizes[4] =
                layout.sizes[5] = layout.sizes[6] = matrix;
            layout.sizes[2] = layout.sizes[7] = rows;
            layout.sizes[3] = layout.sizes[8] = columns;
            for (size_t i = 5; i < 9; ++i) layout.outputs[i] = true;
            work = 1;
        } else {
            return false;
        }
    } else if (std::strncmp(name, "tessera_cuda_training_optimizer_sgd_", 36) == 0) {
        if (nbuf != 3 || ndim != 1) return false;
        size_t bytes = 0;
        if (!checkedBytes(dims[0], trainingStorageBytes(name), bytes))
            return false;
        for (size_t i = 0; i < nbuf; ++i) layout.sizes[i] = bytes;
        layout.outputs[2] = true;
        work = dims[0];
    } else if (std::strncmp(name, "tessera_cuda_training_optimizer_momentum_", 41) == 0 ||
               std::strncmp(name, "tessera_cuda_training_optimizer_nesterov_", 41) == 0) {
        if (nbuf != 5 || ndim != 1) return false;
        size_t storage = 0, state = 0;
        if (!checkedBytes(dims[0], trainingStorageBytes(name), storage) ||
            !checkedBytes(dims[0], sizeof(float), state))
            return false;
        layout.sizes[0] = layout.sizes[1] = layout.sizes[3] = storage;
        layout.sizes[2] = layout.sizes[4] = state;
        layout.outputs[3] = layout.outputs[4] = true;
        work = dims[0];
    } else if (std::strncmp(name, "tessera_cuda_training_optimizer_adam_", 37) == 0 ||
               std::strncmp(name, "tessera_cuda_training_optimizer_adamw_", 38) == 0) {
        if (nbuf != 7 || ndim != 1) return false;
        size_t storage = 0, state = 0;
        if (!checkedBytes(dims[0], trainingStorageBytes(name), storage) ||
            !checkedBytes(dims[0], sizeof(float), state))
            return false;
        layout.sizes[0] = layout.sizes[1] = layout.sizes[4] = storage;
        layout.sizes[2] = layout.sizes[3] =
            layout.sizes[5] = layout.sizes[6] = state;
        layout.outputs[4] = layout.outputs[5] = layout.outputs[6] = true;
        work = dims[0];
    } else if (std::strncmp(
                   name, "tessera_cuda_training_lion_backward_", 36) == 0) {
        // Lion's stop-sign VJP retains p/g/m for operation ownership, carries
        // two output cotangents, and returns f32 dp/dg/dm.
        if (nbuf != 8 || ndim != 1) return false;
        size_t storage = 0, state = 0;
        if (!checkedBytes(dims[0], trainingStorageBytes(name), storage) ||
            !checkedBytes(dims[0], sizeof(float), state))
            return false;
        layout.sizes[0] = layout.sizes[1] = storage;
        for (size_t i = 2; i < nbuf; ++i) layout.sizes[i] = state;
        layout.outputs[5] = layout.outputs[6] = layout.outputs[7] = true;
        work = dims[0];
    } else if (std::strncmp(
                   name, "tessera_cuda_training_deltanet_backward_", 40) == 0) {
        // Versioned four-stage ABI: Q/K, V/gate/dO, beta/decay, and all VJPs.
        if (nbuf != 13 || ndim != 10 || dims[0] <= 0 || dims[1] <= 0 ||
            dims[2] <= 0 || dims[3] <= 0 || dims[4] <= 0 ||
            dims[5] < 0 || dims[5] > 1 || dims[6] < 0 || dims[6] > 1 ||
            dims[7] < 0 || dims[7] > 1 || dims[8] < 0 || dims[8] > 1 ||
            dims[9] < 0 || dims[9] > 1 ||
            dims[0] > LLONG_MAX / dims[1] ||
            dims[0] * dims[1] > LLONG_MAX / dims[2]) return false;
        const long long bh = dims[0] * dims[1];
        const long long bhs = bh * dims[2];
        size_t qBytes = 0, vBytes = 0, scalarBytes = 0;
        if (!checkedBytes(bhs * dims[3], sizeof(float), qBytes) ||
            !checkedBytes(bhs * dims[4], sizeof(float), vBytes) ||
            !checkedBytes(bhs, sizeof(float), scalarBytes)) return false;
        layout.sizes[0] = layout.sizes[1] = layout.sizes[7] = layout.sizes[8] = qBytes;
        layout.sizes[2] = layout.sizes[3] = layout.sizes[6] = layout.sizes[9] = layout.sizes[10] = vBytes;
        layout.sizes[4] = layout.sizes[5] = layout.sizes[11] = layout.sizes[12] = scalarBytes;
        for (size_t i = 7; i < nbuf; ++i) layout.outputs[i] = true;
        work = bh;
        // The recurrence kernel uses only thread zero and maps blockIdx.x to
        // one (batch, head) trajectory.  Do not apply the elementwise /128
        // grid rule here or all but the first 128 trajectories are skipped.
        oneBlockPerWork = true;
    } else if (std::strncmp(name, "tessera_cuda_training_fused_", 28) == 0) {
        if (ndim != 1 || (nbuf != 6 && nbuf != 10)) return false;
        size_t storage = 0, state = 0;
        if (!checkedBytes(dims[0], trainingStorageBytes(name), storage) ||
            !checkedBytes(dims[0], sizeof(float), state))
            return false;
        for (size_t i = 0; i < nbuf; ++i) layout.sizes[i] = storage;
        if (std::strstr(name, "_sum_") || std::strstr(name, "_mean_"))
            layout.sizes[2] = trainingStorageBytes(name);
        if (nbuf == 6) {
            layout.outputs[4] = layout.outputs[5] = true;
        } else {
            layout.sizes[4] = layout.sizes[5] =
                layout.sizes[7] = layout.sizes[8] = state;
            layout.outputs[6] = layout.outputs[7] =
                layout.outputs[8] = layout.outputs[9] = true;
        }
        work = dims[0];
    } else {
        return false;
    }
    const unsigned long long blocks = oneBlockPerWork
        ? static_cast<unsigned long long>(work)
        : (static_cast<unsigned long long>(work) + 127ULL) / 128ULL;
    if (blocks == 0 || blocks > UINT_MAX) return false;
    layout.grid = static_cast<unsigned>(blocks);
    return true;
}

int runTraining(CUfunction fn, const char* name, void** buffers, size_t nbuf,
                const int64_t* dims, size_t ndim, int warmup,
                int repetitions, float* latencyMs) {
    if (!buffers || !dims || nbuf > 16 || ndim > 10 || warmup < 0 ||
        repetitions <= 0)
        return 5;
    TrainingLayout layout;
    if (!trainingLayout(name, nbuf, dims, ndim, layout)) return 5;
    CUdeviceptr device[16] = {};
    CUevent start = nullptr, stop = nullptr;
    int rc = 0;
    for (size_t i = 0; i < nbuf; ++i) {
        if (!buffers[i] ||
            cuMemAlloc(&device[i], layout.sizes[i]) != CUDA_SUCCESS) {
            rc = 3;
            break;
        }
    }
    for (size_t i = 0; !rc && i < nbuf; ++i) {
        if (!layout.outputs[i] &&
            cuMemcpyHtoD(device[i], buffers[i], layout.sizes[i]) != CUDA_SUCCESS)
            rc = 3;
    }
    long long args64[10] = {};
    void* args[26] = {};
    size_t arg = 0;
    for (size_t i = 0; i < nbuf; ++i) args[arg++] = &device[i];
    for (size_t i = 0; i < ndim; ++i) {
        args64[i] = dims[i];
        args[arg++] = &args64[i];
    }
    auto launch = [&]() {
        return cuLaunchKernel(fn, layout.grid, 1, 1, 128, 1, 1, 0, 0, args, 0);
    };
    for (int i = 0; !rc && i < warmup; ++i)
        if (launch() != CUDA_SUCCESS) rc = 3;
    if (!rc && cuCtxSynchronize() != CUDA_SUCCESS) rc = 3;
    if (!rc && latencyMs &&
        (cuEventCreate(&start, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
         cuEventCreate(&stop, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
         cuEventRecord(start, 0) != CUDA_SUCCESS))
        rc = 3;
    for (int i = 0; !rc && i < repetitions; ++i)
        if (launch() != CUDA_SUCCESS) rc = 3;
    if (!rc && latencyMs &&
        (cuEventRecord(stop, 0) != CUDA_SUCCESS ||
         cuEventSynchronize(stop) != CUDA_SUCCESS))
        rc = 3;
    if (!rc && !latencyMs && cuCtxSynchronize() != CUDA_SUCCESS) rc = 3;
    if (!rc && latencyMs) {
        float totalMs = 0.0f;
        if (cuEventElapsedTime(&totalMs, start, stop) != CUDA_SUCCESS)
            rc = 3;
        else
            *latencyMs = totalMs / static_cast<float>(repetitions);
    }
    for (size_t i = 0; !rc && i < nbuf; ++i) {
        if (layout.outputs[i] &&
            cuMemcpyDtoH(buffers[i], device[i], layout.sizes[i]) != CUDA_SUCCESS)
            rc = 3;
    }
    if (start) cuEventDestroy(start);
    if (stop) cuEventDestroy(stop);
    for (CUdeviceptr ptr : device)
        if (ptr) cuMemFree(ptr);
    return rc;
}

size_t align16(size_t value) {
    return (value + 15u) & ~size_t(15u);
}

int runDynamicSharedProbe(CUfunction fn, void** buffers, size_t nbuf,
                          const int64_t* dims, size_t ndim,
                          size_t dynamicSharedBytes, int warmup,
                          int repetitions, float* latencyMs) {
    if (!buffers || !buffers[0] || !dims || nbuf != 1 || ndim != 3 ||
        dims[0] <= 0 || dims[1] <= 0 || (dims[2] != 0 && dims[2] != 1) ||
        dynamicSharedBytes > UINT_MAX || warmup < 0 || repetitions <= 0)
        return 5;
    const auto thenBytes = static_cast<unsigned long long>(dims[0]);
    const auto elseBytes = static_cast<unsigned long long>(dims[1]);
    if (thenBytes > SIZE_MAX - 15u || elseBytes > SIZE_MAX - 15u)
        return 5;
    const size_t required = align16(static_cast<size_t>(
        thenBytes > elseBytes ? thenBytes : elseBytes));
    if (dynamicSharedBytes != required) return 5;
    CUdeviceptr output = 0;
    CUevent start = nullptr, stop = nullptr;
    int rc = cuMemAlloc(&output, 2 * sizeof(float)) == CUDA_SUCCESS ? 0 : 3;
    long long args64[3] = {dims[0], dims[1], dims[2]};
    void* args[] = {&output, &args64[0], &args64[1], &args64[2]};
    auto launch = [&]() {
        return cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1,
                              static_cast<unsigned>(dynamicSharedBytes),
                              0, args, 0);
    };
    for (int i = 0; !rc && i < warmup; ++i)
        if (launch() != CUDA_SUCCESS) rc = 3;
    if (!rc && cuCtxSynchronize() != CUDA_SUCCESS) rc = 3;
    if (!rc && latencyMs &&
        (cuEventCreate(&start, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
         cuEventCreate(&stop, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
         cuEventRecord(start, 0) != CUDA_SUCCESS))
        rc = 3;
    for (int i = 0; !rc && i < repetitions; ++i)
        if (launch() != CUDA_SUCCESS) rc = 3;
    if (!rc && latencyMs &&
        (cuEventRecord(stop, 0) != CUDA_SUCCESS ||
         cuEventSynchronize(stop) != CUDA_SUCCESS))
        rc = 3;
    if (!rc && !latencyMs && cuCtxSynchronize() != CUDA_SUCCESS) rc = 3;
    if (!rc && latencyMs) {
        float total = 0.0f;
        if (cuEventElapsedTime(&total, start, stop) != CUDA_SUCCESS)
            rc = 3;
        else
            *latencyMs = total / static_cast<float>(repetitions);
    }
    if (!rc && cuMemcpyDtoH(buffers[0], output, 2 * sizeof(float)) != CUDA_SUCCESS)
        rc = 3;
    if (start) cuEventDestroy(start);
    if (stop) cuEventDestroy(stop);
    if (output) cuMemFree(output);
    return rc;
}

int runDynamicSharedExpressionProbe(
    CUfunction fn, void** buffers, size_t nbuf, const int64_t* dims,
    size_t ndim, size_t dynamicSharedBytes, int warmup, int repetitions,
    float* latencyMs) {
    if (!buffers || !buffers[0] || !dims || nbuf != 1 || ndim != 5 ||
        dims[0] < 0 || dims[1] < 0 || dims[2] < 0 || dims[3] <= 0 ||
        (dims[4] != 0 && dims[4] != 1) ||
        dynamicSharedBytes > UINT_MAX || warmup < 0 || repetitions <= 0)
        return 5;
    const auto base = static_cast<unsigned long long>(dims[0]);
    const auto factor = static_cast<unsigned long long>(dims[1]);
    const auto bias = static_cast<unsigned long long>(dims[2]);
    const auto fallback = static_cast<unsigned long long>(dims[3]);
    if (base > SIZE_MAX || factor > SIZE_MAX || bias > SIZE_MAX ||
        fallback > SIZE_MAX - 15u)
        return 5;
    if (factor && base > (SIZE_MAX - bias) / factor)
        return 5;
    const size_t computed =
        static_cast<size_t>(base * factor + bias);
    if (computed < 2 || computed > SIZE_MAX - 15u)
        return 5;
    const size_t required = align16(
        computed > fallback ? computed : static_cast<size_t>(fallback));
    if (dynamicSharedBytes != required)
        return 5;

    CUdeviceptr output = 0;
    CUevent start = nullptr, stop = nullptr;
    int rc = cuMemAlloc(&output, 2 * sizeof(float)) == CUDA_SUCCESS ? 0 : 3;
    long long args64[5] = {
        dims[0], dims[1], dims[2], dims[3], dims[4]};
    void* args[] = {
        &output, &args64[0], &args64[1], &args64[2], &args64[3],
        &args64[4]};
    auto launch = [&]() {
        return cuLaunchKernel(
            fn, 1, 1, 1, 1, 1, 1,
            static_cast<unsigned>(dynamicSharedBytes), 0, args, 0);
    };
    for (int i = 0; !rc && i < warmup; ++i)
        if (launch() != CUDA_SUCCESS) rc = 3;
    if (!rc && cuCtxSynchronize() != CUDA_SUCCESS) rc = 3;
    if (!rc && latencyMs &&
        (cuEventCreate(&start, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
         cuEventCreate(&stop, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
         cuEventRecord(start, 0) != CUDA_SUCCESS))
        rc = 3;
    for (int i = 0; !rc && i < repetitions; ++i)
        if (launch() != CUDA_SUCCESS) rc = 3;
    if (!rc && latencyMs &&
        (cuEventRecord(stop, 0) != CUDA_SUCCESS ||
         cuEventSynchronize(stop) != CUDA_SUCCESS))
        rc = 3;
    if (!rc && !latencyMs && cuCtxSynchronize() != CUDA_SUCCESS) rc = 3;
    if (!rc && latencyMs) {
        float total = 0.0f;
        if (cuEventElapsedTime(&total, start, stop) != CUDA_SUCCESS)
            rc = 3;
        else
            *latencyMs = total / static_cast<float>(repetitions);
    }
    if (!rc &&
        cuMemcpyDtoH(buffers[0], output, 2 * sizeof(float)) != CUDA_SUCCESS)
        rc = 3;
    if (start) cuEventDestroy(start);
    if (stop) cuEventDestroy(stop);
    if (output) cuMemFree(output);
    return rc;
}

// Shared launch body behind both the direct C-ABI and the tsrGpuLauncherFn.
int invokeImpl(const char* kernel_name, void** buffers, size_t nbuf,
               const int64_t* dims, size_t ndim,
               size_t dynamicSharedBytes = 0) {
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
        // ragged=true: these two entries are the general mma.sync GEMM, whose
        // kernel now predicates its own boundaries -- M/N by clamping the load
        // and suppressing the store, K by a predicated remainder slab. The
        // `ragged` parameter has existed since this function was written and
        // no caller had ever passed it true, so the M%16/N%8/K%16 guard below
        // rejected every unaligned shape with rc=5 and dispatch fell to the
        // reference.
        //
        // Correction (review on #675): an earlier version of this comment said
        // "the Tile-direct and scheduled kernels are still aligned-only". They
        // never were -- every one of them passes `ragged=true` below, having
        // masked its boundaries in NVIDIALowering.cpp since it was written.
        // What IS specific to these two entries is `requiresEvenK`.
        return invokeMmaGemm16(fn, buffers, nbuf, dims, ndim,
                               /*tileM=*/16, /*tileN=*/8, /*threads=*/32,
                               /*ragged=*/true, /*columnMajorGrid=*/false,
                               /*dimensions64=*/false, /*elementBytes=*/2,
                               /*outputBytes=*/4, /*requiresEvenK=*/true);
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
    if ((std::strncmp(kernel_name, kTileSharedF16,
                      std::strlen(kTileSharedF16)) == 0 ||
         std::strncmp(kernel_name, kTileSharedBf16,
                      std::strlen(kTileSharedBf16)) == 0) &&
        std::strstr(kernel_name, "_outf16") != nullptr)
        return invokeMmaGemm16(fn, buffers, nbuf, dims, ndim,
                               32, 32, 128, true, true, true, 2, 2);
    if (std::strcmp(kernel_name, kTileSharedF16) == 0 ||
        std::strcmp(kernel_name, kTileSharedBf16) == 0)
        return invokeMmaGemm16(fn, buffers, nbuf, dims, ndim,
                               32, 32, 128, true, true, true);
    if (std::strncmp(kernel_name, kScheduledSm120MatmulPrefix,
                     std::strlen(kScheduledSm120MatmulPrefix)) == 0) {
        if (std::strstr(kernel_name, "_fused_") != nullptr)
            return invokeFusedMatmul16(fn, kernel_name, buffers, nbuf, dims, ndim);
        const size_t outputBytes =
            std::strstr(kernel_name, "_outf16") != nullptr ? 2 : 4;
        return std::strstr(kernel_name, "_macro_kernel") != nullptr
            ? invokeMmaGemm16(fn, buffers, nbuf, dims, ndim,
                              32, 32, 128, true, true, true, 2, outputBytes)
            : invokeMmaGemm16(fn, buffers, nbuf, dims, ndim,
                              16, 8, 32, true, true, true, 2, outputBytes);
    }
    if (std::strncmp(kernel_name, "tessera_tile_matmul_fused_", 26) == 0)
        return invokeFusedMatmul16(fn, kernel_name, buffers, nbuf, dims, ndim);
    if (std::strcmp(kernel_name, kTileNvfp4) == 0)
        return invokeNvfp4(fn, buffers, nbuf, dims, ndim);
    if (std::strcmp(kernel_name, kTileInt4) == 0)
        return invokeInt4(fn, buffers, nbuf, dims, ndim);
    if (std::strncmp(kernel_name, kTileCudaIntrinsic,
                     std::strlen(kTileCudaIntrinsic)) == 0)
        return invokeCudaIntrinsic(fn, buffers, nbuf, dims, ndim);
    if (std::strncmp(kernel_name, kTilePackedDecode,
                     std::strlen(kTilePackedDecode)) == 0)
        return invokePackedDecode(fn, buffers, nbuf, dims, ndim);
    if (std::strcmp(kernel_name, kTileMxE2m3) == 0 ||
        std::strcmp(kernel_name, kTileMxE3m2) == 0)
        return invokeMx(fn, buffers, nbuf, dims, ndim, false);
    if (std::strcmp(kernel_name, kTileMxFp4) == 0)
        return invokeMx(fn, buffers, nbuf, dims, ndim, true);
    if (std::strcmp(kernel_name, kTileSoftmaxF16) == 0)
        return invokeSoftmax(fn, buffers, nbuf, dims, ndim, 2);
    if (std::strcmp(kernel_name, kTileSoftmaxBf16) == 0)
        return invokeSoftmax(fn, buffers, nbuf, dims, ndim, 2);
    if (std::strcmp(kernel_name, kTileSoftmaxF32) == 0)
        return invokeSoftmax(fn, buffers, nbuf, dims, ndim, 4);
    if (std::strncmp(kernel_name, kTileReducePrefix,
                     std::strlen(kTileReducePrefix)) == 0) {
        const bool narrow = std::strstr(kernel_name, "_f16_") != nullptr ||
                            std::strstr(kernel_name, "_bf16_") != nullptr;
        return invokeReduce(
            fn, buffers, nbuf, dims, ndim, narrow ? 2 : 4,
            std::strstr(kernel_name, "_cooperative_128") != nullptr);
    }
    if (std::strncmp(kernel_name, kTileNormPrefix,
                     std::strlen(kTileNormPrefix)) == 0) {
        const bool narrow = std::strstr(kernel_name, "_f16_") != nullptr ||
                            std::strstr(kernel_name, "_bf16_") != nullptr;
        return invokeSoftmax(fn, buffers, nbuf, dims, ndim, narrow ? 2 : 4);
    }
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
    if (std::strncmp(kernel_name, kTrainingPrefix,
                     std::strlen(kTrainingPrefix)) == 0)
        return runTraining(fn, kernel_name, buffers, nbuf, dims, ndim,
                           0, 1, nullptr);
    if (std::strncmp(kernel_name, kDynamicSharedPrefix,
                     std::strlen(kDynamicSharedPrefix)) == 0) {
        if (std::strstr(kernel_name, "_local_expr_"))
            return runDynamicSharedExpressionProbe(
                fn, buffers, nbuf, dims, ndim, dynamicSharedBytes,
                0, 1, nullptr);
        return runDynamicSharedProbe(fn, buffers, nbuf, dims, ndim,
                                     dynamicSharedBytes, 0, 1, nullptr);
    }
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

int tessera_nvidia_ptx_invoke_v2(const char* kernel_name, void** buffers,
                                 size_t num_buffers, const int64_t* dims,
                                 size_t num_dims,
                                 size_t dynamic_shared_bytes) {
    return invokeImpl(kernel_name, buffers, num_buffers, dims, num_dims,
                      dynamic_shared_bytes);
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
    if (std::strncmp(kernel_name, kDynamicSharedPrefix,
                     std::strlen(kDynamicSharedPrefix)) == 0)
        return 5;  // dynamic kernels require benchmark_v2.
    if (std::strcmp(kernel_name, kTileMxE2m3) == 0 ||
        std::strcmp(kernel_name, kTileMxE3m2) == 0 ||
        std::strcmp(kernel_name, kTileMxFp4) == 0)
        return benchmarkMx(fn, kernel_name, buffers, num_buffers, dims,
                           num_dims, warmup, repetitions, latency_ms);
    if (std::strncmp(kernel_name, "tessera_tile_softmax_", 21) == 0 ||
        std::strncmp(kernel_name, "tessera_tile_reduce_", 20) == 0 ||
        std::strncmp(kernel_name, kTileNormPrefix,
                     std::strlen(kTileNormPrefix)) == 0)
        return benchmarkUnary(fn, kernel_name, buffers, num_buffers, dims,
                              num_dims, warmup, repetitions, latency_ms);
    if (std::strncmp(kernel_name, kTileAttentionBackwardPrefix,
                     std::strlen(kTileAttentionBackwardPrefix)) == 0)
        return benchmarkAttentionBackward(fn, kernel_name, buffers, num_buffers,
                                          dims, num_dims, warmup, repetitions,
                                          latency_ms);
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
    if (std::strncmp(kernel_name, kTrainingPrefix,
                     std::strlen(kTrainingPrefix)) == 0)
        return runTraining(fn, kernel_name, buffers, num_buffers, dims,
                           num_dims, warmup, repetitions, latency_ms);
    if (std::strncmp(kernel_name, kTilePackedDecode,
                     std::strlen(kTilePackedDecode)) == 0)
        return benchmarkPackedDecode(
            fn, buffers, num_buffers, dims, num_dims, warmup, repetitions,
            latency_ms);
    return benchmarkTileGemm16(fn, kernel_name, buffers, num_buffers, dims,
                               num_dims, warmup, repetitions, latency_ms);
}

int tessera_nvidia_ptx_benchmark_v2(const char* kernel_name, void** buffers,
                                    size_t num_buffers, const int64_t* dims,
                                    size_t num_dims,
                                    size_t dynamic_shared_bytes, int warmup,
                                    int repetitions, float* latency_ms) {
    if (!kernel_name || !buffers || !dims || !latency_ms) return 5;
    if (!ensureContext()) return 2;
    std::lock_guard<std::mutex> lock(g_mu);
    CUfunction fn = getFunctionLocked(kernel_name);
    if (!fn) return g_ptx.count(kernel_name) ? 3 : 4;
    if (std::strncmp(kernel_name, kDynamicSharedPrefix,
                     std::strlen(kDynamicSharedPrefix)) != 0)
        return 5;
    if (std::strstr(kernel_name, "_local_expr_"))
        return runDynamicSharedExpressionProbe(
            fn, buffers, num_buffers, dims, num_dims, dynamic_shared_bytes,
            warmup, repetitions, latency_ms);
    return runDynamicSharedProbe(
        fn, buffers, num_buffers, dims, num_dims, dynamic_shared_bytes,
        warmup, repetitions, latency_ms);
}

int tessera_nvidia_ptx_resources(const char* kernel_name, int block_size,
                                 size_t dynamic_shared_bytes,
                                 int* registers_per_thread,
                                 int* static_shared_bytes,
                                 int* local_bytes,
                                 int* active_blocks_per_sm) {
    if (!kernel_name || block_size <= 0 || dynamic_shared_bytes > UINT_MAX ||
        !registers_per_thread || !static_shared_bytes || !local_bytes ||
        !active_blocks_per_sm)
        return 5;
    if (!ensureContext()) return 2;
    std::lock_guard<std::mutex> lock(g_mu);
    CUfunction fn = getFunctionLocked(kernel_name);
    if (!fn) return g_ptx.count(kernel_name) ? 3 : 4;
    if (cuFuncGetAttribute(registers_per_thread, CU_FUNC_ATTRIBUTE_NUM_REGS,
                           fn) != CUDA_SUCCESS ||
        cuFuncGetAttribute(static_shared_bytes,
                           CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                           fn) != CUDA_SUCCESS ||
        cuFuncGetAttribute(local_bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                           fn) != CUDA_SUCCESS ||
        cuOccupancyMaxActiveBlocksPerMultiprocessor(
            active_blocks_per_sm, fn, block_size,
            dynamic_shared_bytes) != CUDA_SUCCESS)
        return 3;
    return 0;
}

int tessera_nvidia_ptx_device_memory(size_t* total_bytes, size_t* free_bytes) {
    if (!total_bytes || !free_bytes) return 5;
    if (!ensureContext()) return 2;
    size_t total = 0;
    size_t free = 0;
    if (cuMemGetInfo(&free, &total) != CUDA_SUCCESS) return 3;
    *total_bytes = total;
    *free_bytes = free;
    return 0;
}

int tessera_nvidia_register_ptx_launcher(void) {
    if (tsrRegisterGpuLauncher == nullptr) return 2;  // core runtime not in the load
    return tsrRegisterGpuLauncher(gpuLauncher, nullptr) == TSR_STATUS_SUCCESS ? 0 : 1;
}

}  // extern "C"
