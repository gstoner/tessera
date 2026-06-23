// tessera_rocm_flash_attn.cpp — shipped ROCm WMMA flash-attention runtime symbol.
//
// Exposes the stable C-ABI entry points ``tessera_rocm_wmma_flash_attn_f16`` /
// ``_bf16`` that run a real RDNA 3.5 WMMA flash-attention forward pass on the
// AMD GPU. These are the *shipped* runtime symbols the backend_manifest
// ``hardware_verified`` contract requires for the ``flash_attn`` op on the rocm
// target (the numerical proof lives in a checked-in execute_compare_fixture that
// calls them), and the second op — after ``matmul`` — to execute natively on a
// non-Apple backend.
//
// Like the GEMM symbol, the device kernel is compiled at load time with HIPRTC
// for whatever arch the device enumerates (gfx1151 on the Strix Halo box), so
// this object is built by the ordinary host C++ compiler and only needs the HIP
// runtime + HIPRTC at link time — no hipcc-as-compiler.
//
// Algorithm: FA-2-style tiled flash attention, single wave (32 lanes) per
// (query-tile-of-16, batch*head). Both matmuls use RDNA 16x16x16 WMMA
// (f16/bf16 in, f32 accumulate):
//   1. S = scale * Q @ K^T        (WMMA over head_dim chunks)  -> LDS scores
//   2. online softmax over S       (running max m, running sum l, rescale)
//   3. O += P @ V                  (WMMA over head_dim chunks)  -> LDS accumulator
// Scores and the output accumulator are staged in LDS; the row softmax is done
// with one lane per query row. Causal masking and ragged Sq/Sk (zero-pad load +
// -inf score mask + bounds-checked store) are handled. This is the
// correctness-first "rung 0" of attention (the analog of the naive GEMM tile
// before its perf ladder) — the proof that the op executes and matches a CPU
// reference, not a perf-tuned kernel. head_dim must be a multiple of 16.
//
// Fragment/accumulator layout per the RDNA 3.5 ISA (matches tessera_rocm_gemm.cpp
// + rocdl_emit.py): WMMA A frag a[i] = A[row=lane&15, k=i]; B frag b[i] =
// B[k=i, col=lane&15]; C/D frag holds 8 floats/lane, output row = 2*e+(lane>>4),
// col = lane&15.

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace {

// One HIPRTC source template per (storage dtype, head_dim). ``%TYPE%`` /
// ``%WMMA%`` / ``%NAME%`` / ``%D%`` are substituted at load. head_dim (%D%) is a
// literal int so the d-chunk loops fully unroll and the LDS tiles are constexpr.
const char* kFlashTemplate = R"HIPSRC(
typedef %TYPE% wt16 __attribute__((ext_vector_type(16)));
typedef float  f8   __attribute__((ext_vector_type(8)));
extern "C" __global__ void %NAME%(
    const %TYPE%* Q, const %TYPE%* Kk, const %TYPE%* V, float* O,
    int B, int H, int Sq, int Sk, float scale, int causal) {
  constexpr int D = %D%;
  constexpr int DC = D / 16;            // head_dim chunks
  __shared__ %TYPE% sQ[16 * D];         // staged Q tile (row, d)
  __shared__ float  sS[16 * 16];        // scores -> probs (qi, ki)
  __shared__ float  sAcc[16 * D];       // output accumulator (qi, d)
  __shared__ float  sm[16];             // running row max
  __shared__ float  sl[16];             // running row sum
  __shared__ float  scorr[16];          // per-row rescale factor this step
  int tid = threadIdx.x;                // 0..31 (one wave)
  int lane = tid & 31, l15 = lane & 15, half = lane >> 4;
  int qtile = blockIdx.x, bh = blockIdx.y;
  long qbase = (long)bh * Sq * D;       // (b,h) slice of Q / O
  long kbase = (long)bh * Sk * D;       // (b,h) slice of K / V
  int q0 = qtile * 16;
  for (int i = tid; i < 16 * D; i += 32) {
    int r = i / D, c = i % D, gq = q0 + r;
    sQ[i] = (gq < Sq) ? Q[qbase + (long)gq * D + c] : (%TYPE%)0;
    sAcc[i] = 0.0f;
  }
  if (tid < 16) { sm[tid] = -1e30f; sl[tid] = 0.0f; }
  __syncthreads();
  int nKV = (Sk + 15) / 16;
  int lastKt = causal ? ((q0 + 15) / 16) : (nKV - 1);
  if (lastKt > nKV - 1) lastKt = nKV - 1;
  for (int kt = 0; kt <= lastKt; ++kt) {
    int k0 = kt * 16;
    // ---- S = scale * Q @ K^T (WMMA accumulate over d-chunks) ----
    f8 cs = (f8){0,0,0,0,0,0,0,0};
    for (int dc = 0; dc < DC; ++dc) {
      wt16 a, b;
      for (int i = 0; i < 16; ++i) a[i] = sQ[l15 * D + dc * 16 + i];
      int kr = k0 + l15;
      for (int i = 0; i < 16; ++i)
        b[i] = (kr < Sk) ? Kk[kbase + (long)kr * D + dc * 16 + i] : (%TYPE%)0;
      cs = %WMMA%(a, b, cs);
    }
    for (int e = 0; e < 8; ++e) {
      int qi = 2 * e + half, ki = l15, gk = k0 + ki;
      float v = cs[e] * scale;
      if (gk >= Sk) v = -1e30f;
      else if (causal && (q0 + qi) < gk) v = -1e30f;
      sS[qi * 16 + ki] = v;
    }
    __syncthreads();
    // ---- online softmax: lanes 0..15 each own one query row ----
    if (tid < 16) {
      int qi = tid;
      float rmax = -1e30f;
      for (int ki = 0; ki < 16; ++ki) rmax = fmaxf(rmax, sS[qi * 16 + ki]);
      float mold = sm[qi], mnew = fmaxf(mold, rmax);
      float corr = (mold <= -1e30f) ? 0.0f : __expf(mold - mnew);
      float rsum = 0.0f;
      for (int ki = 0; ki < 16; ++ki) {
        float p = __expf(sS[qi * 16 + ki] - mnew);   // -inf - finite -> 0
        sS[qi * 16 + ki] = p;
        rsum += p;
      }
      sl[qi] = sl[qi] * corr + rsum;
      sm[qi] = mnew;
      scorr[qi] = corr;
    }
    __syncthreads();
    for (int i = tid; i < 16 * D; i += 32) sAcc[i] *= scorr[i / D];
    __syncthreads();
    // ---- O += P @ V (WMMA accumulate over d-chunks) ----
    for (int dc = 0; dc < DC; ++dc) {
      wt16 ap, bv;
      for (int i = 0; i < 16; ++i) ap[i] = (%TYPE%)sS[l15 * 16 + i];
      for (int i = 0; i < 16; ++i) {
        int kr = k0 + i;
        bv[i] = (kr < Sk) ? V[kbase + (long)kr * D + dc * 16 + l15] : (%TYPE%)0;
      }
      f8 cpv = (f8){0,0,0,0,0,0,0,0};
      cpv = %WMMA%(ap, bv, cpv);
      for (int e = 0; e < 8; ++e) {
        int qi = 2 * e + half, d = dc * 16 + l15;
        sAcc[qi * D + d] += cpv[e];
      }
    }
    __syncthreads();
  }
  // ---- O = acc / l ----
  for (int i = tid; i < 16 * D; i += 32) {
    int r = i / D, c = i % D, gq = q0 + r;
    if (gq < Sq) {
      float denom = sl[r];
      O[qbase + (long)gq * D + c] = (denom > 0.0f) ? sAcc[i] / denom : 0.0f;
    }
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

// HIPRTC-compile a fully-substituted kernel for device 0's arch; caller owns mod.
bool compileSrc(const std::string& src, const std::string& name,
                hipModule_t* outMod, hipFunction_t* outFn) {
  hipDeviceProp_t props;
  if (hipGetDeviceProperties(&props, 0) != hipSuccess) return false;
  hiprtcProgram prog;
  if (hiprtcCreateProgram(&prog, src.c_str(), "tessera_rocm_flash_attn.hip",
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

// A flash kernel is specialized per (storage dtype, head_dim) because head_dim
// sizes the LDS tiles + unroll. Cache compiled functions keyed by head_dim so a
// repeated call (the common case — head_dim is fixed per model) compiles once.
struct FlashKernel {
  const char* type;   // device element type
  const char* wmma;   // WMMA builtin
  const char* tag;    // unique kernel-name stem
  std::mutex mu;
  std::map<int, hipFunction_t> fns;   // head_dim -> function
  std::vector<hipModule_t> mods;      // owned modules (freed at process exit)
};

FlashKernel g_f16{"__fp16", "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32",
                  "tessera_rocm_wmma_flash_attn_f16_kernel", {}, {}, {}};
FlashKernel g_bf16{"__bf16", "__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32",
                   "tessera_rocm_wmma_flash_attn_bf16_kernel", {}, {}, {}};

// Get (compiling + caching on first use) the kernel function for this head_dim.
// Returns nullptr if HIPRTC compile / module load failed (no device, etc.).
hipFunction_t kernelFor(FlashKernel* k, int D) {
  std::lock_guard<std::mutex> lock(k->mu);
  auto it = k->fns.find(D);
  if (it != k->fns.end()) return it->second;
  std::string name = std::string(k->tag) + "_d" + std::to_string(D);
  std::string src = substitute(kFlashTemplate, "%TYPE%", k->type);
  src = substitute(src, "%WMMA%", k->wmma);
  src = substitute(src, "%NAME%", name);
  src = substitute(src, "%D%", std::to_string(D));
  hipModule_t mod = nullptr;
  hipFunction_t fn = nullptr;
  if (!compileSrc(src, name, &mod, &fn)) return nullptr;
  k->mods.push_back(mod);
  k->fns[D] = fn;
  return fn;
}

// Run the flash-attention forward for one (storage dtype) over [B,H,Sq/Sk,D].
// Q/K/V are row-major f16/bf16 host buffers; O is a row-major f32 host buffer.
// Returns 0 ok / 1 bad shape / 2 no device or compile failed / 3 device op failed.
int runFlash(FlashKernel* k, const void* Q, const void* Kk, const void* V,
             void* O, int B, int H, int Sq, int Sk, int D, float scale,
             int causal, size_t elemBytes) {
  if (B <= 0 || H <= 0 || Sq <= 0 || Sk <= 0 || D <= 0 || (D % 16) != 0)
    return 1;
  hipFunction_t fn = kernelFor(k, D);
  if (!fn) return 2;
  const size_t qkv = (size_t)B * H;
  const size_t qElems = qkv * Sq * D, kElems = qkv * Sk * D;
  const size_t qBytes = qElems * elemBytes, kBytes = kElems * elemBytes;
  const size_t oBytes = qElems * sizeof(float);
  void *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
  int rc = 3;
  do {
    if (hipMalloc(&dQ, qBytes) != hipSuccess) break;
    if (hipMalloc(&dK, kBytes) != hipSuccess) break;
    if (hipMalloc(&dV, kBytes) != hipSuccess) break;
    if (hipMalloc(&dO, oBytes) != hipSuccess) break;
    if (hipMemcpy(dQ, Q, qBytes, hipMemcpyHostToDevice) != hipSuccess) break;
    if (hipMemcpy(dK, Kk, kBytes, hipMemcpyHostToDevice) != hipSuccess) break;
    if (hipMemcpy(dV, V, kBytes, hipMemcpyHostToDevice) != hipSuccess) break;
    unsigned gx = (unsigned)((Sq + 15) / 16);   // query tiles
    unsigned gy = (unsigned)(B * H);             // batch * head
    void* args[] = {&dQ, &dK, &dV, &dO, &B, &H, &Sq, &Sk, &scale, &causal};
    if (hipModuleLaunchKernel(fn, gx, gy, 1, 32, 1, 1, 0, nullptr, args,
                              nullptr) != hipSuccess) break;
    if (hipDeviceSynchronize() != hipSuccess) break;
    if (hipMemcpy(O, dO, oBytes, hipMemcpyDeviceToHost) != hipSuccess) break;
    rc = 0;
  } while (0);
  if (dQ) hipFree(dQ);
  if (dK) hipFree(dK);
  if (dV) hipFree(dV);
  if (dO) hipFree(dO);
  return rc;
}

}  // namespace

// Flash-attention forward, f16 storage / f32 accumulate + output.
// Q[B,H,Sq,D], K[B,H,Sk,D], V[B,H,Sk,D] (f16, row-major); O[B,H,Sq,D] (f32).
// scale multiplies the QK^T scores (pass 1/sqrt(D) for standard attention).
// causal != 0 applies a causal mask (query i attends only to keys <= i).
// head_dim D must be a multiple of 16. Returns 0 ok / 1 bad shape /
// 2 no usable HIP device or HIPRTC / 3 a device memory/launch/copy op failed.
extern "C" int tessera_rocm_wmma_flash_attn_f16(
    const void* Q, const void* K, const void* V, void* O,
    int B, int H, int Sq, int Sk, int D, float scale, int causal) {
  return runFlash(&g_f16, Q, K, V, O, B, H, Sq, Sk, D, scale, causal,
                  sizeof(unsigned short));
}

// As above, bf16 storage / f32 accumulate + output.
extern "C" int tessera_rocm_wmma_flash_attn_bf16(
    const void* Q, const void* K, const void* V, void* O,
    int B, int H, int Sq, int Sk, int D, float scale, int causal) {
  return runFlash(&g_bf16, Q, K, V, O, B, H, Sq, Sk, D, scale, causal,
                  sizeof(unsigned short));
}
