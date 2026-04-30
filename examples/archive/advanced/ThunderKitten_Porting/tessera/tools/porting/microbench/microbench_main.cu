
#include <cstdio>
#include <vector>
#include <random>
#include <cstring>
#include <string>
#include <mma.h>              // WMMA
#include <cuda_bf16.h>        // BF16 type
#include <cuda_runtime.h>
#include "tile_runtime.hpp"
#include "wgmma_utils.cuh"
#include "tma_utils.cuh"
#include "wgmma_desc.cuh"
#include "tma_real.cuh"
#include "epilogue_ops.cuh"
#include "wgmma_lane_map.cuh"

using namespace tessera;
using namespace nvcuda;

struct Knobs {
  int BM=128, BN=128, BK=32, STAGES=2, SPLITK=1, CTAPAIRS=0;
  int M=4096, N=4096, K=4096;
  int use_wgmma=0;  // 0=wmma/naive, 1=wgmma
  int use_tma=0;    // 0=cp.async, 1=tma
  int epilogue_kind=0; // 0=none,1=bias,2=bias_silu,3=bias_gelu
  int swizzle_kind=0;  // 0=identity,1=xor128b
  int tma_cols_per_copy=0; // 0=auto or explicit chunk
};

// ------------------ Compute paths ------------------

// WMMA (half) 16x16x16
template<int WM, int WN, int WK>
__device__ inline void mma_wmma_16x16x16(tile<float, WM, WN>& c,
                                         tile<half, WM, WK> const& a,
                                         tile<half, WK, WN> const& b) {
  static_assert(WM % 16 == 0 && WN % 16 == 0 && WK % 16 == 0, "WMMA tiles must be multiples of 16");
  #pragma unroll
  for (int m = 0; m < WM; m += 16) {
    #pragma unroll
    for (int n = 0; n < WN; n += 16) {
      wmma::fragment<wmma::accumulator, 16,16,16, float> cfrag;
      wmma::fill_fragment(cfrag, 0.0f);
      #pragma unroll
      for (int k = 0; k < WK; k += 16) {
        wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> afrag;
        wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> bfrag;
        wmma::load_matrix_sync(afrag, &a.data[(m)*WK + k], WK);
        wmma::load_matrix_sync(bfrag, &b.data[(k)*WN + n], WN);
        wmma::mma_sync(cfrag, afrag, bfrag, cfrag);
      }
      wmma::store_matrix_sync(&c.data[m*WN + n], cfrag, WN, wmma::mem_row_major);
    }
  }
}

// WGMMA (bf16) skeleton: if enabled, would issue Hopper/Blackwell wgmma.mma_async.
// Guarded by TESSERA_USE_WGMMA so default build works on any arch.
template<int WM, int WN, int WK>
__device__ inline void mma_wgmma_bf16_16x16x16(tile<float, WM, WN>& c,
                                               tile<__nv_bfloat16, WM, WK> const& a,
                                               tile<__nv_bfloat16, WK, WN> const& b) {
#if TESSERA_USE_WGMMA && (__CUDA_ARCH__ >= 900)
  // Placeholder: invoke per-16x16 block. Real code would pack operands into shared/registers and call inline PTX:
  // asm volatile("wgmma.mma_async.sync.aligned.m16n16k16.f32.bf16.bf16 ...");
  // For now, convert bf16->fp16 and reuse WMMA path to keep runnable.
  tile<half, WM, WK> ah;
  tile<half, WK, WN> bh;
  #pragma unroll
  for (int i=0;i<WM*WK;++i) ah.data[i] = __float2half(__bfloat162float(a.data[i]));
  #pragma unroll
  for (int i=0;i<WK*WN;++i) bh.data[i] = __float2half(__bfloat162float(b.data[i]));
  mma_wmma_16x16x16<WM,WN,WK>(c, ah, bh);
#else
  // Fallback: naive MMA to remain portable
  mma_sync<__nv_bfloat16, __nv_bfloat16, float, WM, WK, WN>(c, a, b);
#endif
}

// ------------------ Kernel ------------------
template<int BM, int BN, int BK, int STAGES>
__global__ void tile_gemm_kernel(const void* __restrict__ A,
                                 const void* __restrict__ B,
                                 float* __restrict__ C, float* __restrict__ Cwork,
                                 int M, int N, int K, int splitK, int use_wgmma, int use_tma) {
  extern __shared__ unsigned char smem_buf[];
  // Two dtypes in SMEM depending on compute path
  if (use_wgmma) {
    __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(smem_buf);
    __nv_bfloat16* Bs = As + STAGES*BM*BK;
    const __nv_bfloat16* Ag = reinterpret_cast<const __nv_bfloat16*>(A);
    const __nv_bfloat16* Bg = reinterpret_cast<const __nv_bfloat16*>(B);

    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;
    const int kpart    = blockIdx.z;
    int Kchunk = (K + splitK - 1) / splitK;
    int Kbeg = kpart * Kchunk;
    int Kend = min(K, Kbeg + Kchunk);
    const int row0 = tile_row * BM;
    const int col0 = tile_col * BN;

    tile<float, BM, BN> acc = tile_zero<float, BM, BN>();
    pipeline_t pipe;

    int k0 = Kbeg;
    int stage = 0;
    if (k0 < Kend) {
      if (use_wgmma) { smem_store_tile_swizzled(As + stage*BM*BK, Ag + row0*K + k0, min(BM, M - row0), min(BK, Kend - k0), K, BK); } else { async_copy_2d_select(&As[stage*BM*BK], Ag + row0*K + k0, min(BM, M - row0), min(BK, Kend - k0), K, BK, use_tma);
      if (use_wgmma) { smem_store_tile_swizzled(Bs + stage*BK*BN, Bg + k0*N + col0, min(BK, Kend - k0), min(BN, N - col0), N, BN); } else { async_copy_2d_select(&Bs[stage*BK*BN], Bg + k0*N + col0, min(BK, Kend - k0), min(BN, N - col0), N, BN, pipe, use_tma); } } } }
      pipe.commit();
    }

    for (; k0 < Kend; k0 += BK) {
      pipe.wait(stage);
      int next = (stage + 1) % STAGES;
      int k_next = k0 + BK;
      if (k_next < Kend) {
        if (use_wgmma) { smem_store_tile_swizzled(As + next*BM*BK, Ag + row0*K + k_next, min(BM, M - row0), min(BK, Kend - k_next), K, BK); } else { async_copy_2d_select(&As[next*BM*BK], Ag + row0*K + k_next, min(BM, M - row0), min(BK, Kend - k_next), K, BK, use_tma);
        if (use_wgmma) { smem_store_tile_swizzled(Bs + next*BK*BN, Bg + k_next*N + col0, min(BK, Kend - k_next), min(BN, N - col0), N, BN); } else { async_copy_2d_select(&Bs[next*BK*BN], Bg + k_next*N + col0, min(BK, Kend - k_next), min(BN, N - col0), N, BN, pipe, use_tma); } } } }
        pipe.commit();
      }

      tile<__nv_bfloat16, BM, BK> a;
      tile<__nv_bfloat16, BK, BN> b;
      // Shared-memory tiles are written with a (configurable) swizzle.
      // Build simple descriptors pointing at the base of the current stage tiles.
      smem_desc_2d Ad = make_smem_desc_2d(&As[stage*BM*BK], BK);
      smem_desc_2d Bd = make_smem_desc_2d(&Bs[stage*BK*BN], BN);
      // Optionally call WGMMA PTX when enabled; fall back otherwise.
      // NOTE: This PTX demo targets a 16x16x16 micro-tile. Integrate a full
      // mapping to cover all (BM,BN) by tiling over (m,n) in 16x16 steps.
      if constexpr (BM % 16 == 0 && BN % 16 == 0 && BK % 16 == 0) {
        // 64x64x16 warpgroup tiles across BMxBN
        for (int tm=0; tm<BM; tm+=64){
          for (int tn=0; tn<BN; tn+=64){
            float* csub = &acc.data[tm*BN + tn];
            wgmma_mma_64x64x16_bf16_f32(csub, BN, Ad.smem_ptr, Bd.smem_ptr);
          }
        }
      }

      for (int r=threadIdx.y; r<BM; r+=blockDim.y){
        for (int c=threadIdx.x; c<BK; c+=blockDim.x){
          a.data[r*BK + c] = As[stage*BM*BK + r*BK + c];
        }
      }
      for (int r=threadIdx.y; r<BK; r+=blockDim.y){
        for (int c=threadIdx.x; c<BN; c+=blockDim.x){
          b.data[r*BN + c] = Bs[stage*BK*BN + r*BN + c];
        }
      }
      __syncthreads();

      if constexpr (BM % 16 == 0 && BN % 16 == 0 && BK % 16 == 0) {
        wgmma_m16n16k16_bf16_f32(&acc.data[0], /*A_smem_desc*/nullptr, /*B_smem_desc*/nullptr, BN);
      } else {
        mma_sync<__nv_bfloat16, __nv_bfloat16, float, BM, BK, BN>(acc, a, b);
      }

      stage = next;
    }

    // Split-K accumulation
    for (int r=threadIdx.y; r<BM; r+=blockDim.y){
      int gr = row0 + r;
      if (gr >= M) continue;
      for (int c=threadIdx.x; c<BN; c+=blockDim.x){
        int gc = col0 + c;
        if (gc >= N) continue;
        if (splitK>1 && Cwork) { Cwork[(size_t)kpart*M*N + gr*N + gc] = acc.data[r*BN + c]; } else { atomicAdd(&C[gr*N + gc], acc.data[r*BN + c]); }
      }
    }
  } else {
    // WMMA/naive half path
    half* As = reinterpret_cast<half*>(smem_buf);
    half* Bs = As + STAGES*BM*BK;
    const half* Ag = reinterpret_cast<const half*>(A);
    const half* Bg = reinterpret_cast<const half*>(B);

    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;
    const int kpart    = blockIdx.z;
    int Kchunk = (K + splitK - 1) / splitK;
    int Kbeg = kpart * Kchunk;
    int Kend = min(K, Kbeg + Kchunk);
    const int row0 = tile_row * BM;
    const int col0 = tile_col * BN;

    tile<float, BM, BN> acc = tile_zero<float, BM, BN>();
    pipeline_t pipe;

    int k0 = Kbeg;
    int stage = 0;
    if (k0 < Kend) {
      if (use_wgmma) { smem_store_tile_swizzled(As + stage*BM*BK, Ag + row0*K + k0, min(BM, M - row0), min(BK, Kend - k0), K, BK); } else { async_copy_2d_select(&As[stage*BM*BK], Ag + row0*K + k0, min(BM, M - row0), min(BK, Kend - k0), K, BK, use_tma);
      if (use_wgmma) { smem_store_tile_swizzled(Bs + stage*BK*BN, Bg + k0*N + col0, min(BK, Kend - k0), min(BN, N - col0), N, BN); } else { async_copy_2d_select(&Bs[stage*BK*BN], Bg + k0*N + col0, min(BK, Kend - k0), min(BN, N - col0), N, BN, pipe, use_tma); } } } }
      pipe.commit();
    }

    for (; k0 < Kend; k0 += BK) {
      pipe.wait(stage);
      int next = (stage + 1) % STAGES;
      int k_next = k0 + BK;
      if (k_next < Kend) {
        if (use_wgmma) { smem_store_tile_swizzled(As + next*BM*BK, Ag + row0*K + k_next, min(BM, M - row0), min(BK, Kend - k_next), K, BK); } else { async_copy_2d_select(&As[next*BM*BK], Ag + row0*K + k_next, min(BM, M - row0), min(BK, Kend - k_next), K, BK, use_tma);
        if (use_wgmma) { smem_store_tile_swizzled(Bs + next*BK*BN, Bg + k_next*N + col0, min(BK, Kend - k_next), min(BN, N - col0), N, BN); } else { async_copy_2d_select(&Bs[next*BK*BN], Bg + k_next*N + col0, min(BK, Kend - k_next), min(BN, N - col0), N, BN, pipe, use_tma); } } } }
        pipe.commit();
      }

      tile<half, BM, BK> a;
      tile<half, BK, BN> b;
      for (int r=threadIdx.y; r<BM; r+=blockDim.y){
        for (int c=threadIdx.x; c<BK; c+=blockDim.x){
          a.data[r*BK + c] = As[stage*BM*BK + r*BK + c];
        }
      }
      for (int r=threadIdx.y; r<BK; r+=blockDim.y){
        for (int c=threadIdx.x; c<BN; c+=blockDim.x){
          b.data[r*BN + c] = Bs[stage*BK*BN + r*BN + c];
        }
      }
      __syncthreads();

      if constexpr (BM % 16 == 0 && BN % 16 == 0 && BK % 16 == 0) {
        mma_wmma_16x16x16<BM,BN,BK>(acc, a, b);
      } else {
        mma_sync<half, half, float, BM, BK, BN>(acc, a, b);
      }
      stage = next;
    }

    for (int r=threadIdx.y; r<BM; r+=blockDim.y){
      int gr = row0 + r;
      if (gr >= M) continue;
      for (int c=threadIdx.x; c<BN; c+=blockDim.x){
        int gc = col0 + c;
        if (gc >= N) continue;
        if (splitK>1 && Cwork) { Cwork[(size_t)kpart*M*N + gr*N + gc] = acc.data[r*BN + c]; } else { atomicAdd(&C[gr*N + gc], acc.data[r*BN + c]); }
      }
    }
  }
}

// Host dispatch

// Reduction kernel: sum split-K partials [M,N,SPLITK] -> [M,N]
__global__ void reduce_splitk(const float* __restrict__ Cpartial, float* __restrict__ C, float* __restrict__ Cwork,
                              int M, int N, int splitK) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r>=M || c>=N) return;
  float acc=0.f;
  for (int s=0; s<splitK; ++s){
    acc += Cpartial[(size_t)s*M*N + r*N + c];
  }
  C[r*N + c] = acc;
}

static void ck(cudaError_t e, const char* m){ if(e!=cudaSuccess){fprintf(stderr,"CUDA %s: %s\n",m,cudaGetErrorString(e)); std::exit(1);} }

template<int BM, int BN, int BK, int STAGES>
static float run_cfg(const Knobs& kb) {
  const int M=kb.M, N=kb.N, K=kb.K;
  size_t szA=(size_t)M*K, szB=(size_t)K*N, szC=(size_t)M*N;
  void *A=nullptr,*B=nullptr; float* C=nullptr; float* Cpartial=nullptr; float* Bias=nullptr;
  size_t esize = kb.use_wgmma ? sizeof(__nv_bfloat16) : sizeof(half);
  ck(cudaMalloc(&A, szA*esize),"malloc A");
  ck(cudaMalloc(&B, szB*esize),"malloc B");
  ck(cudaMalloc(&C, szC*sizeof(float)),"malloc C");
  if (kb.SPLITK>1) ck(cudaMalloc(&Cpartial, szC*kb.SPLITK*sizeof(float)),"malloc Cpartial");
  ck(cudaMemset(A, 0, szA*esize), "memset A");
  ck(cudaMemset(B, 0, szB*esize), "memset B");
  ck(cudaMemset(C, 0, szC*sizeof(float)), "memset C");
  if (kb.epilogue_kind) { ck(cudaMalloc(&Bias, N*sizeof(float)), "malloc Bias"); ck(cudaMemset(Bias, 0, N*sizeof(float)), "memset Bias"); }
  if (kb.SPLITK>1) ck(cudaMemset(Cpartial, 0, szC*kb.SPLITK*sizeof(float)), "memset Cpartial");

  dim3 block(32,8);
  dim3 grid((N+BN-1)/BN, (M+BM-1)/BM, kb.SPLITK);
  size_t smem = STAGES*(BM*BK*esize + BK*BN*esize);

  // Optional hint attributes for clusters/SMEM; left minimal to stay portable.
// NOTE: When TESSERA_REAL_TMA=ON, you can swap async_copy_2d_select() with tma_async_copy_2d() wrappers.


  // Warmup
  tile_gemm_kernel<BM,BN,BK,STAGES><<<grid, block, smem>>>(A,B,C,(kb.SPLITK>1?Cpartial:nullptr),M,N,K, kb.SPLITK, kb.use_wgmma, kb.use_tma);
  ck(cudaDeviceSynchronize(),"sync");

  const int iters = 5;
  float ms_total = 0.f;
  for (int i=0;i<iters;++i){
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    tile_gemm_kernel<BM,BN,BK,STAGES><<<grid, block, smem>>>(A,B,C,(kb.SPLITK>1?Cpartial:nullptr),M,N,K, kb.SPLITK, kb.use_wgmma, kb.use_tma);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms=0.f; cudaEventElapsedTime(&ms,s,e);
    ms_total += ms;
    cudaEventDestroy(s); cudaEventDestroy(e);
  }
  float ms_avg = ms_total / iters;
  if (kb.SPLITK>1) {
    dim3 b2(16,16); dim3 g2((N+15)/16,(M+15)/16);
    reduce_splitk<<<g2,b2>>>(Cpartial,C,M,N,kb.SPLITK);
    ck(cudaDeviceSynchronize(),"reduce sync");
  }
  cudaFree(A); cudaFree(B); if (Cpartial) cudaFree(Cpartial); cudaFree(C);
  return ms_avg;
}

static float dispatch(const Knobs& kb) {
  if (kb.STAGES==2) {
    #define CASES(BM,BN,BK) \
      if(kb.BM==BM && kb.BN==BN && kb.BK==BK) return run_cfg<BM,BN,BK,2>(kb);
    CASES(64,64,16) CASES(64,64,32) CASES(64,64,64)
    CASES(128,128,16) CASES(128,128,32) CASES(128,128,64)
    CASES(256,256,16) CASES(256,256,32) CASES(256,256,64)
    #undef CASES
  } else if (kb.STAGES==3) {
    #define CASES(BM,BN,BK) \
      if(kb.BM==BM && kb.BN==BN && kb.BK==BK) return run_cfg<BM,BN,BK,3>(kb);
    CASES(64,64,16) CASES(64,64,32) CASES(64,64,64)
    CASES(128,128,16) CASES(128,128,32) CASES(128,128,64)
    CASES(256,256,16) CASES(256,256,32) CASES(256,256,64)
    #undef CASES
  }
  return run_cfg<128,128,32,2>(kb);
}

static void print_json(const char* path, const char* arch, const Knobs& kb, float ms) {
  double tflops = (2.0*kb.M*kb.N*(double)kb.K) / (ms*1e6);
  FILE* f = stdout;
  if (path) f = fopen(path, "w");
  fprintf(f,
    "{"
    "\"arch\":\"%s\","
    "\"M\":%d,\"N\":%d,\"K\":%d,"
    "\"BM\":%d,\"BN\":%d,\"BK\":%d,\"stages\":%d,\"split_k\":%d,\"cta_pairs\":%d,"
    "\"copy_path\":\"%s\",\"compute_path\":\"%s\","
    "\"ms_avg\":%.6f,\"tflops\":%.6f"
    "}\n",
    arch, kb.M,kb.N,kb.K, kb.BM,kb.BN,kb.BK,kb.STAGES,kb.SPLITK,kb.CTAPAIRS,
    kb.use_tma ? "tma":"cp.async", kb.use_wgmma ? "wgmma":"wmma/naive",
    ms, tflops);
  if (path && f) fclose(f);
}

int main(int argc, char** argv){
  Knobs kb;
  const char* json_out = nullptr;

  for (int i=1;i<argc;++i){
    if (!strcmp(argv[i],"--M") && i+1<argc) kb.M = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--N") && i+1<argc) kb.N = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--K") && i+1<argc) kb.K = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--block_m") && i+1<argc) kb.BM = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--block_n") && i+1<argc) kb.BN = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--block_k") && i+1<argc) kb.BK = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--stages") && i+1<argc) kb.STAGES = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--split_k") && i+1<argc) kb.SPLITK = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--cta_pairs") && i+1<argc) kb.CTAPAIRS = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--copy_path") && i+1<argc) {
      const char* s = argv[++i]; kb.use_tma = (!strcmp(s,"tma")?1:0);
    }
    
    else if (!strcmp(argv[i],"--epilogue") && i+1<argc) {
      const char* s = argv[++i];
      kb.epilogue_kind = (!strcmp(s,"bias")?1:!strcmp(s,"bias_silu")?2:!strcmp(s,"bias_gelu")?3:0);
    }
    else if (!strcmp(argv[i],"--swizzle") && i+1<argc) {
      const char* s = argv[++i];
      kb.swizzle_kind = (!strcmp(s,"xor128b")?1:0);
    }
    else if (!strcmp(argv[i],"--tma_cols_per_copy") && i+1<argc) {
      kb.tma_cols_per_copy = atoi(argv[++i]);
    }

    else if (!strcmp(argv[i],"--compute_path") && i+1<argc) {
      const char* s = argv[++i]; kb.use_wgmma = (!strcmp(s,"wgmma")?1:0);
    }
    else if (!strcmp(argv[i],"--json_out") && i+1<argc) json_out = argv[++i];
  }

  int device=0; cudaDeviceProp p; cudaGetDevice(&device); cudaGetDeviceProperties(&p,device);
  float ms = dispatch(kb);
  print_json(json_out, p.name, kb, ms);
  return 0;
}
