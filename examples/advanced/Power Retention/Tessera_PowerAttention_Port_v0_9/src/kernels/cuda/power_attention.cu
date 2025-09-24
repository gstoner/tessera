
// ===== H100 WGMMA Projection (m64n64k16.bf16.f32) with cp.async pipeline =====
#if __CUDA_ARCH__ >= 900 && defined(TESSERA_ENABLE_WGMMA)
static inline __device__ uint32_t cvta_to_shared_u32(const void* ptr){
  uint32_t smem;
  asm volatile ("cvta.to.shared.u32 %0, %1;" : "=r"(smem) : "l"(ptr));
  return smem;
}

// Prepare SMEM descriptors for WGMMA
static inline __device__ void make_smem_desc(uint32_t &desc, const void* smem_ptr){
  desc = cvta_to_shared_u32(smem_ptr);
}

// Issue one wgmma for m64n64k16 (bf16 -> fp32). We expose a minimal interface (no transpose flags here).
static inline __device__ void wgmma_m64n64k16(uint32_t a_desc, uint32_t b_desc, float* c){
  // Accumulate into 8 FP32 registers representing a 64x64 tile (simplified view).
  asm volatile(
    "wgmma.mma_async.sync.aligned.m64n64k16.bf16.f32 "
    "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, 0; \n"
    "wgmma.commit_group.sync.aligned; \n"
    "wgmma.wait_group.sync.aligned 0; \n"
    : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])
    : "r"(a_desc), "r"(b_desc)
  );
}

// Load an 8x8x4 block via ldmatrix for A (trans) and B (no-trans)
static inline __device__ void ldmatrix_a_x4(uint32_t *regs, const void* smem_ptr){
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(regs[0]),"=r"(regs[1]),"=r"(regs[2]),"=r"(regs[3]) : "r"(cvta_to_shared_u32(smem_ptr)));
}
static inline __device__ void ldmatrix_b_x4(uint32_t *regs, const void* smem_ptr){
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(regs[0]),"=r"(regs[1]),"=r"(regs[2]),"=r"(regs[3]) : "r"(cvta_to_shared_u32(smem_ptr)));
}

// cp.async helpers
static inline __device__ void cp_async(void* smem_dst, const void* gmem_src, int bytes){
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"
               :: "r"(cvta_to_shared_u32(smem_dst)), "l"(gmem_src), "n"(16));
}
static inline __device__ void cp_commit(){ asm volatile("cp.async.commit_group;"); }
static inline __device__ void cp_wait(int N){ asm volatile("cp.async.wait_group %0;" :: "n"(N)); }
#endif


// ===== Hopper SM90+: cp.async, ldmatrix, WGMMA m64n64k16 BF16->FP32 =====
#if __CUDA_ARCH__ >= 900
static inline __device__ void cp_async_bulk(void* smem_dst, const void* gmem_src, size_t bytes){
  asm volatile("cp.async.bulk.shared.global [%0], [%1], %2;"
               :: "r"(smem_dst), "l"(gmem_src), "n"(0) );
  // NOTE: PTX ISA requires literal size; using 0 placeholder here signals "size from barrier".
}

static inline __device__ void cp_async_commit_group(){ asm volatile("cp.async.commit_group;"); }
static inline __device__ void cp_async_wait_group(int N){ asm volatile("cp.async.wait_group %0;" :: "n"(N)); }

// Swizzled ldmatrix for BF16 tiles (A/B):
static inline __device__ void ldmatrix_x4_trans(uint32_t *regs, const void* smem_ptr){
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(regs[0]),"=r"(regs[1]),"=r"(regs[2]),"=r"(regs[3]) : "r"(smem_ptr));
}

// WGMMA: m64n64k16.bf16.f32
static inline __device__ void wgmma_m64n64k16_bf16_fp32(
    float *c, const void* a_desc, const void* b_desc, bool satf=false){
  // a_desc/b_desc are SMEM descriptors prepared via cvta.to.shared etc.
  // Real code would use wgmma.mma_async.* with descriptors; we inline a guarded PTX stub.
  asm volatile(
    "wgmma.mma_async.sync.aligned.m64n64k16.bf16.f32 "
    "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, %10;"
    :
      "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])
    :
      "r"(a_desc), "r"(b_desc), "r"(satf?1:0)
  );
  asm volatile("wgmma.commit_group.sync.aligned;");
  asm volatile("wgmma.wait_group.sync.aligned 0;");
}
#endif


#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math_constants.h>
#include <math.h>
namespace cg = cooperative_groups;

static inline __device__ float bf16_to_f32(__nv_bfloat16 x){ return __bfloat162float(x); }
static inline __device__ __nv_bfloat16 f32_to_bf16(float x){ return __float2bfloat16(x); }

// Map (i,j) with 0<=i<=j<D to linear index in M = D*(D+1)/2
static inline __device__ int spow2_index(int i, int j, int D){
    // number of pairs starting with <  i> : i*D - (i*(i-1))/2
    return i*D - (i*(i-1))/2 + (j - i);
}

// Expand symmetric power degree p=2 for vector x[D] -> phi[M]
// Unweighted product x[i]*x[j] (note: multinomial coeffs could be applied if desired).
static inline __device__ void spow2_expand(const __nv_bfloat16* __restrict__ x, int D, float* __restrict__ phi){
    int M = D*(D+1)/2;
    for(int idx=0; idx<M; ++idx) phi[idx]=0.f;
    for(int i=0;i<D;i++){
        float xi = bf16_to_f32(x[i]);
        // diagonal term (i,i)
        int id = spow2_index(i,i,D);
        phi[id] += xi*xi;
        for(int j=i+1;j<D;j++){
            float xj = bf16_to_f32(x[j]);
            int ij = spow2_index(i,j,D);
            phi[ij] += xi*xj;
        }
    }
}

extern "C" __global__ void power_attn_forward_bf16_deg2(
    const __nv_bfloat16* __restrict__ Q, // [B,H,S,D]
    const __nv_bfloat16* __restrict__ K, // [B,H,S,D]
    const __nv_bfloat16* __restrict__ V, // [B,H,S,Dh] (use D for Dh here)
    __nv_bfloat16* __restrict__ O,       // [B,H,S,Dh]
    int B, int H, int S, int D, int Dh)  // Dh==D for now; generalize as needed
{
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int M = D*(D+1)/2;

    extern __shared__ float smem[];
    float* state = smem;                  // [M*Dh]
    float* sum_phi = state + (size_t)M*Dh;// [M]

    // zero init
    for(int idx=threadIdx.x; idx<M*Dh + M; idx+=blockDim.x){
        state[idx] = 0.f;
    }
    __syncthreads();

    // accumulate state and sum_phi across S
    
// --- Begin per-token output compute with cp.async pipeline over K=M in tiles of 16 ---
for(int s=threadIdx.x; s<S; s+=blockDim.x){
  const __nv_bfloat16* q = &Q[ (((b*H+h)*S + s)*D) ];
  __nv_bfloat16* o = &O[ (((b*H+h)*S + s)*Dh) ];

  // 1) Build phi(q) into SMEM (row-major) for this token.
  extern __shared__ float smem_phi[]; // reuse: after state/sum_phi
  float* phi = smem_phi; // length M
  compute_phi2_q_to_smem_bf16(q, D, phi);

  // 2) Projection: y = (phi^T * state) / (phi^T * sum_phi + eps)
  //    We'll accelerate the numerator with WGMMA when available, else FMA fallback.
  float denom_local = 0.f;

#if __CUDA_ARCH__ >= 900
  // --- PROJECTION_WGMMA_PATH ---
  // Tile K = 16 along M, with M = D(D+1)/2 rounded up to 16.
  int M_pad = ((M + 15)/16)*16;
  // Accumulator registers for N=Dh tile (we conceptually map to 64x64; for Dh<64 loop N too).
  float c_frag[8]; #pragma unroll for (int i=0;i<8;++i) c_frag[i]=0.f;

  // Pseudo descriptor pointers (in practice build DESCs via cvta and smem addressing)
  const void* a_desc = (const void*)(phi);
  const void* b_desc = (const void*)(state);

  // K loop: M_pad / 16
  for (int kblk=0; kblk < M_pad; kblk+=16){
    // cp.async staged loads of A (phi) & B (state) tiles into swizzled SMEM could go here
    // with stages = Cfg::stages (2/3/4). For brevity, we assume in-SMEM already.
    wgmma_m64n64k16_bf16_fp32(c_frag, a_desc, b_desc, false);
  }

  // Reduce denom using FMA on sum_phi (kept simple)
  for (int pair = threadIdx.x; pair < M; pair += blockDim.x){
    float ph = phi[pair];
    denom_local += ph * sum_phi[pair];
  }

  __shared__ float denom_shared;
  if(threadIdx.x==0) denom_shared=0.f;
  __syncthreads();
  atomicAdd(&denom_shared, denom_local);
  __syncthreads();
  float denom = denom_shared + 1e-6f;

  // Store c_frag to o[...] (toy store, map 8 regs -> Dh lanes)
  for (int dh=threadIdx.x; dh<Dh; dh+=blockDim.x){
    float y = (dh < 8 ? c_frag[dh] : 0.f) / denom;
    o[dh] = f32_to_bf16(y);
  }
#else
  // --- FMA FALLBACK PATH ---
  extern __shared__ float redbuf[];
  for(int dh=threadIdx.x; dh<Dh; dh+=blockDim.x) redbuf[dh]=0.f;
  __shared__ float denom_sh;
  if(threadIdx.x==0) denom_sh=0.f;
  __syncthreads();
  for(int pair = threadIdx.x; pair < M; pair += blockDim.x){
    float ph = phi[pair];
    float z  = sum_phi[pair];
    atomicAdd(&denom_sh, ph*z);
    for(int dh=0; dh<Dh; ++dh){
      float st = state[pair*Dh + dh];
      atomicAdd(&redbuf[dh], ph*st);
    }
  }
  __syncthreads();
  float denom = denom_sh + 1e-6f;
  for(int dh=threadIdx.x; dh<Dh; dh+=blockDim.x){
    float y = redbuf[dh] / denom;
    o[dh] = f32_to_bf16(y);
  }
#endif
}

        const __nv_bfloat16* k = &K[ (((b*H+h)*S + s)*D) ];
        const __nv_bfloat16* v = &V[ (((b*H+h)*S + s)*Dh) ];
        // expand phi_k (degree 2)
        extern __shared__ float tmp[]; // alias
        float* phi = tmp; // first M slots reused per thread in sequence (not concurrent)
        // NOTE: threads reuse same buffer; to avoid race, use per-thread registers then write with atomics. Here we compute local arrays in registers and atomically add.
        // For simplicity, allocate per-thread local phi in registers.
    }
    __syncthreads();

    // Due to shared memory reuse complexity above, we instead do a cooperative accumulation by looping S in lock-step:
    for(int s_blk=0; s_blk<S; ++s_blk){
        const __nv_bfloat16* k = &K[ (((b*H+h)*S + s_blk)*D) ];
        const __nv_bfloat16* v = &V[ (((b*H+h)*S + s_blk)*Dh) ];
        // Expand phi_k in registers
        // NOTE: building phi of size M on each thread would be wasteful; we shard across threads over (i,j) pairs.
        for(int pair = threadIdx.x; pair < M; pair += blockDim.x){
            // invert linear index to (i,j) (solve for i)
            int i = 0;
            // binary search i such that start(i)<=pair<start(i+1)
            // start(i)=i*D - i*(i-1)/2
            int lo=0, hi=D-1;
            while(lo<hi){
                int mid=(lo+hi)/2;
                int start = mid*D - (mid*(mid-1))/2;
                if(start <= pair) lo = mid+1; else hi = mid;
            }
            int i_candidate = lo-1;
            if(i_candidate<0) i_candidate=0;
            int start_i = i_candidate*D - (i_candidate*(i_candidate-1))/2;
            while(start_i > pair && i_candidate>0){ i_candidate--; start_i = i_candidate*D - (i_candidate*(i_candidate-1))/2; }
            while(i_candidate+1<D){
                int start_next = (i_candidate+1)*D - ((i_candidate+1)* (i_candidate))/2;
                if(start_next > pair) break;
                i_candidate++;
                start_i = start_next;
            }
            i = i_candidate;
            int j = i + (pair - start_i);

            float ki = bf16_to_f32(k[i]);
            float kj = bf16_to_f32(k[j]);
            float phi_ij = (i==j) ? ki*ki : ki*kj;

            // add to sum_phi
            atomicAdd(&sum_phi[pair], phi_ij);
            // add to state[pair,:] outer with v
            for(int dh=0; dh<Dh; ++dh){
                float vdh = bf16_to_f32(v[dh]);
                atomicAdd(&state[pair*Dh + dh], phi_ij * vdh);
            }
        }
        __syncthreads();
    }

    // Now produce outputs per token
    
// --- Begin per-token output compute with cp.async pipeline over K=M in tiles of 16 ---
for(int s=threadIdx.x; s<S; s+=blockDim.x){
  const __nv_bfloat16* q = &Q[ (((b*H+h)*S + s)*D) ];
  __nv_bfloat16* o = &O[ (((b*H+h)*S + s)*Dh) ];

  // 1) Build phi(q) into SMEM (row-major) for this token.
  extern __shared__ float smem_phi[]; // reuse: after state/sum_phi
  float* phi = smem_phi; // length M
  compute_phi2_q_to_smem_bf16(q, D, phi);

  // 2) Projection: y = (phi^T * state) / (phi^T * sum_phi + eps)
  //    We'll accelerate the numerator with WGMMA when available, else FMA fallback.
  float denom_local = 0.f;

#if __CUDA_ARCH__ >= 900
  // --- PROJECTION_WGMMA_PATH ---
  // Tile K = 16 along M, with M = D(D+1)/2 rounded up to 16.
  int M_pad = ((M + 15)/16)*16;
  // Accumulator registers for N=Dh tile (we conceptually map to 64x64; for Dh<64 loop N too).
  float c_frag[8]; #pragma unroll for (int i=0;i<8;++i) c_frag[i]=0.f;

  // Pseudo descriptor pointers (in practice build DESCs via cvta and smem addressing)
  const void* a_desc = (const void*)(phi);
  const void* b_desc = (const void*)(state);

  // K loop: M_pad / 16
  for (int kblk=0; kblk < M_pad; kblk+=16){
    // cp.async staged loads of A (phi) & B (state) tiles into swizzled SMEM could go here
    // with stages = Cfg::stages (2/3/4). For brevity, we assume in-SMEM already.
    wgmma_m64n64k16_bf16_fp32(c_frag, a_desc, b_desc, false);
  }

  // Reduce denom using FMA on sum_phi (kept simple)
  for (int pair = threadIdx.x; pair < M; pair += blockDim.x){
    float ph = phi[pair];
    denom_local += ph * sum_phi[pair];
  }

  __shared__ float denom_shared;
  if(threadIdx.x==0) denom_shared=0.f;
  __syncthreads();
  atomicAdd(&denom_shared, denom_local);
  __syncthreads();
  float denom = denom_shared + 1e-6f;

  // Store c_frag to o[...] (toy store, map 8 regs -> Dh lanes)
  for (int dh=threadIdx.x; dh<Dh; dh+=blockDim.x){
    float y = (dh < 8 ? c_frag[dh] : 0.f) / denom;
    o[dh] = f32_to_bf16(y);
  }
#else
  // --- FMA FALLBACK PATH ---
  extern __shared__ float redbuf[];
  for(int dh=threadIdx.x; dh<Dh; dh+=blockDim.x) redbuf[dh]=0.f;
  __shared__ float denom_sh;
  if(threadIdx.x==0) denom_sh=0.f;
  __syncthreads();
  for(int pair = threadIdx.x; pair < M; pair += blockDim.x){
    float ph = phi[pair];
    float z  = sum_phi[pair];
    atomicAdd(&denom_sh, ph*z);
    for(int dh=0; dh<Dh; ++dh){
      float st = state[pair*Dh + dh];
      atomicAdd(&redbuf[dh], ph*st);
    }
  }
  __syncthreads();
  float denom = denom_sh + 1e-6f;
  for(int dh=threadIdx.x; dh<Dh; dh+=blockDim.x){
    float y = redbuf[dh] / denom;
    o[dh] = f32_to_bf16(y);
  }
#endif
}

        const __nv_bfloat16* q = &Q[ (((b*H+h)*S + s)*D) ];
        __nv_bfloat16* o = &O[ (((b*H+h)*S + s)*Dh) ];
        // compute phi_q (on the fly, sharded across threads)
        // We'll compute numerator[dh] = sum_pair phi_q[pair]*state[pair,dh]
        // and denom = sum_pair phi_q[pair]*sum_phi[pair]
        extern __shared__ float redbuf[]; // reuse smem: first Dh slots for numerators, 1 for denom (per thread -> atomic reduce)
        // local accum in registers
        float denom_local = 0.f;
        // we will accumulate dh components in registers one-by-one to save regs
        // Compute contribution per pair, then atomically add
        for(int pair = threadIdx.x; pair < M; pair += blockDim.x){
            // invert pair->(i,j)
            int i = 0;
            int lo=0, hi=D-1;
            while(lo<hi){
                int mid=(lo+hi)/2;
                int start = mid*D - (mid*(mid-1))/2;
                if(start <= pair) lo = mid+1; else hi = mid;
            }
            int i_candidate = lo-1;
            if(i_candidate<0) i_candidate=0;
            int start_i = i_candidate*D - (i_candidate*(i_candidate-1))/2;
            while(start_i > pair && i_candidate>0){ i_candidate--; start_i = i_candidate*D - (i_candidate*(i_candidate-1))/2; }
            while(i_candidate+1<D){
                int start_next = (i_candidate+1)*D - ((i_candidate+1)* (i_candidate))/2;
                if(start_next > pair) break;
                i_candidate++;
                start_i = start_next;
            }
            int j = i_candidate;
            j = j + (pair - start_i);

            float qi = bf16_to_f32(q[i]);
            float qj = bf16_to_f32(q[j]);
            float phi_ij = (i==j) ? qi*qi : qi*qj;

            float sumphi = sum_phi[pair];
            denom_local += phi_ij * sumphi;

            // loop Dh
            for(int dh=0; dh<Dh; ++dh){
                float st = state[pair*Dh + dh];
                float contrib = phi_ij * st;
                atomicAdd((float*)&redbuf[dh], contrib);
            }
        }
        __syncthreads();
        // Reduce denom across threads
        __shared__ float denom_shared;
        if(threadIdx.x==0) denom_shared=0.f;
        __syncthreads();
        atomicAdd(&denom_shared, denom_local);
        __syncthreads();
        float denom = denom_shared + 1e-6f;

        
// ===== WGMMA helpers (Hopper, sm_90+) =====
#if __CUDA_ARCH__ >= 900
// Minimal wrappers for WGMMA (BF16 inputs, FP32 acc).
// NOTE: This is a simplified, didactic wrapper; real kernels will use ldmatrix + swizzled SMEM layouts.
static inline __device__ void wgmma_m64n64k16_bf16_fp32(
    const void* a_smem, const void* b_smem, float* c_regs) {
  // Placeholder inline PTX; in production you'd tile across K and accumulate.
  // Using a dummy to keep compilation paths simple in environments without real PTX support.
  // Developers can replace with proper asm volatile blocks.
  // Here, do nothing; c_regs are expected to be pre-accumulated by FMA fallback if PTX is unavailable.
}
#endif

// Compute phi2(q) (degree-2 symmetric power) into SMEM (row-major M) for a single token.
static inline __device__ void compute_phi2_q_to_smem_bf16(
    const __nv_bfloat16* __restrict__ q, int D, float* __restrict__ phi_smem){
  int M = D*(D+1)/2;
  for (int i = threadIdx.x; i < M; i += blockDim.x) phi_smem[i] = 0.f;
  __syncthreads();
  // shard (i,j) pairs across threads
  for(int pair = threadIdx.x; pair < M; pair += blockDim.x){
    // invert to (i,j)
    int lo=0, hi=D-1;
    while(lo<hi){
      int mid=(lo+hi)/2;
      int start = mid*D - (mid*(mid-1))/2;
      if(start <= pair) lo = mid+1; else hi = mid;
    }
    int i = max(0, lo-1);
    int start_i = i*D - (i*(i-1))/2;
    while(start_i > pair && i>0){ i--; start_i = i*D - (i*(i-1))/2; }
    while(i+1<D){
      int start_next = (i+1)*D - ((i+1)*i)/2;
      if(start_next > pair) break;
      i++;
      start_i = start_next;
    }
    int j = i + (pair - start_i);
    float qi = bf16_to_f32(q[i]);
    float qj = bf16_to_f32(q[j]);
    float phi = (i==j) ? qi*qi : qi*qj;
    phi_smem[pair] = phi;
  }
  __syncthreads();
}

// ===== Autotune (Vidrial-style PickBest) =====
struct PowerCfgRT {
  int tok_tile;
  int stages;
  int vec_elems;
};

__device__ __managed__ int g_best_tok_tile = 128;
__device__ __managed__ int g_best_stages   = 2;
__device__ __managed__ int g_best_vec      = 8;

extern "C" __global__ void _tessera_power_reset_autotune(){ g_best_tok_tile=128; g_best_stages=2; g_best_vec=8; }

// write normalized output
        for(int dh=threadIdx.x; dh<Dh; dh+=blockDim.x){
            float num = redbuf[dh];
            redbuf[dh] = 0.f; // clear for reuse
            float y = num / denom;
            o[dh] = f32_to_bf16(y);
        }
        __syncthreads();
    }
}

extern "C" __global__ void retention_infer_bf16_deg2(
    const __nv_bfloat16* __restrict__ Q,    // [B,H,1,D]
    const __nv_bfloat16* __restrict__ K,    // [B,H,1,D]
    const __nv_bfloat16* __restrict__ V,    // [B,H,1,Dh]
    const float* __restrict__ log_G,        // [B,H] or nullptr
    __nv_bfloat16* __restrict__ O,          // [B,H,1,Dh]
    float* __restrict__ state,              // [B,H,M,Dh]
    float* __restrict__ sum_phi,            // [B,H,M]
    int B,int H,int D,int Dh, int step)
{
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int M = D*(D+1)/2;

    const __nv_bfloat16* q = &Q[((b*H+h)*1 + 0)*D];
    const __nv_bfloat16* k = &K[((b*H+h)*1 + 0)*D];
    const __nv_bfloat16* v = &V[((b*H+h)*1 + 0)*Dh];
    float* S = &state[ ((b*H + h)*M*Dh) ];
    float* Z = &sum_phi[ ((b*H + h)*M) ];

    // compute phi_k and apply gating
    float g = 1.f;
    if (log_G) g = expf(log_G[b*H+h]);
    // accumulate into S and Z
    for(int pair = threadIdx.x; pair < M; pair += blockDim.x){
        // recover (i,j)
        int i = 0;
        int lo=0, hi=D-1;
        while(lo<hi){
            int mid=(lo+hi)/2;
            int start = mid*D - (mid*(mid-1))/2;
            if(start <= pair) lo = mid+1; else hi = mid;
        }
        int i_candidate = lo-1;
        if(i_candidate<0) i_candidate=0;
        int start_i = i_candidate*D - (i_candidate*(i_candidate-1))/2;
        while(start_i > pair && i_candidate>0){ i_candidate--; start_i = i_candidate*D - (i_candidate*(i_candidate-1))/2; }
        while(i_candidate+1<D){
            int start_next = (i_candidate+1)*D - ((i_candidate+1)* (i_candidate))/2;
            if(start_next > pair) break;
            i_candidate++;
            start_i = start_next;
        }
        int i2=i_candidate;
        int j = i2 + (pair - start_i);
        float ki = bf16_to_f32(k[i2]);
        float kj = bf16_to_f32(k[j]);
        float phi = (i2==j) ? ki*ki : ki*kj;
        float w = g*phi;
        atomicAdd(&Z[pair], w);
        for(int dh=0; dh<Dh; ++dh){
            float vdh = bf16_to_f32(v[dh]);
            atomicAdd(&S[pair*Dh + dh], w * vdh);
        }
    }
    __syncthreads();

    // query
    __nv_bfloat16* o = &O[((b*H+h)*1 + 0)*Dh];
    extern __shared__ float redbuf[];
    for(int dh=threadIdx.x; dh<Dh; dh+=blockDim.x) redbuf[dh]=0.f;
    __shared__ float denom_sh;
    if(threadIdx.x==0) denom_sh=0.f;
    __syncthreads();

    float denom_local=0.f;
    for(int pair = threadIdx.x; pair < M; pair += blockDim.x){
        // recover (i,j)
        int i = 0;
        int lo=0, hi=D-1;
        while(lo<hi){
            int mid=(lo+hi)/2;
            int start = mid*D - (mid*(mid-1))/2;
            if(start <= pair) lo = mid+1; else hi = mid;
        }
        int i_candidate = lo-1;
        if(i_candidate<0) i_candidate=0;
        int start_i = i_candidate*D - (i_candidate*(i_candidate-1))/2;
        while(start_i > pair && i_candidate>0){ i_candidate--; start_i = i_candidate*D - (i_candidate*(i_candidate-1))/2; }
        while(i_candidate+1<D){
            int start_next = (i_candidate+1)*D - ((i_candidate+1)* (i_candidate))/2;
            if(start_next > pair) break;
            i_candidate++;
            start_i = start_next;
        }
        int i2=i_candidate;
        int j = i2 + (pair - start_i);
        float qi = bf16_to_f32(q[i2]);
        float qj = bf16_to_f32(q[j]);
        float phi = (i2==j) ? qi*qi : qi*qj;
        float z = Z[pair];
        denom_local += phi * z;
        for(int dh=0; dh<Dh; ++dh){
            float st = S[pair*Dh + dh];
            atomicAdd(&redbuf[dh], phi * st);
        }
    }
    __syncthreads();
    atomicAdd(&denom_sh, denom_local);
    __syncthreads();
    float denom = denom_sh + 1e-6f;
    for(int dh=threadIdx.x; dh<Dh; dh+=blockDim.x){
        float y = redbuf[dh] / denom;
        o[dh] = f32_to_bf16(y);
    }
}


extern "C" void tessera_power_attn_cuda_forward_cfg(
    const void* q,const void* k,const void* v,void* o,
    int B,int H,int S,int D,int Dh,int M,int window,int causal,
    int tok_tile,int stages,int vec_elems, void* stream_void)
{
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
  dim3 block(256);
  int tiles = (S + tok_tile - 1)/tok_tile;
  dim3 grid(B*H*tiles);
  size_t shmem = size_t(M) * size_t(Dh) * sizeof(float) + size_t(M)*sizeof(float) + size_t(M)*sizeof(float);
  // Select template based on tok_tile
  if (tok_tile == 128) {
    power_attn_forward_kernel_cfg<PowerCfg<128,2,8,false>><<<grid, block, shmem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<__nv_bfloat16*>(o),
      B,H,S,D,Dh,M,window,causal);
  } else {
    power_attn_forward_kernel_cfg<PowerCfg<256,2,8,false>><<<grid, block, shmem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<__nv_bfloat16*>(o),
      B,H,S,D,Dh,M,window,causal);
  }
}


extern "C" void tessera_power_attn_cuda_forward_cfg(
    const void* q,const void* k,const void* v,void* o,
    int B,int H,int S,int D,int Dh,int M,int window,int causal,
    int tok_tile,int stages,int vec_elems,int ldsm, void* stream_void)
{
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
  dim3 block(256);
  int tiles = (S + tok_tile - 1)/tok_tile;
  dim3 grid(B*H*tiles);
  size_t shmem = size_t(M) * size_t(Dh) * sizeof(float) + size_t(M)*sizeof(float);

  // Dispatch based on tok_tile and ldsm (stages/vec_elems are currently advisory placeholders in this scaffold).
  if (tok_tile == 128 && ldsm==0) {
    power_attn_forward_kernel_cfg<PowerCfg<128,2,8,false>><<<grid, block, shmem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<__nv_bfloat16*>(o),
      B,H,S,D,Dh,M,window,causal);
  } else if (tok_tile == 128 && ldsm==1) {
    power_attn_forward_kernel_cfg<PowerCfg<128,2,8,true>><<<grid, block, shmem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<__nv_bfloat16*>(o),
      B,H,S,D,Dh,M,window,causal);
  } else if (tok_tile == 192 && ldsm==0) {
    power_attn_forward_kernel_cfg<PowerCfg<192,2,8,false>><<<grid, block, shmem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<__nv_bfloat16*>(o),
      B,H,S,D,Dh,M,window,causal);
  } else if (tok_tile == 192 && ldsm==1) {
    power_attn_forward_kernel_cfg<PowerCfg<192,2,8,true>><<<grid, block, shmem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<__nv_bfloat16*>(o),
      B,H,S,D,Dh,M,window,causal);
  } else if (tok_tile == 256 && ldsm==0) {
    power_attn_forward_kernel_cfg<PowerCfg<256,2,8,false>><<<grid, block, shmem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<__nv_bfloat16*>(o),
      B,H,S,D,Dh,M,window,causal);
  } else { // 256 & ldsm==1
    power_attn_forward_kernel_cfg<PowerCfg<256,2,8,true>><<<grid, block, shmem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<__nv_bfloat16*>(o),
      B,H,S,D,Dh,M,window,causal);
  }
}


#if __CUDA_ARCH__ >= 900 && defined(TESSERA_ENABLE_WGMMA)
extern "C" __global__ void power_attn_forward_wgmma_bf16_deg2(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    int B, int H, int S, int D, int Dh)
{
  int bh = blockIdx.x;
  int b = bh / H;
  int h = bh % H;
  int M = D*(D+1)/2;
  int Mpad = ((M + 15)/16)*16;

  extern __shared__ float smem[];
  float* state = smem;                  // [M*Dh]
  float* sum_phi = state + (size_t)Mpad*Dh; // [Mpad]
  float* phi_buf = sum_phi + (size_t)Mpad;  // [Mpad], per-token phi(q)

  // zero init
  for(int idx=threadIdx.x; idx<Mpad*Dh + Mpad; idx+=blockDim.x) state[idx] = 0.f;
  __syncthreads();

  // Build state & sum_phi (FMA path for correctness; cp.async could be added here similarly)
  for(int s_blk=0; s_blk<S; ++s_blk){
    const __nv_bfloat16* k = &K[ (((b*H+h)*S + s_blk)*D) ];
    const __nv_bfloat16* v = &V[ (((b*H+h)*S + s_blk)*Dh) ];
    // shard (i,j) pairs over threads
    for(int pair = threadIdx.x; pair < M; pair += blockDim.x){
      // recover (i,j)
      int lo=0, hi=D-1;
      while(lo<hi){
        int mid=(lo+hi)/2;
        int start = mid*D - (mid*(mid-1))/2;
        if(start <= pair) lo = mid+1; else hi = mid;
      }
      int i = max(0, lo-1);
      int start_i = i*D - (i*(i-1))/2;
      while(start_i > pair && i>0){ i--; start_i = i*D - (i*(i-1))/2; }
      while(i+1<D){
        int start_next = (i+1)*D - ((i+1)*i)/2;
        if(start_next > pair) break;
        i++; start_i = start_next;
      }
      int j = i + (pair - start_i);
      float ki = bf16_to_f32(k[i]);
      float kj = bf16_to_f32(k[j]);
      float phi = (i==j) ? ki*ki : ki*kj;
      atomicAdd(&sum_phi[pair], phi);
      for(int dh=0; dh<Dh; ++dh){
        float vdh = bf16_to_f32(v[dh]);
        atomicAdd(&state[pair*Dh + dh], phi * vdh);
      }
    }
    __syncthreads();
  }

  // Per-token projection: cp.async pipeline over K=Mpad in steps of 16
  for(int s=threadIdx.x; s<S; s+=blockDim.x){
    const __nv_bfloat16* q = &Q[ (((b*H+h)*S + s)*D) ];
    __nv_bfloat16* o = &O[ (((b*H+h)*S + s)*Dh) ];
    // compute phi(q) into phi_buf
    compute_phi2_q_to_smem_bf16(q, D, phi_buf);
    for (int t = threadIdx.x + M; t < Mpad; t += blockDim.x) phi_buf[t] = 0.f;
    __syncthreads();

    // Accumulator
    float c_frag[8]; #pragma unroll for(int i=0;i<8;++i) c_frag[i]=0.f;
    // Denominator
    __shared__ float denom_sh;
    if(threadIdx.x==0) denom_sh=0.f;
    __syncthreads();

    // K-loop in blocks of 16
    for (int kblk=0; kblk<Mpad; kblk+=16){
      // In a production kernel: cp.async next tiles of phi_buf[k:k+16] and state[k:k+16,:] into double/triple buffers
      // and use ldmatrix to feed wgmma. Here, we simplify descriptor creation per step.
      uint32_t a_desc, b_desc;
      make_smem_desc(a_desc, phi_buf + kblk);
      make_smem_desc(b_desc, state   + kblk*Dh);

      wgmma_m64n64k16(a_desc, b_desc, c_frag);
      __syncthreads();
    }

    // denom
    float denom_local = 0.f;
    for (int pair = threadIdx.x; pair < M; pair += blockDim.x){
      denom_local += phi_buf[pair] * sum_phi[pair];
    }
    atomicAdd(&denom_sh, denom_local);
    __syncthreads();
    float denom = denom_sh + 1e-6f;

    // store out (map 8 acc regs to Dh lanes for demonstration)
    for (int dh=threadIdx.x; dh<Dh; dh+=blockDim.x){
      float y = (dh < 8 ? c_frag[dh] : 0.f) / denom;
      o[dh] = f32_to_bf16(y);
    }
    __syncthreads();
  }
}
#endif


extern "C" void tessera_power_attn_cuda_forward_wgmma_cfg(
    const void* q,const void* k,const void* v,void* o,
    int B,int H,int S,int D,int Dh,int M,int window,int causal,
    int tok_tile,int stages,int vec_elems,int ldsm, void* stream_void)
{
#if __CUDA_ARCH__ >= 900 && defined(TESSERA_ENABLE_WGMMA)
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
  dim3 block(256);
  int tiles = (S + tok_tile - 1)/tok_tile;
  dim3 grid(B*H*tiles);
  // Shared memory: Mpad*Dh + Mpad + Mpad (floats)
  int Mpad = ((M + 15)/16)*16;
  size_t shmem = (size_t)Mpad * (size_t)Dh * sizeof(float) + (size_t)Mpad * sizeof(float) + (size_t)Mpad * sizeof(float);
  power_attn_forward_wgmma_bf16_deg2<<<grid, block, shmem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<__nv_bfloat16*>(o),
      B,H,S,D,Dh);
#else
  // Fallback to FMA path if not available
  tessera_power_attn_cuda_forward_cfg(q,k,v,o,B,H,S,D,Dh,M,window,causal,
                                      tok_tile,stages,vec_elems,ldsm, stream_void);
#endif
}
