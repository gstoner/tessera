#include "common.cuh"
#include <torch/extension.h>
#include <curand_kernel.h>
#include <cmath>
#include <type_traits>
#include <cuda_fp16.h>

// Broadcast lane 0 value across warp
__inline__ __device__ float warp_bcast0(float v){
  return __shfl_sync(0xffffffff, v, 0);
}

template <typename T>
__device__ inline void vec_accum_dot(const T* __restrict__ q,
                                     const T* __restrict__ k,
                                     int D, int tid, int nthreads,
                                     float &acc) {
  // Scalar fallback
  for (int d = tid; d < D; d += nthreads){
    acc += (float)q[d] * (float)k[d];
  }
}

// float specialization: float4
template <>
__device__ inline void vec_accum_dot<float>(const float* __restrict__ q,
                                            const float* __restrict__ k,
                                            int D, int tid, int nthreads,
                                            float &acc) {
  int V = 4;
  int DL = D / V;
  const float4* q4 = reinterpret_cast<const float4*>(q);
  const float4* k4 = reinterpret_cast<const float4*>(k);
  for (int dl = tid; dl < DL; dl += nthreads){
    float4 aq = q4[dl];
    float4 bk = k4[dl];
    acc += aq.x * bk.x + aq.y * bk.y + aq.z * bk.z + aq.w * bk.w;
  }
  // tail
  for (int d = DL*V + tid; d < D; d += nthreads){
    acc += q[d] * k[d];
  }
}

// half specialization: half2 (convert to float2)
template <>
__device__ inline void vec_accum_dot<half>(const half* __restrict__ q,
                                           const half* __restrict__ k,
                                           int D, int tid, int nthreads,
                                           float &acc) {
  int V = 2;
  int DL = D / V;
  const half2* q2 = reinterpret_cast<const half2*>(q);
  const half2* k2 = reinterpret_cast<const half2*>(k);
  for (int dl = tid; dl < DL; dl += nthreads){
    half2 aq = q2[dl];
    half2 bk = k2[dl];
    float2 a = __half22float2(aq);
    float2 b = __half22float2(bk);
    acc += a.x * b.x + a.y * b.y;
  }
  // tail
  for (int d = DL*V + tid; d < D; d += nthreads){
    acc += __half2float(q[d]) * __half2float(k[d]);
  }
}

// Philox helper: one RNG per (b,h,i,j) attention prob
__device__ inline float dropout_keep_prob(uint64_t seed, int bh, int i, int j, float p_drop){
  if (p_drop <= 0.0f) return 1.0f;
  curandStatePhilox4_32_10_t state;
  // unique counter from indices; subsequence=j, offset=i
  curand_init((unsigned long long)seed + (unsigned long long)bh, j, i, &state);
  float4 r = curand_uniform4(&state);
  float u = r.x; // one random per (i,j)
  float keep = (u >= p_drop) ? 1.0f : 0.0f;
  return keep / (1.0f - p_drop); // scale by 1/keep_prob
}

template <typename T, int BM, int BN>
__global__ __launch_bounds__(256, 2)
void fa_fwd_tiled_kernel(const T* __restrict__ Q,
                         const T* __restrict__ K,
                         const T* __restrict__ V,
                         const float* __restrict__ mask, // [B,H,S,S] or null
                         T* __restrict__ Out,
                         int B, int H, int S, int D,
                         float scale, float dropout_p, bool is_causal,
                         uint64_t seed)
{
  const int b = blockIdx.z;
  const int h = blockIdx.y;
  const int i0 = blockIdx.x * BM;
  if (b>=B || h>=H || i0>=S) return;

  extern __shared__ char smem_raw[];
  T* Ktile = reinterpret_cast<T*>(smem_raw);
  T* Vtile = reinterpret_cast<T*>(smem_raw + BN * D * sizeof(T));

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int nthreads = blockDim.x;

  float m_row[BM];
  float l_row[BM];
  #pragma unroll
  for (int ii=0; ii<BM; ++ii){ m_row[ii] = -1e30f; l_row[ii] = 0.0f; }

  const int head_stride = S * D;
  const T* Qhead = Q + ((b*H + h) * head_stride);
  const T* Khead = K + ((b*H + h) * head_stride);
  const T* Vhead = V + ((b*H + h) * head_stride);
  T* Ohead = Out + ((b*H + h) * head_stride);

  const int bh = b*H + h;

  for (int j0=0; j0<S; j0+=BN){
    if (is_causal && j0 > i0 + BM - 1) break;

    // Load K,V tile vectorized
    {
      TESSERA_NVTX_RANGE_COLORED("LDS:KV", 0xFFAA7733);
      int vec = std::is_same<T,float>::value ? 4 : 2;
      int elems = BN * D;
      int iters = (elems + vec-1)/vec;
      for (int t = tid; t < iters; t += nthreads){
        int base = t * vec;
        if (base < elems){
          int j = base / D;
          int d = base % D;
          int jj = j0 + j;
          #pragma unroll
          for (int u=0; u<vec; ++u){
            int dd = d + u;
            if (dd < D){
              if (jj < S){
                Ktile[j*D + dd] = Khead[jj*D + dd];
                Vtile[j*D + dd] = Vhead[jj*D + dd];
              } else {
                Ktile[j*D + dd] = (T)0;
                Vtile[j*D + dd] = (T)0;
              }
            }
          }
        }
      }
    }
    __syncthreads();

    // Pass 1: row max
    {
      TESSERA_NVTX_RANGE_COLORED("QK^T:max", 0xFF2288FF);
      for (int ii=0; ii<BM; ++ii){
        int i = i0 + ii;
        if (i>=S) break;
        float m_local = -1e30f;
        for (int j=0; j<BN; ++j){
          int jj = j0 + j;
          if (jj>=S) break;
          if (is_causal && jj>i) break;
          float dot = 0.0f;
          vec_accum_dot<T>(Qhead + i*D, Ktile + j*D, D, tid, nthreads, dot);
          // reduce across warp
          for (int offs=16; offs>0; offs>>=1)
            dot += __shfl_down_sync(0xffffffff, dot, offs);
          if (lane == 0){
            float z = dot * scale + (mask ? mask[(bh)*S*S + i*S + jj] : 0.0f);
            m_local = fmaxf(m_local, z);
          }
        }
        if (lane == 0){
          float m_old = m_row[ii];
          m_row[ii] = fmaxf(m_old, m_local);
        }
      }
    }
    __syncthreads();

    // Pass 2: exp-sum + O accumulation
    {
      TESSERA_NVTX_RANGE_COLORED("softmax+PV", 0xFF66CC22);
      for (int ii=0; ii<BM; ++ii){
        int i = i0 + ii; if (i>=S) break;
        float m_i = m_row[ii];
        float l_add = 0.0f;
        for (int j=0; j<BN; ++j){
          int jj = j0 + j;
          if (jj>=S) break;
          if (is_causal && jj>i) break;
          float dot = 0.0f;
          vec_accum_dot<T>(Qhead + i*D, Ktile + j*D, D, tid, nthreads, dot);
          for (int offs=16; offs>0; offs>>=1)
            dot += __shfl_down_sync(0xffffffff, dot, offs);
          float p = 0.0f;
          if (lane == 0){
            float z = dot * scale + (mask ? mask[(bh)*S*S + i*S + jj] : 0.0f);
            p = __expf(z - m_i);
            // inline dropout if requested
            if (dropout_p > 0.0f){
              float keep = dropout_keep_prob(seed, bh, i, jj, dropout_p);
              p *= keep;
            }
            l_add += p;
          }
          p = warp_bcast0(p);
          // accumulate O with vectorized stride on d
          for (int d = tid; d < D; d += nthreads){
            float add = p * (float)Vtile[j*D + d];
            atomicAdd(reinterpret_cast<float*>(&Ohead[i*D + d]), add);
          }
        }
        if (lane == 0) l_row[ii] += l_add;
      }
    }
    __syncthreads();
  }

  // Normalize
  {
    TESSERA_NVTX_RANGE_COLORED("normalize O", 0xFFBB33DD);
    for (int ii=0; ii<BM; ++ii){
      int i = i0 + ii; if (i>=S) break;
      float rinv = 1.0f / (l_row[ii] + 1e-20f);
      for (int d = tid; d < D; d += blockDim.x){
        float v = (float)Ohead[i*D + d] * rinv;
        Ohead[i*D + d] = (T)v;
      }
    }
  }
}

void flashattn_fwd_tiled_launcher(at::Tensor Q, at::Tensor K, at::Tensor V,
                                  c10::optional<at::Tensor> attn_mask,
                                  c10::optional<at::Tensor> /*dropout_mask_unused*/,
                                  double scale, double dropout_p, bool is_causal,
                                  at::Tensor Out,
                                  long long rng_seed)
{
  TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda() && Out.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(Q.scalar_type()==at::kFloat || Q.scalar_type()==at::kHalf, "float16/float32 only");
  TORCH_CHECK(Q.sizes()==K.sizes() && Q.sizes()==V.sizes() && Q.sizes()==Out.sizes(), "shape mismatch");
  int B = Q.size(0), H = Q.size(1), S = Q.size(2), D = Q.size(3);

  const float* mask_ptr = nullptr;
  if (attn_mask.has_value() && attn_mask->defined()){
    TORCH_CHECK(attn_mask->scalar_type()==at::kFloat && attn_mask->sizes()==std::vector<int64_t>({B,H,S,S}), "mask [B,H,S,S] float");
    mask_ptr = attn_mask->data_ptr<float>();
  }

  dim3 grid(div_up(S, 64), H, B);
  dim3 block(256);
  size_t smem = (size_t)(64 * D * sizeof(float)) * 2;

  // seed for Philox (host provides nondet seed through torch generator if needed; we use clock64 fallback)
  uint64_t seed = (rng_seed==0LL) ? (uint64_t)clock64() : (uint64_t)rng_seed;

  if (Q.scalar_type()==at::kFloat){
    fa_fwd_tiled_kernel<float,64,64><<<grid, block, smem>>>(
      Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
      mask_ptr, Out.data_ptr<float>(),
      B,H,S,D, (float)scale, (float)dropout_p, is_causal, seed);
  } else {
    smem = (size_t)(64 * D * sizeof(at::Half)) * 2;
    fa_fwd_tiled_kernel<half,64,64><<<grid, block, smem>>>(
      reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
      mask_ptr, reinterpret_cast<half*>(Out.data_ptr<at::Half>()),
      B,H,S,D, (float)scale, (float)dropout_p, is_causal, seed);
  }
  CUDA_CHECK(cudaGetLastError());
}
