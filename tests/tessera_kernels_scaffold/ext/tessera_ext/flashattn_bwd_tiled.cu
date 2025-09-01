#include "common.cuh"
#include <torch/extension.h>
#include <curand_kernel.h>
#include <cmath>
#include <type_traits>
#include <cuda_fp16.h>

__inline__ __device__ float warp_bcast0(float v){
  return __shfl_sync(0xffffffff, v, 0);
}

template <typename T>
__device__ inline void vec_accum_dot(const T* __restrict__ a,
                                     const T* __restrict__ b,
                                     int D, int tid, int nthreads,
                                     float &acc) {
  for (int d = tid; d < D; d += nthreads) acc += (float)a[d] * (float)b[d];
}

template <>
__device__ inline void vec_accum_dot<float>(const float* __restrict__ a,
                                            const float* __restrict__ b,
                                            int D, int tid, int nthreads,
                                            float &acc) {
  int V = 4, DL = D / V;
  const float4* a4 = reinterpret_cast<const float4*>(a);
  const float4* b4 = reinterpret_cast<const float4*>(b);
  for (int dl = tid; dl < DL; dl += nthreads){
    float4 x = a4[dl], y = b4[dl];
    acc += x.x*y.x + x.y*y.y + x.z*y.z + x.w*y.w;
  }
  for (int d = DL*V + tid; d < D; d += nthreads) acc += a[d]*b[d];
}

template <>
__device__ inline void vec_accum_dot<half>(const half* __restrict__ a,
                                           const half* __restrict__ b,
                                           int D, int tid, int nthreads,
                                           float &acc) {
  int V = 2, DL = D / V;
  const half2* a2 = reinterpret_cast<const half2*>(a);
  const half2* b2 = reinterpret_cast<const half2*>(b);
  for (int dl = tid; dl < DL; dl += nthreads){
    half2 x = a2[dl], y = b2[dl];
    float2 xf = __half22float2(x);
    float2 yf = __half22float2(y);
    acc += xf.x*yf.x + xf.y*yf.y;
  }
  for (int d=DL*V + tid; d < D; d += nthreads)
    acc += __half2float(a[d]) * __half2float(b[d]);
}

__device__ inline float dropout_keep_prob(uint64_t seed, int bh, int i, int j, float p_drop){
  if (p_drop <= 0.0f) return 1.0f;
  curandStatePhilox4_32_10_t state;
  curand_init((unsigned long long)seed + (unsigned long long)bh, j, i, &state);
  float4 r = curand_uniform4(&state);
  float u = r.x;
  float keep = (u >= p_drop) ? 1.0f : 0.0f;
  return keep / (1.0f - p_drop);
}

template <typename T, int BM, int BN>
__global__ __launch_bounds__(256, 2)
void fa_bwd_tiled_kernel(const T* __restrict__ Q,
                         const T* __restrict__ K,
                         const T* __restrict__ V,
                         const T* __restrict__ dOut,
                         const float* __restrict__ mask, // [B,H,S,S] or null
                         T* __restrict__ dQ,
                         T* __restrict__ dK,
                         T* __restrict__ dV,
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
  float* dKtile = reinterpret_cast<float*>(smem_raw + (BN * D * sizeof(T)) * 2);
  float* dVtile = dKtile + BN * D;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int nthreads = blockDim.x;

  const int head_stride = S * D;
  const T* Qhead = Q + ((b*H + h) * head_stride);
  const T* Khead = K + ((b*H + h) * head_stride);
  const T* Vhead = V + ((b*H + h) * head_stride);
  const T* dOhead = dOut + ((b*H + h) * head_stride);
  T* dQhead = dQ + ((b*H + h) * head_stride);
  T* dKhead = dK + ((b*H + h) * head_stride);
  T* dVhead = dV + ((b*H + h) * head_stride);

  const int bh = b*H + h;

  // Zero dQ rows handled by this block
  for (int ii=0; ii<BM; ++ii){
    int i = i0 + ii; if (i>=S) break;
    for (int d = tid; d < D; d += nthreads){
      dQhead[i*D + d] = (T)0;
    }
  }

  for (int j0=0; j0<S; j0+=BN){
    if (is_causal && j0 > i0 + BM - 1) break;

    // Load K/V tile and zero shared partials
    {
      TESSERA_NVTX_RANGE_COLORED("bwd:LDS:KV+zero", 0xFFAA7733);
      int vec = std::is_same<T,float>::value ? 4 : 2;
      int elems = BN * D;
      int iters = (elems + vec-1)/vec;
      for (int t = tid; t < iters; t += nthreads){
        int base = t * vec;
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
              Ktile[j*D + dd] = (T)0; Vtile[j*D + dd] = (T)0;
            }
            dKtile[j*D + dd] = 0.0f;
            dVtile[j*D + dd] = 0.0f;
          }
        }
      }
    }
    __syncthreads();

    for (int ii=0; ii<BM; ++ii){
      int i = i0 + ii; if (i>=S) break;
      // Pass 1: row max
      float m = -1e30f;
      for (int j=0; j<BN; ++j){
        int jj = j0 + j; if (jj>=S) break; if (is_causal && jj>i) break;
        float dot = 0.0f;
        vec_accum_dot<T>(Qhead + i*D, Ktile + j*D, D, tid, nthreads, dot);
        for (int offs=16; offs>0; offs>>=1) dot += __shfl_down_sync(0xffffffff, dot, offs);
        if (lane == 0){
          float z = dot * scale + (mask ? mask[(bh)*S*S + i*S + jj] : 0.0f);
          m = fmaxf(m, z);
        }
      }
      // Pass 2: l and s_dot
      float l = 0.0f, s_dot = 0.0f;
      for (int j=0; j<BN; ++j){
        int jj = j0 + j; if (jj>=S) break; if (is_causal && jj>i) break;
        float dot = 0.0f, dP = 0.0f;
        vec_accum_dot<T>(Qhead + i*D, Ktile + j*D, D, tid, nthreads, dot);
        vec_accum_dot<T>(dOhead + i*D, Vtile + j*D, D, tid, nthreads, dP);
        for (int offs=16; offs>0; offs>>=1){
          dot += __shfl_down_sync(0xffffffff, dot, offs);
          dP  += __shfl_down_sync(0xffffffff, dP,  offs);
        }
        if (lane == 0){
          float z = dot * scale + (mask ? mask[(bh)*S*S + i*S + jj] : 0.0f);
          float p = __expf(z - m);
          if (dropout_p > 0.0f){
            float keep = dropout_keep_prob(seed, bh, i, jj, dropout_p);
            p *= keep;
          }
          l += p; s_dot += p * dP;
        }
      }
      float rinv_l = 0.f;
      if (lane == 0) rinv_l = 1.0f / (l + 1e-20f);
      rinv_l = __shfl_sync(0xffffffff, rinv_l, 0);
      float m_bcast = __shfl_sync(0xffffffff, m, 0);

      // Pass 3: dQ, dKtile, dVtile
      for (int j=0; j<BN; ++j){
        int jj = j0 + j; if (jj>=S) break; if (is_causal && jj>i) break;
        float dot = 0.0f, dP = 0.0f;
        vec_accum_dot<T>(Qhead + i*D, Ktile + j*D, D, tid, nthreads, dot);
        vec_accum_dot<T>(dOhead + i*D, Vtile + j*D, D, tid, nthreads, dP);
        for (int offs=16; offs>0; offs>>=1){
          dot += __shfl_down_sync(0xffffffff, dot, offs);
          dP  += __shfl_down_sync(0xffffffff, dP,  offs);
        }
        float dZ = 0.0f;
        if (lane == 0){
          float z = dot * scale + (mask ? mask[(bh)*S*S + i*S + jj] : 0.0f);
          float p = __expf(z - m_bcast) * rinv_l;
          if (dropout_p > 0.0f){
            float keep = dropout_keep_prob(seed, bh, i, jj, dropout_p);
            p *= keep;
          }
          dZ = p * (dP - s_dot * rinv_l);
        }
        dZ = warp_bcast0(dZ);

        // dQ += dZ*K*scale
        for (int d = tid; d < D; d += nthreads){
          float add = dZ * (float)Ktile[j*D + d] * scale;
          atomicAdd(reinterpret_cast<float*>(&dQhead[i*D + d]), add);
        }
        // shared partials
        for (int d = tid; d < D; d += nthreads){
          dKtile[j*D + d] += dZ * (float)Qhead[i*D + d] * scale;
          dVtile[j*D + d] += (float)dOhead[i*D + d] * (/*p*/ dZ * 0 + 0); // dV uses p * dO; but p = dZ/something. For correctness-first we keep previous stage accumulation below.
        }
      }
      __syncthreads();
      // For correctness, recompute p to accumulate dV (avoids storing p per element)
      for (int j=0; j<BN; ++j){
        int jj = j0 + j; if (jj>=S) break; if (is_causal && jj>i) break;
        float dot = 0.0f;
        vec_accum_dot<T>(Qhead + i*D, Ktile + j*D, D, tid, nthreads, dot);
        for (int offs=16; offs>0; offs>>=1) dot += __shfl_down_sync(0xffffffff, dot, offs);
        float p = 0.0f;
        if (lane == 0){
          float z = dot * scale + (mask ? mask[(bh)*S*S + i*S + jj] : 0.0f);
          p = __expf(z - m_bcast) * rinv_l;
          if (dropout_p > 0.0f){
            float keep = dropout_keep_prob(seed, bh, i, jj, dropout_p);
            p *= keep;
          }
        }
        p = warp_bcast0(p);
        for (int d=tid; d<D; d+=nthreads){
          dVtile[j*D + d] += p * (float)dOhead[i*D + d];
        }
      }
      __syncthreads();
    } // ii

    // Commit partials
    {
      TESSERA_NVTX_RANGE_COLORED("bwd:commit dK/dV", 0xFFEEDD22);
      for (int idx = tid; idx < BN*D; idx += nthreads){
        int j = idx / D, d = idx % D;
        int jj = j0 + j;
        if (jj < S){
          atomicAdd(reinterpret_cast<float*>(&dKhead[jj*D + d]), dKtile[idx]);
          atomicAdd(reinterpret_cast<float*>(&dVhead[jj*D + d]), dVtile[idx]);
        }
      }
    }
    __syncthreads();
  } // tiles
}

void flashattn_bwd_tiled_launcher(at::Tensor Q, at::Tensor K, at::Tensor V,
                                  at::Tensor dOut,
                                  c10::optional<at::Tensor> attn_mask,
                                  c10::optional<at::Tensor> /*dropout_mask_unused*/,
                                  double scale, double dropout_p, bool is_causal,
                                  at::Tensor dQ, at::Tensor dK, at::Tensor dV,
                                  long long rng_seed)
{
  TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(dOut.is_cuda() && dQ.is_cuda() && dK.is_cuda() && dV.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(Q.scalar_type()==at::kFloat || Q.scalar_type()==at::kHalf, "float16/float32 only");
  int B=Q.size(0), H=Q.size(1), S=Q.size(2), D=Q.size(3);

  const float* mask_ptr = nullptr;
  if (attn_mask.has_value() && attn_mask->defined()){
    TORCH_CHECK(attn_mask->scalar_type()==at::kFloat && attn_mask->sizes()==std::vector<int64_t>({B,H,S,S}), "mask [B,H,S,S] float");
    mask_ptr = attn_mask->data_ptr<float>();
  }

  dim3 grid(div_up(S, 64), H, B);
  dim3 block(256);
  size_t smem = (size_t)(64*D*sizeof(float))*4;

  // Zero global dK/dV
  dK.zero_(); dV.zero_();

  uint64_t seed = (rng_seed==0LL) ? (uint64_t)clock64() : (uint64_t)rng_seed;

  if (Q.scalar_type()==at::kFloat){
    fa_bwd_tiled_kernel<float,64,64><<<grid, block, smem>>>(
      Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
      dOut.data_ptr<float>(), mask_ptr,
      dQ.data_ptr<float>(), dK.data_ptr<float>(), dV.data_ptr<float>(),
      B,H,S,D, (float)scale, (float)dropout_p, is_causal, seed);
  } else {
    smem = (size_t)(64*D*sizeof(at::Half))*2 + (size_t)(64*D*sizeof(float))*2;
    fa_bwd_tiled_kernel<half,64,64><<<grid, block, smem>>>(
      reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(dOut.data_ptr<at::Half>()),
      mask_ptr,
      reinterpret_cast<half*>(dQ.data_ptr<at::Half>()),
      reinterpret_cast<half*>(dK.data_ptr<at::Half>()),
      reinterpret_cast<half*>(dV.data_ptr<at::Half>()),
      B,H,S,D, (float)scale, (float)dropout_p, is_causal, seed);
  }
  CUDA_CHECK(cudaGetLastError());
}
