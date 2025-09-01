#include "common.cuh"
#include <torch/extension.h>
#include <cmath>

template <typename T, int BM, int BN>
__global__ __launch_bounds__(256, 2)
void fa_bwd_tiled_kernel(const T* __restrict__ Q,
                         const T* __restrict__ K,
                         const T* __restrict__ V,
                         const T* __restrict__ dOut,
                         const float* __restrict__ mask,          // [B,H,S,S] or null
                         const float* __restrict__ dropout_mask,  // [B,H,S,S] or null
                         T* __restrict__ dQ,
                         T* __restrict__ dK,
                         T* __restrict__ dV,
                         int B, int H, int S, int D,
                         float scale, float dropout_p, bool is_causal)
{
  const int b = blockIdx.z;
  const int h = blockIdx.y;
  const int i0 = blockIdx.x * BM;
  if (b>=B || h>=H || i0>=S) return;

  extern __shared__ char smem_raw[];
  T* Ktile = reinterpret_cast<T*>(smem_raw);
  T* Vtile = reinterpret_cast<T*>(smem_raw + BN * D * sizeof(T));
  float* dKtile = reinterpret_cast<float*>(smem_raw + (BN * D * sizeof(T))*2);
  float* dVtile = dKtile + BN * D;

  const int tid = threadIdx.x;
  const int lane = tid & 31;

  // Base pointers
  const int head_stride = S * D;
  const T* Qhead = Q + ((b*H + h) * head_stride);
  const T* Khead = K + ((b*H + h) * head_stride);
  const T* Vhead = V + ((b*H + h) * head_stride);
  const T* dOhead = dOut + ((b*H + h) * head_stride);
  T* dQhead = dQ + ((b*H + h) * head_stride);
  T* dKhead = dK + ((b*H + h) * head_stride);
  T* dVhead = dV + ((b*H + h) * head_stride);

  // Zero dQ for rows this block owns
  for (int ii=0; ii<BM; ++ii){
    int i = i0 + ii;
    if (i>=S) break;
    for (int d = tid; d < D; d += blockDim.x){
      dQhead[i*D + d] = (T)0;
    }
  }

  // Process K/V tiles
  for (int j0=0; j0<S; j0+=BN){
    if (is_causal && j0 > i0 + BM - 1) break;

    // Load K/V tile and zero partial dK/dV tile in shared
    {
      TESSERA_NVTX_RANGE_COLORED("bwd:LDS:KV+zero", 0xFFAA7733);
      for (int idx = tid; idx < BN * D; idx += blockDim.x){
        int j = idx / D, d = idx % D;
        int jj = j0 + j;
        if (jj < S){
          Ktile[j*D + d] = Khead[jj*D + d];
          Vtile[j*D + d] = Vhead[jj*D + d];
        } else {
          Ktile[j*D + d] = (T)0; Vtile[j*D + d] = (T)0;
        }
        dKtile[idx] = 0.0f;
        dVtile[idx] = 0.0f;
      }
    }
    __syncthreads();

    // For each query row i in this block, compute p_ij, softmax stats, and accumulate dQ, dKtile, dVtile
    for (int ii=0; ii<BM; ++ii){
      int i = i0 + ii;
      if (i>=S) break;
      // First pass: max
      float m = -1e30f;
      for (int j=0; j<BN; ++j){
        int jj = j0 + j;
        if (jj>=S) break;
        if (is_causal && jj>i) break;
        float dot = 0.0f;
        for (int d = tid; d < D; d += blockDim.x)
          dot += (float)Qhead[i*D + d] * (float)Ktile[j*D + d];
        for (int offs=16; offs>0; offs>>=1) dot += __shfl_down_sync(0xffffffff, dot, offs);
        if (lane == 0){
          float z = dot * scale + (mask ? mask[(b*H + h)*S*S + i*S + jj] : 0.0f);
          m = fmaxf(m, z);
        }
      }
      // Second: l and s_dot (sum p*dP)
      float l = 0.0f, s_dot = 0.0f;
      for (int j=0; j<BN; ++j){
        int jj = j0 + j;
        if (jj>=S) break;
        if (is_causal && jj>i) break;
        float dot = 0.0f, dP = 0.0f;
        for (int d=tid; d<D; d+=blockDim.x){
          dot += (float)Qhead[i*D + d] * (float)Ktile[j*D + d];
          dP  += (float)dOhead[i*D + d] * (float)Vtile[j*D + d];
        }
        for (int offs=16; offs>0; offs>>=1){
          dot += __shfl_down_sync(0xffffffff, dot, offs);
          dP  += __shfl_down_sync(0xffffffff, dP,  offs);
        }
        if (lane == 0){
          float z = dot * scale + (mask ? mask[(b*H + h)*S*S + i*S + jj] : 0.0f);
          float p = __expf(z - m);
          if (dropout_mask){
            float keep = dropout_mask[(b*H + h)*S*S + i*S + jj];
            float inv_keep = (dropout_p>0.f) ? (1.f/(1.f-dropout_p)) : 1.f;
            p *= keep * inv_keep;
          }
          l += p; s_dot += p * dP;
        }
      }
      float rinv_l = 0.f;
      if (lane == 0) rinv_l = 1.0f / (l + 1e-20f);
      rinv_l = __shfl_sync(0xffffffff, rinv_l, 0);
      float m_bcast = __shfl_sync(0xffffffff, m, 0);

      // Final: accumulate dQ_i, dKtile_j, dVtile_j
      for (int j=0; j<BN; ++j){
        int jj = j0 + j;
        if (jj>=S) break;
        if (is_causal && jj>i) break;
        float dot = 0.0f, dP = 0.0f;
        for (int d=tid; d<D; d+=blockDim.x){
          dot += (float)Qhead[i*D + d] * (float)Ktile[j*D + d];
          dP  += (float)dOhead[i*D + d] * (float)Vtile[j*D + d];
        }
        for (int offs=16; offs>0; offs>>=1){
          dot += __shfl_down_sync(0xffffffff, dot, offs);
          dP  += __shfl_down_sync(0xffffffff, dP,  offs);
        }
        if (lane == 0){
          float z = dot * scale + (mask ? mask[(b*H + h)*S*S + i*S + jj] : 0.0f);
          float p = __expf(z - m_bcast) * rinv_l;
          if (dropout_mask){
            float keep = dropout_mask[(b*H + h)*S*S + i*S + jj];
            float inv_keep = (dropout_p>0.f) ? (1.f/(1.f-dropout_p)) : 1.f;
            p *= keep * inv_keep;
          }
          float dZ = p * (dP - s_dot * rinv_l);
          // dQ += dZ * K * scale
          for (int d=tid; d<D; d+=blockDim.x){
            float add = dZ * (float)Ktile[j*D + d] * scale;
            atomicAdd(reinterpret_cast<float*>(&dQhead[i*D + d]), add);
          }
          // dKtile += dZ * Q * scale; dVtile += p * dO
          for (int d=tid; d<D; d+=blockDim.x){
            dKtile[j*D + d] += dZ * (float)Qhead[i*D + d] * scale;
            dVtile[j*D + d] += p * (float)dOhead[i*D + d];
          }
        }
      }
      __syncthreads();
    } // rows ii

    // Commit shared dK/dV tile to global
    {
      TESSERA_NVTX_RANGE_COLORED("bwd:commit dK/dV", 0xFFEEDD22);
      for (int idx=tid; idx<BN*D; idx+=blockDim.x){
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
                                  c10::optional<at::Tensor> dropout_mask,
                                  double scale, double dropout_p, bool is_causal,
                                  at::Tensor dQ, at::Tensor dK, at::Tensor dV)
{
  TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(dOut.is_cuda() && dQ.is_cuda() && dK.is_cuda() && dV.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(Q.scalar_type()==at::kFloat || Q.scalar_type()==at::kHalf, "float16/float32 only");
  int B=Q.size(0), H=Q.size(1), S=Q.size(2), D=Q.size(3);

  const float* mask_ptr = nullptr;
  const float* drop_ptr = nullptr;
  if (attn_mask.has_value() && attn_mask->defined()){
    TORCH_CHECK(attn_mask->scalar_type()==at::kFloat && attn_mask->sizes()==std::vector<int64_t>({B,H,S,S}), "mask [B,H,S,S] float");
    mask_ptr = attn_mask->data_ptr<float>();
  }
  if (dropout_mask.has_value() && dropout_mask->defined()){
    TORCH_CHECK(dropout_mask->scalar_type()==at::kFloat && dropout_mask->sizes()==std::vector<int64_t>({B,H,S,S}), "dropout_mask [B,H,S,S] float");
    drop_ptr = dropout_mask->data_ptr<float>();
  }

  dim3 grid(div_up(S, 64), H, B);
  dim3 block(256);
  size_t smem = (size_t)(64*D*sizeof(float))*4;

  // Zero dK/dV
  dK.zero_();
  dV.zero_();

  if (Q.scalar_type()==at::kFloat){
    fa_bwd_tiled_kernel<float,64,64><<<grid, block, smem>>>(
      Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
      dOut.data_ptr<float>(),
      mask_ptr, drop_ptr,
      dQ.data_ptr<float>(), dK.data_ptr<float>(), dV.data_ptr<float>(),
      B,H,S,D, (float)scale, (float)dropout_p, is_causal);
  } else {
    smem = (size_t)(64*D*sizeof(at::Half))*2 + (size_t)(64*D*sizeof(float))*2;
    fa_bwd_tiled_kernel<half,64,64><<<grid, block, smem>>>(
      reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(dOut.data_ptr<at::Half>()),
      mask_ptr, drop_ptr,
      reinterpret_cast<half*>(dQ.data_ptr<at::Half>()),
      reinterpret_cast<half*>(dK.data_ptr<at::Half>()),
      reinterpret_cast<half*>(dV.data_ptr<at::Half>()),
      B,H,S,D, (float)scale, (float)dropout_p, is_causal);
  }
  CUDA_CHECK(cudaGetLastError());
}
