#include "common.cuh"
#include <torch/extension.h>
#include <cmath>

// Row-wise naive fused backward for attention with optional mask/causal/dropout.
// Shapes: Q,K,V,Out,dOut: [B,H,S,D]
// mask (optional): [B,H,S,S] additive (e.g., -1e9 for masked)
// dropout_mask (optional): [B,H,S,S] multiplier {0,1}
// scale: usually 1/sqrt(D)
// is_causal: only attend to j <= i if true
template <typename T>
__global__ void flash_bwd_naive_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    const T* __restrict__ dOut,
    const float* __restrict__ mask,          // nullable
    const float* __restrict__ dropout_mask,  // nullable
    T* __restrict__ dQ,
    T* __restrict__ dK,
    T* __restrict__ dV,
    int B, int H, int S, int D,
    float scale,
    float dropout_p,
    bool is_causal)
{
  TESSERA_NVTX_RANGE("flash_bwd_naive_kernel");
  int b = blockIdx.z;
  int h = blockIdx.y;
  int i = blockIdx.x; // query index
  if (b>=B || h>=H || i>=S) return;

  const T* q_i = Q + (((b*H + h)*S + i) * D);
  const T* k_head = K + ((b*H + h) * S * D);
  const T* v_head = V + ((b*H + h) * S * D);
  const T* dO_i = dOut + (((b*H + h)*S + i) * D);

  T* dQ_i = dQ + (((b*H + h)*S + i) * D);

  float inv_keep = (dropout_p > 0.0f) ? (1.0f / (1.0f - dropout_p)) : 1.0f;

  extern __shared__ float smem[]; // optional future use

  // Compute logits, max, exp-sum (online softmax in 2 passes)
  float m = -1e30f;
  for (int j=0;j<S;++j){
    if (is_causal && j>i) break;
    float dot = 0.0f;
    #pragma unroll 4
    for (int d=0; d<D; ++d) dot += (float)q_i[d] * (float)k_head[j*D + d];
    if (mask) dot += mask[(b*H + h)*S*S + i*S + j];
    dot *= scale;
    m = fmaxf(m, dot);
  }

  // l = sum_j exp(z_ij - m) * (dropout? mask_ij/keep : 1)
  float l = 0.0f;
  for (int j=0;j<S;++j){
    if (is_causal && j>i) break;
    float dot = 0.0f;
    #pragma unroll 4
    for (int d=0; d<D; ++d) dot += (float)q_i[d] * (float)k_head[j*D + d];
    if (mask) dot += mask[(b*H + h)*S*S + i*S + j];
    dot *= scale;
    float p = expf(dot - m);
    if (dropout_mask){
      p *= dropout_mask[(b*H + h)*S*S + i*S + j] * inv_keep;
    }
    l += p;
  }
  float rinv_l = 1.0f / (l + 1e-20f);

  // Compute p_ij and <dP, P> term: s = sum_j p_ij * (dP_ij)
  // Here dP_ij = dO_i · V_j
  float s_dot = 0.0f;
  // We will also accumulate dV here: dV_j += p_ij * dO_i
  for (int j=0;j<S;++j){
    if (is_causal && j>i) break;
    // p_ij
    float dot = 0.0f;
    #pragma unroll 4
    for (int d=0; d<D; ++d) dot += (float)q_i[d] * (float)k_head[j*D + d];
    if (mask) dot += mask[(b*H + h)*S*S + i*S + j];
    dot *= scale;
    float p = expf(dot - m) * rinv_l;
    if (dropout_mask){
      p *= dropout_mask[(b*H + h)*S*S + i*S + j] * inv_keep;
    }
    // dP_ij = dO_i · V_j
    float dP = 0.0f;
    #pragma unroll 4
    for (int d=0; d<D; ++d){
      dP += (float)dO_i[d] * (float)v_head[j*D + d];
    }
    s_dot += p * dP;
  }

  // Accumulate dQ_i and dK_j, dV_j
  // dV_j = sum_i p_ij * dO_i   (here we're processing fixed i)
  // dZ_ij = p_ij * (dP_ij - s_dot)
  // dQ_i = sum_j dZ_ij * K_j * scale
  // dK_j = sum_i dZ_ij * Q_i * scale  (here we add for current i)
  // We'll accumulate in global memory with atomicAdd for dK/dV since j loops overlap across i.
  for (int d=0; d<D; ++d) dQ_i[d] = (T)0;

  for (int j=0;j<S;++j){
    if (is_causal && j>i) break;
    float dot = 0.0f;
    #pragma unroll 4
    for (int d=0; d<D; ++d) dot += (float)q_i[d] * (float)k_head[j*D + d];
    if (mask) dot += mask[(b*H + h)*S*S + i*S + j];
    dot *= scale;
    float p = expf(dot - m) * rinv_l;
    if (dropout_mask){
      p *= dropout_mask[(b*H + h)*S*S + i*S + j] * inv_keep;
    }
    float dP = 0.0f;
    #pragma unroll 4
    for (int d=0; d<D; ++d){
      dP += (float)dO_i[d] * (float)v_head[j*D + d];
    }
    float dZ = p * (dP - s_dot);

    // dQ_i += dZ * K_j * scale
    #pragma unroll 4
    for (int d=0; d<D; ++d){
      float add = dZ * (float)k_head[j*D + d] * scale;
      dQ_i[d] = (T)((float)dQ_i[d] + add);
    }

    // dK_j += dZ * Q_i * scale
    T* dK_j = dK + (((b*H + h)*S + j) * D);
    #pragma unroll 4
    for (int d=0; d<D; ++d){
      float add = dZ * (float)q_i[d] * scale;
      atomicAdd(reinterpret_cast<float*>(&dK_j[d]), add);
    }

    // dV_j += p * dO_i
    T* dV_j = dV + (((b*H + h)*S + j) * D);
    #pragma unroll 4
    for (int d=0; d<D; ++d){
      float add = p * (float)dO_i[d];
      atomicAdd(reinterpret_cast<float*>(&dV_j[d]), add);
    }
  }
}

void flashattn_bwd_fused_launcher(
    at::Tensor Q, at::Tensor K, at::Tensor V,
    at::Tensor dOut,
    c10::optional<at::Tensor> attn_mask,
    c10::optional<at::Tensor> dropout_mask,
    double scale, double dropout_p, bool is_causal,
    at::Tensor dQ, at::Tensor dK, at::Tensor dV)
{
  TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda() && dOut.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(dQ.is_cuda() && dK.is_cuda() && dV.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(Q.scalar_type()==at::kFloat || Q.scalar_type()==at::kHalf, "float32/float16 only");
  TORCH_CHECK(Q.sizes()==K.sizes() && Q.sizes()==V.sizes() && Q.sizes()==dQ.sizes() && Q.sizes()==dV.sizes(), "mismatched shapes");
  TORCH_CHECK(dOut.sizes()==Q.sizes(), "dOut shape");

  int B = Q.size(0), H = Q.size(1), S = Q.size(2), D = Q.size(3);
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

  // Zero dK/dV since we atomicAdd into them
  dK.zero_();
  dV.zero_();

  dim3 grid(S, H, B);
  dim3 block(1,1,1);
  if (Q.scalar_type()==at::kFloat){
    flash_bwd_naive_kernel<float><<<grid, block>>>(
      Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
      dOut.data_ptr<float>(),
      mask_ptr, drop_ptr,
      dQ.data_ptr<float>(), dK.data_ptr<float>(), dV.data_ptr<float>(),
      B,H,S,D,(float)scale,(float)dropout_p,is_causal);
  } else {
    flash_bwd_naive_kernel<half><<<grid, block>>>(
      reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(dOut.data_ptr<at::Half>()),
      mask_ptr, drop_ptr,
      reinterpret_cast<half*>(dQ.data_ptr<at::Half>()),
      reinterpret_cast<half*>(dK.data_ptr<at::Half>()),
      reinterpret_cast<half*>(dV.data_ptr<at::Half>()),
      B,H,S,D,(float)scale,(float)dropout_p,is_causal);
  }
  CUDA_CHECK(cudaGetLastError());
}
