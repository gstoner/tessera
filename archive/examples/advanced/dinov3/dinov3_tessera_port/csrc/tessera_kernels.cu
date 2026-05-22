
#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <string>

// --- Helper: get CUDA blocks/threads ---
static inline dim3 threads2D(int tx=16, int ty=16) { return dim3(tx, ty, 1); }
static inline dim3 blocks2D(int m, int n, int BM=64, int BN=64) {
  return dim3((n + BN - 1) / BN, (m + BM - 1) / BM, 1);
}

template <typename T>
__device__ __forceinline__ T gelu(T x) {
  // Approximate GELU
  return static_cast<T>(0.5) * x * (static_cast<T>(1.0) + tanh(static_cast<T>(0.79788456) * (x + static_cast<T>(0.044715) * x * x * x)));
}

// --- Tiled GEMM: C[M,N] = alpha * A[M,K] * B[K,N] + beta * C ---
template <typename scalar_t, int BM, int BN, int BK>
__global__ void tile_gemm_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta) {
  __shared__ scalar_t As[BM][BK];
  __shared__ scalar_t Bs[BK][BN];

  int row0 = blockIdx.y * BM;
  int col0 = blockIdx.x * BN;
  int ty = threadIdx.y;
  int tx = threadIdx.x;

  scalar_t acc = 0;

  for (int k0 = 0; k0 < K; k0 += BK) {
    int a_row = row0 + ty;
    int a_col = k0 + tx;
    int b_row = k0 + ty;
    int b_col = col0 + tx;

    // load with bounds check
    As[ty][tx] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : static_cast<scalar_t>(0);
    Bs[ty][tx] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : static_cast<scalar_t>(0);
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      acc += As[ty][kk] * Bs[kk][tx];
    }
    __syncthreads();
  }

  int row = row0 + ty;
  int col = col0 + tx;
  if (row < M && col < N) {
    scalar_t c = (beta == 0.0f) ? 0 : C[row * N + col];
    C[row * N + col] = static_cast<scalar_t>(alpha) * acc + static_cast<scalar_t>(beta) * c;
  }
}

// --- LayerNorm over last dim D ---
template <typename scalar_t>
__global__ void layer_norm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ w,   // optional, can be null
    const scalar_t* __restrict__ b,   // optional, can be null
    scalar_t* __restrict__ y,
    int M, int D, double eps) {
  // one block per row (M), 256 threads do a reduction
  extern __shared__ float smem[];
  float* ssum = smem;
  float* ssum2 = smem + blockDim.x;

  int row = blockIdx.x;
  if (row >= M) return;

  // parallel sum
  float thread_sum = 0.0f;
  float thread_sum2 = 0.0f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float v = static_cast<float>(x[row * D + i]);
    thread_sum += v;
    thread_sum2 += v * v;
  }
  ssum[threadIdx.x] = thread_sum;
  ssum2[threadIdx.x] = thread_sum2;
  __syncthreads();

  // reduce
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      ssum[threadIdx.x] += ssum[threadIdx.x + stride];
      ssum2[threadIdx.x] += ssum2[threadIdx.x + stride];
    }
    __syncthreads();
  }

  float mean = ssum[0] / D;
  float var = ssum2[0] / D - mean * mean;
  float inv_std = rsqrtf(var + static_cast<float>(eps));

  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float v = static_cast<float>(x[row * D + i]);
    float h = (v - mean) * inv_std;
    if (w) h *= static_cast<float>(w[i]);
    if (b) h += static_cast<float>(b[i]);
    y[row * D + i] = static_cast<scalar_t>(h);
  }
}

// --- Rowwise softmax over last dim N ---
template <typename scalar_t>
__global__ void rowwise_softmax_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    int M, int N) {
  extern __shared__ float smem[];
  float* smax = smem;
  float* ssum = smem + blockDim.x;

  int row = blockIdx.x;
  if (row >= M) return;

  // find max
  float tmax = -1e30f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float v = static_cast<float>(x[row * N + i]);
    if (v > tmax) tmax = v;
  }
  smax[threadIdx.x] = tmax;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  float mx = smax[0];

  // sum exp
  float tsum = 0.0f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float v = expf(static_cast<float>(x[row * N + i]) - mx);
    tsum += v;
    y[row * N + i] = static_cast<scalar_t>(v); // store exp temporarily
  }
  ssum[threadIdx.x] = tsum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      ssum[threadIdx.x] += ssum[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float denom = ssum[0];

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float v = static_cast<float>(y[row * N + i]) / denom;
    y[row * N + i] = static_cast<scalar_t>(v);
  }
}

// --- Launcher: TileLinear ---
torch::Tensor tile_linear(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> bias_opt, c10::optional<std::string> act_opt) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "tile_linear: tensors must be CUDA");
  TORCH_CHECK(x.dtype() == torch::kFloat32 && w.dtype() == torch::kFloat32, "tile_linear: only float32 supported");
  TORCH_CHECK(x.dim() == 3, "x must be (B,N,K)");
  TORCH_CHECK(w.dim() == 2, "w must be (M,K)");

  auto B = x.size(0);
  auto N = x.size(1);
  auto K = x.size(2);
  auto M = w.size(0);
  TORCH_CHECK(w.size(1) == K, "w.shape[1] must equal x.shape[2]");

  auto y = torch::empty({B, N, M}, x.options());

  // Flatten to (M_, K) x (K, N_) -> (M_, N_) with M_ = B*N
  auto x2 = x.reshape({B * N, K}).contiguous();
  auto y2 = y.reshape({B * N, M}).contiguous();
  auto wT = w.t().contiguous(); // (K,M) row-major

  const int BM = 64, BN = 64, BK = 32;
  dim3 block(16, 16);
  dim3 grid = blocks2D(x2.size(0), y2.size(1), BM, BN);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "tile_gemm_launch", ([&] {
    tile_gemm_kernel<scalar_t, BM, BN, BK><<<grid, block>>>(
        x2.data_ptr<scalar_t>(),
        wT.data_ptr<scalar_t>(),
        y2.data_ptr<scalar_t>(),
        x2.size(0), y2.size(1), K, 1.0f, 0.0f);
  }));

  // bias + activation (on y2)
  if (bias_opt.has_value()) {
    auto b = bias_opt.value();
    TORCH_CHECK(b.is_cuda() && b.dim() == 1 && b.size(0) == M, "bias shape mismatch");
    y2.add_(b); // broadcast over row dimension
  }
  if (act_opt.has_value()) {
    auto act = act_opt.value();
    if (act == "gelu") {
      y2 = torch::gelu(y2);
    } else if (act == "relu") {
      y2 = torch::relu(y2);
    }
  }

  return y2.reshape({B, N, M});
}

// --- Launcher: LayerNorm (forward) ---
torch::Tensor layer_norm(torch::Tensor x, double eps, c10::optional<torch::Tensor> w_opt, c10::optional<torch::Tensor> b_opt) {
  TORCH_CHECK(x.is_cuda(), "layer_norm: CUDA only for this build");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "layer_norm: float32 only");
  TORCH_CHECK(x.dim() >= 2, "layer_norm: x must be at least 2D");
  auto D = x.size(-1);
  auto M = x.numel() / D;

  auto x2 = x.reshape({M, D}).contiguous();
  auto y2 = torch::empty_like(x2);

  const float *wptr = nullptr, *bptr = nullptr;
  torch::Tensor w, b;
  if (w_opt.has_value()) { w = w_opt.value().contiguous(); TORCH_CHECK(w.is_cuda() && w.numel()==D, "w mismatch"); wptr = w.data_ptr<float>(); }
  if (b_opt.has_value()) { b = b_opt.value().contiguous(); TORCH_CHECK(b.is_cuda() && b.numel()==D, "b mismatch"); bptr = b.data_ptr<float>(); }

  int threads = 256;
  int blocks = M;
  size_t smem = threads * sizeof(float) * 2;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layer_norm_launch", ([&] {
    layer_norm_kernel<float><<<blocks, threads, smem>>>(
      x2.data_ptr<float>(),
      wptr, bptr,
      y2.data_ptr<float>(),
      M, D, eps);
  }));

  return y2.reshape_as(x);
}

// --- Launcher: Rowwise Softmax ---
torch::Tensor rowwise_softmax(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "rowwise_softmax: CUDA tensor required");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "rowwise_softmax: float32 only");
  TORCH_CHECK(x.dim() == 2, "rowwise_softmax: x must be (M,N)");
  int M = x.size(0), N = x.size(1);
  auto y = torch::empty_like(x);

  int threads = 256;
  int blocks = M;
  size_t smem = threads * sizeof(float) * 2;
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "rowwise_softmax_launch", ([&] {
    rowwise_softmax_kernel<float><<<blocks, threads, smem>>>(
      x.data_ptr<float>(),
      y.data_ptr<float>(),
      M, N);
  }));
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tile_linear", &tile_linear, "Tessera TileLinear (CUDA)");
  m.def("layer_norm", &layer_norm, "Tessera LayerNorm (CUDA)");
  m.def("rowwise_softmax", &rowwise_softmax, "Tessera Rowwise Softmax (CUDA)");
}


// ===================== Batched GEMM (A[M,K] x B[K,N]) =====================

template <typename scalar_t, int BM, int BN, int BK, bool TRANS_B>
__global__ void tile_gemm_batched_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int BATCH, int M, int N, int K,
    int strideA, int strideB, int strideC) {

  __shared__ scalar_t As[BM][BK];
  __shared__ scalar_t Bs[BK][BN];

  int b = blockIdx.z;
  if (b >= BATCH) return;

  int row0 = blockIdx.y * BM;
  int col0 = blockIdx.x * BN;
  int ty = threadIdx.y;
  int tx = threadIdx.x;

  scalar_t acc = 0;

  const scalar_t* A_b = A + b * strideA;
  const scalar_t* B_b = B + b * strideB;
  scalar_t* C_b = C + b * strideC;

  for (int k0 = 0; k0 < K; k0 += BK) {
    int a_row = row0 + ty;
    int a_col = k0 + tx;

    int b_row = k0 + ty;
    int b_col = col0 + tx;

    // load tiles with bounds checks
    As[ty][tx] = (a_row < M && a_col < K) ? A_b[a_row * K + a_col] : static_cast<scalar_t>(0);

    if constexpr (TRANS_B) {
      // B is provided as (N,K) logically (transposed), so element at (b_row, b_col) refers to [col, row] in original
      // We index B_b as row-major [N,K]
      Bs[ty][tx] = (b_row < K && b_col < N) ? B_b[b_col * K + b_row] : static_cast<scalar_t>(0);
    } else {
      // B is [K,N] row-major
      Bs[ty][tx] = (b_row < K && b_col < N) ? B_b[b_row * N + b_col] : static_cast<scalar_t>(0);
    }
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      acc += As[ty][kk] * Bs[kk][tx];
    }
    __syncthreads();
  }

  int row = row0 + ty;
  int col = col0 + tx;
  if (row < M && col < N) {
    C_b[row * N + col] = acc;
  }
}

torch::Tensor batched_gemm(torch::Tensor A, torch::Tensor B, bool trans_b) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "batched_gemm: CUDA tensors required");
  TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "batched_gemm: float32 only");
  TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "batched_gemm: A and B must be 3D");
  int BATCH = A.size(0);
  int M = A.size(1);
  int K = A.size(2);

  if (!trans_b) {
    TORCH_CHECK(B.size(0) == BATCH && B.size(1) == K, "B must be (B,K,N)");
  }
  int N = trans_b ? B.size(1) : B.size(2);
  if (trans_b) {
    TORCH_CHECK(B.size(0) == BATCH && B.size(2) == K, "B^T must be (B,N,K)");
  }

  auto C = torch::empty({BATCH, M, N}, A.options());
  auto Acont = A.contiguous();
  auto Bcont = B.contiguous();
  auto Ccont = C.contiguous();

  const int BM = 64, BN = 64, BK = 32;
  dim3 block(16, 16, 1);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, BATCH);

  int strideA = M * K;
  int strideB = (trans_b ? (N * K) : (K * N));
  int strideC = M * N;

  if (trans_b) {
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "batched_gemm_tB_launch", ([&] {
      tile_gemm_batched_kernel<scalar_t, BM, BN, BK, true><<<grid, block>>>(
          Acont.data_ptr<scalar_t>(),
          Bcont.data_ptr<scalar_t>(),
          Ccont.data_ptr<scalar_t>(),
          BATCH, M, N, K, strideA, strideB, strideC);
    }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "batched_gemm_launch", ([&] {
      tile_gemm_batched_kernel<scalar_t, BM, BN, BK, false><<<grid, block>>>(
          Acont.data_ptr<scalar_t>(),
          Bcont.data_ptr<scalar_t>(),
          Ccont.data_ptr<scalar_t>(),
          BATCH, M, N, K, strideA, strideB, strideC);
    }));
  }

  return Ccont;
}

// ===================== Small Fused FlashAttention (forward, float32, no mask) =====================
// One block per (g,i) where g is head-batch index and i is query index in [0,N).
// This is a correctness-first fused kernel (not heavily optimized).

template <int BN>
__global__ void flash_attn_forward_kernel(
    const float* __restrict__ Q,  // [G, N, D]
    const float* __restrict__ K,  // [G, N, D]
    const float* __restrict__ V,  // [G, N, D]
    float* __restrict__ O,        // [G, N, D]
    int G, int N, int D, float scale)
{
  int gid = blockIdx.y;      // head-batch index in [0,G)
  int qi = blockIdx.x;       // query index in [0,N)
  if (gid >= G || qi >= N) return;

  extern __shared__ float smem[];
  float* Kblk = smem;                 // BN * D
  float* Vblk = smem + BN * D;        // BN * D

  const float* Qg = Q + gid * N * D;
  const float* Kg = K + gid * N * D;
  const float* Vg = V + gid * N * D;
  float* Og = O + gid * N * D;

  const float* q = Qg + qi * D;       // query row
  float m_i = -1e30f;
  float l_i = 0.0f;

  // accumulator for output
  // each thread accumulates a subset, then we write back coalesced
  // here we do simple per-thread striding
  for (int t = threadIdx.x; t < D; t += blockDim.x) {
    Og[qi * D + t] = 0.0f;
  }
  __syncthreads();

  for (int k0 = 0; k0 < N; k0 += BN) {
    int bn = min(BN, N - k0);

    // load K and V tiles to shared
    for (int t = threadIdx.x; t < bn * D; t += blockDim.x) {
      Kblk[t] = Kg[k0 * D + t];
      Vblk[t] = Vg[k0 * D + t];
    }
    __syncthreads();

    // compute scores for this block: s_j = <q, K_j> * scale
    // find block max
    float blk_max = -1e30f;
    for (int j = 0; j < bn; ++j) {
      // dot product q ⋅ Kblk[j]
      float dot = 0.0f;
      for (int t = threadIdx.x; t < D; t += blockDim.x) {
        dot += q[t] * Kblk[j * D + t];
      }
      // warp-reduce within blockDim.x using shared mem atomic add? We'll do naive: store per-thread partial in shared then reduce
      __shared__ float partial;
      if (threadIdx.x == 0) partial = 0.0f;
      __syncthreads();
      atomicAdd(&partial, dot);
      __syncthreads();
      float s_j = partial * scale;
      if (threadIdx.x == 0) {
        if (s_j > blk_max) blk_max = s_j;
      }
      __syncthreads();
    }
    // broadcast blk_max
    __shared__ float smx;
    if (threadIdx.x == 0) smx = blk_max;
    __syncthreads();
    float new_m = fmaxf(m_i, smx);
    float exp_m_diff = expf(m_i - new_m);

    // update denominator and output accumulator
    float blk_l = 0.0f;
    for (int j = 0; j < bn; ++j) {
      float dot = 0.0f;
      for (int t = threadIdx.x; t < D; t += blockDim.x) {
        dot += q[t] * Kblk[j * D + t];
      }
      __shared__ float partial2;
      if (threadIdx.x == 0) partial2 = 0.0f;
      __syncthreads();
      atomicAdd(&partial2, dot);
      __syncthreads();
      float s_j = partial2 * scale;
      float p = expf(s_j - new_m);
      blk_l += p;

      // accumulate output
      for (int t = threadIdx.x; t < D; t += blockDim.x) {
        float prev = Og[qi * D + t] * exp_m_diff;
        float contrib = p * Vblk[j * D + t];
        Og[qi * D + t] = prev + contrib;
      }
      __syncthreads();
    }

    l_i = l_i * exp_m_diff + blk_l;
    m_i = new_m;
    __syncthreads();
  }

  // normalize
  for (int t = threadIdx.x; t < D; t += blockDim.x) {
    Og[qi * D + t] = Og[qi * D + t] / l_i;
  }
}

// Wrapper: expects Q,K,V shaped [B,H,N,D]; returns same
torch::Tensor flash_attn_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
  TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "flash_attn_forward: CUDA tensors required");
  TORCH_CHECK(Q.dtype() == torch::kFloat32, "flash_attn_forward: float32 only");
  TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4, "flash_attn_forward: Q,K,V = [B,H,N,D]");

  int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
  TORCH_CHECK(K.size(0)==B && K.size(1)==H && K.size(2)==N && K.size(3)==D, "K shape mismatch");
  TORCH_CHECK(V.size(0)==B && V.size(1)==H && V.size(2)==N && V.size(3)==D, "V shape mismatch");

  float scale = 1.0f / std::sqrt((float)D);

  // collapse G = B*H
  auto Qc = Q.contiguous().view({B*H, N, D});
  auto Kc = K.contiguous().view({B*H, N, D});
  auto Vc = V.contiguous().view({B*H, N, D});
  auto O = torch::empty_like(Qc);

  dim3 grid(N, B*H, 1);
  dim3 block(128, 1, 1);
  size_t smem = (64 * D + 64 * D) * sizeof(float); // BN=64
  flash_attn_forward_kernel<64><<<grid, block, smem>>>(
      Qc.data_ptr<float>(), Kc.data_ptr<float>(), Vc.data_ptr<float>(),
      O.data_ptr<float>(), B*H, N, D, scale);

  return O.view({B, H, N, D});
}


// ===================== 2D GEMM helpers for backward (float32) =====================
template <typename scalar_t, int BM, int BN, int BK>
__global__ void tile_gemm2d_kernel(
    const scalar_t* __restrict__ A, // [M,K]
    const scalar_t* __restrict__ B, // [K,N]
    scalar_t* __restrict__ C,       // [M,N]
    int M, int N, int K) {
  __shared__ scalar_t As[BM][BK];
  __shared__ scalar_t Bs[BK][BN];
  int row0 = blockIdx.y * BM;
  int col0 = blockIdx.x * BN;
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  float acc = 0.0f;
  for (int k0 = 0; k0 < K; k0 += BK) {
    int a_row = row0 + ty;
    int a_col = k0 + tx;
    int b_row = k0 + ty;
    int b_col = col0 + tx;
    As[ty][tx] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : (scalar_t)0;
    Bs[ty][tx] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : (scalar_t)0;
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      acc += static_cast<float>(As[ty][kk]) * static_cast<float>(Bs[kk][tx]);
    }
    __syncthreads();
  }
  int row = row0 + ty;
  int col = col0 + tx;
  if (row < M && col < N) {
    C[row * N + col] = static_cast<scalar_t>(acc);
  }
}

// C = A^T * B where A is [M,K], B is [M,N]  -> C [K,N]
template <typename scalar_t, int BK, int BN, int BM>
__global__ void tile_gemm_AT_B_kernel(
    const scalar_t* __restrict__ A, // [M,K]
    const scalar_t* __restrict__ B, // [M,N]
    scalar_t* __restrict__ C,       // [K,N]
    int M, int N, int K) {
  __shared__ scalar_t As[BM][BK];
  __shared__ scalar_t Bs[BM][BN];
  int k0 = blockIdx.y * BK;
  int n0 = blockIdx.x * BN;
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  float acc = 0.0f;
  for (int m0 = 0; m0 < M; m0 += BM) {
    int a_row = m0 + ty;
    int a_col = k0 + tx;
    int b_row = m0 + ty;
    int b_col = n0 + tx;
    As[ty][tx] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : (scalar_t)0;
    Bs[ty][tx] = (b_row < M && b_col < N) ? B[b_row * N + b_col] : (scalar_t)0;
    __syncthreads();
    #pragma unroll
    for (int mm = 0; mm < BM; ++mm) {
      acc += static_cast<float>(As[mm][ty]) * static_cast<float>(Bs[mm][tx]); // As transposed on the fly
    }
    __syncthreads();
  }
  int k = k0 + ty;
  int n = n0 + tx;
  if (k < K && n < N) {
    C[k * N + n] = static_cast<scalar_t>(acc);
  }
}

torch::Tensor linear_bw_input(torch::Tensor dY, torch::Tensor W) {
  TORCH_CHECK(dY.is_cuda() && W.is_cuda(), "linear_bw_input: CUDA tensors required");
  TORCH_CHECK(dY.dtype() == torch::kFloat32 && W.dtype() == torch::kFloat32, "float32 only");
  TORCH_CHECK(dY.dim() == 2 && W.dim() == 2, "dY [M,N], W [N,K]");
  int M = dY.size(0), N = dY.size(1), K = W.size(1);
  TORCH_CHECK(W.size(0) == N, "W shape mismatch");
  auto dX = torch::empty({M, K}, dY.options());
  dim3 block(16,16,1);
  dim3 grid((K+63)/64, (M+63)/64, 1);
  AT_DISPATCH_FLOATING_TYPES(dY.scalar_type(), "linear_bw_input", ([&] {
    tile_gemm2d_kernel<scalar_t,64,64,32><<<grid, block>>>(
      dY.data_ptr<scalar_t>(), W.data_ptr<scalar_t>(), dX.data_ptr<scalar_t>(), M, K, N);
  }));
  return dX;
}

torch::Tensor linear_bw_weight(torch::Tensor dY, torch::Tensor X) {
  TORCH_CHECK(dY.is_cuda() && X.is_cuda(), "linear_bw_weight: CUDA tensors required");
  TORCH_CHECK(dY.dtype() == torch::kFloat32 && X.dtype() == torch::kFloat32, "float32 only");
  TORCH_CHECK(dY.dim() == 2 && X.dim() == 2, "dY [M,N], X [M,K]");
  int M = dY.size(0), N = dY.size(1), K = X.size(1);
  auto dW = torch::empty({N, K}, dY.options());
  dim3 block(16,16,1);
  dim3 grid((K+63)/64, (N+63)/64, 1);
  AT_DISPATCH_FLOATING_TYPES(dY.scalar_type(), "linear_bw_weight", ([&] {
    tile_gemm_AT_B_kernel<scalar_t,64,64,64><<<grid, block>>>(
      dY.data_ptr<scalar_t>(), X.data_ptr<scalar_t>(), dW.data_ptr<scalar_t>(), M, N, K);
  }));
  return dW;
}

// ===================== LayerNorm backward =====================
__global__ void layer_norm_bw_kernel(
    const float* __restrict__ x,
    const float* __restrict__ dy,
    const float* __restrict__ w,  // optional
    float* __restrict__ dx,
    float* __restrict__ dw,       // optional
    float* __restrict__ db,       // optional
    int M, int D, double eps)
{
  extern __shared__ float smem[];
  float* ssum = smem;
  float* ssum2 = smem + blockDim.x;
  int row = blockIdx.x;
  if (row >= M) return;

  // compute mean/var
  float ts1 = 0.f, ts2 = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float v = x[row*D + i];
    ts1 += v;
    ts2 += v*v;
  }
  ssum[threadIdx.x] = ts1;
  ssum2[threadIdx.x] = ts2;
  __syncthreads();
  for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      ssum[threadIdx.x] += ssum[threadIdx.x + s];
      ssum2[threadIdx.x] += ssum2[threadIdx.x + s];
    }
    __syncthreads();
  }
  float mean = ssum[0] / D;
  float var = ssum2[0] / D - mean*mean;
  float inv_std = rsqrtf(var + (float)eps);

  // compute yhat and dŷ
  float sum_dy = 0.f, sum_dy_yhat = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float yhat = (x[row*D + i] - mean) * inv_std;
    float dyi = dy[row*D + i] * (w ? w[i] : 1.f);
    sum_dy += dyi;
    sum_dy_yhat += dyi * yhat;
  }
  ssum[threadIdx.x] = sum_dy;
  ssum2[threadIdx.x] = sum_dy_yhat;
  __syncthreads();
  for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      ssum[threadIdx.x] += ssum[threadIdx.x + s];
      ssum2[threadIdx.x] += ssum2[threadIdx.x + s];
    }
    __syncthreads();
  }
  float S1 = ssum[0];
  float S2 = ssum2[0];

  // dx
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float yhat = (x[row*D + i] - mean) * inv_std;
    float dyi = dy[row*D + i] * (w ? w[i] : 1.f);
    float dx_i = (dyi - (S1/D) - yhat * (S2/D)) * inv_std;
    dx[row*D + i] = dx_i;
  }

  // dw, db (reduce across rows outside if needed)
}

std::vector<torch::Tensor> layer_norm_bw(torch::Tensor x, torch::Tensor dy, c10::optional<torch::Tensor> w_opt, double eps) {
  TORCH_CHECK(x.is_cuda() && dy.is_cuda(), "layer_norm_bw: CUDA only");
  TORCH_CHECK(x.dtype() == torch::kFloat32 && dy.dtype() == torch::kFloat32, "float32 only");
  int D = x.size(-1);
  int M = x.numel() / D;
  auto x2 = x.reshape({M, D}).contiguous();
  auto dy2 = dy.reshape({M, D}).contiguous();
  auto dx2 = torch::empty_like(x2);

  torch::Tensor w;
  const float* wptr = nullptr;
  if (w_opt.has_value()) {
    w = w_opt.value().contiguous();
    TORCH_CHECK(w.is_cuda() && w.numel()==D, "w mismatch");
    wptr = w.data_ptr<float>();
  }

  int threads = 256;
  int blocks = M;
  size_t smem = threads*sizeof(float)*2;
  layer_norm_bw_kernel<<<blocks, threads, smem>>>(
    x2.data_ptr<float>(), dy2.data_ptr<float>(), wptr, dx2.data_ptr<float>(), nullptr, nullptr, M, D, eps
  );
  return {dx2.reshape_as(x), torch::Tensor(), torch::Tensor()};
}

// ===================== FlashAttention backward (CUDA-assisted using our primitives) =====================
std::vector<torch::Tensor> flash_attn_backward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                               torch::Tensor dO, bool causal,
                                               c10::optional<torch::Tensor> dropout_mask_opt) {
  TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda() && dO.is_cuda(), "flash_attn_backward: CUDA only");
  TORCH_CHECK(Q.dtype() == torch::kFloat32, "float32 only");
  int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
  int G = B*H;

  auto Qg = Q.view({G,N,D}).contiguous();
  auto Kg = K.view({G,N,D}).contiguous();
  auto Vg = V.view({G,N,D}).contiguous();
  auto dOg = dO.view({G,N,D}).contiguous();

  // scores = QK^T
  auto scores = batched_gemm(Qg, Kg, true); // [G,N,N]
  if (causal) {
    // apply causal mask (upper triangle to -inf)
    auto iu = torch::ones({N, N}, scores.options()).triu(1) * 1e9;
    scores = scores - iu; // subtract large number -> -inf approx
  }
  // softmax
  auto probs2d = scores.reshape({G*N, N});
  auto probs = rowwise_softmax(probs2d).reshape({G,N,N}); // P

  if (dropout_mask_opt.has_value()) {
    auto mask = dropout_mask_opt.value().contiguous();
    TORCH_CHECK(mask.sizes() == probs.sizes(), "dropout mask shape must match probs");
    probs = probs * mask;
  }

  // dV = P^T @ dO
  auto dV = batched_gemm(probs.transpose(1,2).contiguous(), dOg, false);

  // dP = dO @ V^T
  auto dP = batched_gemm(dOg, Vg, true);

  // softmax backward per row: z = P ⊙ (dP - sum(dP ⊙ P))
  auto tmp = (dP * probs).sum(dim=2, keepdim=True);
  auto z = probs * (dP - tmp);  // [G,N,N]

  float scale = 1.0f / std::sqrt((float)D);
  // dQ = z @ K * scale
  auto dQ = batched_gemm(z, Kg, false) * scale;
  // dK = z^T @ Q * scale
  auto dK = batched_gemm(z.transpose(1,2).contiguous(), Qg, false) * scale;

  return {dQ.view({B,H,N,D}), dK.view({B,H,N,D}), dV.view({B,H,N,D})};
}

// ===================== Fused QKV + Bias + GELU (forward only micro-kernel) =====================
// Here we pack three linears (Q,K,V) into one GEMM by concatenating weights; for simplicity, launch three GEMMs.
torch::Tensor qkv_bias_gelu(torch::Tensor X, torch::Tensor Wq, torch::Tensor bq,
                            torch::Tensor Wk, torch::Tensor bk,
                            torch::Tensor Wv, torch::Tensor bv) {
  TORCH_CHECK(X.is_cuda() && Wq.is_cuda() && Wk.is_cuda() && Wv.is_cuda(), "CUDA only");
  TORCH_CHECK(X.dtype() == torch::kFloat32, "float32 only");
  int B = X.size(0), N = X.size(1);
  int K = X.size(2), Dq = Wq.size(0), Dk = Wk.size(0), Dv = Wv.size(0);

  auto X2 = X.reshape({B*N, K}).contiguous();
  auto Yq = torch::empty({B*N, Dq}, X.options());
  auto Yk = torch::empty({B*N, Dk}, X.options());
  auto Yv = torch::empty({B*N, Dv}, X.options());

  // Y = X @ W^T + b
  {
    dim3 block(16,16,1);
    dim3 grid((Dq+63)/64, (B*N+63)/64, 1);
    AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "gemm_q", ([&] {
      tile_gemm2d_kernel<scalar_t,64,64,32><<<grid, block>>>(
        X2.data_ptr<scalar_t>(),
        Wq.t().contiguous().data_ptr<scalar_t>(),
        Yq.data_ptr<scalar_t>(), B*N, Dq, K);
    }));
    Yq.add_(bq);
  }
  {
    dim3 block(16,16,1);
    dim3 grid((Dk+63)/64, (B*N+63)/64, 1);
    AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "gemm_k", ([&] {
      tile_gemm2d_kernel<scalar_t,64,64,32><<<grid, block>>>(
        X2.data_ptr<scalar_t>(),
        Wk.t().contiguous().data_ptr<scalar_t>(),
        Yk.data_ptr<scalar_t>(), B*N, Dk, K);
    }));
    Yk.add_(bk);
  }
  {
    dim3 block(16,16,1);
    dim3 grid((Dv+63)/64, (B*N+63)/64, 1);
    AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "gemm_v", ([&] {
      tile_gemm2d_kernel<scalar_t,64,64,32><<<grid, block>>>(
        X2.data_ptr<scalar_t>(),
        Wv.t().contiguous().data_ptr<scalar_t>(),
        Yv.data_ptr<scalar_t>(), B*N, Dv, K);
    }));
    Yv.add_(bv);
  }

  // GELU on Q only (common micro-fusion before split_heads), leave K,V linear
  Yq = torch::gelu(Yq);

  return torch::stack({Yq, Yk, Yv}, 0).reshape({3, B, N, -1});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tile_linear", &tile_linear, "Tessera TileLinear (CUDA)");
  m.def("linear_bw_input", &linear_bw_input, "TileLinear dX (CUDA)");
  m.def("linear_bw_weight", &linear_bw_weight, "TileLinear dW (CUDA)");
  m.def("layer_norm", &layer_norm, "Tessera LayerNorm (CUDA)");
  m.def("layer_norm_bw", &layer_norm_bw, "Tessera LayerNorm backward (CUDA)");
  m.def("rowwise_softmax", &rowwise_softmax, "Rowwise Softmax (CUDA)");
  m.def("batched_gemm", &batched_gemm, "Batched GEMM (CUDA)");
  m.def("flash_attn_forward", &flash_attn_forward, "Fused FlashAttention forward (CUDA)");
  m.def("flash_attn_backward", &flash_attn_backward, "FlashAttention backward (CUDA-assisted)");
  m.def("qkv_bias_gelu", &qkv_bias_gelu, "Fused QKV + Bias + GELU (forward micro-kernel)");
}


#include <mma.h>
using namespace nvcuda;

// ===================== WMMA (FP16) GEMM =====================
// Computes C[M,N] = A[M,K] * B[K,N] in half-precision, accumulates in FP32.
// Requirements: M,N,K multiples of 16; row-major A, row-major B (we load B as col-major by using its transpose).
__global__ void wmma_gemm_kernel(const half* __restrict__ A, const half* __restrict__ Bt, float* __restrict__ C,
                                 int M, int N, int K) {
  int tile_m = blockIdx.y;
  int tile_n = blockIdx.x;

  // one warp per 16x16 tile
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
  wmma::fill_fragment(c, 0.0f);

  for (int tile_k = 0; tile_k < K; tile_k += 16) {
    // A tile: row-major
    const half* Aptr = A + (tile_m * 16) * K + tile_k;
    // B^T tile: row-major (means original B is col-major), so we effectively compute A * B
    const half* BptrT = Bt + (tile_n * 16) * K + tile_k;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b; // loading B^T row-major equals B col-major

    wmma::load_matrix_sync(a, Aptr, K);
    wmma::load_matrix_sync(b, BptrT, K);
    wmma::mma_sync(c, a, b, c);
  }

  float* Cptr = C + (tile_m * 16) * N + tile_n * 16;
  wmma::store_matrix_sync(Cptr, c, N, wmma::mem_row_major);
}

torch::Tensor tile_linear_wmma(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> bias_opt, c10::optional<std::string> act_opt) {
  // x: [B,N,K] half; w: [M,K] half
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "tile_linear_wmma: CUDA tensors required");
  TORCH_CHECK(x.dtype() == torch::kHalf && w.dtype() == torch::kHalf, "WMMA path requires FP16");
  TORCH_CHECK(x.dim() == 3 && w.dim() == 2, "x: [B,N,K], w: [M,K]");

  int B = x.size(0), N = x.size(1), K = x.size(2), M = w.size(0);
  TORCH_CHECK(w.size(1) == K, "w.shape[1] must equal x.shape[2]");
  TORCH_CHECK((N % 16)==0 and (M % 16)==0 and (K % 16)==0, "WMMA requires dims multiple of 16");

  auto x2 = x.reshape({B*N, K}).contiguous();
  auto wT = w.t().contiguous(); // [K,M] half
  auto y2 = torch::empty({B*N, M}, x.options().dtype(torch::kFloat32)); // accumulate in fp32

  dim3 grid(M/16, (B*N)/16, 1);
  dim3 block(32, 1, 1); // one warp
  wmma_gemm_kernel<<<grid, block>>>(
    reinterpret_cast<const half*>(x2.data_ptr<at::Half>()),
    reinterpret_cast<const half*>(wT.data_ptr<at::Half>()),
    y2.data_ptr<float>(),
    B*N, M, K);

  // bias + activation
  if (bias_opt.has_value()) {
    auto b = bias_opt.value().to(y2.dtype());
    y2.add_(b);
  }
  if (act_opt.has_value()) {
    auto act = act_opt.value();
    if (act == "gelu") y2 = torch::gelu(y2);
    else if (act == "relu") y2 = torch::relu(y2);
  }
  return y2.to(x.dtype()).reshape({B, N, M});
}

// ===================== Fused QKV single GEMM pack =====================
// X [B,N,K], Wcat [3D,K] (rows=3D), bcat [3D]
torch::Tensor qkv_pack_gemm(torch::Tensor x, torch::Tensor Wcat, torch::Tensor bcat, bool gelu_q) {
  TORCH_CHECK(x.is_cuda() && Wcat.is_cuda() && bcat.is_cuda(), "qkv_pack_gemm: CUDA only");
  TORCH_CHECK(x.dtype() == torch::kFloat32 && Wcat.dtype() == torch::kFloat32 && bcat.dtype() == torch::kFloat32, "float32 only");
  int B = x.size(0), N = x.size(1), K = x.size(2);
  int M3 = Wcat.size(0);
  TORCH_CHECK(Wcat.size(1) == K && bcat.numel() == M3, "shape mismatch");

  auto x2 = x.reshape({B*N, K}).contiguous();
  auto y2 = torch::empty({B*N, M3}, x.options());

  // Y = X @ Wcat^T
  dim3 block(16,16,1);
  dim3 grid((M3+63)/64, (B*N+63)/64, 1);
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "qkv_pack_gemm_launch", ([&] {
    tile_gemm2d_kernel<scalar_t,64,64,32><<<grid, block>>>(
      x2.data_ptr<scalar_t>(),
      Wcat.t().contiguous().data_ptr<scalar_t>(),
      y2.data_ptr<scalar_t>(), B*N, M3, K);
  }));

  // add bias and GELU on Q slice
  y2.add_(bcat);
  int D = M3 / 3;
  auto yq = y2.narrow(1, 0, D);
  if (gelu_q) yq = torch::gelu(yq);
  auto yk = y2.narrow(1, D, D);
  auto yv = y2.narrow(1, 2*D, D);

  return torch::stack({yq, yk, yv}, 0).reshape({3, B, N, D});
}

// ===================== Fused FlashAttention forward with dropout/causal =====================

__device__ inline float rng_uniform01(uint64_t seed, int gid, int qi, int j, int lane) {
  // XorShift-style hash -> float in (0,1)
  uint64_t x = seed ^ ((uint64_t)gid << 32) ^ ((uint64_t)qi << 16) ^ ((uint64_t)j << 4) ^ (uint64_t)lane;
  x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
  uint64_t m = x * 2685821657736338717ULL;
  uint32_t u = (uint32_t)((m >> 32) & 0xffffffffu);
  return (u / 4294967296.0f); // [0,1)
}

template <int BN>
__global__ void flash_attn_fwd_ex_kernel(
    const float* __restrict__ Q,  // [G, N, D]
    const float* __restrict__ K,  // [G, N, D]
    const float* __restrict__ V,  // [G, N, D]
    float* __restrict__ O,        // [G, N, D]
    int G, int N, int D, float scale, float dropout_p, bool causal, uint64_t seed)
{
  int gid = blockIdx.y;      // head-batch index
  int qi = blockIdx.x;       // query index
  if (gid >= G || qi >= N) return;

  extern __shared__ float smem[];
  float* Kblk = smem;                 // BN * D
  float* Vblk = smem + BN * D;        // BN * D

  const float* Qg = Q + gid * N * D;
  const float* Kg = K + gid * N * D;
  const float* Vg = V + gid * N * D;
  float* Og = O + gid * N * D;

  const float* q = Qg + qi * D;
  float m_i = -1e30f;
  float l_i = 0.0f;

  for (int t = threadIdx.x; t < D; t += blockDim.x) {
    Og[qi * D + t] = 0.0f;
  }
  __syncthreads();

  for (int k0 = 0; k0 < N; k0 += BN) {
    int bn = min(BN, N - k0);

    // load tiles
    for (int t = threadIdx.x; t < bn * D; t += blockDim.x) {
      Kblk[t] = Kg[k0 * D + t];
      Vblk[t] = Vg[k0 * D + t];
    }
    __syncthreads();

    // block max
    float blk_max = -1e30f;
    for (int j = 0; j < bn; ++j) {
      int key_idx = k0 + j;
      if (causal && key_idx > qi) break;
      float dot = 0.0f;
      for (int t = threadIdx.x; t < D; t += blockDim.x) {
        dot += q[t] * Kblk[j * D + t];
      }
      __shared__ float partial;
      if (threadIdx.x == 0) partial = 0.0f;
      __syncthreads();
      atomicAdd(&partial, dot);
      __syncthreads();
      float s_j = partial * scale;
      if (threadIdx.x == 0) {
        if (s_j > blk_max) blk_max = s_j;
      }
      __syncthreads();
    }
    __shared__ float smx;
    if (threadIdx.x == 0) smx = blk_max;
    __syncthreads();
    float new_m = fmaxf(m_i, smx);
    float exp_m_diff = expf(m_i - new_m);

    // update denominator and output
    float blk_l = 0.0f;
    for (int j = 0; j < bn; ++j) {
      int key_idx = k0 + j;
      if (causal && key_idx > qi) break;
      float dot = 0.0f;
      for (int t = threadIdx.x; t < D; t += blockDim.x) {
        dot += q[t] * Kblk[j * D + t];
      }
      __shared__ float partial2;
      if (threadIdx.x == 0) partial2 = 0.0f;
      __syncthreads();
      atomicAdd(&partial2, dot);
      __syncthreads();
      float s_j = partial2 * scale;
      float p = expf(s_j - new_m);
      if (dropout_p > 0.f) {
        float r = rng_uniform01(seed, gid, qi, key_idx, threadIdx.x);
        if (r < dropout_p) p = 0.f;
        else p = p / (1.f - dropout_p);
      }
      blk_l += p;

      for (int t = threadIdx.x; t < D; t += blockDim.x) {
        float prev = Og[qi * D + t] * exp_m_diff;
        float contrib = p * Vblk[j * D + t];
        Og[qi * D + t] = prev + contrib;
      }
      __syncthreads();
    }

    l_i = l_i * exp_m_diff + blk_l;
    m_i = new_m;
    __syncthreads();
  }

  for (int t = threadIdx.x; t < D; t += blockDim.x) {
    Og[qi * D + t] = Og[qi * D + t] / l_i;
  }
}

torch::Tensor flash_attn_forward_ex(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double dropout_p, bool causal, uint64_t seed) {
  TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "flash_attn_forward_ex: CUDA tensors required");
  TORCH_CHECK(Q.dtype() == torch::kFloat32, "float32 only for this kernel");
  int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);

  float scale = 1.0f / std::sqrt((float)D);
  auto Qc = Q.contiguous().view({B*H, N, D});
  auto Kc = K.contiguous().view({B*H, N, D});
  auto Vc = V.contiguous().view({B*H, N, D});
  auto O = torch::empty_like(Qc);

  dim3 grid(N, B*H, 1);
  dim3 block(128, 1, 1);
  size_t smem = (64 * D + 64 * D) * sizeof(float); // BN=64
  flash_attn_fwd_ex_kernel<64><<<grid, block, smem>>>(
      Qc.data_ptr<float>(), Kc.data_ptr<float>(), Vc.data_ptr<float>(),
      O.data_ptr<float>(), B*H, N, D, scale, (float)dropout_p, causal, seed);

  return O.view({B, H, N, D});
}

// Bindings: add WMMA and new flash_ex and qkv_pack
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tile_linear", &tile_linear, "Tessera TileLinear (CUDA)");
  m.def("tile_linear_wmma", &tile_linear_wmma, "Tessera TileLinear WMMA (CUDA)");
  m.def("linear_bw_input", &linear_bw_input, "TileLinear dX (CUDA)");
  m.def("linear_bw_weight", &linear_bw_weight, "TileLinear dW (CUDA)");
  m.def("layer_norm", &layer_norm, "Tessera LayerNorm (CUDA)");
  m.def("layer_norm_bw", &layer_norm_bw, "Tessera LayerNorm backward (CUDA)");
  m.def("rowwise_softmax", &rowwise_softmax, "Rowwise Softmax (CUDA)");
  m.def("batched_gemm", &batched_gemm, "Batched GEMM (CUDA)");
  m.def("flash_attn_forward", &flash_attn_forward, "Fused FlashAttention forward (CUDA)");
  m.def("flash_attn_forward_ex", &flash_attn_forward_ex, "Fused FlashAttention forward (dropout, causal) (CUDA)");
  m.def("flash_attn_backward", &flash_attn_backward, "FlashAttention backward (CUDA-assisted)");
  m.def("qkv_bias_gelu", &qkv_bias_gelu, "Fused QKV + Bias + GELU (forward micro-kernel)");
  m.def("qkv_pack_gemm", &qkv_pack_gemm, "Fused QKV single GEMM pack (forward)");
}

#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;

// ===================== WMMA BF16 (using __nv_bfloat16) =====================
// Accumulates in FP32. Requires M,N,K multiples of 16.
__global__ void wmma_gemm_kernel_bf16(const __nv_bfloat16* __restrict__ A,
                                      const __nv_bfloat16* __restrict__ Bt, // B^T
                                      float* __restrict__ C,
                                      int M, int N, int K) {
  int tile_m = blockIdx.y;
  int tile_n = blockIdx.x;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
  wmma::fill_fragment(c, 0.0f);

  for (int tile_k = 0; tile_k < K; tile_k += 16) {
    const __nv_bfloat16* Aptr = A + (tile_m * 16) * K + tile_k;
    const __nv_bfloat16* BptrT = Bt + (tile_n * 16) * K + tile_k;

    // Load A row-major and B^T row-major (equivalent to B col-major)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;

    wmma::load_matrix_sync(a, Aptr, K);
    wmma::load_matrix_sync(b, BptrT, K);
    wmma::mma_sync(c, a, b, c);
  }

  float* Cptr = C + (tile_m * 16) * N + tile_n * 16;
  wmma::store_matrix_sync(Cptr, c, N, wmma::mem_row_major);
}

torch::Tensor tile_linear_wmma_bf16(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> bias_opt, c10::optional<std::string> act_opt) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "tile_linear_wmma_bf16: CUDA tensors required");
  TORCH_CHECK(x.dtype() == torch::kBFloat16 && w.dtype() == torch::kBFloat16, "BF16 WMMA requires bfloat16");
  TORCH_CHECK(x.dim() == 3 && w.dim() == 2, "x: [B,N,K], w: [M,K]");

  int B = x.size(0), N = x.size(1), K = x.size(2), M = w.size(0);
  TORCH_CHECK(w.size(1) == K, "w.shape[1] must equal x.shape[2]");
  TORCH_CHECK((N % 16)==0 && (M % 16)==0 && (K % 16)==0, "WMMA requires dims multiple of 16");

  auto x2 = x.reshape({B*N, K}).contiguous();
  auto wT = w.t().contiguous(); // [K,M] bf16
  auto y2 = torch::empty({B*N, M}, x.options().dtype(torch::kFloat32)); // accumulate in fp32

  dim3 grid(M/16, (B*N)/16, 1);
  dim3 block(32, 1, 1);
  wmma_gemm_kernel_bf16<<<grid, block>>>(
    reinterpret_cast<const __nv_bfloat16*>(x2.data_ptr<at::BFloat16>()),
    reinterpret_cast<const __nv_bfloat16*>(wT.data_ptr<at::BFloat16>()),
    y2.data_ptr<float>(),
    B*N, M, K);

  if (bias_opt.has_value()) {
    auto b = bias_opt.value().to(y2.dtype());
    y2.add_(b);
  }
  if (act_opt.has_value()) {
    auto act = act_opt.value();
    if (act == "gelu") y2 = torch::gelu(y2);
    else if (act == "relu") y2 = torch::relu(y2);
  }
  return y2.to(x.dtype()).reshape({B, N, M});
}

// ===================== Batched GEMM WMMA (FP16/BF16) =====================
__global__ void wmma_batched_kernel_fp16(const half* __restrict__ A, const half* __restrict__ Bt,
                                         float* __restrict__ C, int BATCH, int M, int N, int K, int strideA, int strideBt, int strideC) {
  int b = blockIdx.z;
  if (b >= BATCH) return;
  int tile_m = blockIdx.y;
  int tile_n = blockIdx.x;

  const half* Ab = A + b * strideA;
  const half* Bt_b = Bt + b * strideBt;
  float* Cb = C + b * strideC;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
  wmma::fill_fragment(c, 0.0f);
  for (int tile_k = 0; tile_k < K; tile_k += 16) {
    const half* Aptr = Ab + (tile_m * 16) * K + tile_k;
    const half* BptrT = Bt_b + (tile_n * 16) * K + tile_k;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
    wmma::load_matrix_sync(a, Aptr, K);
    wmma::load_matrix_sync(b, BptrT, K);
    wmma::mma_sync(c, a, b, c);
  }
  float* Cptr = Cb + (tile_m * 16) * N + tile_n * 16;
  wmma::store_matrix_sync(Cptr, c, N, wmma::mem_row_major);
}

__global__ void wmma_batched_kernel_bf16(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ Bt,
                                         float* __restrict__ C, int BATCH, int M, int N, int K, int strideA, int strideBt, int strideC) {
  int b = blockIdx.z;
  if (b >= BATCH) return;
  int tile_m = blockIdx.y;
  int tile_n = blockIdx.x;

  const __nv_bfloat16* Ab = A + b * strideA;
  const __nv_bfloat16* Bt_b = Bt + b * strideBt;
  float* Cb = C + b * strideC;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
  wmma::fill_fragment(c, 0.0f);
  for (int tile_k = 0; tile_k < K; tile_k += 16) {
    const __nv_bfloat16* Aptr = Ab + (tile_m * 16) * K + tile_k;
    const __nv_bfloat16* BptrT = Bt_b + (tile_n * 16) * K + tile_k;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
    wmma::load_matrix_sync(a, Aptr, K);
    wmma::load_matrix_sync(b, BptrT, K);
    wmma::mma_sync(c, a, b, c);
  }
  float* Cptr = Cb + (tile_m * 16) * N + tile_n * 16;
  wmma::store_matrix_sync(Cptr, c, N, wmma::mem_row_major);
}

torch::Tensor batched_gemm_wmma(torch::Tensor A, torch::Tensor B_t) {
  // A: [B,M,K], B_t: [B,N,K] (== B transposed)
  TORCH_CHECK(A.is_cuda() && B_t.is_cuda(), "batched_gemm_wmma: CUDA tensors required");
  TORCH_CHECK(A.dim() == 3 && B_t.dim() == 3, "A [B,M,K], B_t [B,N,K]");
  int BATCH = A.size(0), M = A.size(1), K = A.size(2), N = B_t.size(1);
  TORCH_CHECK(B_t.size(2) == K, "B_t last dim must be K");
  TORCH_CHECK((M%16)==0 && (N%16)==0 && (K%16)==0, "WMMA requires multiples of 16");

  auto C = torch::empty({BATCH, M, N}, A.options().dtype(torch::kFloat32)); // accumulate fp32

  dim3 grid(N/16, M/16, BATCH);
  dim3 block(32, 1, 1);
  int strideA = M*K;
  int strideBt = N*K;
  int strideC = M*N;

  if (A.dtype() == torch::kFloat16 && B_t.dtype() == torch::kFloat16) {
    wmma_batched_kernel_fp16<<<grid, block>>>(
      reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(B_t.data_ptr<at::Half>()),
      C.data_ptr<float>(), BATCH, M, N, K, strideA, strideBt, strideC);
  } else if (A.dtype() == torch::kBFloat16 && B_t.dtype() == torch::kBFloat16) {
    wmma_batched_kernel_bf16<<<grid, block>>>(
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(B_t.data_ptr<at::BFloat16>()),
      C.data_ptr<float>(), BATCH, M, N, K, strideA, strideBt, strideC);
  } else {
    TORCH_CHECK(false, "batched_gemm_wmma: dtype must be FP16 or BF16");
  }
  return C.to(A.dtype());
}

// ===================== QKV pack fused backward =====================
__device__ inline float gelu_deriv(float x) {
  const float kAlpha = 0.7978845608f; // sqrt(2/pi)
  const float kC = 0.044715f;
  float x3 = x*x*x;
  float u = kAlpha * (x + kC * x3);
  float t = tanhf(u);
  float sech2 = 1.0f - t*t;
  return 0.5f * (1.0f + t) + 0.5f * x * sech2 * kAlpha * (1.0f + 3.0f * kC * x*x);
}

// In-place: apply GELU' to first D columns of dY using saved pre-activation Yq_pre
__global__ void apply_gelu_grad_firstD(float* __restrict__ dY, const float* __restrict__ Yq_pre,
                                       int M, int M3, int D) {
  int row = blockIdx.x;
  if (row >= M) return;
  for (int col = threadIdx.x; col < D; col += blockDim.x) {
    float g = gelu_deriv(Yq_pre[row * D + col]);
    dY[row * M3 + col] *= g; // modify Q slice
  }
}

std::vector<torch::Tensor> qkv_pack_gemm_fwd(torch::Tensor x, torch::Tensor Wcat, torch::Tensor bcat, bool gelu_q) {
  // returns [packed(3,B,N,D), yq_pre(B*N,D)]
  TORCH_CHECK(x.is_cuda() && Wcat.is_cuda() && bcat.is_cuda(), "qkv_pack_gemm_fwd: CUDA only");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "float32 only");
  int B = x.size(0), N = x.size(1), K = x.size(2), M3 = Wcat.size(0);
  int D = M3 / 3;
  auto x2 = x.reshape({B*N, K}).contiguous();
  auto y2 = torch::empty({B*N, M3}, x.options());
  // single GEMM
  dim3 block(16,16,1);
  dim3 grid((M3+63)/64, (B*N+63)/64, 1);
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "qkv_pack_gemm_fwd", ([&] {
    tile_gemm2d_kernel<scalar_t,64,64,32><<<grid, block>>>(
      x2.data_ptr<scalar_t>(),
      Wcat.t().contiguous().data_ptr<scalar_t>(),
      y2.data_ptr<scalar_t>(), B*N, M3, K);
  }));
  y2.add_(bcat);
  auto yq_pre = y2.narrow(1, 0, D).contiguous(); // save pre-activation
  // GELU on Q slice
  if (gelu_q) {
    auto yq = torch::gelu(yq_pre);
    y2.narrow(1, 0, D).copy_(yq);
  }
  auto yk = y2.narrow(1, D, D);
  auto yv = y2.narrow(1, 2*D, D);
  auto packed = torch::stack({y2.narrow(1,0,D), yk, yv}, 0).reshape({3, B, N, D});
  return {packed, yq_pre.reshape({B,N,D})};
}

std::vector<torch::Tensor> qkv_pack_gemm_bw(torch::Tensor x, torch::Tensor Wcat, torch::Tensor yq_pre,
                                            torch::Tensor dPacked, bool gelu_q) {
  // returns {dX, dWcat, dbcat}
  TORCH_CHECK(x.is_cuda() && Wcat.is_cuda() && dPacked.is_cuda(), "qkv_pack_gemm_bw: CUDA only");
  TORCH_CHECK(x.dtype() == torch::kFloat32 && Wcat.dtype() == torch::kFloat32 && dPacked.dtype() == torch::kFloat32, "float32 only");
  int B = x.size(0), N = x.size(1), K = x.size(2), M3 = Wcat.size(0);
  int D = M3 / 3;
  int M = B * N;

  auto x2 = x.reshape({M, K}).contiguous();
  auto dYcat = dPacked.reshape({3, B, N, D}).permute({1,2,0,3}).reshape({M, M3}).contiguous(); // [M,3D]
  // apply GELU' to Q slice in-place
  if (gelu_q) {
    auto yq_pre2 = yq_pre.reshape({M, D}).contiguous();
    dim3 block(256,1,1);
    dim3 grid(M,1,1);
    apply_gelu_grad_firstD<<<grid, block>>>(dYcat.data_ptr<float>(), yq_pre2.data_ptr<float>(), M, M3, D);
  }
  // dWcat = dYcat^T @ X
  auto dW = torch::empty({M3, K}, x.options());
  dim3 block1(16,16,1);
  dim3 grid1((K+63)/64, (M3+63)/64, 1);
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "qkv_pack_gemm_bw_dW", ([&] {
    tile_gemm_AT_B_kernel<scalar_t,64,64,64><<<grid1, block1>>>(
      dYcat.data_ptr<scalar_t>(), x2.data_ptr<scalar_t>(), dW.data_ptr<scalar_t>(),
      M, M3, K);
  }));
  // dX = dYcat @ Wcat
  auto dX2 = torch::empty({M, K}, x.options());
  dim3 block2(16,16,1);
  dim3 grid2((K+63)/64, (M+63)/64, 1);
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "qkv_pack_gemm_bw_dX", ([&] {
    tile_gemm2d_kernel<scalar_t,64,64,32><<<grid2, block2>>>(
      dYcat.data_ptr<scalar_t>(), Wcat.data_ptr<scalar_t>(), dX2.data_ptr<scalar_t>(), M, K, M3);
  }));
  // dbcat = rowwise sum of dYcat
  auto db = dYcat.sum(0);
  return {dX2.reshape({B,N,K}), dW, db};
}

// ===================== FlashAttention backward with mask+dropout =====================
std::vector<torch::Tensor> flash_attn_backward_ex(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                                  torch::Tensor dO, bool causal, double dropout_p, uint64_t seed) {
  TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda() && dO.is_cuda(), "flash_attn_backward_ex: CUDA only");
  TORCH_CHECK(Q.dtype() == torch::kFloat32, "float32 only");

  int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
  int G = B*H;
  float scale = 1.0f / std::sqrt((float)D);

  auto Qg = Q.view({G,N,D}).contiguous();
  auto Kg = K.view({G,N,D}).contiguous();
  auto Vg = V.view({G,N,D}).contiguous();
  auto dOg = dO.view({G,N,D}).contiguous();

  // scores and softmax with causal+dropout (re-create mask via seed like fwd_ex)
  auto scores = batched_gemm(Qg, Kg, true); // [G,N,N]
  if (causal) {
    auto iu = torch::ones({N, N}, scores.options()).triu(1) * 1e9;
    scores = scores - iu;
  }
  scores = scores * scale;

  auto probs2d = scores.reshape({G*N, N});
  auto probs = rowwise_softmax(probs2d).reshape({G,N,N}); // softmax P

  if (dropout_p > 0.0) {
    // regenerate deterministic mask
    auto mask = torch::empty_like(probs);
    // simple host-side RNG: not perfect, but deterministic via seed
    auto gen = at::detail::createCPUGenerator();
    gen->set_current_seed(seed);
    mask.uniform_(0.0, 1.0, gen);
    mask = (mask >= dropout_p).to(probs.dtype()) / (1.0 - dropout_p);
    probs = probs * mask.to(probs.device());
  }

  // dV = P^T @ dO
  auto dV = batched_gemm(probs.transpose(1,2).contiguous(), dOg, false);

  // dP = dO @ V^T
  auto dP = batched_gemm(dOg, Vg, true);

  // z = P ⊙ (dP - sum(dP ⊙ P))
  auto tmp = (dP * probs).sum(dim=2, keepdim=True);
  auto z = probs * (dP - tmp);

  auto dQ = batched_gemm(z, Kg, false) ;
  auto dK = batched_gemm(z.transpose(1,2).contiguous(), Qg, false) ;

  dQ = dQ * scale;
  dK = dK * scale;

  return {dQ.view({B,H,N,D}), dK.view({B,H,N,D}), dV.view({B,H,N,D})};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // --- existing bindings are already defined earlier in the file; append new ones
  m.def("tile_linear_wmma_bf16", &tile_linear_wmma_bf16, "Tessera TileLinear WMMA BF16 (CUDA)");
  m.def("batched_gemm_wmma", &batched_gemm_wmma, "Batched GEMM WMMA (CUDA)");
  m.def("qkv_pack_gemm_fwd", &qkv_pack_gemm_fwd, "Fused QKV pack forward (returns packed + yq_pre)");
  m.def("qkv_pack_gemm_bw", &qkv_pack_gemm_bw, "Fused QKV pack backward (returns dX, dWcat, dbcat)");
  m.def("flash_attn_backward_ex", &flash_attn_backward_ex, "FlashAttention backward (mask+dropout)");
}

// ---- v8 BF16 WMMA backward additions ----

// ======== WMMA BF16 backward for Linear ========
// dX = dY [M,N] @ W [N,K]  (A=dY, Bt=W^T[K,N])
torch::Tensor linear_bw_input_wmma_bf16(torch::Tensor dY_bf16, torch::Tensor W_bf16) {
  TORCH_CHECK(dY_bf16.is_cuda() && W_bf16.is_cuda(), "linear_bw_input_wmma_bf16: CUDA only");
  TORCH_CHECK(dY_bf16.dtype() == torch::kBFloat16 && W_bf16.dtype() == torch::kBFloat16, "BF16 required");
  TORCH_CHECK(dY_bf16.dim()==2 && W_bf16.dim()==2, "dY [M,N], W [N,K]");
  int M = dY_bf16.size(0), N = dY_bf16.size(1), K = W_bf16.size(1);
  TORCH_CHECK(W_bf16.size(0) == N, "shape mismatch");
  TORCH_CHECK((M%16)==0 && (N%16)==0 && (K%16)==0, "WMMA requires multiples of 16");

  auto dX32 = torch::empty({M, K}, dY_bf16.options().dtype(torch::kFloat32));
  // Use the same WMMA kernel as forward: A[M,K] x B^T[N,K] => C[M,N]
  // Here: A=dY (M,N) so K=N; Bt=W^T (K,N) so Bt dims [K,N]
  dim3 grid(K/16, M/16, 1);
  dim3 block(32, 1, 1);
  wmma_batched_kernel_bf16<<<grid, block>>>(
    reinterpret_cast<const __nv_bfloat16*>(dY_bf16.contiguous().data_ptr<at::BFloat16>()),
    reinterpret_cast<const __nv_bfloat16*>(W_bf16.t().contiguous().data_ptr<at::BFloat16>()),
    dX32.data_ptr<float>(),
    1, M, K, N, M*N, K*N, M*K);
  return dX32.to(dY_bf16.dtype());
}

// dW = dY^T [N,M] @ X [M,K]  (A=dY^T, Bt=X^T[K,M])
torch::Tensor linear_bw_weight_wmma_bf16(torch::Tensor dY_bf16, torch::Tensor X_bf16) {
  TORCH_CHECK(dY_bf16.is_cuda() && X_bf16.is_cuda(), "linear_bw_weight_wmma_bf16: CUDA only");
  TORCH_CHECK(dY_bf16.dtype() == torch::kBFloat16 && X_bf16.dtype() == torch::kBFloat16, "BF16 required");
  TORCH_CHECK(dY_bf16.dim()==2 && X_bf16.dim()==2, "dY [M,N], X [M,K]");
  int M = dY_bf16.size(0), N = dY_bf16.size(1), K = X_bf16.size(1);
  TORCH_CHECK((M%16)==0 && (N%16)==0 && (K%16)==0, "WMMA requires multiples of 16");

  auto dW32 = torch::empty({N, K}, dY_bf16.options().dtype(torch::kFloat32));
  dim3 grid(K/16, N/16, 1);
  dim3 block(32, 1, 1);
  // A=dY^T [N,M], Bt=X^T [K,M] -> C [N,K]
  wmma_batched_kernel_bf16<<<grid, block>>>(
    reinterpret_cast<const __nv_bfloat16*>(dY_bf16.t().contiguous().data_ptr<at::BFloat16>()),
    reinterpret_cast<const __nv_bfloat16*>(X_bf16.t().contiguous().data_ptr<at::BFloat16>()),
    dW32.data_ptr<float>(),
    1, N, K, M, N*M, K*M, N*K);
  return dW32.to(dY_bf16.dtype());

  m.def("linear_bw_input_wmma_bf16", &linear_bw_input_wmma_bf16, "Linear backward input WMMA BF16");
  m.def("linear_bw_weight_wmma_bf16", &linear_bw_weight_wmma_bf16, "Linear backward weight WMMA BF16");
}

