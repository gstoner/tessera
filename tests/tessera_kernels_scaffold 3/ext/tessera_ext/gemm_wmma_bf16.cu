#include "common.cuh"
#include <mma.h>
#include <torch/extension.h>
using namespace nvcuda;

#if defined(TESSERA_ENABLE_BF16)
namespace {
constexpr int WM = 128;
constexpr int WN = 128;

__global__ void gemm_wmma_bf16_kernel(const __nv_bfloat16* __restrict__ A,
                                      const __nv_bfloat16* __restrict__ B,
                                      float* __restrict__ C,
                                      int M, int N, int K,
                                      float alpha, float beta)
{
  TESSERA_NVTX_RANGE("gemm_wmma_bf16_kernel");
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;
  int warpId = threadIdx.x / 32;
  int tilesPerBlockRow = WN / 16;
  int tileRow = warpId / tilesPerBlockRow;
  int tileCol = warpId % tilesPerBlockRow;
  int row = blockRow * WM + tileRow * 16;
  int col = blockCol * WN + tileCol * 16;

  wmma::fragment<wmma::accumulator, 16,16,16, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  for (int kb=0; kb<K; kb+=16){
    wmma::fragment<wmma::matrix_a, 16,16,16, __nv_bfloat16, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, 16,16,16, __nv_bfloat16, wmma::row_major> bFrag;
    const __nv_bfloat16* aTile = A + (row * K + kb);
    const __nv_bfloat16* bTile = B + (kb * N + col);
    wmma::load_matrix_sync(aFrag, aTile, K);
    wmma::load_matrix_sync(bFrag, bTile, N);
    wmma::mma_sync(acc, aFrag, bFrag, acc);
  }

  float cTile[16*16];
  wmma::store_matrix_sync(cTile, acc, 16, wmma::mem_row_major);
  for (int i=0;i<16;i++){
    int r = row + i;
    if (r < M){
      for (int j=0;j<16;j++){
        int c = col + j;
        if (c < N){
          float prev = beta != 0.0f ? C[r * N + c] : 0.0f;
          C[r * N + c] = alpha * cTile[i*16 + j] + beta * prev;
        }
      }
    }
  }
}
} // namespace

void gemm_wmma_bf16_launcher(at::Tensor A, at::Tensor B, at::Tensor C, float alpha, float beta){
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(A.scalar_type()==at::kBFloat16 && B.scalar_type()==at::kBFloat16, "A,B must be bfloat16");
  TORCH_CHECK(C.scalar_type()==at::kFloat, "C must be float32");
  int64_t M = A.size(0), K = A.size(1), N = B.size(1);
  TORCH_CHECK(B.size(0)==K, "K mismatch");

  dim3 grid(div_up((int)N, WN), div_up((int)M, WM));
  dim3 block(128);
  gemm_wmma_bf16_kernel<<<grid, block>>>(
    reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
    reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<at::BFloat16>()),
    C.data_ptr<float>(), (int)M,(int)N,(int)K, alpha, beta);
  CUDA_CHECK(cudaGetLastError());
}
#else
void gemm_wmma_bf16_launcher(at::Tensor, at::Tensor, at::Tensor, float, float){
  TORCH_CHECK(false, "Build without TESSERA_ENABLE_BF16");
}
#endif
