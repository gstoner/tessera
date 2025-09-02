#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

// Simple WMMA GEMM: C = alpha * A @ B + beta * C
// Tiles 16x16x16, row-major A, col-major B, row-major C
extern "C" __global__
void wmma_fp16_kernel(const __half* A, const __half* B, float* C,
                      int M, int N, int K, float alpha, float beta)
{
    // Each warp computes one 16x16 tile of C
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x);
    // Warp indices in tiles:
    int tileM = warpM;
    int tileN = warpN;

    if (tileM*16 >= M || tileN*16 >= N) return;

    wmma::fragment<wmma::matrix_a, 16,16,16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16,16,16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16,16,16, float> c;

    wmma::fill_fragment(c, 0.0f);

    for (int k=0; k<K; k+=16) {
        int aRow = tileM*16;
        int aCol = k;
        int bRow = k;
        int bCol = tileN*16;

        wmma::load_matrix_sync(a, A + aRow*K + aCol, K);
        wmma::load_matrix_sync(b, B + bRow*N + bCol, N);
        wmma::mma_sync(c, a, b, c);
    }

    // Scale and store
    // Load original C tile for beta
    float out[16*16];
    for (int i=0;i<16;i++) for (int j=0;j<16;j++) {
        int row = tileM*16 + i;
        int col = tileN*16 + j;
        float cv = 0.0f;
        if (row < M && col < N) cv = C[row*N + col];
        out[i*16+j] = alpha * c.x[i*16+j] + beta * cv;
    }
    for (int i=0;i<16;i++) for (int j=0;j<16;j++) {
        int row = tileM*16 + i;
        int col = tileN*16 + j;
        if (row < M && col < N) C[row*N + col] = out[i*16+j];
    }
}

void launch_wmma_fp16_gemm(const __half* A, const __half* B, float* C,
                           int M, int N, int K, float alpha, float beta, cudaStream_t s)
{
    dim3 block(4, 4); // 16 warps? For simplicity, launch 4x4 warps per block (assuming 32 threads/warp mapping)
    dim3 grid( (N + 16 - 1)/16, (M + 16 - 1)/16 );
    wmma_fp16_kernel<<<grid, dim3(32,32,1), 0, s>>>(A,B,C,M,N,K,alpha,beta);
}
