#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;

// WMMA BF16 kernel (sm80+)
extern "C" __global__
void wmma_bf16_kernel(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
                      int M, int N, int K, float alpha, float beta)
{
#if __CUDA_ARCH__ >= 800
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x);
    int tileM = warpM;
    int tileN = warpN;
    if (tileM*16 >= M || tileN*16 >= N) return;

    wmma::fragment<wmma::matrix_a, 16,16,16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16,16,16, __nv_bfloat16, wmma::col_major> b;
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
#else
    (void)A;(void)B;(void)C;(void)M;(void)N;(void)K;(void)alpha;(void)beta;
#endif
}

void launch_wmma_bf16_gemm(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
                           int M, int N, int K, float alpha, float beta, cudaStream_t s)
{
    dim3 block(4,4);
    dim3 grid( (N + 16 - 1)/16, (M + 16 - 1)/16 );
    wmma_bf16_kernel<<<grid, dim3(32,32,1), 0, s>>>(A,B,C,M,N,K,alpha,beta);
}
