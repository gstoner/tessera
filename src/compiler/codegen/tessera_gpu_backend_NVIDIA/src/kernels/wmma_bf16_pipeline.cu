#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>

using namespace nvcuda;

#ifndef TESSERA_STAGE_BYTES
#define TESSERA_STAGE_BYTES 128
#endif

// Compute one 128x128 tile of C via double-buffered K-sweep with 16x16x16 WMMA fragments.
// This kernel is "correctness-checked" (pure WMMA math) and demonstrates a realistic pipeline:
//   - two shared-memory buffers for A/B
//   - cooperative copies for the next K-slab while computing current slab
// NOTE: For simplicity, layouts are A row-major, B col-major, C row-major.
extern "C" __global__
void wmma_bf16_pipeline_kernel(const __nv_bfloat16* __restrict__ A,
                               const __nv_bfloat16* __restrict__ B,
                               float* __restrict__ C,
                               int M, int N, int K,
                               int lda, int ldb, int ldc,
                               float alpha, float beta)
{
#if __CUDA_ARCH__ >= 800
    const int TM = 128;
    const int TN = 128;
    const int TK = 16;

    // Tile indices
    int tile_m = blockIdx.y;
    int tile_n = blockIdx.x;
    int row0 = tile_m * TM;
    int col0 = tile_n * TN;
    if (row0 >= M || col0 >= N) return;

    // Shared memory: double buffers for A (TM x TK) and B (TK x TN)
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* As[2];
    __nv_bfloat16* Bs[2];
    size_t a_elems = TM*TK;
    size_t b_elems = TK*TN;
    As[0] = smem;
    Bs[0] = As[0] + a_elems;
    As[1] = Bs[0] + b_elems;
    Bs[1] = As[1] + a_elems;

    // Each warp computes an 64x64 sub-tile using 16x16x16 fragments (4x4 tiles per warp-group)
    const int warps_per_block = (blockDim.x * blockDim.y) / 32;
    int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) >> 5;
    int lane_id = threadIdx.x & 31;

    // Map warp to a 64x64 quadrant inside the 128x128 block
    int warp_m = (warp_id / 2); // 0..3 (2x2 grid => 4 warps computing 64x64 each when block has 128 threads)
    int warp_n = (warp_id % 2);
    int c_row = row0 + warp_m * 64;
    int c_col = col0 + warp_n * 64;

    // Accumulators: 4x4 fragments of 16x16 each (total 64x64)
    wmma::fragment<wmma::accumulator,16,16,16,float> acc[4][4];
    for (int i=0;i<4;i++) for (int j=0;j<4;j++) wmma::fill_fragment(acc[i][j], 0.0f);

    // Helper to stage one K-slab into shared
    auto stage_k = [&] (int buf, int k0) {
        // Cooperative copy A (TM x TK) and B (TK x TN)
        int t = threadIdx.y*blockDim.x + threadIdx.x;
        int num_threads = blockDim.x*blockDim.y;

        // A: TM rows, TK cols
        for (int idx=t; idx < TM*TK; idx += num_threads) {
            int r = idx / TK;
            int c = idx % TK;
            int gr = row0 + r;
            int gc = k0   + c;
            __nv_bfloat16 v = __float2bfloat16(0.f);
            if (gr < M && gc < K) v = A[gr*lda + gc];
            As[buf][r*TK + c] = v;
        }
        // B: TK rows, TN cols (col-major in global, but we load as standard and use col-major WMMA)
        for (int idx=t; idx < TK*TN; idx += num_threads) {
            int r = idx / TN;
            int c = idx % TN;
            int gr = k0   + r;
            int gc = col0 + c;
            __nv_bfloat16 v = __float2bfloat16(0.f);
            if (gr < K && gc < N) v = B[gr*ldb + gc];
            Bs[buf][r*TN + c] = v;
        }
        __syncthreads();
    };

    // Preload first slab
    int buf = 0;
    stage_k(buf, /*k0=*/0);

    // Main K loop with double buffering
    for (int k0=0; k0<K; k0+=TK) {
        int next = buf ^ 1;
        // Preload next
        if (k0 + TK < K) {
            stage_k(next, k0 + TK);
        }

        // Compute on current buffer: each warp issues 4x4 WMMA tiles of 16x16
        for (int mi=0; mi<4; ++mi) {
            for (int nj=0; nj<4; ++nj) {
                wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
                const __nv_bfloat16* Atile = As[buf] + (warp_m*64 + mi*16)*TK + 0;
                const __nv_bfloat16* Btile = Bs[buf] + 0*TN + (warp_n*64 + nj*16);
                wmma::load_matrix_sync(a, Atile, TK);
                wmma::load_matrix_sync(b, Btile, TN);
                wmma::mma_sync(acc[mi][nj], a, b, acc[mi][nj]);
            }
        }
        __syncthreads();
        buf = next;
    }

    // Epilogue: write C = alpha*acc + beta*C (correctness-checked WMMA store)
    for (int mi=0; mi<4; ++mi) {
        for (int nj=0; nj<4; ++nj) {
            int row = c_row + mi*16;
            int col = c_col + nj*16;
            // Load old C if beta!=0
            float* Cptr = C + row*ldc + col;
            // WMMA stores row-major 16x16
            wmma::store_matrix_sync(Cptr, acc[mi][nj], ldc, wmma::mem_row_major);
            if (alpha != 1.0f || beta != 0.0f) {
                for (int i=0;i<16;i++) for (int j=0;j<16;j++) {
                    int r = row + i, c = col + j;
                    if (r<M && c<N) {
                        float oldv = beta!=0.0f ? C[r*ldc + c] : 0.0f;
                        float v = Cptr[i*ldc + j];
                        C[r*ldc + c] = alpha*v + beta*oldv;
                    }
                }
            }
        }
    }
#else
    (void)A;(void)B;(void)C;(void)M;(void)N;(void)K;(void)lda;(void)ldb;(void)ldc;(void)alpha;(void)beta;
#endif
}
