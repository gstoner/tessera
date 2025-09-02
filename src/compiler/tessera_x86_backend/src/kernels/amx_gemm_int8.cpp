#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <algorithm>
#include "tessera/x86/amx_runtime.h"

// Helper: compute a single 16x16 block C += A(16xK) * B(Kx16) using AMX dpbssd
static void amx_block_s8s8_s32(const int8_t* A, const int8_t* B, int32_t* C,
                               int lda, int ldb, int ldc, int K)
{
    using namespace tessera::x86;
    // Configure tiles: T0 accumulator 16x64 (s32 stride 64*4), T1 A 16x64 (s8 stride 64*1), T2 B 64x16 (s8 stride 16*1)
    AMXTileConfig cfg;
    amx_build_default_bf16_config(cfg);
    // Override types: layout is same bytes/row; integer ops use same tile dims
    cfg.rows[0] = 16; cfg.colsb[0] = 64*4; // s32
    cfg.rows[1] = 16; cfg.colsb[1] = 64*1; // s8
    cfg.rows[2] = 64; cfg.colsb[2] = 16*1; // s8
    amx_load_config(cfg);

    _tile_zero(0);
    const int TK = 64;
    for (int k=0; k<K; k+=TK) {
        _tile_loadd(1, A + k, lda); // lda in bytes for s8
        _tile_loadd(2, B + k*ldb, ldb);
        _tile_dpbssd(0, 1, 2); // s8*s8 -> s32
    }
    _tile_stored(0, C, ldc); // store s32
    _tile_release();
}

// Public API: handles edge tiles by zero-padding into stack buffers.
extern "C" void tessera_x86_amx_gemm_s8s8_s32(const int8_t* A, const int8_t* B, int32_t* C,
                                              int M, int N, int K, int beta)
{
    using namespace tessera::x86;
    if (!tessera_x86_amx_int8_supported() || !tessera_x86_amx_enable_linux()) {
        return;
    }

    // Scale C by beta (integer)
    if (beta == 0) {
        std::memset(C, 0, sizeof(int32_t)*(size_t)M*(size_t)N);
    } else if (beta != 1) {
        for (int i=0;i<M*N;i++) C[i] *= beta;
    }

    const int BM = 16, BN = 16;
    const int TK = 64; // K step in AMX tile
    for (int m=0; m<M; m+=BM) {
        int mb = std::min(BM, M - m);
        for (int n=0; n<N; n+=BN) {
            int nb = std::min(BN, N - n);

            if (mb==BM && nb==BN && (K % TK)==0) {
                // Fast path: direct tiles from original matrices
                // Note: lda/ldb/ldc for _tile_loadd/_tile_stored are bytes-per-row
                amx_block_s8s8_s32(A + m*K, B + n, C + m*N + n, K, N, N*4, K);
            } else {
                // Edge: pack into padded buffers
                alignas(64) int8_t  Ablk[BM*TK];
                alignas(64) int8_t  Bblk[TK*BN];
                alignas(64) int32_t Cblk[BM*BN];

                // Zero buffers
                std::memset(Ablk, 0, sizeof(Ablk));
                std::memset(Bblk, 0, sizeof(Bblk));
                std::memset(Cblk, 0, sizeof(Cblk));

                // Pack A (mb x K) in 16x64 chunks
                for (int i=0; i<mb; ++i) {
                    const int8_t* src = A + (m+i)*K;
                    int8_t* dst = Ablk + i*TK;
                    int kk = std::min(K, TK);
                    std::memcpy(dst, src, kk);
                }
                // Pack B (K x nb)
                for (int k=0; k<std::min(K,TK); ++k) {
                    const int8_t* src = B + k*N + n;
                    int8_t* dst = Bblk + k*BN;
                    std::memcpy(dst, src, nb);
                }

                amx_block_s8s8_s32(Ablk, Bblk, Cblk, TK, BN, BN*4, std::min(K,TK));

                // Accumulate into C
                for (int i=0;i<mb;i++) {
                    std::memcpy(C + (m+i)*N + n, Cblk + i*BN, nb*sizeof(int32_t));
                }
            }
        }
    }
}
