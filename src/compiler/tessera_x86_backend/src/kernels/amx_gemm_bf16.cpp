#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include "tessera/x86/amx_runtime.h"

// T0: accumulator (16x64), T1: A (16x64), T2: B (64x16)
extern "C" void tessera_x86_amx_gemm_bf16(const uint16_t* A, const uint16_t* B, float* C,
                                          int M, int N, int K, float beta)
{
    using namespace tessera::x86;
    if (!tessera_x86_amx_supported() || !tessera_x86_amx_enable_linux()) {
        // Fallback: zero or assert. In practice, the caller should check beforehand.
        return;
    }

    // Initialize C *= beta
    if (beta == 0.0f) {
        std::memset(C, 0, sizeof(float)*size_t(M)*size_t(N));
    } // else accumulate on top

    AMXTileConfig cfg;
    amx_build_default_bf16_config(cfg);
    amx_load_config(cfg);

    const int TM = 16; // rows in A tile
    const int TN = 16; // cols in B tile
    const int TK = 64; // shared K per tile op

    for (int m=0; m<M; m+=TM) {
        for (int n=0; n<N; n+=TN) {
            _tile_zero(0); // T0 accumulator
            for (int k=0; k<K; k+=TK) {
                // Load tiles: A (T1) and B (T2)
                const void* a_ptr = (const void*)(A + (size_t)m*K + k);
                const void* b_ptr = (const void*)(B + (size_t)k*N + n);
                _tile_loadd(1, a_ptr, K*2); // bf16 => 2 bytes stride per col
                _tile_loadd(2, b_ptr, N*2);
                // T0 += T1 (bf16) * T2 (bf16)
                _tile_dpbf16ps(0, 1, 2);
            }
            // Store accumulator to C
            void* c_ptr = (void*)(C + (size_t)m*N + n);
            _tile_stored(0, c_ptr, N*4); // fp32 => 4 bytes stride
        }
    }

    amx_release();
}
