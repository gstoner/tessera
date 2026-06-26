// AVX-512 elementwise bitwise kernels (i32) for the Tessera x86 backend.
//
// Applies a pointwise bitwise op over i32 buffers, producing an [n] i32 result —
// the optimized CPU lane for the S2 bitwise family, the AVX-512 analog of the
// ROCm `generate-rocm-bitwise-kernel` lane. A scalar reference is provided
// alongside for on-device validation.
//
//   kind 0 = and   a & b       kind 2 = xor   a ^ b   (binary)
//   kind 1 = or    a | b       kind 3 = not   ~a      (unary, B ignored)
//
// Operands act on the full integer bit pattern (no normalization). 16 i32 lanes
// per __m512i; the tail (n % 16) is scalar.

#include <immintrin.h>
#include <cstdint>

namespace {
constexpr int kAnd = 0;
constexpr int kOr = 1;
constexpr int kXor = 2;
constexpr int kNot = 3;

inline int32_t scalar_bitwise(int32_t a, int32_t b, int kind) {
    switch (kind) {
    case kAnd: return a & b;
    case kOr:  return a | b;
    case kXor: return a ^ b;
    case kNot: return ~a;
    default:   return a;
    }
}
}  // namespace

extern "C" void tessera_x86_reference_bitwise_i32(const int32_t* A,
                                                  const int32_t* B, int64_t n,
                                                  int32_t* out, int kind) {
    for (int64_t i = 0; i < n; ++i)
        out[i] = scalar_bitwise(A[i], kind == kNot ? 0 : B[i], kind);
}

extern "C" void tessera_x86_avx512_bitwise_i32(const int32_t* A,
                                               const int32_t* B, int64_t n,
                                               int32_t* out, int kind) {
    const int64_t vstep = 16;  // i32 lanes per __m512i
    const __m512i ones = _mm512_set1_epi32(-1);
    int64_t i = 0;
    for (; i + vstep <= n; i += vstep) {
        __m512i a = _mm512_loadu_si512(A + i);
        __m512i y;
        if (kind == kNot) {
            y = _mm512_xor_si512(a, ones);   // ~a
        } else {
            __m512i b = _mm512_loadu_si512(B + i);
            switch (kind) {
            case kAnd: y = _mm512_and_si512(a, b); break;
            case kOr:  y = _mm512_or_si512(a, b); break;
            case kXor: y = _mm512_xor_si512(a, b); break;
            default:   y = a; break;
            }
        }
        _mm512_storeu_si512(out + i, y);
    }
    for (; i < n; ++i)
        out[i] = scalar_bitwise(A[i], kind == kNot ? 0 : B[i], kind);
}
