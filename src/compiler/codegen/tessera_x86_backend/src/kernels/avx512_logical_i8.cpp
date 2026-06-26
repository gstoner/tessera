// AVX-512 elementwise logical kernels (i8 bool) for the Tessera x86 backend.
//
// Applies a pointwise logical op over i8 boolean buffers, producing an [n] uint8
// mask (0/1) — the optimized CPU lane for the S2 logical family, the AVX-512
// analog of the ROCm `generate-rocm-logical-kernel` lane. A scalar reference is
// provided alongside for on-device validation.
//
//   kind 0 = and   a && b      kind 2 = xor   a ^ b   (binary)
//   kind 1 = or    a || b      kind 3 = not   !a      (unary, B ignored)
//
// Inputs are normalized to a boolean via `x != 0` (matching numpy, where any
// nonzero value is true), so the kernel is correct for arbitrary i8 inputs.
// 64 i8 lanes per __m512i (needs AVX512BW); the tail (n % 64) is scalar.

#include <immintrin.h>
#include <cstdint>

namespace {
constexpr int kAnd = 0;
constexpr int kOr = 1;
constexpr int kXor = 2;
constexpr int kNot = 3;

inline uint8_t scalar_logical(uint8_t a, uint8_t b, int kind) {
    bool ba = a != 0, bb = b != 0;
    switch (kind) {
    case kAnd: return (ba && bb) ? 1 : 0;
    case kOr:  return (ba || bb) ? 1 : 0;
    case kXor: return (ba != bb) ? 1 : 0;
    case kNot: return ba ? 0 : 1;
    default:   return 0;
    }
}
}  // namespace

extern "C" void tessera_x86_reference_logical_i8(const uint8_t* A,
                                                 const uint8_t* B, int64_t n,
                                                 uint8_t* out, int kind) {
    for (int64_t i = 0; i < n; ++i)
        out[i] = scalar_logical(A[i], kind == kNot ? 0 : B[i], kind);
}

extern "C" void tessera_x86_avx512_logical_i8(const uint8_t* A,
                                              const uint8_t* B, int64_t n,
                                              uint8_t* out, int kind) {
    const int64_t vstep = 64;  // i8 lanes per __m512i
    const __m512i zero = _mm512_setzero_si512();
    int64_t i = 0;
    for (; i + vstep <= n; i += vstep) {
        __m512i a = _mm512_loadu_si512(A + i);
        // na = (a != 0) ? 1 : 0
        __m512i na = _mm512_maskz_set1_epi8(
            _mm512_cmpneq_epi8_mask(a, zero), 1);
        __m512i y;
        if (kind == kNot) {
            // !a = (a == 0) ? 1 : 0
            y = _mm512_maskz_set1_epi8(_mm512_cmpeq_epi8_mask(a, zero), 1);
        } else {
            __m512i b = _mm512_loadu_si512(B + i);
            __m512i nb = _mm512_maskz_set1_epi8(
                _mm512_cmpneq_epi8_mask(b, zero), 1);
            switch (kind) {
            case kAnd: y = _mm512_and_si512(na, nb); break;
            case kOr:  y = _mm512_or_si512(na, nb); break;
            case kXor: y = _mm512_xor_si512(na, nb); break;
            default:   y = na; break;
            }
        }
        _mm512_storeu_si512(out + i, y);
    }
    for (; i < n; ++i)
        out[i] = scalar_logical(A[i], kind == kNot ? 0 : B[i], kind);
}
