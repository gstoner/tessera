// X86-GEMM-ALIGN-1 design-phase harness: candidate alignment fixes for
// tessera_x86_avx512_gemm_f32, each checked bitwise against v0 (the pre-fix
// kernel) on the same inputs.
//   v0 pre-fix kernel          v1 whole-B copy, padded ldb   v2 per-strip K x 16 panel
//   v3 NB-strip panels, one accumulator (NB = argv[7])     v4 v3 + thread_local buffer
//   v5 8 accumulators, direct loads (runtime strip count)   v6 v5 over a packed panel
//   v7 v6 with the strip count a template parameter (shipped for M > 1)
//   v8 v5 with the strip count a template parameter (shipped for M == 1)
// usage: design_variants <variant 0..8> M N K offB reps [NB]
// build: see run_design_sweep.sh
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <vector>

static double now_ns() { timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW, &ts); return ts.tv_sec*1e9+ts.tv_nsec; }

// V0: production kernel as of d8da67f7 (index helper inlined).
static void v0(const float* A, const float* B, int64_t M, int64_t N, int64_t K, float* C) {
  for (int64_t m = 0; m < M; ++m) {
    const float* a = A + m*K; float* c = C + m*N; int64_t n = 0;
    for (; n + 16 <= N; n += 16) {
      __m512 acc = _mm512_setzero_ps();
      for (int64_t k = 0; k < K; ++k)
        acc = _mm512_fmadd_ps(_mm512_set1_ps(a[k]), _mm512_loadu_ps(B + k*N + n), acc);
      _mm512_storeu_ps(c + n, acc);
    }
    if (n < N) {
      __mmask16 tail = (__mmask16)((1u << (unsigned)(N - n)) - 1u);
      __m512 acc = _mm512_setzero_ps();
      for (int64_t k = 0; k < K; ++k)
        acc = _mm512_fmadd_ps(_mm512_set1_ps(a[k]), _mm512_maskz_loadu_ps(tail, B + k*N + n), acc);
      _mm512_mask_storeu_ps(c + n, tail, acc);
    }
  }
}

// V1: copy B once into a 64-byte-aligned buffer with ldb = roundup(N,16), zero pad; same loop.
static void v1(const float* A, const float* B, int64_t M, int64_t N, int64_t K, float* C) {
  int64_t ldb = (N + 15) / 16 * 16;
  float* P = (float*)aligned_alloc(64, (size_t)(K*ldb*4 + 63) / 64 * 64);
  for (int64_t k = 0; k < K; ++k) {
    memcpy(P + k*ldb, B + k*N, N*4);
    for (int64_t j = N; j < ldb; ++j) P[k*ldb + j] = 0.f;
  }
  for (int64_t m = 0; m < M; ++m) {
    const float* a = A + m*K; float* c = C + m*N; int64_t n = 0;
    for (; n + 16 <= N; n += 16) {
      __m512 acc = _mm512_setzero_ps();
      for (int64_t k = 0; k < K; ++k)
        acc = _mm512_fmadd_ps(_mm512_set1_ps(a[k]), _mm512_load_ps(P + k*ldb + n), acc);
      _mm512_storeu_ps(c + n, acc);
    }
    if (n < N) {
      __mmask16 tail = (__mmask16)((1u << (unsigned)(N - n)) - 1u);
      __m512 acc = _mm512_setzero_ps();
      for (int64_t k = 0; k < K; ++k)
        acc = _mm512_fmadd_ps(_mm512_set1_ps(a[k]), _mm512_load_ps(P + k*ldb + n), acc);
      _mm512_mask_storeu_ps(c + n, tail, acc);
    }
  }
  free(P);
}

// V2: per 16-column strip, pack B[:, n:n+16] into an aligned K x 16 panel, then all rows.
static void v2(const float* A, const float* B, int64_t M, int64_t N, int64_t K, float* C) {
  float* P = (float*)aligned_alloc(64, (size_t)K*64);
  for (int64_t n = 0; n < N; n += 16) {
    unsigned w = (unsigned)std::min<int64_t>(16, N - n);
    __mmask16 mask = w == 16 ? (__mmask16)0xffff : (__mmask16)((1u << w) - 1u);
    for (int64_t k = 0; k < K; ++k)
      _mm512_store_ps(P + k*16, _mm512_maskz_loadu_ps(mask, B + k*N + n));
    for (int64_t m = 0; m < M; ++m) {
      const float* a = A + m*K;
      __m512 acc = _mm512_setzero_ps();
      for (int64_t k = 0; k < K; ++k)
        acc = _mm512_fmadd_ps(_mm512_set1_ps(a[k]), _mm512_load_ps(P + k*16), acc);
      _mm512_mask_storeu_ps(C + m*N + n, mask, acc);
    }
  }
  free(P);
}


// V3: pack NB 16-column strips per pass (contiguous row segments), aligned panels, then all rows per strip.
static int g_nb = 8;
static void v3(const float* A, const float* B, int64_t M, int64_t N, int64_t K, float* C) {
  const int64_t NB = g_nb;
  float* P = (float*)aligned_alloc(64, (size_t)K*64*NB);
  for (int64_t n0 = 0; n0 < N; n0 += 16*NB) {
    int64_t strips = std::min<int64_t>(NB, (N - n0 + 15) / 16);
    for (int64_t k = 0; k < K; ++k) {
      const float* row = B + k*N + n0;
      for (int64_t j = 0; j < strips; ++j) {
        int64_t w = std::min<int64_t>(16, N - n0 - 16*j);
        __mmask16 mask = w == 16 ? (__mmask16)0xffff : (__mmask16)((1u << (unsigned)w) - 1u);
        _mm512_store_ps(P + (j*K + k)*16, _mm512_maskz_loadu_ps(mask, row + 16*j));
      }
    }
    for (int64_t j = 0; j < strips; ++j) {
      int64_t n = n0 + 16*j; int64_t w = std::min<int64_t>(16, N - n);
      __mmask16 mask = w == 16 ? (__mmask16)0xffff : (__mmask16)((1u << (unsigned)w) - 1u);
      const float* panel = P + j*K*16;
      for (int64_t m = 0; m < M; ++m) {
        const float* a = A + m*K;
        __m512 acc = _mm512_setzero_ps();
        for (int64_t k = 0; k < K; ++k)
          acc = _mm512_fmadd_ps(_mm512_set1_ps(a[k]), _mm512_load_ps(panel + k*16), acc);
        _mm512_mask_storeu_ps(C + m*N + n, mask, acc);
      }
    }
  }
  free(P);
}

// V4 (thread_local reused panel buffer): pack NB 16-column strips per pass (contiguous row segments), aligned panels, then all rows per strip.
static void v4(const float* A, const float* B, int64_t M, int64_t N, int64_t K, float* C) {
  const int64_t NB = g_nb;
  static thread_local float* buf = nullptr; static thread_local size_t cap = 0;
  size_t need = (size_t)K*64*NB;
  if (need > cap) { free(buf); buf = (float*)aligned_alloc(64, need); cap = need; }
  float* P = buf;
  for (int64_t n0 = 0; n0 < N; n0 += 16*NB) {
    int64_t strips = std::min<int64_t>(NB, (N - n0 + 15) / 16);
    for (int64_t k = 0; k < K; ++k) {
      const float* row = B + k*N + n0;
      for (int64_t j = 0; j < strips; ++j) {
        int64_t w = std::min<int64_t>(16, N - n0 - 16*j);
        __mmask16 mask = w == 16 ? (__mmask16)0xffff : (__mmask16)((1u << (unsigned)w) - 1u);
        _mm512_store_ps(P + (j*K + k)*16, _mm512_maskz_loadu_ps(mask, row + 16*j));
      }
    }
    for (int64_t j = 0; j < strips; ++j) {
      int64_t n = n0 + 16*j; int64_t w = std::min<int64_t>(16, N - n);
      __mmask16 mask = w == 16 ? (__mmask16)0xffff : (__mmask16)((1u << (unsigned)w) - 1u);
      const float* panel = P + j*K*16;
      for (int64_t m = 0; m < M; ++m) {
        const float* a = A + m*K;
        __m512 acc = _mm512_setzero_ps();
        for (int64_t k = 0; k < K; ++k)
          acc = _mm512_fmadd_ps(_mm512_set1_ps(a[k]), _mm512_load_ps(panel + k*16), acc);
        _mm512_mask_storeu_ps(C + m*N + n, mask, acc);
      }
    }
  }
}

static inline __mmask16 strip_mask(int64_t w) { return w >= 16 ? (__mmask16)0xffff : (__mmask16)((1u << (unsigned)w) - 1u); }
// V5: no pack; 8 strips register-blocked, direct loadu from B.
static void v5(const float* A, const float* B, int64_t M, int64_t N, int64_t K, float* C) {
  for (int64_t n0 = 0; n0 < N; n0 += 128) {
    int64_t strips = std::min<int64_t>(8, (N - n0 + 15) / 16);
    __mmask16 mk[8]; for (int j = 0; j < 8; ++j) mk[j] = j < strips ? strip_mask(N - n0 - 16*j) : 0;
    for (int64_t m = 0; m < M; ++m) {
      const float* a = A + m*K;
      __m512 acc[8]; for (int j = 0; j < 8; ++j) acc[j] = _mm512_setzero_ps();
      for (int64_t k = 0; k < K; ++k) {
        __m512 av = _mm512_set1_ps(a[k]); const float* row = B + k*N + n0;
        for (int j = 0; j < 8; ++j) if (j < strips) acc[j] = _mm512_fmadd_ps(av, _mm512_maskz_loadu_ps(mk[j], row + 16*j), acc[j]);
      }
      for (int j = 0; j < strips; ++j) _mm512_mask_storeu_ps(C + m*N + n0 + 16*j, mk[j], acc[j]);
    }
  }
}
// V6: pack K x (8*16) block row-major into aligned buffer, 8 register-blocked chains per row of A.
static void v6(const float* A, const float* B, int64_t M, int64_t N, int64_t K, float* C) {
  float* P = (float*)aligned_alloc(64, (size_t)K*512);
  for (int64_t n0 = 0; n0 < N; n0 += 128) {
    int64_t strips = std::min<int64_t>(8, (N - n0 + 15) / 16);
    __mmask16 mk[8]; for (int j = 0; j < 8; ++j) mk[j] = j < strips ? strip_mask(N - n0 - 16*j) : 0;
    for (int64_t k = 0; k < K; ++k) {
      const float* row = B + k*N + n0;
      for (int64_t j = 0; j < strips; ++j) _mm512_store_ps(P + k*128 + 16*j, _mm512_maskz_loadu_ps(mk[j], row + 16*j));
    }
    for (int64_t m = 0; m < M; ++m) {
      const float* a = A + m*K;
      __m512 acc[8]; for (int j = 0; j < 8; ++j) acc[j] = _mm512_setzero_ps();
      if (strips == 8) {
        for (int64_t k = 0; k < K; ++k) {
          __m512 av = _mm512_set1_ps(a[k]); const float* pr = P + k*128;
          for (int j = 0; j < 8; ++j) acc[j] = _mm512_fmadd_ps(av, _mm512_load_ps(pr + 16*j), acc[j]);
        }
      } else {
        for (int64_t k = 0; k < K; ++k) {
          __m512 av = _mm512_set1_ps(a[k]); const float* pr = P + k*128;
          for (int j = 0; j < 8; ++j) if (j < strips) acc[j] = _mm512_fmadd_ps(av, _mm512_load_ps(pr + 16*j), acc[j]);
        }
      }
      for (int j = 0; j < strips; ++j) _mm512_mask_storeu_ps(C + m*N + n0 + 16*j, mk[j], acc[j]);
    }
  }
  free(P);
}

// V7/V8: strip count as a template parameter so acc[] stays in registers.
template <int S, bool Packed>
static inline void block_rows(const float* A, const float* src, int64_t ld, int64_t M, int64_t N, int64_t K,
                              float* C, int64_t n0, const __mmask16* mk) {
  for (int64_t m = 0; m < M; ++m) {
    const float* a = A + m*K;
    __m512 acc[S];
    for (int j = 0; j < S; ++j) acc[j] = _mm512_setzero_ps();
    for (int64_t k = 0; k < K; ++k) {
      const __m512 av = _mm512_set1_ps(a[k]);
      const float* r = src + k*ld;
      for (int j = 0; j < S; ++j)
        acc[j] = _mm512_fmadd_ps(av, Packed ? _mm512_load_ps(r + 16*j) : _mm512_maskz_loadu_ps(mk[j], r + 16*j), acc[j]);
    }
    for (int j = 0; j < S; ++j) _mm512_mask_storeu_ps(C + m*N + n0 + 16*j, mk[j], acc[j]);
  }
}
template <bool Packed>
static void dispatch(int S, const float* A, const float* src, int64_t ld, int64_t M, int64_t N, int64_t K, float* C, int64_t n0, const __mmask16* mk) {
  switch (S) {
    case 1: block_rows<1, Packed>(A, src, ld, M, N, K, C, n0, mk); break;
    case 2: block_rows<2, Packed>(A, src, ld, M, N, K, C, n0, mk); break;
    case 3: block_rows<3, Packed>(A, src, ld, M, N, K, C, n0, mk); break;
    case 4: block_rows<4, Packed>(A, src, ld, M, N, K, C, n0, mk); break;
    case 5: block_rows<5, Packed>(A, src, ld, M, N, K, C, n0, mk); break;
    case 6: block_rows<6, Packed>(A, src, ld, M, N, K, C, n0, mk); break;
    case 7: block_rows<7, Packed>(A, src, ld, M, N, K, C, n0, mk); break;
    default: block_rows<8, Packed>(A, src, ld, M, N, K, C, n0, mk); break;
  }
}
static void v7(const float* A, const float* B, int64_t M, int64_t N, int64_t K, float* C) {
  float* P = (float*)aligned_alloc(64, (size_t)K*512);
  for (int64_t n0 = 0; n0 < N; n0 += 128) {
    int S = (int)std::min<int64_t>(8, (N - n0 + 15) / 16);
    __mmask16 mk[8]; for (int j = 0; j < 8; ++j) mk[j] = j < S ? strip_mask(N - n0 - 16*j) : 0;
    for (int64_t k = 0; k < K; ++k) {
      const float* row = B + k*N + n0;
      for (int j = 0; j < S; ++j) _mm512_store_ps(P + k*16*S + 16*j, _mm512_maskz_loadu_ps(mk[j], row + 16*j));
    }
    dispatch<true>(S, A, P, 16*S, M, N, K, C, n0, mk);
  }
  free(P);
}
static void v8(const float* A, const float* B, int64_t M, int64_t N, int64_t K, float* C) {
  for (int64_t n0 = 0; n0 < N; n0 += 128) {
    int S = (int)std::min<int64_t>(8, (N - n0 + 15) / 16);
    __mmask16 mk[8]; for (int j = 0; j < 8; ++j) mk[j] = j < S ? strip_mask(N - n0 - 16*j) : 0;
    dispatch<false>(S, A, B + n0, N, M, N, K, C, n0, mk);
  }
}

// V9: v7 with K blocked by KC (env KC, default 256): the panel stays KC x 128;
// accumulators continue through C between K blocks (exact: fp32 store/load).
template <int S>
static inline void block_rows_kc(const float* A, const float* P, int64_t M, int64_t N, int64_t K,
                                 float* C, int64_t n0, int64_t k0, int64_t kc, const __mmask16* mk) {
  for (int64_t m = 0; m < M; ++m) {
    const float* a = A + m*K + k0;
    float* c = C + m*N + n0;
    __m512 acc[S];
    for (int j = 0; j < S; ++j) acc[j] = k0 == 0 ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(mk[j], c + 16*j);
    for (int64_t k = 0; k < kc; ++k) {
      const __m512 av = _mm512_set1_ps(a[k]);
      const float* r = P + k*16*S;
      for (int j = 0; j < S; ++j) acc[j] = _mm512_fmadd_ps(av, _mm512_load_ps(r + 16*j), acc[j]);
    }
    for (int j = 0; j < S; ++j) _mm512_mask_storeu_ps(c + 16*j, mk[j], acc[j]);
  }
}
static void v9(const float* A, const float* B, int64_t M, int64_t N, int64_t K, float* C) {
  static int64_t KC = getenv("KC") ? atol(getenv("KC")) : 256;
  const int64_t kcap = std::min<int64_t>(KC, K);
  float* P = (float*)aligned_alloc(64, (size_t)kcap*512);
  for (int64_t n0 = 0; n0 < N; n0 += 128) {
    int S = (int)std::min<int64_t>(8, (N - n0 + 15) / 16);
    __mmask16 mk[8]; for (int j = 0; j < 8; ++j) mk[j] = j < S ? strip_mask(N - n0 - 16*j) : 0;
    for (int64_t k0 = 0; k0 < K; k0 += kcap) {
      const int64_t kc = std::min<int64_t>(kcap, K - k0);
      for (int64_t k = 0; k < kc; ++k) {
        const float* row = B + (k0 + k)*N + n0;
        for (int j = 0; j < S; ++j) _mm512_store_ps(P + k*16*S + 16*j, _mm512_maskz_loadu_ps(mk[j], row + 16*j));
      }
      switch (S) {
        case 1: block_rows_kc<1>(A, P, M, N, K, C, n0, k0, kc, mk); break;
        case 2: block_rows_kc<2>(A, P, M, N, K, C, n0, k0, kc, mk); break;
        case 3: block_rows_kc<3>(A, P, M, N, K, C, n0, k0, kc, mk); break;
        case 4: block_rows_kc<4>(A, P, M, N, K, C, n0, k0, kc, mk); break;
        case 5: block_rows_kc<5>(A, P, M, N, K, C, n0, k0, kc, mk); break;
        case 6: block_rows_kc<6>(A, P, M, N, K, C, n0, k0, kc, mk); break;
        case 7: block_rows_kc<7>(A, P, M, N, K, C, n0, k0, kc, mk); break;
        default: block_rows_kc<8>(A, P, M, N, K, C, n0, k0, kc, mk); break;
      }
    }
  }
  free(P);
}

int main(int argc, char** argv) {
  if (argc < 7) { fprintf(stderr, "usage\n"); return 2; }
  int var = atoi(argv[1]); int64_t M = atoll(argv[2]), N = atoll(argv[3]), K = atoll(argv[4]);
  long offB = atol(argv[5]); int reps = atoi(argv[6]);
  auto place = [](size_t bytes, long off) { char* r = (char*)aligned_alloc(4096, (bytes + off + 8191) / 4096 * 4096); return (float*)(r + off); };
  float* A = place(M*K*4, 0); float* B = place(K*N*4, offB); float* C = place(M*N*4, 0); float* R = place(M*N*4, 0);
  srand(20260927);
  for (int64_t i = 0; i < M*K; ++i) A[i] = (float)rand() / RAND_MAX - 0.5f;
  for (int64_t i = 0; i < K*N; ++i) B[i] = (float)rand() / RAND_MAX - 0.5f;
  v0(A, B, M, N, K, R);
  void (*f)(const float*, const float*, int64_t, int64_t, int64_t, float*) = var == 0 ? v0 : var == 1 ? v1 : var == 2 ? v2 : var == 3 ? v3 : var == 4 ? v4 : var == 5 ? v5 : var == 6 ? v6 : var == 7 ? v7 : var == 8 ? v8 : v9;
  if (argc > 7) g_nb = atoi(argv[7]);
  f(A, B, M, N, K, C);
  bool bitwise = memcmp(C, R, M*N*4) == 0;
  std::vector<double> s;
  for (int i = 0; i < 9; ++i) { double t0 = now_ns(); for (int r = 0; r < reps; ++r) f(A, B, M, N, K, C); s.push_back((now_ns() - t0) / reps / 1e3); }
  std::sort(s.begin(), s.end());
  printf("nb=%d v%d M=%ld N=%ld K=%ld offB=%ld med_us=%.2f min_us=%.2f bitwise=%d\n", g_nb, var, (long)M, (long)N, (long)K, offB, s[4], s[0], bitwise);
}
