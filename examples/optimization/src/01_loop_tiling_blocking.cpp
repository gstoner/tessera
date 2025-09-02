
// Baseline vs cache-blocked GEMM (row-major). Not production-grade; illustrative only.
#include <vector>
#include <random>
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <immintrin.h>

static inline double now_ms() {
  using C = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(C::now().time_since_epoch()).count();
}

float gemm_baseline(const float* A, const float* B, float* C, int M, int N, int K) {
  double t0 = now_ms();
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k) {
      float a = A[i*K + k];
      for (int j = 0; j < N; ++j)
        C[i*N + j] += a * B[k*N + j];
    }
  double t1 = now_ms();
  return float(t1 - t0);
}

float gemm_blocked(const float* A, const float* B, float* C, int M, int N, int K, int BM=128, int BN=128, int BK=64) {
  double t0 = now_ms();
  for (int i0 = 0; i0 < M; i0 += BM)
    for (int k0 = 0; k0 < K; k0 += BK)
      for (int j0 = 0; j0 < N; j0 += BN) {
        int iMax = std::min(i0+BM, M);
        int kMax = std::min(k0+BK, K);
        int jMax = std::min(j0+BN, N);
        for (int i = i0; i < iMax; ++i) {
          for (int k = k0; k < kMax; ++k) {
            // vectorized load of B row (when BN%8==0) with AVX2 as an example
            const float a = A[i*K + k];
            int j = j0;
#if defined(__AVX2__)
            for (; j+8 <= jMax; j+=8) {
              __m256 b = _mm256_loadu_ps(&B[k*N + j]);
              __m256 c = _mm256_loadu_ps(&C[i*N + j]);
              c = _mm256_add_ps(c, _mm256_mul_ps(_mm256_set1_ps(a), b));
              _mm256_storeu_ps(&C[i*N + j], c);
            }
#endif
            for (; j < jMax; ++j)
              C[i*N + j] += a * B[k*N + j];
          }
        }
      }
  double t1 = now_ms();
  return float(t1 - t0);
}

int main() {
  int M=1024, N=1024, K=1024;
  std::vector<float> A(M*K), B(K*N), C0(M*N), C1(M*N);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1,1);
  std::generate(A.begin(), A.end(), [&]{return dist(rng);});
  std::generate(B.begin(), B.end(), [&]{return dist(rng);});
  std::fill(C0.begin(), C0.end(), 0.f);
  std::fill(C1.begin(), C1.end(), 0.f);

  float t0 = gemm_baseline(A.data(), B.data(), C0.data(), M,N,K);
  float t1 = gemm_blocked(A.data(), B.data(), C1.data(), M,N,K);

  // simple checksum
  double s0=0, s1=0;
  for (auto v: C0) s0+=v;
  for (auto v: C1) s1+=v;

  printf("baseline: %.2f ms  blocked+AVX: %.2f ms\n", t0, t1);
  printf("checksum diff: %.6f\n", std::abs(s0-s1));
  return 0;
}
