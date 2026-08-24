#include "tessera/x86/target.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

extern "C" void tessera_x86_avx512_gemm_f32(const float *, const float *,
                                             int64_t, int64_t, int64_t,
                                             float *);

int main() {
  constexpr int M = 5;
  constexpr int N = 19;
  constexpr int K = 7;

  std::vector<float> af(M * K), bf(K * N), cf(M * N), rf(M * N);
  std::vector<double> ad(M * K), bd(K * N), cd(M * N), rd(M * N);
  std::vector<uint8_t> au(M * K);
  std::vector<int8_t> bi(K * N);
  std::vector<int32_t> ci(M * N), ri(M * N);

  for (int i = 0; i < M * K; ++i) {
    af[i] = float((i % 11) - 5) / 7.0f;
    ad[i] = double((i % 13) - 6) / 9.0;
    au[i] = static_cast<uint8_t>((i * 3 + 1) % 17);
  }
  for (int i = 0; i < K * N; ++i) {
    bf[i] = float((i % 9) - 4) / 5.0f;
    bd[i] = double((i % 15) - 7) / 11.0;
    bi[i] = static_cast<int8_t>((i * 5 + 2) % 19 - 9);
  }

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float facc = 0.0f;
      double dacc = 0.0;
      int32_t iacc = 0;
      for (int k = 0; k < K; ++k) {
        facc += af[m * K + k] * bf[k * N + n];
        dacc += ad[m * K + k] * bd[k * N + n];
        iacc += int32_t(au[m * K + k]) * int32_t(bi[k * N + n]);
      }
      rf[m * N + n] = facc;
      rd[m * N + n] = dacc;
      ri[m * N + n] = iacc;
    }
  }

  tessera_x86_avx512_gemm_f32(af.data(), bf.data(), M, N, K, cf.data());
  tessera_x86_avx512_gemm_f64(ad.data(), bd.data(), M, N, K, cd.data());
  tessera_x86_avx512_vnni_gemm_u8s8_s32(au.data(), bi.data(), ci.data(), M, N,
                                        K, 0);

  for (int i = 0; i < M * N; ++i) {
    assert(std::fabs(cf[i] - rf[i]) < 1.0e-5f);
    assert(std::fabs(cd[i] - rd[i]) < 1.0e-12);
    assert(ci[i] == ri[i]);
  }
  return 0;
}
