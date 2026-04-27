#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "tessera/x86/target.h"

static uint16_t fp32_to_bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, sizeof(u));
  return static_cast<uint16_t>((u + 0x00008000u) >> 16);
}

static float bf16_to_fp32(uint16_t x) {
  uint32_t u = uint32_t(x) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

int main() {
  constexpr int M = 17, N = 19, K = 33;
  std::vector<uint16_t> a(M * K), b(K * N);
  std::vector<float> c(M * N, 1.25f), ref(M * N, 1.25f);
  for (int i = 0; i < M * K; ++i)
    a[i] = fp32_to_bf16(float((i % 13) - 6) / 7.0f);
  for (int i = 0; i < K * N; ++i)
    b[i] = fp32_to_bf16(float((i % 11) - 5) / 5.0f);

  tessera::x86::tessera_x86_avx512_gemm_bf16(a.data(), b.data(), c.data(), M, N, K, 0.5f);

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.5f * ref[m * N + n];
      for (int k = 0; k < K; ++k)
        acc += bf16_to_fp32(a[m * K + k]) * bf16_to_fp32(b[k * N + n]);
      ref[m * N + n] = acc;
    }
  }

  for (int i = 0; i < M * N; ++i)
    assert(std::fabs(c[i] - ref[i]) < 1.0e-4f);
  return 0;
}
