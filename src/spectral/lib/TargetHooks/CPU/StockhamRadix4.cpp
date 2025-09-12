// Minimal illustrative CPU radix-4 Stockham step (scalar placeholder)
#include <complex>
#include <vector>
extern "C" void ts_stockham_radix4_scalar(const std::complex<float>* in,
                                         std::complex<float>* out,
                                         int N, int stride) {
  for (int i = 0; i + 3*stride < N; i += 4*stride) {
    auto a = in[i + 0*stride];
    auto b = in[i + 1*stride];
    auto c = in[i + 2*stride];
    auto d = in[i + 3*stride];
    auto t0 = a + c;
    auto t1 = a - c;
    auto t2 = b + d;
    auto t3 = std::complex<float>(b.imag() - d.imag(), d.real() - b.real());
    out[i + 0*stride] = t0 + t2;
    out[i + 1*stride] = t1 + t3;
    out[i + 2*stride] = t0 - t2;
    out[i + 3*stride] = t1 - t3;
  }
}
