
// Vectorized transform: y[i] = a*x[i] + b with AVX2/AVX-512 fallback to scalar.
#include <immintrin.h>
#include <vector>
#include <random>
#include <cstdio>
#include <chrono>
#include <algorithm>
static inline double now_ms(){using C=std::chrono::high_resolution_clock;return std::chrono::duration<double,std::milli>(C::now().time_since_epoch()).count();}

int main(){
  const int N = 1<<24;
  const float a=1.2345f, b=0.6789f;
  std::vector<float> x(N), y(N);
  std::mt19937 rng(123); std::uniform_real_distribution<float> d(-1,1);
  std::generate(x.begin(), x.end(), [&]{return d(rng);} );
  double t0=now_ms();
#if defined(__AVX512F__)
  int i=0; __m512 va=_mm512_set1_ps(a), vb=_mm512_set1_ps(b);
  for(; i+16<=N; i+=16){
    __m512 vx=_mm512_loadu_ps(&x[i]);
    __m512 vy=_mm512_fmadd_ps(va, vx, vb);
    _mm512_storeu_ps(&y[i], vy);
  }
#elif defined(__AVX2__)
  int i=0; __m256 va=_mm256_set1_ps(a), vb=_mm256_set1_ps(b);
  for(; i+8<=N; i+=8){
    __m256 vx=_mm256_loadu_ps(&x[i]);
#if defined(__FMA__)
    __m256 vy=_mm256_fmadd_ps(va, vx, vb);
#else
    __m256 vy=_mm256_add_ps(_mm256_mul_ps(va,vx), vb);
#endif
    _mm256_storeu_ps(&y[i], vy);
  }
#else
  int i=0;
#endif
  // scalar cleanup tail
  for (int i=(N/16)*16; i<N; ++i) y[i]=a*x[i]+b;
  double t1=now_ms();
  double sum=0; for (auto v: y) sum+=v;
  printf("vectorized ax+b over %d elems in %.2f ms, checksum=%.3f\n",N,t1-t0,sum);
  return 0;
}
