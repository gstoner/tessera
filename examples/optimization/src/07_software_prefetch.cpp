
// Double-buffered streaming transform with software prefetch (gcc/clang __builtin_prefetch).
#include <vector>
#include <cstdio>
#include <chrono>
#include <algorithm>

static inline double now_ms(){using C=std::chrono::high_resolution_clock;return std::chrono::duration<double,std::milli>(C::now().time_since_epoch()).count();}

int main(){
  const int N = 1<<26; // ~67M floats
  std::vector<float> x(N,1.0f), y(N,0.0f);
  double t0=now_ms();
  const int PF_DIST = 256; // elements ahead
  for (int i=0;i<N;++i){
#if defined(__GNUC__) || defined(__clang__)
    if (i+PF_DIST < N) __builtin_prefetch(&x[i+PF_DIST], 0, 3);
#endif
    y[i] = 2.0f*x[i] + y[i];
  }
  double t1=now_ms();
  double s=0; for (auto v:y) s+=v;
  printf("prefetch stream time: %.2f ms  sum=%.3f\\n", t1-t0, s);
  return 0;
}
