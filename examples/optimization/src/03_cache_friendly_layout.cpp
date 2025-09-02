
// AoS vs SoA: compute v = a*x + y on 3D points to illustrate layout impact.
#include <vector>
#include <random>
#include <cstdio>
#include <chrono>
#include <algorithm>

struct PointAoS { float x,y,z; };

static inline double now_ms(){using C=std::chrono::high_resolution_clock;return std::chrono::duration<double,std::milli>(C::now().time_since_epoch()).count();}

int main(){
  const int N = 1<<22; const float a=1.1f;
  // AoS
  std::vector<PointAoS> P(N); std::vector<PointAoS> Q(N);
  // SoA
  std::vector<float> X(N),Y(N),Z(N), U(N),V(N),W(N);

  for(int i=0;i<N;++i){P[i]={float(i%13),float(i%7),float(i%5)}; Q[i]={0,0,0};}
  for(int i=0;i<N;++i){X[i]=float(i%13);Y[i]=float(i%7);Z[i]=float(i%5);U[i]=V[i]=W[i]=0;}

  double t0=now_ms();
  for(int i=0;i<N;++i){ Q[i].x = a*P[i].x + Q[i].x; Q[i].y = a*P[i].y + Q[i].y; Q[i].z = a*P[i].z + Q[i].z; }
  double t1=now_ms();

  double t2=now_ms();
  for(int i=0;i<N;++i){ U[i] = a*X[i] + U[i]; V[i] = a*Y[i] + V[i]; W[i] = a*Z[i] + W[i]; }
  double t3=now_ms();

  double sA=0,sS=0; for(auto& q:Q){sA+=q.x+q.y+q.z;} for(int i=0;i<N;++i){sS+=U[i]+V[i]+W[i];}
  printf("AoS: %.2f ms  SoA: %.2f ms  speedup: %.2fx  diff=%.3f\n", t1-t0, t3-t2, (t1-t0)/(t3-t2+1e-9), sA-sS);
  return 0;
}
