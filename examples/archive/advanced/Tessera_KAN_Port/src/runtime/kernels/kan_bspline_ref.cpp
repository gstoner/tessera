// Reference CPU bspline evaluator (no threading). For validation only.
#include <vector>
#include <cmath>
#include <algorithm>

extern "C" void kan_bspline_eval_ref(
    const float* x, // [B*I]
    int B, int I,
    int degree, float gmin, float gmax, int grid_size,
    float* phi     // [B*I*M]
) {
  const int k = degree;
  const int M = grid_size + k - 1;
  const int K = grid_size + 2*k - 1; // knots length
  std::vector<float> knots; knots.reserve(K);
  for (int r=0;r<k;r++) knots.push_back(gmin);
  if (grid_size>1) {
    for (int c=1;c<grid_size;c++){
      knots.push_back(gmin + (gmax-gmin) * (float(c)/float(grid_size)));
    }
  }
  for (int r=0;r<k;r++) knots.push_back(gmax);

  auto idx = [&](int b,int i){ return b*I + i; };
  auto pidx = [&](int b,int i,int m){ return (b*I + i)*M + m; };

  // degree-0 init
  for (int b=0;b<B;b++){
    for (int i0=0;i0<I;i0++){
      float xv = std::min(std::max(x[idx(b,i0)], gmin), gmax);
      for (int m=0;m<M;m++){
        float t0 = knots[m], t1 = knots[m+1];
        bool in = (m < M-1) ? (xv >= t0 && xv < t1) : (xv >= t0 && xv <= t1);
        phi[pidx(b,i0,m)] = in ? 1.f : 0.f;
      }
    }
  }
  // elevate degree
  std::vector<float> buf(M);
  for (int d=1; d<=k; ++d){
    for (int b=0;b<B;b++){
      for (int i0=0;i0<I;i0++){
        float xv = std::min(std::max(x[idx(b,i0)], gmin), gmax);
        for (int m=0;m<M;m++){
          float t_m = knots[m], t_md = knots[m+d];
          float term1 = 0.f;
          float denom1 = (t_md - t_m);
          if (denom1 != 0.f) term1 = ((xv - t_m)/denom1) * phi[pidx(b,i0,m)];
          float term2 = 0.f;
          if (m+1 < M){
            float t_md1 = knots[m+d+1], t_m1 = knots[m+1];
            float denom2 = (t_md1 - t_m1);
            if (denom2 != 0.f) term2 = ((t_md1 - xv)/denom2) * phi[pidx(b,i0,m+1)];
          }
          buf[m] = term1 + term2;
        }
        for (int m=0;m<M;m++) phi[pidx(b,i0,m)] = buf[m];
      }
    }
  }
}