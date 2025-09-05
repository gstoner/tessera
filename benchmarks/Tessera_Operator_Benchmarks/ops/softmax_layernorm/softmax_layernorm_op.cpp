#include "ops/softmax_layernorm/softmax_layernorm_op.h"
#include "harness/op_utils.h"
#include "common/timer.h"
#include <vector>
#include <cmath>
#include <algorithm>

static OpResult sl_impl(const OpArgs& a){
  int B = std::max(1,a.M);
  int S = std::max(1,a.N);
  int D = std::max(1,a.K);
  std::vector<float> x(B*S*D), y(B*S*D), yref(B*S*D);
  fill_uniform(x, a.seed);

  auto run = [&](std::vector<float>& out){
    // softmax across S for each (B,D)
    for(int b=0;b<B;++b){
      for(int d=0; d<D; ++d){
        float m=-1e9f;
        for(int s=0;s<S;++s) m = std::max(m, x[(b*S+s)*D + d]);
        float z=0.0f;
        for(int s=0;s<S;++s){ z += std::exp(x[(b*S+s)*D + d]-m); }
        for(int s=0;s<S;++s){ out[(b*S+s)*D + d] = std::exp(x[(b*S+s)*D + d]-m)/z; }
      }
    }
    // layernorm over D for each (B,S)
    for(int b=0;b<B;++b)
      for(int s=0;s<S;++s){
        double mean=0.0, var=0.0;
        for(int d=0; d<D; ++d) mean += out[(b*S+s)*D + d];
        mean /= D;
        for(int d=0; d<D; ++d){
          double dv = out[(b*S+s)*D + d] - mean;
          var += dv*dv;
        }
        var/=D;
        double inv = 1.0/std::sqrt(var + 1e-5);
        for(int d=0; d<D; ++d){
          out[(b*S+s)*D + d] = float((out[(b*S+s)*D + d]-mean)*inv);
        }
      }
  };

  run(y);
  OpTimer T; double total=0.0;
  for(int it=0; it<a.iters; ++it){ T.start(); run(y); total += T.stop_ms(); }
  run(yref);
  double l2 = l2_error(y, yref);

  double bytes = 2.0*(double)B*S*D*sizeof(float);
  double gbps = (bytes/1e9) / ((total/a.iters)/1000.0);
  return { total/a.iters, 0.0, gbps, l2 };
}

void register_softmax_layernorm(){
  OpRegistry::instance().add({"softmax_layernorm", sl_impl, "Softmax over S then LayerNorm over D (CPU)"});
}
