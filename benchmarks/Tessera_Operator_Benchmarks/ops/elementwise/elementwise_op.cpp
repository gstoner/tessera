#include "ops/elementwise/elementwise_op.h"
#include "harness/op_utils.h"
#include "common/timer.h"
#include <vector>
#include <cmath>

static OpResult ew_impl(const OpArgs& a){
  int64_t N = (int64_t)std::max(1,a.M) * std::max(1,a.N) * std::max(1,a.K);
  if(N<=0) N = 1<<24;
  std::vector<float> x(N), y(N), yref(N);
  fill_uniform(x, a.seed);

  auto run = [&](std::vector<float>& out){
    for(int64_t i=0;i<N;++i){
      float v = x[i];
      out[i] = std::tanh(v) + 0.1f*v;
    }
  };

  run(y);
  OpTimer T; double total=0.0;
  for(int it=0; it<a.iters; ++it){ T.start(); run(y); total += T.stop_ms(); }
  run(yref);
  double l2 = l2_error(y, yref);

  double bytes = 2.0*(double)N*sizeof(float); // read x, write y
  double gbps = (bytes/1e9) / ((total/a.iters)/1000.0);
  return { total/a.iters, 0.0, gbps, l2 };
}

void register_elementwise(){
  OpRegistry::instance().add({"elementwise", ew_impl, "tanh(x)+0.1x (CPU)"});
}
