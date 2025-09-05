#include "ops/reduce/reduce_op.h"
#include "harness/op_utils.h"
#include "common/timer.h"
#include <vector>
#include <stdexcept>
#include <numeric>

static OpResult reduce_impl(const OpArgs& a){
  // Reduce over M*N*K flattened
  int64_t N = (int64_t)std::max(1,a.M) * std::max(1,a.N) * std::max(1,a.K);
  if(N<=0) N = 1<<24; // default 16M elements if not set
  std::vector<float> x(N), y(1);
  fill_uniform(x, a.seed);

  auto run = [&](std::vector<float>& out){
    float acc=0.0f;
    for(int64_t i=0;i<N;++i) acc += x[i];
    out[0]=acc;
  };

  run(y);
  OpTimer T; double total=0.0;
  for(int it=0; it<a.iters; ++it){ T.start(); run(y); total += T.stop_ms(); }

  float ref=0.0f; for(auto v: x) ref+=v;
  double l2 = std::abs(y[0]-ref);

  double bytes = (double)N*sizeof(float);
  double gbps = (bytes/1e9) / ((total/a.iters)/1000.0);
  return { total/a.iters, 0.0, gbps, l2 };
}

void register_reduce(){ OpRegistry::instance().add({"reduce", reduce_impl, "Sum reduce (CPU)"}); }
