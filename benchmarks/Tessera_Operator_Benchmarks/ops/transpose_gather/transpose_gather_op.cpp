#include "ops/transpose_gather/transpose_gather_op.h"
#include "harness/op_utils.h"
#include "common/timer.h"
#include <vector>
#include <stdexcept>

static OpResult tg_impl(const OpArgs& a){
  int B = std::max(1,a.M);
  int S = std::max(1,a.N);
  int D = std::max(1,a.K);
  std::vector<float> x(B*S*D), y(B*D*S), yref(B*D*S);
  fill_uniform(x, a.seed);

  auto run = [&](std::vector<float>& out){
    for(int b=0;b<B;++b)
      for(int s=0;s<S;++s)
        for(int d=0; d<D; ++d){
          out[(b*D + d)*S + s] = x[(b*S + s)*D + d];
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

void register_transpose_gather(){
  OpRegistry::instance().add({"transpose_gather", tg_impl, "Transpose (B,S,D) -> (B,D,S) (CPU)"});
}
