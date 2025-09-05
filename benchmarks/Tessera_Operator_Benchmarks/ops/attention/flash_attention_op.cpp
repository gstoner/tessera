#include "ops/attention/flash_attention_op.h"
#include "harness/op_utils.h"
#include "common/timer.h"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

static OpResult fa_impl(const OpArgs& a){
  int B=a.B, H=a.heads, S=a.seq, D=a.dim;
  if(B<=0||H<=0||S<=0||D<=0) throw std::runtime_error("flash_attention: set --B --heads --seq --dim");

  std::vector<float> Q(B*H*S*D), K(B*H*S*D), V(B*H*S*D), O(B*H*S*D), Oref(B*H*S*D);
  fill_uniform(Q, a.seed);
  fill_uniform(K, a.seed+1);
  fill_uniform(V, a.seed+2);

  auto run = [&](std::vector<float>& Out){
    const float scale = 1.0f/std::sqrt((float)D);
    for(int b=0;b<B;++b)
      for(int h=0;h<H;++h)
        for(int i=0;i<S;++i){
          // scores = Q[i] · K^T
          std::vector<float> scores(S);
          for(int j=0;j<S;++j){
            float acc=0.0f;
            for(int d=0;d<D;++d){
              float q = Q[(((b*H+h)*S+i)*D)+d];
              float k = K[(((b*H+h)*S+j)*D)+d];
              acc += q*k;
            }
            // causal mask: j>i -> -inf
            if(j>i) acc = -1e9f;
            scores[j]=acc*scale;
          }
          // softmax
          float m = *std::max_element(scores.begin(), scores.end());
          float sum = 0.0f;
          for(int j=0;j<S;++j){ scores[j] = std::exp(scores[j]-m); sum += scores[j]; }
          for(int j=0;j<S;++j) scores[j] /= sum;
          // Out = softmax(scores) * V
          for(int d=0; d<D; ++d){
            float acc=0.0f;
            for(int j=0;j<S;++j){
              float v = V[(((b*H+h)*S+j)*D)+d];
              acc += scores[j]*v;
            }
            Out[(((b*H+h)*S+i)*D)+d] = acc;
          }
        }
  };

  run(O);
  OpTimer T; double total=0.0;
  for(int it=0; it<a.iters; ++it){ T.start(); run(O); total += T.stop_ms(); }
  run(Oref);
  double l2 = l2_error(O, Oref);

  // rough flops: QK^T (B*H*S*S*D*2), softmax (B*H*S*S*exp), PV (B*H*S*S*D*2)
  double flops = 2.0*(double)B*H*S*S*D + 2.0*(double)B*H*S*S*D;
  double gflops = (flops/1e9) / ((total/a.iters)/1000.0);
  return { total/a.iters, gflops, 0.0, l2 };
}

void register_flash_attention(){
  OpRegistry::instance().add({"flash_attention", fa_impl, "FlashAttention (naive CPU reference, causal)"});
}
