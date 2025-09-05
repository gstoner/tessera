#include "ops/conv2d/conv2d_op.h"
#include "harness/op_utils.h"
#include "common/timer.h"
#include <vector>
#include <stdexcept>

static OpResult conv_impl(const OpArgs& a){
  int N=a.Nn,H=a.H,W=a.W,C=a.C,Kc=a.Kc,R=a.R,S=a.S;
  int sh=a.stride_h, sw=a.stride_w, ph=a.pad_h, pw=a.pad_w;
  if(N<=0||H<=0||W<=0||C<=0||Kc<=0||R<=0||S<=0) throw std::runtime_error("conv2d: set dims");

  int Ho = (H + 2*ph - R)/sh + 1;
  int Wo = (W + 2*pw - S)/sw + 1;

  std::vector<float> X(N*H*W*C), Wt(R*S*C*Kc), Y(N*Ho*Wo*Kc), Yref(N*Ho*Wo*Kc);
  fill_uniform(X, a.seed);
  fill_uniform(Wt, a.seed+7);

  auto conv = [&](std::vector<float>& Ydst){
    std::fill(Ydst.begin(), Ydst.end(), 0.0f);
    for(int n=0;n<N;++n)
      for(int ho=0;ho<Ho;++ho)
        for(int wo=0;wo<Wo;++wo)
          for(int k=0;k<Kc;++k){
            float acc=0.0f;
            for(int r=0;r<R;++r)
              for(int s=0;s<S;++s)
                for(int c=0;c<C;++c){
                  int h = ho*sh + r - ph;
                  int w = wo*sw + s - pw;
                  if(h<0||h>=H||w<0||w>=W) continue;
                  float x = X[((n*H + h)*W + w)*C + c];
                  float wv= Wt[((r*S + s)*C + c)*Kc + k];
                  acc += x*wv;
                }
            Ydst[((n*Ho + ho)*Wo + wo)*Kc + k] = acc;
          }
  };

  conv(Y);
  OpTimer T; double total=0.0;
  for(int it=0; it<a.iters; ++it){
    T.start(); conv(Y); total += T.stop_ms();
  }
  conv(Yref);
  double l2 = l2_error(Y, Yref);

  // Very rough op count
  double flops = 2.0 * (double)N*Ho*Wo*Kc * (double)R*S*C;
  double gflops = (flops/1e9) / ((total/a.iters)/1000.0);
  return { total/a.iters, gflops, 0.0, l2 };
}

void register_conv2d(){
  OpRegistry::instance().add({"conv2d", conv_impl, "NHWC direct conv (CPU reference)"});
}
