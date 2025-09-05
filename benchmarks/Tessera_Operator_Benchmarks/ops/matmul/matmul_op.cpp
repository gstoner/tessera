#include "ops/matmul/matmul_op.h"
#include "harness/op_registry.h"
#include "harness/op_utils.h"
#include "common/timer.h"
#include <vector>
#include <cmath>

static OpResult matmul_impl(const OpArgs& a){
  int M=a.M, N=a.N, K=a.K;
  if(M<=0||N<=0||K<=0) { throw std::runtime_error("matmul requires --m --n --k"); }

  std::vector<float> A(M*K), B(K*N), C(M*N), Cref(M*N);
  fill_uniform(A, a.seed, -1.0f, 1.0f);
  fill_uniform(B, a.seed+1, -1.0f, 1.0f);

  auto gemm = [&](std::vector<float>& Cout){
    for(int m=0;m<M;++m){
      for(int n=0;n<N;++n){
        float acc=0.0f;
        for(int k=0;k<K;++k) acc += A[m*K+k]*B[k*N+n];
        Cout[m*N+n]=acc;
      }
    }
  };

  // Warmup + timing
  gemm(C);
  OpTimer T; double total=0.0;
  for(int it=0;it<a.iters;++it){
    T.start();
    gemm(C);
    total += T.stop_ms();
  }
  gemm(Cref);
  double l2 = l2_error(C, Cref);

  // FLOPs: 2*M*N*K
  double flops = 2.0*(double)M*N*K;
  double gflops = (flops/1e9) / ( (total/a.iters)/1000.0 );
  return { total/a.iters, gflops, 0.0, l2 };
}

void register_matmul(){
  OpRegistry::instance().add({"matmul", matmul_impl, "Matrix multiply (CPU reference)"});
}
