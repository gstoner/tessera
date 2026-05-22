#include <cuda_runtime.h>
extern "C" __global__ void power_attn_forward_kernel(const __half*,const __half*,const __half*,__half*,
    int,int,int,int,int,int,int){}
extern "C" void tessera_power_attn_cuda_forward(const void* q,const void* k,const void* v,void* o,
    int B,int H,int S,int D,int M,int window,int causal,int dtype, void* stream){
  (void)q;(void)k;(void)v;(void)o;(void)B;(void)H;(void)S;(void)D;(void)M;(void)window;(void)causal;(void)dtype;(void)stream;
}
