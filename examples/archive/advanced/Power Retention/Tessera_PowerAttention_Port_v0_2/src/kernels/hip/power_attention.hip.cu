#include <hip/hip_runtime.h>
extern "C" __global__ void power_attn_forward_kernel_hip(const __half*,const __half*,const __half*,__half*,
    int,int,int,int,int,int,int){}
extern "C" void tessera_power_attn_hip_forward(const void*,const void*,const void*,void*,
    int,int,int,int,int,int,int,int, void*){
}
