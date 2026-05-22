extern "C" {
#ifdef TESSERA_POWER_USE_CUDA
void tessera_power_attn_cuda_forward(const void*, const void*, const void*, void*,
    int,int,int,int,int,int,int,int, void*);
#endif
#ifdef TESSERA_POWER_USE_HIP
void tessera_power_attn_hip_forward(const void*, const void*, const void*, void*,
    int,int,int,int,int,int,int,int, void*);
#endif
}
