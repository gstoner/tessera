
#include <cstdint>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include "Autotune.h"

extern "C" void tessera_power_attn_cuda_forward_cfg(
    const void* q,const void* k,const void* v,void* o,
    int B,int H,int S,int D,int Dh,int M,int window,int causal,
    int tok_tile,int stages,int vec_elems, void* stream);

namespace tessera { namespace power {
enum DType { F16=0, BF16=1, FP8_E4M3=2 };

void launch_power_attention(
    const void* q, const void* k, const void* v, void* o,
    int B,int H,int S,int D,int M,int window,bool causal,
    DType dtype, void* stream)
{
#ifdef TESSERA_POWER_USE_CUDA
  // Determine arch
  int device=0; cudaGetDevice(&device);
  cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, device);
  int arch = prop.major*10 + prop.minor; // e.g., 90,100
  int Dh = D; // for now
  int Mphi = D*(D+1)/2;
  // Autotune
  std::string cache = "autotune_cache.txt";
  std::string csv   = "autotune_runs.csv";
  PowerAutotuner tuner(cache, csv);
  TunePick pick = tuner.pick_or_run(TuneKey{arch,D,Dh,S}, q,k,v,o, B,H,S,D,Dh,Mphi,window, causal?1:0, stream);
  tessera_power_attn_cuda_forward_cfg(q,k,v,o,B,H,S,D,Dh,Mphi,window,causal?1:0,
      pick.cfg.tok_tile, pick.cfg.stages, pick.cfg.vec_elems, stream);
#elif defined(TESSERA_POWER_USE_HIP)
  // TODO: add HIP autotune path
  (void)q;(void)k;(void)v;(void)o;(void)B;(void)H;(void)S;(void)D;(void)M;(void)window;(void)causal;(void)dtype;(void)stream;
  throw std::runtime_error("HIP path not implemented yet for autotune.");
#else
  (void)q;(void)k;(void)v;(void)o;(void)B;(void)H;(void)S;(void)D;(void)M;(void)window;(void)causal;(void)dtype;(void)stream;
  throw std::runtime_error("No backend enabled for tessera_power.");
#endif
}
}} // ns
