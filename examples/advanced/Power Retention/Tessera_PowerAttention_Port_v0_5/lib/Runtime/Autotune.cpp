#include "Autotune.h"
#include <map>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>

// Kernel launcher with explicit config (implemented in CUDA TU)
extern "C" void tessera_power_attn_cuda_forward_cfg(
    const void* q,const void* k,const void* v,void* o,
    int B,int H,int S,int D,int Dh,int M,int window,int causal,
    int tok_tile,int stages,int vec_elems, void* stream);

PowerAutotuner::PowerAutotuner(const std::string& cache_path, const std::string& csv_path)
: cache_path_(cache_path), csv_path_(csv_path) { load(); }

void PowerAutotuner::load(){
  std::ifstream f(cache_path_);
  if(!f.good()) return;
  int arch,D,Dh,S,tok,stg,vec; float ms;
  while (f>>arch>>D>>Dh>>S>>tok>>stg>>vec>>ms){
    TuneKey k{arch,D,Dh,S};
    cache_[k] = TunePick{PowerCfgRT{tok,stg,vec}, ms};
  }
}
void PowerAutotuner::save(){
  if(!dirty_) return;
  std::ofstream f(cache_path_, std::ios::trunc);
  for (auto& it : cache_){
    const TuneKey& k = it.first;
    const TunePick& p = it.second;
    f<<k.arch<<" "<<k.D<<" "<<k.Dh<<" "<<k.S<<" "
     <<p.cfg.tok_tile<<" "<<p.cfg.stages<<" "<<p.cfg.vec_elems<<" "<<p.ms<<"\n";
  }
  dirty_=false;
}
void PowerAutotuner::append_csv(const TuneKey& key, const TunePick& pick){
  std::ofstream f(csv_path_, std::ios::app);
  f<<"arch,D,Dh,S,tok_tile,stages,vec_elems,ms\n";
  f<<key.arch<<","<<key.D<<","<<key.Dh<<","<<key.S<<","
   <<pick.cfg.tok_tile<<","<<pick.cfg.stages<<","<<pick.cfg.vec_elems<<","<<pick.ms<<"\n";
}

TunePick PowerAutotuner::run_sweep(const TuneKey& key,
                     const void* q,const void* k,const void* v,void* o,
                     int B,int H,int S,int D,int Dh,int M,int window,int causal,
                     void* stream)
{
  std::vector<PowerCfgRT> grid = {
    {128,2,8},{128,3,8},{256,2,8},{256,3,8},{256,2,16}
  };
  float best_ms = 1e9f; PowerCfgRT best=grid[0];
  for (auto cfg : grid){
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a, (cudaStream_t)stream);
    // warmup + 3 runs
    for(int w=0; w<4; ++w){
      tessera_power_attn_cuda_forward_cfg(q,k,v,o,B,H,S,D,Dh,M,window,causal,
                                          cfg.tok_tile,cfg.stages,cfg.vec_elems, stream);
    }
    cudaEventRecord(b, (cudaStream_t)stream);
    cudaEventSynchronize(b);
    float ms=0; cudaEventElapsedTime(&ms,a,b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    ms /= 4.0f;
    if (ms < best_ms){ best_ms = ms; best = cfg; }
  }
  TunePick pick{best, best_ms};
  return pick;
}

TunePick PowerAutotuner::pick_or_run(const TuneKey& key,
                       const void* q,const void* k,const void* v,void* o,
                       int B,int H,int S,int D,int Dh,int M,int window,int causal,
                       void* stream)
{
  auto it = cache_.find(key);
  if (it != cache_.end()) return it->second;
  TunePick pick = run_sweep(key,q,k,v,o,B,H,S,D,Dh,M,window,causal,stream);
  cache_[key] = pick; dirty_ = true; save(); append_csv(key,pick);
  return pick;
}
