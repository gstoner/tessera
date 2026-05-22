#include "Autotune.h"
#include <map>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "nlohmann_json.hpp"
// NOTE: replace with real nlohmann/json in your build
using json = nlohmann::json;

// Kernel launcher with explicit config (implemented in CUDA TU)
extern "C" void tessera_power_attn_cuda_forward_cfg(
    const void* q,const void* k,const void* v,void* o,
    int B,int H,int S,int D,int Dh,int M,int window,int causal,
    int tok_tile,int stages,int vec_elems,int ldsm, void* stream);

PowerAutotuner::PowerAutotuner(const std::string& cache_txt, const std::string& cache_json, const std::string& csv_path)
: cache_txt_(cache_txt), cache_json_(cache_json), csv_path_(csv_path) { load(); }

static void write_txt(std::ofstream& f, const std::pair<const TuneKey, TunePick>& it){
  const TuneKey& k = it.first; const TunePick& p = it.second;
  f<<k.arch<<" "<<k.D<<" "<<k.Dh<<" "<<k.S<<" "<<k.M<<" "
   <<p.cfg.tok_tile<<" "<<p.cfg.stages<<" "<<p.cfg.vec_elems<<" "<<p.cfg.ldsm<<" "<<p.ms<<"\n";
}

void PowerAutotuner::load(){
  // Prefer JSON
  std::ifstream jf(cache_json_);
  if (jf.good()){
    json J; jf>>J;
    for (auto& e : J){
      TuneKey k{e["arch"], e["D"], e["Dh"], e["S"], e["M"]};
      PowerCfgRT c{e["tok_tile"], e["stages"], e["vec_elems"], e["ldsm"]};
      float ms = e["ms"];
      cache_[k] = TunePick{c, ms};
    }
    return;
  }
  // Fallback to txt
  std::ifstream f(cache_txt_);
  if(!f.good()) return;
  int arch,D,Dh,S,M,tok,stg,vec,ldsm; float ms;
  while (f>>arch>>D>>Dh>>S>>M>>tok>>stg>>vec>>ldsm>>ms){
    TuneKey k{arch,D,Dh,S,M};
    cache_[k] = TunePick{PowerCfgRT{tok,stg,vec,ldsm}, ms};
  }
}

void PowerAutotuner::save(){
  if(!dirty_) return;
  // Save JSON
  json J = json::array();
  for (auto& it : cache_){
    const TuneKey& k = it.first;
    const TunePick& p = it.second;
    J.push_back({
      {"arch",k.arch},{"D",k.D},{"Dh",k.Dh},{"S",k.S},{"M",k.M},
      {"tok_tile",p.cfg.tok_tile},{"stages",p.cfg.stages},
      {"vec_elems",p.cfg.vec_elems},{"ldsm",p.cfg.ldsm},{"ms",p.ms}
    });
  }
  std::ofstream jf(cache_json_, std::ios::trunc);
  jf << J.dump(2);
  // Also write txt for human-grep
  std::ofstream f(cache_txt_, std::ios::trunc);
  for (auto& it : cache_) write_txt(f,it);
  dirty_=false;
}

void PowerAutotuner::append_csv(const TuneKey& key, const TunePick& pick){
  bool add_header = !std::ifstream(csv_path_).good();
  std::ofstream f(csv_path_, std::ios::app);
  if (add_header)
    f<<"arch,D,Dh,S,M,tok_tile,stages,vec_elems,ldsm,ms,tflops,tok_per_s\n";
  f<<key.arch<<","<<key.D<<","<<key.Dh<<","<<key.S<<","<<key.M<<","
   <<pick.cfg.tok_tile<<","<<pick.cfg.stages<<","<<pick.cfg.vec_elems<<","<<pick.cfg.ldsm<<","<<pick.ms<<","<<0<<","<<0<<"\n"; // placeholders set below
}

TunePick PowerAutotuner::run_sweep(const TuneKey& key,
                     const void* q,const void* k,const void* v,void* o,
                     int B,int H,int S,int D,int Dh,int M,int window,int causal,
                     void* stream)
{
  // Expanded grid
  std::vector<PowerCfgRT> grid;
  int tok_tiles[] = {128,192,256};
  int stages[] = {2,3,4};
  int vecs[] = {8,16};
  int ldsms[] = {0,1};
  for (int tt : tok_tiles)
    for (int st : stages)
      for (int ve : vecs)
        for (int ld : ldsms)
          grid.push_back(PowerCfgRT{tt,st,ve,ld});

  float best_ms = 1e9f; PowerCfgRT best=grid[0];
  for (auto cfg : grid){
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a, (cudaStream_t)stream);
    // warmup + 3 runs
    for(int w=0; w<4; ++w){
      tessera_power_attn_cuda_forward_cfg(q,k,v,o,B,H,S,D,Dh,M,window,causal,
                                          cfg.tok_tile,cfg.stages,cfg.vec_elems,cfg.ldsm, stream);
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
  // Env toggle
  const char* env = std::getenv("TESSERA_POWER_AUTOTUNE");
  bool do_tune = (env==nullptr || std::string(env)!="0");
  if (!do_tune){
    TunePick pick{PowerCfgRT{128,2,8,0}, 0.f};
    return pick;
  }

  auto it = cache_.find(key);
  if (it != cache_.end()) return it->second;
  TunePick pick = run_sweep(key,q,k,v,o,B,H,S,D,Dh,M,window,causal,stream);
  // Estimate FLOPs for projection: flops ≈ 2 * M * Dh * S (per call).
  // ms measured covers 4 runs average; use pick.ms
  double flops = 2.0 * (double)M * (double)Dh * (double)S;
  double seconds = (double)pick.ms / 1e3;
  double tflops = (flops / seconds) / 1e12;
  double tok_per_s = (double)S / seconds;
  // Rewrite the CSV line with metrics (simplest: append a second line with metrics)
  append_csv(key, pick); // existing line
  // Append metrics line
  std::ofstream f(csv_path_, std::ios::app);
  f<<"#metrics,"<<tflops<<","<<tok_per_s<<"\n";

  cache_[key] = pick; dirty_ = true; save(); append_csv(key,pick);
  return pick;
}
