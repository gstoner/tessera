#include "Autotune.h"
#include <map>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "nlohmann_json.hpp"
using json = nlohmann::json;

// FMA cfg launcher
extern "C" void tessera_power_attn_cuda_forward_cfg(
    const void* q,const void* k,const void* v,void* o,
    int B,int H,int S,int D,int Dh,int M,int window,int causal,
    int tok_tile,int stages,int vec_elems,int ldsm, void* stream);

// WGMMA cfg launcher (H100)
extern "C" void tessera_power_attn_cuda_forward_wgmma_cfg(
    const void* q,const void* k,const void* v,void* o,
    int B,int H,int S,int D,int Dh,int M,int window,int causal,
    int tok_tile,int stages,int vec_elems,int ldsm, void* stream);

PowerAutotuner::PowerAutotuner(const std::string& cache_txt, const std::string& cache_json, const std::string& csv_path)
: cache_txt_(cache_txt), cache_json_(cache_json), csv_path_(csv_path) { load(); }

static void write_txt(std::ofstream& f, const std::pair<const TuneKey, TunePick>& it){
  const TuneKey& k = it.first; const TunePick& p = it.second;
  f<<k.arch<<" "<<k.D<<" "<<k.Dh<<" "<<k.S<<" "<<k.M<<" "
   <<p.cfg.tok_tile<<" "<<p.cfg.stages<<" "<<p.cfg.vec_elems<<" "<<p.cfg.ldsm<<" "
   <<p.cfg.variant<<" "<<p.ms<<" "<<p.tflops<<" "<<p.tok_per_s<<"\n";
}

void PowerAutotuner::load(){
  // Prefer JSON
  std::ifstream jf(cache_json_);
  if (jf.good()){
    json J; jf>>J;
    for (auto& e : J){
      TuneKey k{e["arch"], e["D"], e["Dh"], e["S"], e["M"]};
      PowerCfgRT c{e["tok_tile"], e["stages"], e["vec_elems"], e["ldsm"], e["variant"]};
      TunePick p{c, e["ms"], e["tflops"], e["tok_per_s"]};
      cache_[k] = p;
    }
    return;
  }
  // Fallback to txt
  std::ifstream f(cache_txt_);
  if(!f.good()) return;
  int arch,D,Dh,S,M,tok,stg,vec,ldsm; float ms; double tflops,tps; std::string variant;
  while (f>>arch>>D>>Dh>>S>>M>>tok>>stg>>vec>>ldsm>>variant>>ms>>tflops>>tps){
    TuneKey k{arch,D,Dh,S,M};
    cache_[k] = TunePick{PowerCfgRT{tok,stg,vec,ldsm,variant}, ms, tflops, tps};
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
      {"vec_elems",p.cfg.vec_elems},{"ldsm",p.cfg.ldsm},{"variant",p.cfg.variant},
      {"ms",p.ms},{"tflops",p.tflops},{"tok_per_s",p.tok_per_s}
    });
  }
  std::ofstream jf(cache_json_, std::ios::trunc);
  jf << J.dump(2);
  // Also write txt for human-grep
  std::ofstream f(cache_txt_, std::ios::trunc);
  for (auto& it : cache_) write_txt(f,it);
  dirty_=false;
}

void PowerAutotuner::append_csv_header(){
  if (!std::ifstream(csv_path_).good()){
    std::ofstream f(csv_path_, std::ios::app);
    f<<"arch,D,Dh,S,M,tok_tile,stages,vec_elems,ldsm,variant,ms,tflops,tok_per_s\n";
  }
}

void PowerAutotuner::append_csv(const TuneKey& key, const TunePick& pick){
  append_csv_header();
  std::ofstream f(csv_path_, std::ios::app);
  f<<key.arch<<","<<key.D<<","<<key.Dh<<","<<key.S<<","<<key.M<<","
   <<pick.cfg.tok_tile<<","<<pick.cfg.stages<<","<<pick.cfg.vec_elems<<","<<pick.cfg.ldsm<<","
   <<pick.cfg.variant<<","<<pick.ms<<","<<pick.tflops<<","<<pick.tok_per_s<<"\n";
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
  // include FMA and WGMMA variants
  std::vector<std::string> variants = {"fma"};
#if defined(TESSERA_ENABLE_WGMMA)
  if (key.arch >= 90) variants.push_back("wgmma");
#endif
  for (int tt : tok_tiles)
    for (int st : stages)
      for (int ve : vecs)
        for (int ld : ldsms)
          for (auto& var : variants)
            grid.push_back(PowerCfgRT{tt,st,ve,ld,var});

  append_csv_header();
  float best_ms = 1e9f; PowerCfgRT best=grid[0]; double best_tflops=0.0, best_tps=0.0;
  for (auto cfg : grid){
    // Time 4 runs and average
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a, (cudaStream_t)stream);
    for(int w=0; w<4; ++w){
      if (cfg.variant == "wgmma"){
#if defined(TESSERA_ENABLE_WGMMA)
        tessera_power_attn_cuda_forward_wgmma_cfg(q,k,v,o,B,H,S,D,Dh,M,window,causal,
                                                  cfg.tok_tile,cfg.stages,cfg.vec_elems,cfg.ldsm, stream);
#else
        // skip if not compiled
        tessera_power_attn_cuda_forward_cfg(q,k,v,o,B,H,S,D,Dh,M,window,causal,
                                            cfg.tok_tile,cfg.stages,cfg.vec_elems,cfg.ldsm, stream);
#endif
      } else {
        tessera_power_attn_cuda_forward_cfg(q,k,v,o,B,H,S,D,Dh,M,window,causal,
                                            cfg.tok_tile,cfg.stages,cfg.vec_elems,cfg.ldsm, stream);
      }
    }
    cudaEventRecord(b, (cudaStream_t)stream);
    cudaEventSynchronize(b);
    float ms=0; cudaEventElapsedTime(&ms,a,b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    ms /= 4.0f;

    // Metrics: projection-only FLOPs ≈ 2 * M * Dh * S
    double flops = 2.0 * (double)M * (double)Dh * (double)S;
    double seconds = (double)ms / 1e3;
    double tflops = (seconds>0) ? (flops / seconds) / 1e12 : 0.0;
    double tok_per_s = (seconds>0) ? (double)S / seconds : 0.0;

    // Log every candidate
    append_csv(key, TunePick{cfg, ms, tflops, tok_per_s});

    if (ms < best_ms){ best_ms = ms; best = cfg; best_tflops=tflops; best_tps=tok_per_s; }
  }
  TunePick pick{best, best_ms, best_tflops, best_tps};
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
    TunePick pick{PowerCfgRT{128,2,8,0,"fma"}, 0.f, 0.0, 0.0};
    return pick;
  }
  auto it = cache_.find(key);
  if (it != cache_.end()) return it->second;
  TunePick pick = run_sweep(key,q,k,v,o,B,H,S,D,Dh,M,window,causal,stream);
  cache_[key] = pick; dirty_ = true; save(); append_csv(key,pick);
  return pick;
}
