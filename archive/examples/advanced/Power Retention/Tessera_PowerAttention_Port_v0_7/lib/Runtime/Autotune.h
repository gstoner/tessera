#pragma once
#include <string>
#include <vector>
#include <tuple>
#include <map>

struct PowerCfgRT {
  int tok_tile;
  int stages;
  int vec_elems;
  int ldsm; // 0/1
};

struct TuneKey {
  int arch;   // 90, 100, etc.
  int D;
  int Dh;
  int S;
  int M;
  bool operator<(const TuneKey& o) const {
    if (arch!=o.arch) return arch<o.arch;
    if (D!=o.D) return D<o.D;
    if (Dh!=o.Dh) return Dh<o.Dh;
    if (S!=o.S) return S<o.S;
    return M<o.M;
  }
};

struct TunePick {
  PowerCfgRT cfg;
  float ms;
};

class PowerAutotuner {
 public:
  PowerAutotuner(const std::string& cache_txt, const std::string& cache_json, const std::string& csv_path);
  TunePick pick_or_run(const TuneKey& key,
                       const void* q,const void* k,const void* v,void* o,
                       int B,int H,int S,int D,int Dh,int M,int window,int causal,
                       void* stream);
 private:
  std::string cache_txt_;
  std::string cache_json_;
  std::string csv_path_;
  bool dirty_=false;
  std::map<TuneKey,TunePick> cache_;
  void load();
  void save();
  void append_csv(const TuneKey& key, const TunePick& pick);
  TunePick run_sweep(const TuneKey& key,
                     const void* q,const void* k,const void* v,void* o,
                     int B,int H,int S,int D,int Dh,int M,int window,int causal,
                     void* stream);
};
