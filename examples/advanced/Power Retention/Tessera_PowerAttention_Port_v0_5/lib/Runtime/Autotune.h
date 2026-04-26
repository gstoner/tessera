#pragma once
#include <string>
#include <vector>
#include <tuple>

struct PowerCfgRT {
  int tok_tile;
  int stages;
  int vec_elems;
};

struct TuneKey {
  int arch;   // sm version (e.g., 90/100)
  int D;
  int Dh;
  int S;
  bool operator<(const TuneKey& o) const {
    if (arch!=o.arch) return arch<o.arch;
    if (D!=o.D) return D<o.D;
    if (Dh!=o.Dh) return Dh<o.Dh;
    return S<o.S;
  }
};

struct TunePick {
  PowerCfgRT cfg;
  float ms;
};

class PowerAutotuner {
 public:
  PowerAutotuner(const std::string& cache_path, const std::string& csv_path);
  TunePick pick_or_run(const TuneKey& key,
                       const void* q,const void* k,const void* v,void* o,
                       int B,int H,int S,int D,int Dh,int M,int window,int causal,
                       void* stream);
 private:
  std::string cache_path_;
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
