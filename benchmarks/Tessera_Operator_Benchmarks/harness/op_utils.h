#pragma once
#include <vector>
#include <random>
#include <algorithm>

template<typename T>
void fill_uniform(std::vector<T>& v, uint64_t seed, T lo=T(-1), T hi=T(1)){
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist((double)lo, (double)hi);
  for(auto& x: v) x = (T)dist(rng);
}

template<typename T>
double l2_error(const std::vector<T>& a, const std::vector<T>& b){
  double e=0.0;
  for(size_t i=0;i<a.size();++i){
    double d = double(a[i]) - double(b[i]);
    e += d*d;
  }
  return std::sqrt(e);
}
