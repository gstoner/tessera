#pragma once
#include <chrono>

struct OpTimer {
  using clock = std::chrono::high_resolution_clock;
  clock::time_point t0;
  void start() { t0 = clock::now(); }
  double stop_ms() const {
    auto t1 = clock::now();
    std::chrono::duration<double, std::milli> d = t1 - t0;
    return d.count();
  }
};
