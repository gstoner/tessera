#pragma once
#include <string>
#include <vector>
#include <random>

inline void opbench_device_init() {
  // Hook: initialize backends if needed (Tessera, CUDA, ROCm, etc).
}

inline uint64_t seed_from_args(uint64_t base_seed, int variant) {
  std::seed_seq seq{(uint32_t)base_seed, (uint32_t)variant};
  std::vector<uint32_t> v(2);
  seq.generate(v.begin(), v.end());
  return ((uint64_t)v[0] << 32) | v[1];
}
