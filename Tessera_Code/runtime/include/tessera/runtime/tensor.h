#pragma once
#include <cstdint>
#include <vector>

namespace tessera {
enum class DType { FP32=0, BF16=1, FP16=2, INT32=3 };
struct Tensor {
  void* data{nullptr};
  DType dtype{DType::FP32};
  int device{-1}; // -1 host
  std::vector<int64_t> shape;
};
} // namespace tessera
