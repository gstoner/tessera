#pragma once
#include <string>
#include <functional>
#include <unordered_map>
#include <vector>

struct OpArgs {
  // Generic arg bag; parsed from CLI.
  int iters = 50;
  uint64_t seed = 123;
  // Matmul
  int M=0,N=0,K=0;
  // Conv2d NHWC
  int Nn=0,H=0,W=0,C=0, Kc=0, R=0, S=0, stride_h=1, stride_w=1, pad_h=0, pad_w=0;
  // Attention
  int B=0, heads=0, seq=0, dim=0;
};

struct OpResult {
  double avg_ms=0.0;
  double gflops=0.0;
  double gbps=0.0;
  double l2_ref=0.0;
};

using OpFn = std::function<OpResult(const OpArgs&)>;

struct OpInfo {
  std::string name;
  OpFn fn;
  std::string help;
};

class OpRegistry {
public:
  static OpRegistry& instance();
  void add(const OpInfo& info);
  std::vector<OpInfo> list() const;
  OpFn find(const std::string& name) const;
private:
  std::unordered_map<std::string,OpInfo> map_;
};
