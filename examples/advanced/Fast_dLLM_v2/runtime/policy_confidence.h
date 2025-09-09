#pragma once
// Pseudocode header for confidence-aware parallel decoding policy and cache COW

namespace tessera { namespace runtime {

struct ConfidenceParams {
  float tau;      // entropy/top-p agreement threshold
  int   window;   // tokens per validation window
  int   max_K;    // number of speculative branches
};

struct PrefixDecision {
  int lcp_len;    // commit prefix length
  bool keep[8];   // whether to keep branch k after merge
};

PrefixDecision validate_and_merge(const float* stats[], int K, const ConfidenceParams& p);

}} // namespace
