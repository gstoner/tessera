#include <cstdio>
#include <vector>
#include <random>
#include <cstdint>
#include <algorithm>
#include <string>

// Extremely small microbench that simulates LLaDA-style steps with a block cache.
// It does not call real kernels; it exercises the policy and prints reuse stats.

struct CacheBlock {
  int t0, t1;
  int hits = 0;
  int writes = 0;
};

int main(int argc, char** argv) {
  int steps = 256;
  int B_tok = 16;
  int K = 2;
  if (argc > 1) steps = std::stoi(argv[1]);
  if (argc > 2) B_tok = std::stoi(argv[2]);
  if (argc > 3) K = std::stoi(argv[3]);

  std::vector<CacheBlock> blocks;
  blocks.reserve((steps+B_tok-1)/B_tok);

  int commits = 0;
  for (int t = 0; t < steps; ++t) {
    int blk = t / B_tok;
    if ((int)blocks.size() <= blk) {
      blocks.push_back({blk*B_tok, blk*B_tok+B_tok, 0, 0});
    }
    // Simulate K branches reading this block
    blocks[blk].hits += K;
    // Occasionally write/update boundary stripes
    if ((t % B_tok) == 0) blocks[blk].writes++;

    // Every 8 tokens, pretend to validate & commit a prefix of length 4
    if ((t % 8) == 7) commits += 4;
  }

  int total_hits = 0, total_writes = 0;
  for (auto &b : blocks) { total_hits += b.hits; total_writes += b.writes; }
  std::printf("steps=%d B_tok=%d K=%d blocks=%zu hits=%d writes=%d commits=%d\n",
              steps, B_tok, K, blocks.size(), total_hits, total_writes, commits);
  return 0;
}
