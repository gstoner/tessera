#ifndef TESSERA_TARGET_METALIUM_BUFFER_PLANNER_H
#define TESSERA_TARGET_METALIUM_BUFFER_PLANNER_H

#include <cstdint>
#include <string>
#include <vector>
#include <optional>

namespace tessera_metalium_planner {

enum class Space { DRAM, SRAM };

struct BufferBinding {
  std::string name;
  Space space;
  int64_t offsetBytes = 0; // within its space's arena
  int64_t sizeBytes   = 0;
  int64_t strideBytes = 0; // row stride for 2D tiles
};

struct DMADescriptor {
  std::string name;
  std::string direction;  // "dram_to_sram", "sram_to_dram"
  int64_t rows = 0, cols = 0;
  int64_t srcRowStride = 0, srcColStride = 0;
  int64_t dstRowStride = 0, dstColStride = 0;
  int64_t elementBytes = 0;
  int64_t burst = 256;
};

struct AttentionTilePlan {
  // Buffers allocated in SRAM for the tiles per-core
  BufferBinding qTile, kTile, vTile, oTile;
  // DMA descriptors to move tiles in/out
  DMADescriptor qLoad, kLoad, vLoad, oStore;
  // Tile shape (M,N,K) for matmul
  int64_t tileM = 64, tileN = 64, tileK = 32;
};

struct KVCachePlan {
  BufferBinding kDRAM, vDRAM;  // persistent in DRAM (full sequence)
  BufferBinding kTileSRAM, vTileSRAM; // sliding tiles in SRAM
  DMADescriptor kLoad, vLoad;
  int64_t tileSeq = 256; // how many tokens per tile
};

/// Simple byte-size helper (does not need MLIR; caller passes element width)
inline int64_t bytesForMatrix(int64_t rows, int64_t cols, int64_t elemBytes, int64_t rowStride = 0) {
  if (rowStride == 0) rowStride = cols * elemBytes;
  return rows * rowStride;
}

/// Plan SRAM tile buffers for attention (Q,K,V,O) under a byte budget.
AttentionTilePlan planAttentionTiles(int64_t headDim,
                                     int64_t tileM, int64_t tileN, int64_t tileK,
                                     int64_t elemBytes, int64_t sramBudgetBytes);

/// Plan KV-cache DRAM buffers and SRAM staging tiles.
KVCachePlan planKVCache(int64_t maxSeq, int64_t headDim, int64_t elemBytes,
                        int64_t sramBudgetBytes);

} // namespace tessera_metalium_planner

#endif // TESSERA_TARGET_METALIUM_BUFFER_PLANNER_H
