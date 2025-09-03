#include "Tessera/Target/Metalium/MetaliumBufferPlanner.h"
#include <algorithm>

using namespace tessera_metalium_planner;

AttentionTilePlan planAttentionTiles(int64_t headDim,
                                     int64_t tileM, int64_t tileN, int64_t tileK,
                                     int64_t elemBytes, int64_t sramBudgetBytes) {
  AttentionTilePlan P;
  P.tileM = tileM; P.tileN = tileN; P.tileK = tileK;

  // Allocate SRAM tiles: Q[MxK], K[KxN], V[NxK] modeled in row-major bytes.
  int64_t qBytes = bytesForMatrix(tileM, tileK, elemBytes);
  int64_t kBytes = bytesForMatrix(tileK, tileN, elemBytes);
  int64_t vBytes = bytesForMatrix(tileN, tileK, elemBytes);
  int64_t oBytes = bytesForMatrix(tileM, tileN, elemBytes);

  int64_t total = qBytes + kBytes + vBytes + oBytes;
  // Simple shrink loop if over budget: reduce tileN first, then K.
  while (total > sramBudgetBytes && (P.tileN > 16 || P.tileK > 16)) {
    if (P.tileN > 16) P.tileN >>= 1; else P.tileK >>= 1;
    qBytes = bytesForMatrix(P.tileM, P.tileK, elemBytes);
    kBytes = bytesForMatrix(P.tileK, P.tileN, elemBytes);
    vBytes = bytesForMatrix(P.tileN, P.tileK, elemBytes);
    oBytes = bytesForMatrix(P.tileM, P.tileN, elemBytes);
    total = qBytes + kBytes + vBytes + oBytes;
  }

  int64_t cursor = 0;
  P.qTile = {"Q_sram", Space::SRAM, cursor, qBytes, P.tileK * elemBytes}; cursor += qBytes;
  P.kTile = {"K_sram", Space::SRAM, cursor, kBytes, P.tileN * elemBytes}; cursor += kBytes;
  P.vTile = {"V_sram", Space::SRAM, cursor, vBytes, P.tileK * elemBytes}; cursor += vBytes;
  P.oTile = {"O_sram", Space::SRAM, cursor, oBytes, P.tileN * elemBytes}; cursor += oBytes;

  // DMA descriptors (dram→sram for Q,K,V; sram→dram for O)
  P.qLoad = {"Q_load", "dram_to_sram", P.tileM, P.tileK,
             headDim * elemBytes, elemBytes,
             P.qTile.strideBytes, elemBytes, elemBytes, 256};
  P.kLoad = {"K_load", "dram_to_sram", P.tileK, P.tileN,
             elemBytes * P.tileN, elemBytes, P.kTile.strideBytes, elemBytes, elemBytes, 256};
  P.vLoad = {"V_load", "dram_to_sram", P.tileN, P.tileK,
             headDim * elemBytes, elemBytes, P.vTile.strideBytes, elemBytes, elemBytes, 256};
  P.oStore= {"O_store","sram_to_dram", P.tileM, P.tileN,
             P.oTile.strideBytes, elemBytes, headDim * elemBytes, elemBytes, elemBytes, 256};

  return P;
}

KVCachePlan planKVCache(int64_t maxSeq, int64_t headDim, int64_t elemBytes,
                        int64_t sramBudgetBytes) {
  KVCachePlan K;
  // DRAM buffers for full sequence per head (layout e.g., [Seq, HeadDim]).
  int64_t rowStride = headDim * elemBytes;
  int64_t totalBytes = bytesForMatrix(maxSeq, headDim, elemBytes, rowStride);
  K.kDRAM = {"K_dram", Space::DRAM, 0, totalBytes, rowStride};
  K.vDRAM = {"V_dram", Space::DRAM, totalBytes, totalBytes, rowStride};

  // Choose a tileSeq that fits two SRAM staging tiles (K and V)
  int64_t tileSeq = std::min<int64_t>(maxSeq, std::max<int64_t>(64, sramBudgetBytes / (2 * rowStride)));
  // Round to a power-of-two chunk for nicer DMA bursts
  int64_t p2 = 1; while (p2 * 2 <= tileSeq) p2 <<= 1; tileSeq = p2;
  K.tileSeq = tileSeq;

  int64_t kTileBytes = bytesForMatrix(tileSeq, headDim, elemBytes, rowStride);
  int64_t vTileBytes = kTileBytes;
  int64_t cursor = 0;
  K.kTileSRAM = {"K_sram_tile", Space::SRAM, cursor, kTileBytes, rowStride}; cursor += kTileBytes;
  K.vTileSRAM = {"V_sram_tile", Space::SRAM, cursor, vTileBytes, rowStride};

  // DMA descriptors to load K/V segments into SRAM tiles
  K.kLoad = {"K_cache_load", "dram_to_sram", tileSeq, headDim,
             rowStride, elemBytes, rowStride, elemBytes, elemBytes, 256};
  K.vLoad = {"V_cache_load", "dram_to_sram", tileSeq, headDim,
             rowStride, elemBytes, rowStride, elemBytes, elemBytes, 256};

  return K;
}
