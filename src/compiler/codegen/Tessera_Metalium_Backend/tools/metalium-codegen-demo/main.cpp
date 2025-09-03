
#include "Tessera/Target/Metalium/MetaliumBufferPlanner.h"
#include <iostream>
using namespace tessera_metalium_planner;

static void printPlan(const AttentionTilePlan &P) {
  std::cout << "Attention Tile Plan: M="<<P.tileM<<" N="<<P.tileN<<" K="<<P.tileK<<"\n";
  std::cout << "  Q SRAM bytes: " << P.qTile.sizeBytes << ", stride: " << P.qTile.strideBytes << "\n";
  std::cout << "  K SRAM bytes: " << P.kTile.sizeBytes << ", stride: " << P.kTile.strideBytes << "\n";
  std::cout << "  V SRAM bytes: " << P.vTile.sizeBytes << ", stride: " << P.vTile.strideBytes << "\n";
  std::cout << "  O SRAM bytes: " << P.oTile.sizeBytes << ", stride: " << P.oTile.strideBytes << "\n";
}

static void printKV(const KVCachePlan &K) {
  std::cout << "KV Cache Plan: tileSeq="<<K.tileSeq<<"\n";
  std::cout << "  K DRAM bytes: " << K.kDRAM.sizeBytes << " stride: " << K.kDRAM.strideBytes << "\n";
  std::cout << "  V DRAM bytes: " << K.vDRAM.sizeBytes << " stride: " << K.vDRAM.strideBytes << "\n";
  std::cout << "  K SRAM tile bytes: " << K.kTileSRAM.sizeBytes << "\n";
  std::cout << "  V SRAM tile bytes: " << K.vTileSRAM.sizeBytes << "\n";
}

int main(); // keep existing
int main() {
  // Existing demo prints program JSON + enqueues; now add planner samples.
  auto program = tessera_metalium_shim::emitProgramFromModule(nullptr);
  auto json = tessera_metalium_shim::toJson(program);
  std::cout << "Program JSON:\n" << json << std::endl;

  tessera_metalium_shim::Queue q;
  for (auto &k : program.kernels) {
    auto handle = tessera_metalium_shim::enqueue(q, k);
    std::cout << "Enqueued: " << handle << std::endl;
  }
  std::cout << "Queue size: " << q.commands.size() << std::endl;

  // Planner demos
  auto att = planAttentionTiles(/*headDim=*/128, /*M=*/64, /*N=*/64, /*K=*/32,
                                /*elemBytes=*/2 /*bf16*/, /*sramBudget=*/(64*1024));
  printPlan(att);

  auto kv = planKVCache(/*maxSeq=*/8192, /*headDim=*/128, /*elemBytes=*/2, /*sramBudget=*/(64*1024));
  printKV(kv);
  return 0;
}
