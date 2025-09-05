#pragma once
#include <cstdint>
#include <string>
#include <cstdlib>

namespace tessera { namespace collective {

enum class Algo { Auto, Ring, Tree, Hier };
enum class Path { Auto, NVLINK, PCIE, RDMA };

struct Topology {
  bool hasNVLink = true;
  bool hasRDMA   = false;
  double bwNVLinkGBs = 300.0;
  double bwPCIeGBs   = 32.0;
  double bwRDMA_GBs  = 100.0;
};

struct Policy {
  uint64_t smallMax = 256ull << 10;
  uint64_t largeMin = 8ull   << 20;
  Algo smallAlgo = Algo::Tree;
  Algo midAlgo   = Algo::Tree;
  Algo largeAlgo = Algo::Ring;
  Path intraPath = Path::NVLINK;
  Path interPath = Path::RDMA;

  static Policy fromEnv() {
    Policy p;
    if (const char* s = std::getenv("TESSERA_COMM_SMALL_MAX")) p.smallMax = std::strtoull(s,nullptr,10);
    if (const char* s = std::getenv("TESSERA_COMM_LARGE_MIN")) p.largeMin = std::strtoull(s,nullptr,10);
    if (const char* s = std::getenv("TESSERA_COMM_INTRA_PATH")) {
      std::string v=s; if (v=="pcie") p.intraPath=Path::PCIE; else if (v=="nvlink") p.intraPath=Path::NVLINK;
    }
    if (const char* s = std::getenv("TESSERA_COMM_INTER_PATH")) {
      std::string v=s; if (v=="rdma") p.interPath=Path::RDMA; else if (v=="pcie") p.interPath=Path::PCIE;
    }
    return p;
  }

  Algo chooseAlgo(uint64_t bytes) const {
    if (bytes >= largeMin) return largeAlgo;
    if (bytes <= smallMax) return smallAlgo;
    return midAlgo;
  }
  Path choosePath(bool intraNode, const Topology& topo) const {
    if (intraNode) return topo.hasNVLink ? intraPath : Path::PCIE;
    return topo.hasRDMA ? interPath : Path::PCIE;
  }
};

}} // ns
