\
#include "Tessera/Target/Metalium/MetaliumCodegen.h"
#include <sstream>

using namespace tessera_metalium_shim;

Program emitProgramFromModule(/*mlir::ModuleOp*/ void *moduleOpaque) {
  // This is a stub: in a real integration you would walk the MLIR module,
  // find tessera_metalium.{dma,load2d,store2d,matmul} ops, and group them into kernels.
  Program p;
  Kernel k;
  k.name = "demo_kernel";
  k.coreRange = "[0,0]-[7,11]";
  k.ir = "tessera_metalium.matmul(tile=[64,64,32]) \\n tessera_metalium.dma(...)";
  p.kernels.push_back(k);
  return p;
}

std::string enqueue(Queue &q, const Kernel &k) {
  std::ostringstream oss;
  oss << "LAUNCH " << k.name << " on " << k.coreRange;
  q.commands.push_back(oss.str());
  return oss.str();
}

std::string toJson(const Program &p) {
  std::ostringstream oss;
  oss << "{\\n  \\\"kernels\\\": [\\n";
  for (size_t i = 0; i < p.kernels.size(); ++i) {
    const auto &k = p.kernels[i];
    oss << "    {\\n";
    oss << "      \\\"name\\\": \\\"" << k.name << "\\\",\\n";
    oss << "      \\\"coreRange\\\": \\\"" << k.coreRange << "\\\",\\n";
    oss << "      \\\"ir\\\": \\\"" << k.ir << "\\\"\\n";
    oss << "    }" << (i + 1 == p.kernels.size() ? "" : ",") << "\\n";
  }
  oss << "  ]\\n}\\n";
  return oss.str();
}
