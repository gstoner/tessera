// passes/tprof_markers_stub.cpp
#include <string>
#include <vector>
#include <iostream>
namespace tprof_mlir {
struct op_t { std::string name; };
void apply_stub(std::vector<op_t>& ops) {
  std::cerr << "[tprof_mlir] stub injected markers for " << ops.size() << " ops\n";
}
} // namespace tprof_mlir
