\
#ifndef TESSERA_TARGET_METALIUM_CODEGEN_H
#define TESSERA_TARGET_METALIUM_CODEGEN_H

#include <string>
#include <vector>
#include <optional>

namespace mlir {
class ModuleOp; // fwd decl
}

namespace tessera_metalium_shim {

struct Kernel {
  std::string name;
  std::string coreRange;   // e.g. "[0,0]-[7,11]"
  std::string ir;          // textual IR snippet or JSON for the kernel
};

struct Program {
  std::vector<Kernel> kernels;
};

struct Queue {
  std::vector<std::string> commands;
};

/// Emit a mock Program from an MLIR Module (walk tessera_metalium ops and serialize).
Program emitProgramFromModule(/*mlir::ModuleOp*/ void *moduleOpaque);

/// Enqueue a kernel launch. Returns a handle string in this mock.
std::string enqueue(Queue &q, const Kernel &k);

/// Serialize the Program as JSON (for demo/debug).
std::string toJson(const Program &p);

} // namespace tessera_metalium_shim

#endif // TESSERA_TARGET_METALIUM_CODEGEN_H
