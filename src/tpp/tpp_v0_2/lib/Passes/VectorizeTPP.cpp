
#include "mlir/Pass/Pass.h"
using namespace mlir;
namespace {
struct VectorizeTPP : public PassWrapper<VectorizeTPP, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "tpp-vectorize"; }
  StringRef getDescription() const final { return "Vectorize/Tile mapping (stub)"; }
  void runOnOperation() final {}
};
}
std::unique_ptr<Pass> createVectorizeTPPPass(){ return std::make_unique<VectorizeTPP>(); }
