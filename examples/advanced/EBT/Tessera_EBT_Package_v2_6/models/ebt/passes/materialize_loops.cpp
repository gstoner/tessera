#include "materialize_loops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
using namespace mlir;

namespace tessera { namespace ebt {
struct MaterializeLoopsPass : PassWrapper<MaterializeLoopsPass, OperationPass<ModuleOp>> {
  int K=4, T=4;
  MaterializeLoopsPass() = default;
  MaterializeLoopsPass(MaterializeLoopsOptions o){ K=o.K; T=o.T; }
  void getDependentDialects(DialectRegistry &r) const override { r.insert<scf::SCFDialect>(); }
  void runOnOperation() override {
    ModuleOp m = getOperation();
    // Very small demo: find a func named @driver and wrap its body in scf.for K,T markers.
    func::FuncOp f = m.lookupSymbol<func::FuncOp>("driver");
    if (!f) return;
    OpBuilder b(f.getBody());
    auto loc = f.getLoc();
    Value k0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value kN = b.create<arith::ConstantIndexOp>(loc, K);
    Value t0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value tN = b.create<arith::ConstantIndexOp>(loc, T);
    // Create loops before first op
    Operation *first = &f.getBody().front().front();
    b.setInsertionPoint(first);
    auto kLoop = b.create<scf::ForOp>(loc, k0, kN, 1);
    b.setInsertionPointToStart(kLoop.getBody());
    auto tLoop = b.create<scf::ForOp>(loc, t0, tN, 1);
    // Move original ops into inner loop (best-effort demo)
    while (&f.getBody().front().front() != tLoop) {
      Operation *op = &f.getBody().front().front();
      op->moveBefore(&tLoop.getBody()->front());
    }
    kLoop->setAttr("mapping", b.getStringAttr("candidates"));
    tLoop->setAttr("mapping", b.getStringAttr("steps"));
  }
};

std::unique_ptr<mlir::Pass> createMaterializeLoopsPass(MaterializeLoopsOptions o) {
  return std::make_unique<MaterializeLoopsPass>(o);
}
void registerMaterializeLoopsPipeline() {
  PassPipelineRegistration<MaterializeLoopsOptions>(
    "tessera-ebt-materialize-loops","Materialize KxT loops",
    [](OpPassManager& pm, MaterializeLoopsOptions const& o){
      pm.addPass(createMaterializeLoopsPass(o));
    });
}
}} // ns
