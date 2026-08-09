#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "tessera/Dialect/Collective/IR/CollectiveDialect.h"
#include "tessera/Dialect/Collective/IR/CollectivePasses.h"
using namespace mlir;
int main(int argc, char **argv){
  DialectRegistry R;
  registerAllDialects(R);
  registerAllPasses();
  tessera::collective::registerCollectiveDialect(R);
  tessera::collective::registerCollectivePasses();
  return failed(MlirOptMain(argc, argv, "Tessera collective opt\n", R));
}
