#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/MlirOptMain.h"
using namespace mlir;
int main(int argc, char **argv){
  DialectRegistry R; registerAllDialects(R); registerAllPasses();
  return failed(MlirOptMain(argc, argv, "Tessera collective opt\n", R));
}
