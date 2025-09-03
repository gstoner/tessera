#include "loader.h"
#include <iostream>
using namespace tessera::rocm;
int main(int argc, char** argv){
  if (argc<3){ std::cerr << "usage: launch_demo <hsaco> <kernel>\n"; return 1; }
  Loader L; if(!L.loadFile(argv[1])) return 2; if(!L.getKernel(argv[2])) return 3;
  std::vector<KernelArg> args; if(!L.launch(argv[2], dim3{256,1,1}, dim3{256,1,1}, args)) return 4;
  std::cout << "OK\n"; return 0;
}
