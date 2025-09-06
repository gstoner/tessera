#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include "common/manifest.hpp"

static void print_usage() {
  std::cerr << "Usage: tessera-compile [options] <inputs>\n";
}

int main(int argc, char** argv) {
  std::string out_dir = "out";
  bool json = false;
  std::vector<std::string> inputs;

  for (int i=1;i<argc;++i) {
    std::string a = argv[i];
    if (a == "--out-dir" && i+1<argc) { out_dir = argv[++i]; }
    else if (a == "--json") { json = true; }
    else if (a == "-h" || a == "--help") { print_usage(); return 0; }
    else if (a.rfind("-",0)==0) { /* ignore unknown for skeleton */ }
    else inputs.push_back(a);
  }
  auto paths = makeArtifactLayout(out_dir);
  int rc = 0;
  std::string msg;
  try {

    // Skeleton: pretend to compile and emit manifest + a dummy kernel.
    std::string kernel = "// .ptx (skeleton)\n// entry: demo_kernel\n";
    writeFile(paths.kernels_dir + "/demo.ptx", kernel);
    std::string host = R"(// host stub (skeleton)
extern "C" int demo_launch() { return 0; }
)";
    writeFile(paths.host_dir + "/launch.cu", host);
    std::string manifest = R"({
  "tessera": { "version": "0.3.1" },
  "artifacts": {
    "kernels": ["kernels/demo.ptx"],
    "host": ["host/launch.cu"]
  }
})";
    writeFile(paths.meta_dir + "/compile.json", manifest);
    std::cerr << "[tessera-compile] artifacts at " << paths.out_dir << "\n";

  } catch (const std::exception& e) {
    rc = 1;
    msg = e.what();
  }
  if (json) {
    std::cout << "{\"tool\":\"tessera-compile\",\"out_dir\":\""<< out_dir <<"\",\"ok\":"<< (rc==0?"true":"false") <<",\"time\":\""<< nowIso8601() <<"\"}" << std::endl;
  }
  return rc;
}
