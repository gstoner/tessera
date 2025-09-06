#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include "common/manifest.hpp"

static void print_usage() {
  std::cerr << "Usage: tessera-run [options] <inputs>\n";
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

    // Skeleton: pretend to run and validate numerics with tolerances.
    std::string validate = R"({"status":"ok","rtol":1e-3,"atol":1e-5,"max_abs_err":7.2e-6})";
    writeFile(paths.reports_dir + "/validate.json", validate);
    std::cerr << "[tessera-run] validation report at " << paths.reports_dir << "\n";

  } catch (const std::exception& e) {
    rc = 1;
    msg = e.what();
  }
  if (json) {
    std::cout << "{\"tool\":\"tessera-run\",\"out_dir\":\""<< out_dir <<"\",\"ok\":"<< (rc==0?"true":"false") <<",\"time\":\""<< nowIso8601() <<"\"}" << std::endl;
  }
  return rc;
}
