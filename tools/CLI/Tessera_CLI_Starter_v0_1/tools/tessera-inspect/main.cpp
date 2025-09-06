#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include "common/manifest.hpp"

static void print_usage() {
  std::cerr << "Usage: tessera-inspect [options] <inputs>\n";
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

    // Skeleton: read kernel name(s) and emit a one-row table in markdown & JSON.
    std::string md = "| kernel | regs | smem_kb | occupancy | size_bytes |\n"
                     "|--------|------|---------|-----------|------------|\n"
                     "| demo_kernel | 128 | 48 | 0.62 | 4096 |\n";
    writeFile(paths.reports_dir + "/inspect.md", md);
    std::string js = R"({"kernels":[{"name":"demo_kernel","regs":128,"smem_kb":48,"occupancy":0.62,"size_bytes":4096}]})";
    writeFile(paths.reports_dir + "/inspect.json", js);
    std::cerr << "[tessera-inspect] wrote inspect.{md,json}\n";

  } catch (const std::exception& e) {
    rc = 1;
    msg = e.what();
  }
  if (json) {
    std::cout << "{\"tool\":\"tessera-inspect\",\"out_dir\":\""<< out_dir <<"\",\"ok\":"<< (rc==0?"true":"false") <<",\"time\":\""<< nowIso8601() <<"\"}" << std::endl;
  }
  return rc;
}
