#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include "common/manifest.hpp"

static void print_usage() {
  std::cerr << "Usage: tessera-profiler [options] <inputs>\n";
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

    // Skeleton: generate roofline.csv + perfetto.json + HTML report.
    std::string csv = "kernel,flops,bytes,time_ms,ai,achieved_tflops,achieved_gbps\n"
                      "demo_kernel,6.4e12,1.2e9,0.83,5.33,7.7,950\n";
    writeFile(paths.reports_dir + "/roofline.csv", csv);
    std::string perfetto = R"({"traceEvents":[{"ph":"X","name":"demo_kernel","ts":0,"dur":830000}],"displayTimeUnit":"ns"})";
    writeFile(paths.reports_dir + "/perfetto.json", perfetto);
    std::string html = R"(<html><head><meta charset="utf-8"><title>Tessera Roofline</title></head>
<body><h1>Tessera Roofline (skeleton)</h1>
<p>See <code>roofline.csv</code> and load <code>perfetto.json</code> into Perfetto UI.</p>
</body></html>)";
    writeFile(paths.reports_dir + "/roofline.html", html);
    std::cerr << "[tessera-profiler] reports at " << paths.reports_dir << "\n";

  } catch (const std::exception& e) {
    rc = 1;
    msg = e.what();
  }
  if (json) {
    std::cout << "{\"tool\":\"tessera-profiler\",\"out_dir\":\""<< out_dir <<"\",\"ok\":"<< (rc==0?"true":"false") <<",\"time\":\""<< nowIso8601() <<"\"}" << std::endl;
  }
  return rc;
}
