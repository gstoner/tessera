#include "tprof/tprof_runtime.h"
#include <iostream>
#include <cstring>

static void usage() {
  std::cout << "tprof --demo-out <file.json> [--perfetto-out file.json] [--report-out file.html]\n";
}

int main(int argc, char** argv) {
  const char* out = nullptr;
  const char* perfetto = nullptr;
  const char* report = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--demo-out") == 0 && i + 1 < argc) {
      out = argv[++i];
    } else if (std::strcmp(argv[i], "--perfetto-out") == 0 && i + 1 < argc) {
      perfetto = argv[++i];
    } else if (std::strcmp(argv[i], "--report-out") == 0 && i + 1 < argc) {
      report = argv[++i];
    } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
      usage();
      return 0;
    }
  }
  if (!out && !perfetto && !report) { usage(); return 1; }

  tprof::config_t cfg; cfg.mode = tprof::config_t::FAST;
  tprof::enable(cfg);

  {
    tprof::range_t r0("demo.kernel");
    for (int i = 0; i < 3; ++i) {
      tprof::range_t r1("stage.compute");
      tprof::counter_add("bytes_hbm", 1e6);
      tprof::marker("mbarrier.try_wait.parity");
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  if (out && !tprof::export_chrome(out)) {
    std::cerr << "Failed to write: " << out << "\n";
    return 2;
  }
  if (perfetto && !tprof::export_perfetto(perfetto)) {
    std::cerr << "Failed to write: " << perfetto << "\n";
    return 3;
  }
  if (report) {
    // Defer to Python report script if present.
    std::string input = out ? out : (perfetto ? perfetto : "demo.trace.json");
    std::string cmd = std::string("python3 tools/profiler/scripts/tprof_report.py --in ") + input + " --out " + report + " --peak-flops 2.0e14 --hbm-gbs 3000";
    int rc = std::system(cmd.c_str());
    if (rc != 0) std::cerr << "Report generation returned code " << rc << "\n";
  }

  if (out) std::cout << "Wrote Chrome trace: " << out << "\n";
  if (perfetto) std::cout << "Wrote Perfetto trace: " << perfetto << "\n";
  if (report) std::cout << "Wrote HTML report: " << report << "\n";
  return 0;
}
