#include "tprof/tprof_runtime.h"
#include <iostream>
#include <cstring>

static void usage() {
  std::cout << "tprof --demo-out <file.json>\n";
}

int main(int argc, char** argv) {
  const char* out = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--demo-out") == 0 && i + 1 < argc) {
      out = argv[++i];
    } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
      usage();
      return 0;
    }
  }
  if (!out) {
    usage();
    return 1;
  }

  tprof::Config cfg;
  cfg.mode = tprof::Config::FAST;
  tprof::enable(cfg);

  {
    tprof::Range r0("demo.kernel");
    for (int i = 0; i < 3; ++i) {
      tprof::Range r1("stage.compute");
      tprof::counter_add("bytes_hbm", 1e6);
      tprof::marker("mbarrier.try_wait.parity");
      // simulate work
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  if (!tprof::export_chrome(out)) {
    std::cerr << "Failed to write: " << out << "\n";
    return 2;
  }

  std::cout << "Wrote Chrome trace: " << out << "\n";
  return 0;
}
