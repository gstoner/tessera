\
#include "tprof/tprof_runtime.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <cctype>
#include <map>

static void usage() {
  std::cout << "tprof [--demo-out chrome.json] [--perfetto-out file.json] [--report-out file.html]"
               " [--peaks peaks.yaml] [--arch sm90]\n"
               "tprof peaks print --peaks peaks.yaml [--arch sm90]\n";
}

// Minimal YAML-ish parser: supports either top-level {arch:{peak_flops,hbm_gbs}} or 'devices:' root.
static bool parse_peaks_yaml(const std::string& path, std::map<std::string, std::pair<double,double>>& out) {
  std::ifstream is(path);
  if (!is) return false;
  std::string line;
  bool under_devices = false;
  std::string cur_arch;
  while (std::getline(is, line)) {
    // trim
    auto ltrim = [](std::string& s){ s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch){return !std::isspace(ch);})); };
    auto rtrim = [](std::string& s){ s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch){return !std::isspace(ch);}).base(), s.end()); };
    rtrim(line); ltrim(line);
    if (line.empty() || line[0] == '#') continue;
    if (line == "devices:" || line == "devices: {}") { under_devices = true; cur_arch.clear(); continue; }
    if (line.back() == ':' && line.find(':') == line.size()-1) {
      // section header (e.g., sm90:)
      cur_arch = line.substr(0, line.size()-1);
      out.emplace(cur_arch, std::make_pair(0.0, 0.0));
      continue;
    }
    // key: value inside a section
    auto pos = line.find(':');
    if (pos != std::string::npos) {
      std::string key = line.substr(0, pos);
      std::string val = line.substr(pos+1);
      rtrim(key); rtrim(val); ltrim(key); ltrim(val);
      auto to_double = [](const std::string& v)->double{
        std::string s; s.reserve(v.size());
        for (char c: v) if (c!='_' && c!='"' && c!='\'') s.push_back(c);
        try { return std::stod(s); } catch(...) { return 0.0; }
      };
      if (key == "peak_flops") {
        if (!cur_arch.empty()) out[cur_arch].first = to_double(val);
      } else if (key == "hbm_gbs") {
        if (!cur_arch.empty()) out[cur_arch].second = to_double(val);
      } else if (key == "sm" || key == "arch") {
        // ignore
      } else {
        // Could be "sm90: { peak_flops: ..., hbm_gbs: ... }" single-line mapping
        // Try to detect arch value in "key: { ... }" when current arch is empty.
        if (val.size() && val[0] == '{') {
          std::string arch = key;
          double pf=0.0, hb=0.0;
          auto pfpos = val.find("peak_flops");
          if (pfpos != std::string::npos) {
            auto cpos = val.find(':', pfpos);
            if (cpos != std::string::npos) {
              auto end = val.find_first_of(",}", cpos+1);
              pf = to_double(val.substr(cpos+1, end-(cpos+1)));
            }
          }
          auto hbpos = val.find("hbm_gbs");
          if (hbpos != std::string::npos) {
            auto cpos = val.find(':', hbpos);
            if (cpos != std::string::npos) {
              auto end = val.find_first_of(",}", cpos+1);
              hb = to_double(val.substr(cpos+1, end-(cpos+1)));
            }
          }
          out[arch] = std::make_pair(pf, hb);
        }
      }
    }
  }
  return !out.empty();
}

int main(int argc, char** argv) {
  if (argc >= 3 && std::strcmp(argv[1], "peaks") == 0 && std::strcmp(argv[2], "print") == 0) {
    const char* peaks = nullptr;
    const char* arch = nullptr;
    for (int i = 3; i < argc; ++i) {
      if (std::strcmp(argv[i], "--peaks") == 0 && i + 1 < argc) peaks = argv[++i];
      else if (std::strcmp(argv[i], "--arch") == 0 && i + 1 < argc) arch = argv[++i];
    }
    if (!peaks) {
      std::cerr << "Usage: tprof peaks print --peaks peaks.yaml [--arch sm90]\n";
      return 1;
    }
    std::map<std::string, std::pair<double,double>> table;
    if (!parse_peaks_yaml(peaks, table)) {
      std::cerr << "Failed to parse peaks file: " << peaks << "\n";
      return 2;
    }
    std::pair<double,double> chosen{0.0,0.0};
    std::string chosen_key;
    if (arch && table.count(arch)) {
      chosen = table[arch]; chosen_key = arch;
    } else if (const char* env = std::getenv("TPROF_ARCH")) {
      std::string k = env;
      if (table.count(k)) { chosen = table[k]; chosen_key = k; }
    }
    if (chosen_key.empty()) {
      // pick first non-zero or first entry
      for (auto& kv : table) { chosen_key = kv.first; chosen = kv.second; break; }
    }
    std::cout << "arch=" << (chosen_key.empty() ? "unknown" : chosen_key)
              << " peak_flops=" << chosen.first
              << " hbm_gbs=" << chosen.second << "\n";
    return 0;
  }

  // Normal CLI path
  const char* out = nullptr;
  const char* perfetto = nullptr;
  const char* report = nullptr;
  const char* peaks = nullptr;
  const char* arch = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--demo-out") == 0 && i + 1 < argc) out = argv[++i];
    else if (std::strcmp(argv[i], "--perfetto-out") == 0 && i + 1 < argc) perfetto = argv[++i];
    else if (std::strcmp(argv[i], "--report-out") == 0 && i + 1 < argc) report = argv[++i];
    else if (std::strcmp(argv[i], "--peaks") == 0 && i + 1 < argc) peaks = argv[++i];
    else if (std::strcmp(argv[i], "--arch") == 0 && i + 1 < argc) arch = argv[++i];
    else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
      usage(); return 0;
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
    std::string input = out ? out : (perfetto ? perfetto : "demo.trace.json");
    std::string cmd = std::string("python3 tools/profiler/scripts/tprof_report.py --in ") + input + " --out " + report;
    if (peaks) { cmd += std::string(" --peaks ") + peaks; }
    if (arch) { cmd += std::string(" --arch ") + arch; }
    int rc = std::system(cmd.c_str());
    if (rc != 0) std::cerr << "Report generation returned code " << rc << "\n";
  }

  if (out) std::cout << "Wrote Chrome trace: " << out << "\n";
  if (perfetto) std::cout << "Wrote Perfetto trace: " << perfetto << "\n";
  if (report) std::cout << "Wrote HTML report: " << report << "\n";
  return 0;
}
