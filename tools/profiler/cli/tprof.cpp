#include "tprof/tprof_runtime.h"
#include "tprof/x86_timing_provider.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <utility>

namespace {

using peaks_t = std::map<std::string, std::pair<double, double>>;

void usage() {
  std::cout << "tprof [--demo-out chrome.json] [--perfetto-out file.json] "
               "[--report-out file.html] [--peaks peaks.yaml] [--arch sm90]\n"
               "tprof peaks print --peaks peaks.yaml [--arch sm90]\n"
               "tprof x86 timing-status\n";
}

void trim(std::string& value) {
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) {
                return !std::isspace(ch);
              }));
  value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) {
                return !std::isspace(ch);
              }).base(),
              value.end());
}

double yaml_number(const std::string& value) {
  std::string normalized;
  normalized.reserve(value.size());
  for (char ch : value) {
    if (ch != '_' && ch != '"' && ch != '\'') {
      normalized.push_back(ch);
    }
  }
  try {
    return std::stod(normalized);
  } catch (...) {
    return 0.0;
  }
}

void parse_inline_mapping(const std::string& arch, const std::string& value, peaks_t& out) {
  auto read_field = [&](const char* field) {
    const auto field_pos = value.find(field);
    if (field_pos == std::string::npos) return 0.0;
    const auto colon_pos = value.find(':', field_pos);
    if (colon_pos == std::string::npos) return 0.0;
    const auto end_pos = value.find_first_of(",}", colon_pos + 1);
    return yaml_number(value.substr(colon_pos + 1, end_pos - (colon_pos + 1)));
  };

  out[arch] = {read_field("peak_flops"), read_field("hbm_gbs")};
}

bool parse_peaks_yaml(const std::string& path, peaks_t& out) {
  std::ifstream is(path);
  if (!is) return false;

  std::string line;
  std::string current_arch;
  while (std::getline(is, line)) {
    trim(line);
    if (line.empty() || line[0] == '#') continue;
    if (line == "devices:" || line == "devices: {}") {
      current_arch.clear();
      continue;
    }
    if (line.back() == ':') {
      current_arch = line.substr(0, line.size() - 1);
      out.emplace(current_arch, std::make_pair(0.0, 0.0));
      continue;
    }

    const auto pos = line.find(':');
    if (pos == std::string::npos) continue;

    std::string key = line.substr(0, pos);
    std::string value = line.substr(pos + 1);
    trim(key);
    trim(value);

    if (value.empty()) {
      current_arch = key;
      out.emplace(current_arch, std::make_pair(0.0, 0.0));
    } else if (value[0] == '{') {
      parse_inline_mapping(key, value, out);
    } else if (!current_arch.empty() && key == "peak_flops") {
      out[current_arch].first = yaml_number(value);
    } else if (!current_arch.empty() && key == "hbm_gbs") {
      out[current_arch].second = yaml_number(value);
    }
  }
  return !out.empty();
}

std::pair<std::string, std::pair<double, double>> select_peaks(
    const peaks_t& table, const char* arch) {
  if (arch != nullptr) {
    auto it = table.find(arch);
    if (it != table.end()) return *it;
  }
  if (const char* env_arch = std::getenv("TPROF_ARCH")) {
    auto it = table.find(env_arch);
    if (it != table.end()) return *it;
  }
  if (!table.empty()) return *table.begin();
  return {"unknown", {0.0, 0.0}};
}

std::string shell_quote(const std::string& value) {
  std::string quoted = "'";
  for (char ch : value) {
    if (ch == '\'') {
      quoted += "'\\''";
    } else {
      quoted += ch;
    }
  }
  quoted += "'";
  return quoted;
}

} // namespace

int main(int argc, char** argv) {
  if (argc >= 3 && std::strcmp(argv[1], "x86") == 0 &&
      std::strcmp(argv[2], "timing-status") == 0) {
    const auto caps = tprof::x86_timing_capabilities();
    tprof::x86_perf_group_t perf;
    const bool perf_open = perf.open();
    const bool perf_started = perf_open && perf.start();
    const auto start = tprof::x86_clock_snapshot();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const auto end = tprof::x86_clock_snapshot();
    const auto counters = perf_started ? perf.stop() : tprof::x86_perf_snapshot_t{};
    const bool affinity_stable = start.tsc_valid && end.tsc_valid &&
                                 start.logical_cpu == end.logical_cpu;
    const uint64_t raw_delta =
        end.monotonic_raw_ns >= start.monotonic_raw_ns
            ? end.monotonic_raw_ns - start.monotonic_raw_ns
            : 0;
    const uint64_t steady_delta = end.steady_ns >= start.steady_ns
                                      ? end.steady_ns - start.steady_ns
                                      : 0;
    const uint64_t tsc_delta = end.tsc_cycles >= start.tsc_cycles
                                   ? end.tsc_cycles - start.tsc_cycles
                                   : 0;
    const double tsc_hz = raw_delta > 0
                              ? static_cast<double>(tsc_delta) * 1.0e9 /
                                    static_cast<double>(raw_delta)
                              : 0.0;
    const uint64_t clock_difference = steady_delta > raw_delta
                                          ? steady_delta - raw_delta
                                          : raw_delta - steady_delta;
    const bool clock_agreement =
        raw_delta > 0 && steady_delta > 0 &&
        static_cast<double>(clock_difference) /
                static_cast<double>(raw_delta) <=
            0.05;
    std::cout << std::boolalpha << std::setprecision(17)
              << "{\n"
              << "  \"schema\": \"tessera.x86_timing_status.v1\",\n"
              << "  \"x86_64\": " << caps.x86_64 << ",\n"
              << "  \"avx512_visible\": " << caps.avx512f << ",\n"
              << "  \"monotonic_raw_valid\": "
              << (start.monotonic_raw_valid && end.monotonic_raw_valid) << ",\n"
              << "  \"rdtscp_valid\": "
              << (start.tsc_valid && end.tsc_valid) << ",\n"
              << "  \"invariant_tsc\": " << caps.invariant_tsc << ",\n"
              << "  \"affinity_stable\": " << affinity_stable << ",\n"
              << "  \"clock_agreement_valid\": " << clock_agreement << ",\n"
              << "  \"perf_event_open\": " << perf_open << ",\n"
              << "  \"perf_sample_valid\": " << counters.valid << ",\n"
              << "  \"perf_event_paranoid\": " << caps.perf_event_paranoid << ",\n"
              << "  \"perf_error\": "
              << (perf_open ? counters.error_code : perf.error_code()) << ",\n"
              << "  \"raw_interval_ns\": " << raw_delta << ",\n"
              << "  \"steady_interval_ns\": " << steady_delta << ",\n"
              << "  \"tsc_interval_cycles\": " << tsc_delta << ",\n"
              << "  \"calibrated_tsc_hz\": " << tsc_hz << ",\n"
              << "  \"perf_task_clock_ns\": " << counters.task_clock_ns << ",\n"
              << "  \"perf_time_enabled_ns\": " << counters.time_enabled_ns << ",\n"
              << "  \"perf_time_running_ns\": " << counters.time_running_ns << ",\n"
              << "  \"perf_multiplex_scale\": " << counters.multiplex_scale
              << ",\n"
              << "  \"perf_available_events_mask\": "
              << counters.available_events_mask << ",\n"
              << "  \"perf_unavailable_events_mask\": "
              << counters.unavailable_events_mask
              << "\n}\n";
    return 0;
  }

  if (argc >= 3 && std::strcmp(argv[1], "peaks") == 0 &&
      std::strcmp(argv[2], "print") == 0) {
    const char* peaks = nullptr;
    const char* arch = nullptr;
    for (int i = 3; i < argc; ++i) {
      if (std::strcmp(argv[i], "--peaks") == 0 && i + 1 < argc) peaks = argv[++i];
      else if (std::strcmp(argv[i], "--arch") == 0 && i + 1 < argc) arch = argv[++i];
    }
    if (peaks == nullptr) {
      std::cerr << "Usage: tprof peaks print --peaks peaks.yaml [--arch sm90]\n";
      return 1;
    }

    peaks_t table;
    if (!parse_peaks_yaml(peaks, table)) {
      std::cerr << "Failed to parse peaks file: " << peaks << "\n";
      return 2;
    }

    const auto chosen = select_peaks(table, arch);
    std::cout << "arch=" << chosen.first << " peak_flops=" << chosen.second.first
              << " hbm_gbs=" << chosen.second.second << "\n";
    return 0;
  }

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
      usage();
      return 0;
    }
  }

  if (out == nullptr && perfetto == nullptr && report == nullptr) {
    usage();
    return 1;
  }

  tprof::config_t cfg;
  cfg.mode = tprof::config_t::FAST;
  tprof::enable(cfg);
  {
    tprof::range_t r0("demo.kernel");
    tprof::runtime_api("tsrLaunchKernel", "{\"target\":\"demo\",\"kernel\":\"demo.kernel\"}");
    for (int i = 0; i < 3; ++i) {
      tprof::range_t r1("stage.compute");
      tprof::counter_add("bytes_hbm", 1e6);
      tprof::counter_add("flops", 2.0e9);
      tprof::marker("mbarrier.try_wait.parity");
      tprof::device_activity("demo.kernel.dispatch", 1000.0, "{\"queue\":\"demo\"}");
      tprof::intra_kernel_sample("warp_phase", static_cast<double>(i), "{\"phase\":\"compute\"}");
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  tprof::disable();

  if (out != nullptr && !tprof::export_chrome(out)) {
    std::cerr << "Failed to write: " << out << "\n";
    return 2;
  }
  if (perfetto != nullptr && !tprof::export_perfetto(perfetto)) {
    std::cerr << "Failed to write: " << perfetto << "\n";
    return 3;
  }
  if (report != nullptr) {
    std::string input = out != nullptr ? out : (perfetto != nullptr ? perfetto : "demo.trace.json");
    std::string cmd = "python3 tools/profiler/scripts/tprof_report.py --in " +
                      shell_quote(input) + " --out " + shell_quote(report);
    if (peaks != nullptr) cmd += " --peaks " + shell_quote(peaks);
    if (arch != nullptr) cmd += " --arch " + shell_quote(arch);
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
      std::cerr << "Report generation returned code " << rc << "\n";
      return 4;
    }
  }

  if (out != nullptr) std::cout << "Wrote Chrome trace: " << out << "\n";
  if (perfetto != nullptr) std::cout << "Wrote Perfetto trace: " << perfetto << "\n";
  if (report != nullptr) std::cout << "Wrote HTML report: " << report << "\n";
  return 0;
}
