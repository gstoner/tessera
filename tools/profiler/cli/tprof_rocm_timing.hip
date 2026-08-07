#include "tprof/rocm_timing_provider.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

std::string json_escape(const std::string& value) {
  std::string escaped;
  for (char ch : value) {
    if (ch == '"' || ch == '\\') escaped.push_back('\\');
    escaped.push_back(ch);
  }
  return escaped;
}

}  // namespace

int main(int argc, char** argv) {
  tprof::rocm_timing_probe_config_t config;
  for (int index = 1; index < argc; ++index) {
    if (std::string(argv[index]) == "--repetitions" && index + 1 < argc)
      config.repetitions = std::atoi(argv[++index]);
    else if (std::string(argv[index]) == "--spin-iterations" &&
             index + 1 < argc)
      config.spin_iterations = std::atoi(argv[++index]);
  }
  const auto result = tprof::collect_rocm_timing_probe(config);
  std::cout << std::boolalpha << std::setprecision(17)
            << "{\n"
            << "  \"schema\": \"tessera.rocm_clock_probe.v1\",\n"
            << "  \"architecture\": \"" << json_escape(result.architecture)
            << "\",\n"
            << "  \"device_name\": \"" << json_escape(result.device_name)
            << "\",\n"
            << "  \"error\": \"" << json_escape(result.error) << "\",\n"
            << "  \"exact_gfx1151\": " << result.exact_gfx1151 << ",\n"
            << "  \"wsl\": " << result.wsl << ",\n"
            << "  \"repetitions\": " << result.repetitions << ",\n"
            << "  \"blocks\": " << result.blocks << ",\n"
            << "  \"threads\": " << result.threads << ",\n"
            << "  \"host_wall_ns\": " << result.host_wall_ns << ",\n"
            << "  \"empty_host_wall_ns\": " << result.empty_host_wall_ns
            << ",\n"
            << "  \"hip_event_ms\": " << result.hip_event_ms << ",\n"
            << "  \"empty_hip_event_ms\": " << result.empty_hip_event_ms
            << ",\n"
            << "  \"device_wall_clock_ticks\": "
            << result.device_wall_clock_ticks << ",\n"
            << "  \"device_wall_clock_ns\": "
            << result.device_wall_clock_ns << ",\n"
            << "  \"wall_clock_rate_khz\": "
            << result.wall_clock_rate_khz << ",\n"
            << "  \"host_wall_valid\": " << result.host_wall_valid << ",\n"
            << "  \"hip_event_valid\": " << result.hip_event_valid << ",\n"
            << "  \"device_wall_clock_valid\": "
            << result.device_wall_clock_valid << ",\n"
            << "  \"clock_agreement_valid\": "
            << result.clock_agreement_valid << ",\n"
            << "  \"eligible_for_regression\": "
            << result.eligible_for_regression << ",\n"
            << "  \"eligible_for_promotion\": "
            << result.eligible_for_promotion << "\n}\n";
  return result.error.empty() ? 0 : 1;
}
