#include "tprof/tprof_runtime.h"
#include <fstream>
namespace tprof {
bool chrome_export(const std::vector<event_t>& events, const std::string& path) {
  std::ofstream os(path, std::ios::out | std::ios::trunc);
  if (!os) return false;
  os << "{\n  \"displayTimeUnit\": \"ns\",\n  \"traceEvents\": [\n";
  bool first = true;
  for (const auto& e : events) {
    if (!first) os << ",\n";
    first = false;
    switch (e.type) {
      case event_t::RANGE_B:
        os << "    { \"name\": \"" << (e.name ? e.name : "range")
           << "\", \"ph\": \"B\", \"ts\": " << (e.ts_ns / 1000.0)
           << ", \"pid\": 0, \"tid\": " << e.tid << " }";
        break;
      case event_t::RANGE_E:
        os << "    { \"ph\": \"E\", \"ts\": " << (e.ts_ns / 1000.0)
           << ", \"pid\": 0, \"tid\": " << e.tid << " }";
        break;
      case event_t::MARKER:
        os << "    { \"name\": \"" << (e.name ? e.name : "marker")
           << "\", \"ph\": \"i\", \"s\": \"g\", \"ts\": " << (e.ts_ns / 1000.0)
           << ", \"pid\": 0, \"tid\": " << e.tid << " }";
        break;
      case event_t::COUNTER:
        os << "    { \"name\": \"" << (e.name ? e.name : "counter")
           << "\", \"ph\": \"C\", \"ts\": " << (e.ts_ns / 1000.0)
           << ", \"pid\": 0, \"args\": { \"value\": " << e.value << " } }";
        break;
    }
  }
  os << "\n  ]\n}\n";
  return true;
}
} // namespace tprof
