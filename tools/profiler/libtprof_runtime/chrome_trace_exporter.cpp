#include "tprof/tprof_runtime.h"
#include <fstream>
#include <string>

namespace tprof {

// Minimal Chrome trace writer with B/E phases + M/P counters.
bool chrome_export(const std::vector<Event>& events, const std::string& path) {
  std::ofstream os(path, std::ios::out | std::ios::trunc);
  if (!os) return false;

  os << "{\n  \"traceEvents\": [\n";

  bool first = true;
  for (const auto& e : events) {
    if (!first) os << ",\n";
    first = false;
    switch (e.type) {
      case Event::RANGE_B:
        os << "    { \"name\": \"" << (e.name ? e.name : "range") << "\", \"ph\": \"B\", "
              "\"ts\": " << (e.ts_ns / 1000.0) << ", \"pid\": 0, \"tid\": " << e.tid << " }";
        break;
      case Event::RANGE_E:
        os << "    { \"ph\": \"E\", \"ts\": " << (e.ts_ns / 1000.0) << ", \"pid\": 0, \"tid\": " << e.tid << " }";
        break;
      case Event::MARKER:
        os << "    { \"name\": \"" << (e.name ? e.name : "marker") << "\", \"ph\": \"i\", "
              "\"s\": \"g\", \"ts\": " << (e.ts_ns / 1000.0) << ", \"pid\": 0, \"tid\": " << e.tid << " }";
        break;
      case Event::COUNTER:
        os << "    { \"name\": \"" << (e.name ? e.name : "counter") << "\", \"ph\": \"C\", "
              "\"ts\": " << (e.ts_ns / 1000.0) << ", \"pid\": 0, \"args\": { \"value\": " << e.value << " } }";
        break;
    }
  }
  os << "\n  ]\n}\n";
  return true;
}

} // namespace tprof
