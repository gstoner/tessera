#include "tprof/tprof_runtime.h"
#include <fstream>
#include <string>

namespace tprof {

namespace {
std::string json_escape(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (char ch : value) {
    switch (ch) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += ch; break;
    }
  }
  return out;
}
} // namespace

// Minimal "Trace Event JSON" accepted by Perfetto UI. Adds a process_name metadata record.
bool perfetto_export(const std::vector<event_t>& events, const std::string& path) {
  std::ofstream os(path, std::ios::out | std::ios::trunc);
  if (!os) return false;
  os << "{\n  \"displayTimeUnit\": \"ns\",\n  \"traceEvents\": [\n";
  // Metadata first
  os << "    { \"ph\": \"M\", \"name\": \"process_name\", \"pid\": 0, \"args\": { \"name\": \"tessera prof\" } }";
  for (const auto& e : events) {
    os << ",\n";
    switch (e.type) {
      case event_t::RANGE_B:
        os << "    { \"name\": \"" << json_escape(e.name)
           << "\", \"ph\": \"B\", \"ts\": " << (e.ts_ns / 1000.0)
           << ", \"pid\": 0, \"tid\": " << e.tid << " }";
        break;
      case event_t::RANGE_E:
        os << "    { \"ph\": \"E\", \"ts\": " << (e.ts_ns / 1000.0)
           << ", \"pid\": 0, \"tid\": " << e.tid << " }";
        break;
      case event_t::MARKER:
        os << "    { \"name\": \"" << json_escape(e.name)
           << "\", \"ph\": \"i\", \"s\": \"g\", \"ts\": " << (e.ts_ns / 1000.0)
           << ", \"pid\": 0, \"tid\": " << e.tid << " }";
        break;
      case event_t::COUNTER:
        os << "    { \"name\": \"" << json_escape(e.name)
           << "\", \"ph\": \"C\", \"ts\": " << (e.ts_ns / 1000.0)
           << ", \"pid\": 0, \"tid\": " << e.tid
           << ", \"args\": { \"value\": " << e.value << " } }";
        break;
    }
  }
  os << "\n  ]\n}\n";
  return true;
}
} // namespace tprof
