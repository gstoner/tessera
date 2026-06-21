#pragma once

#include <cstdint>
#include <sstream>
#include <string>

namespace tprof {

inline std::string tprof_json_escape(const char* value) {
  std::string out;
  if (!value) return out;
  for (const char ch : std::string(value)) {
    switch (ch) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += ch; break;
    }
  }
  return out;
}

inline std::string provider_payload(const char* provider,
                                    const char* kind,
                                    const char* name,
                                    uint64_t correlation_id,
                                    const char* detail_key,
                                    const char* detail_value,
                                    const char* extra_json) {
  std::ostringstream os;
  os << "{\"provider\":\"" << tprof_json_escape(provider)
     << "\",\"kind\":\"" << tprof_json_escape(kind)
     << "\",\"name\":\"" << tprof_json_escape(name)
     << "\",\"correlation_id\":" << correlation_id;
  if (detail_key && detail_value) {
    os << ",\"" << tprof_json_escape(detail_key) << "\":\""
       << tprof_json_escape(detail_value) << "\"";
  }
  if (extra_json && extra_json[0] != '\0') {
    os << ",\"extra\":" << extra_json;
  }
  os << "}";
  return os.str();
}

inline bool tprof_contains_token(const char* value, const char* token) {
  if (!token || token[0] == '\0') return false;
  if (!value) return false;
  return std::string(value).find(token) != std::string::npos;
}

inline bool tprof_passes_filters(const char* value,
                                 const char* include_token,
                                 const char* exclude_token) {
  if (include_token && include_token[0] != '\0' &&
      !tprof_contains_token(value, include_token)) {
    return false;
  }
  if (exclude_token && exclude_token[0] != '\0' &&
      tprof_contains_token(value, exclude_token)) {
    return false;
  }
  return true;
}

} // namespace tprof
