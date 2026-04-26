#pragma once
#include <string>
#include <vector>
#include "tprof/tprof_runtime.h"
namespace tprof {
bool perfetto_export(const std::vector<event_t>& events, const std::string& path);
} // namespace tprof
