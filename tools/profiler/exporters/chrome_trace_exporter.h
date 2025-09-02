#pragma once
#include <string>
#include <vector>
#include "tprof/tprof_runtime.h"

namespace tprof {
bool chrome_export(const std::vector<Event>& events, const std::string& path);
} // namespace tprof
