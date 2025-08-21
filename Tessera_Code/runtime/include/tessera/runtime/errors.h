#pragma once
#include <stdexcept>
#include <string>

namespace tessera {
struct Error : public std::runtime_error {
  int code;
  explicit Error(int c, const std::string& what) : std::runtime_error(what), code(c) {}
};
} // namespace tessera
