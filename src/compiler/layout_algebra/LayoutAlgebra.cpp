#include "tessera/LayoutAlgebra.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Group = std::vector<std::string>;

int fail(int code, const std::string &message, char *error, size_t capacity) {
  if (error && capacity) {
    std::strncpy(error, message.c_str(), capacity - 1);
    error[capacity - 1] = '\0';
  }
  return code;
}

bool checkedMul(int64_t lhs, int64_t rhs, int64_t &out) {
  if (lhs <= 0 || rhs <= 0 || lhs > INT64_MAX / rhs)
    return false;
  out = lhs * rhs;
  return true;
}

bool validName(const std::string &name) {
  if (name.empty() || !(std::isalpha(static_cast<unsigned char>(name[0])) ||
                        name[0] == '_'))
    return false;
  return std::all_of(name.begin() + 1, name.end(), [](char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
  });
}

bool parseSide(const std::string &text, std::vector<Group> &groups,
               std::string &why) {
  size_t i = 0;
  while (i < text.size()) {
    while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])))
      ++i;
    if (i == text.size())
      break;
    Group group;
    bool parenthesized = text[i] == '(';
    if (parenthesized)
      ++i;
    while (i < text.size()) {
      while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])))
        ++i;
      if (parenthesized && i < text.size() && text[i] == ')') {
        ++i;
        break;
      }
      if (!parenthesized && !group.empty())
        break;
      size_t begin = i;
      while (i < text.size() &&
             (std::isalnum(static_cast<unsigned char>(text[i])) || text[i] == '_'))
        ++i;
      std::string name = text.substr(begin, i - begin);
      if (!validName(name)) {
        why = "expected an axis name";
        return false;
      }
      group.push_back(name);
      if (!parenthesized)
        break;
    }
    if (parenthesized && (i == 0 || text[i - 1] != ')')) {
      why = "unterminated axis group";
      return false;
    }
    if (group.empty()) {
      why = "empty axis group";
      return false;
    }
    groups.push_back(group);
  }
  if (groups.empty()) {
    why = "layout side is empty";
    return false;
  }
  return true;
}

bool parseBindings(const char *csv, std::map<std::string, int64_t> &bindings,
                   std::string &why) {
  if (!csv || !*csv)
    return true;
  std::stringstream stream(csv);
  std::string item;
  while (std::getline(stream, item, ',')) {
    size_t eq = item.find('=');
    if (eq == std::string::npos || item.find('=', eq + 1) != std::string::npos) {
      why = "axis bindings must be comma-separated name=positive_integer pairs";
      return false;
    }
    std::string name = item.substr(0, eq);
    name.erase(std::remove_if(name.begin(), name.end(), [](char c) {
      return std::isspace(static_cast<unsigned char>(c));
    }), name.end());
    std::string value = item.substr(eq + 1);
    char *end = nullptr;
    errno = 0;
    long long parsed = std::strtoll(value.c_str(), &end, 10);
    while (end && *end && std::isspace(static_cast<unsigned char>(*end)))
      ++end;
    if (!validName(name) || errno || !end || *end || parsed <= 0) {
      why = "axis binding must have a valid name and positive integer extent";
      return false;
    }
    if (!bindings.emplace(name, static_cast<int64_t>(parsed)).second) {
      why = "axis binding appears more than once";
      return false;
    }
  }
  return true;
}

std::vector<std::string> flatten(const std::vector<Group> &groups) {
  std::vector<std::string> result;
  for (const Group &group : groups)
    result.insert(result.end(), group.begin(), group.end());
  return result;
}

} // namespace

extern "C" {

const char *tessera_layout_algebra_version_v1(void) {
  return "tessera.layout_algebra.v1";
}

int tessera_layout_size_v1(const int64_t *shape, size_t rank, int64_t *result) {
  if (!shape || !result || rank == 0)
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  int64_t total = 1;
  for (size_t i = 0; i < rank; ++i)
    if (!checkedMul(total, shape[i], total))
      return shape[i] <= 0 ? TESSERA_LAYOUT_DYNAMIC_UNRESOLVED
                           : TESSERA_LAYOUT_OVERFLOW;
  *result = total;
  return TESSERA_LAYOUT_OK;
}

int tessera_layout_cosize_v1(const int64_t *shape, const int64_t *stride,
                             size_t rank, int64_t *result) {
  if (!shape || !stride || !result || rank == 0)
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  int64_t span = 1;
  for (size_t i = 0; i < rank; ++i) {
    if (shape[i] <= 0)
      return TESSERA_LAYOUT_DYNAMIC_UNRESOLVED;
    if (stride[i] < 0)
      return TESSERA_LAYOUT_INVALID_ARGUMENT;
    if (shape[i] - 1 > 0 && stride[i] > (INT64_MAX - span) / (shape[i] - 1))
      return TESSERA_LAYOUT_OVERFLOW;
    span += (shape[i] - 1) * stride[i];
  }
  *result = span;
  return TESSERA_LAYOUT_OK;
}

int tessera_layout_crd2idx_v1(const int64_t *shape, const int64_t *stride,
                              const int64_t *coord, size_t rank,
                              int64_t *result) {
  if (!shape || !stride || !coord || !result || rank == 0)
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  int64_t index = 0;
  for (size_t i = 0; i < rank; ++i) {
    if (shape[i] <= 0)
      return TESSERA_LAYOUT_DYNAMIC_UNRESOLVED;
    if (coord[i] < 0 || coord[i] >= shape[i] || stride[i] < 0)
      return TESSERA_LAYOUT_INVALID_ARGUMENT;
    if (coord[i] && stride[i] > (INT64_MAX - index) / coord[i])
      return TESSERA_LAYOUT_OVERFLOW;
    index += coord[i] * stride[i];
  }
  *result = index;
  return TESSERA_LAYOUT_OK;
}

int tessera_layout_idx2crd_v1(const int64_t *shape, size_t rank, int64_t index,
                              int64_t *coord, size_t coord_capacity) {
  if (!shape || !coord || rank == 0 || coord_capacity < rank || index < 0)
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  int64_t total = 0;
  int status = tessera_layout_size_v1(shape, rank, &total);
  if (status != TESSERA_LAYOUT_OK)
    return status;
  if (index >= total)
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  for (size_t i = 0; i < rank; ++i) {
    coord[i] = index % shape[i];
    index /= shape[i];
  }
  return TESSERA_LAYOUT_OK;
}

int tessera_layout_rearrange_plan_v1(
    const char *spec, const int64_t *input_shape, size_t input_rank,
    const char *bindings_csv, int64_t *expanded_shape,
    size_t expanded_capacity, int64_t *permutation,
    size_t permutation_capacity, int64_t *output_shape,
    size_t output_capacity, size_t *atomic_rank, size_t *output_rank,
    char *error, size_t error_capacity) {
  if (!spec || !input_shape || !atomic_rank || !output_rank)
    return fail(TESSERA_LAYOUT_INVALID_ARGUMENT, "null rearrange argument", error,
                error_capacity);
  std::string expression(spec);
  size_t arrow = expression.find("->");
  if (arrow == std::string::npos || expression.find("->", arrow + 2) != std::string::npos)
    return fail(TESSERA_LAYOUT_MALFORMED_SPEC,
                "layout must contain exactly one ->", error, error_capacity);
  std::vector<Group> lhs, rhs;
  std::string why;
  if (!parseSide(expression.substr(0, arrow), lhs, why) ||
      !parseSide(expression.substr(arrow + 2), rhs, why))
    return fail(TESSERA_LAYOUT_MALFORMED_SPEC, why, error, error_capacity);
  if (lhs.size() != input_rank)
    return fail(TESSERA_LAYOUT_MALFORMED_SPEC,
                "left layout group count must equal input rank", error,
                error_capacity);

  std::vector<std::string> lhsAxes = flatten(lhs), rhsAxes = flatten(rhs);
  std::set<std::string> lhsSet(lhsAxes.begin(), lhsAxes.end());
  std::set<std::string> rhsSet(rhsAxes.begin(), rhsAxes.end());
  if (lhsSet.size() != lhsAxes.size() || rhsSet.size() != rhsAxes.size() ||
      lhsSet != rhsSet)
    return fail(TESSERA_LAYOUT_MALFORMED_SPEC,
                "each axis must appear exactly once on both sides", error,
                error_capacity);

  std::map<std::string, int64_t> extents;
  if (!parseBindings(bindings_csv, extents, why))
    return fail(TESSERA_LAYOUT_MALFORMED_SPEC, why, error, error_capacity);
  for (const auto &entry : extents)
    if (!lhsSet.count(entry.first))
      return fail(TESSERA_LAYOUT_MALFORMED_SPEC,
                  "axis binding names an axis absent from the layout", error,
                  error_capacity);

  for (size_t dim = 0; dim < lhs.size(); ++dim) {
    if (input_shape[dim] == 0 || input_shape[dim] < -1)
      return fail(TESSERA_LAYOUT_DYNAMIC_UNRESOLVED,
                  "input extents must be positive or the dynamic sentinel -1",
                  error, error_capacity);
    int64_t knownProduct = 1;
    std::vector<std::string> unknown;
    for (const std::string &axis : lhs[dim]) {
      auto found = extents.find(axis);
      if (found == extents.end())
        unknown.push_back(axis);
      else if (!checkedMul(knownProduct, found->second, knownProduct))
        return fail(TESSERA_LAYOUT_OVERFLOW, "axis extent product overflow", error,
                    error_capacity);
    }
    if (input_shape[dim] == -1) {
      for (const std::string &axis : unknown)
        extents[axis] = -1;
      continue;
    }
    if (unknown.size() > 1)
      return fail(TESSERA_LAYOUT_DYNAMIC_UNRESOLVED,
                  "grouped input dimension has more than one unresolved axis",
                  error, error_capacity);
    if (input_shape[dim] % knownProduct)
      return fail(TESSERA_LAYOUT_MALFORMED_SPEC,
                  "axis factors do not divide the input dimension", error,
                  error_capacity);
    if (unknown.empty()) {
      if (knownProduct != input_shape[dim])
        return fail(TESSERA_LAYOUT_MALFORMED_SPEC,
                    "axis factors do not multiply to the input dimension", error,
                    error_capacity);
    } else {
      extents[unknown.front()] = input_shape[dim] / knownProduct;
    }
  }

  *atomic_rank = lhsAxes.size();
  *output_rank = rhs.size();
  if (expanded_capacity < *atomic_rank || permutation_capacity < *atomic_rank ||
      output_capacity < *output_rank)
    return fail(TESSERA_LAYOUT_BUFFER_TOO_SMALL, "rearrange output buffer too small",
                error, error_capacity);
  for (size_t i = 0; i < lhsAxes.size(); ++i)
    expanded_shape[i] = extents[lhsAxes[i]];
  for (size_t i = 0; i < rhsAxes.size(); ++i) {
    auto found = std::find(lhsAxes.begin(), lhsAxes.end(), rhsAxes[i]);
    permutation[i] = static_cast<int64_t>(found - lhsAxes.begin());
  }
  for (size_t i = 0; i < rhs.size(); ++i) {
    int64_t extent = 1;
    bool dynamic = false;
    for (const std::string &axis : rhs[i]) {
      if (extents[axis] == -1) {
        dynamic = true;
        continue;
      }
      if (!checkedMul(extent, extents[axis], extent))
        return fail(TESSERA_LAYOUT_OVERFLOW, "output extent overflow", error,
                    error_capacity);
    }
    output_shape[i] = dynamic ? -1 : extent;
  }
  if (error && error_capacity)
    error[0] = '\0';
  return TESSERA_LAYOUT_OK;
}

} // extern "C"
