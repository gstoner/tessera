#include "tessera/LayoutAlgebra.h"
#include "tessera/Rank2Index.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Group = std::vector<std::string>;

bool checkedMul(int64_t lhs, int64_t rhs, int64_t &out);

struct Tree {
  int64_t value = 0;
  std::vector<Tree> children;

  bool isLeaf() const { return children.empty(); }
};

bool sameProfile(const Tree &shape, const Tree &stride) {
  if (shape.isLeaf() != stride.isLeaf())
    return false;
  if (shape.isLeaf())
    return true;
  if (shape.children.size() != stride.children.size())
    return false;
  for (size_t i = 0; i < shape.children.size(); ++i)
    if (!sameProfile(shape.children[i], stride.children[i]))
      return false;
  return true;
}

bool decodeTree(const TesseraLayoutNodeV1 *nodes, size_t count, size_t &cursor,
                Tree &out) {
  if (!nodes || cursor >= count)
    return false;
  const TesseraLayoutNodeV1 node = nodes[cursor++];
  out.value = node.value;
  out.children.clear();
  if (node.child_count == 0)
    return true;
  if (node.value != 0 || node.child_count > count - cursor)
    return false;
  out.children.resize(node.child_count);
  for (Tree &child : out.children)
    if (!decodeTree(nodes, count, cursor, child))
      return false;
  return true;
}

bool decodeLayout(const TesseraLayoutNodeV1 *shapeNodes, size_t shapeCount,
                  const TesseraLayoutNodeV1 *strideNodes, size_t strideCount,
                  Tree &shape, Tree &stride) {
  size_t shapeCursor = 0, strideCursor = 0;
  return decodeTree(shapeNodes, shapeCount, shapeCursor, shape) &&
         decodeTree(strideNodes, strideCount, strideCursor, stride) &&
         shapeCursor == shapeCount && strideCursor == strideCount &&
         sameProfile(shape, stride);
}

void encodeTree(const Tree &tree, std::vector<TesseraLayoutNodeV1> &out) {
  out.push_back({tree.isLeaf() ? tree.value : 0,
                 static_cast<uint32_t>(tree.children.size())});
  for (const Tree &child : tree.children)
    encodeTree(child, out);
}

void flattenLeaves(const Tree &tree, std::vector<int64_t> &out) {
  if (tree.isLeaf()) {
    out.push_back(tree.value);
    return;
  }
  for (const Tree &child : tree.children)
    flattenLeaves(child, out);
}

bool staticPositive(const Tree &tree) {
  if (tree.isLeaf())
    return tree.value > 0;
  return std::all_of(tree.children.begin(), tree.children.end(), staticPositive);
}

bool validShapeLeaves(const Tree &tree) {
  if (tree.isLeaf())
    return tree.value > 0 || tree.value == -1;
  return std::all_of(tree.children.begin(), tree.children.end(), validShapeLeaves);
}

bool validStrideLeaves(const Tree &tree) {
  if (tree.isLeaf())
    return tree.value >= 0 || tree.value == -1;
  return std::all_of(tree.children.begin(), tree.children.end(), validStrideLeaves);
}

bool writeTrees(const Tree &shape, const Tree &stride,
                TesseraLayoutNodeV1 *outputShape, size_t outputShapeCapacity,
                TesseraLayoutNodeV1 *outputStride, size_t outputStrideCapacity,
                size_t *outputShapeCount, size_t *outputStrideCount) {
  std::vector<TesseraLayoutNodeV1> encodedShape, encodedStride;
  encodeTree(shape, encodedShape);
  encodeTree(stride, encodedStride);
  *outputShapeCount = encodedShape.size();
  *outputStrideCount = encodedStride.size();
  if (encodedShape.size() > outputShapeCapacity ||
      encodedStride.size() > outputStrideCapacity)
    return false;
  std::copy(encodedShape.begin(), encodedShape.end(), outputShape);
  std::copy(encodedStride.begin(), encodedStride.end(), outputStride);
  return true;
}

Tree makeFlatTree(const std::vector<int64_t> &values) {
  if (values.size() == 1)
    return Tree{values.front(), {}};
  Tree root;
  root.children.reserve(values.size());
  for (int64_t value : values)
    root.children.push_back(Tree{value, {}});
  return root;
}

Tree makeGroup(std::vector<Tree> children) {
  if (children.size() == 1)
    return children.front();
  return Tree{0, std::move(children)};
}

Tree flatLeavesTree(const std::vector<int64_t> &values) {
  std::vector<Tree> children;
  children.reserve(values.size());
  for (int64_t value : values)
    children.push_back(Tree{value, {}});
  return makeGroup(std::move(children));
}

bool scaledTree(const Tree &input, int64_t scale, Tree &out) {
  out.value = input.value;
  out.children.clear();
  if (input.isLeaf())
    return checkedMul(input.value, scale, out.value);
  out.children.resize(input.children.size());
  for (size_t i = 0; i < input.children.size(); ++i)
    if (!scaledTree(input.children[i], scale, out.children[i]))
      return false;
  return true;
}

bool compactColumnMajor(const std::vector<int64_t> &shape,
                        const std::vector<int64_t> &stride) {
  int64_t expected = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (stride[i] != expected || !checkedMul(expected, shape[i], expected))
      return false;
  }
  return true;
}

bool complementFlat(const std::vector<int64_t> &shape,
                    const std::vector<int64_t> &stride, int64_t cotarget,
                    std::vector<int64_t> &resultShape,
                    std::vector<int64_t> &resultStride) {
  struct Mode { int64_t extent, stride; };
  std::vector<Mode> modes;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] <= 0 || stride[i] < 0)
      return false;
    modes.push_back({shape[i], stride[i]});
  }
  std::sort(modes.begin(), modes.end(), [](const Mode &lhs, const Mode &rhs) {
    return lhs.stride < rhs.stride;
  });
  int64_t span = 1;
  for (const Mode &mode : modes) {
    if (mode.stride < span || mode.stride % span)
      return false;
    const int64_t gap = mode.stride / span;
    if (gap > 1) {
      resultShape.push_back(gap);
      resultStride.push_back(span);
      if (!checkedMul(span, gap, span))
        return false;
    }
    if (!checkedMul(span, mode.extent, span))
      return false;
  }
  if (cotarget <= 0 || cotarget > INT64_MAX - (span - 1))
    return false;
  resultShape.push_back((cotarget + span - 1) / span);
  resultStride.push_back(span);
  return true;
}

bool productResult(const Tree &lhsShape, const Tree &lhsStride,
                   const Tree &rhsShape, const Tree &rhsStride, int variant,
                   Tree &outShape, Tree &outStride) {
  std::vector<int64_t> lhsShapeFlat, lhsStrideFlat, rhsShapeFlat, rhsStrideFlat;
  flattenLeaves(lhsShape, lhsShapeFlat);
  flattenLeaves(lhsStride, lhsStrideFlat);
  flattenLeaves(rhsShape, rhsShapeFlat);
  flattenLeaves(rhsStride, rhsStrideFlat);
  if (variant < TESSERA_LAYOUT_LOGICAL || variant > TESSERA_LAYOUT_RAKED)
    return false;
  if ((variant == TESSERA_LAYOUT_BLOCKED || variant == TESSERA_LAYOUT_RAKED) &&
      lhsShapeFlat.size() != rhsShapeFlat.size())
    return false;
  int64_t lhsCosize = 0;
  if (tessera_layout_cosize_v1(lhsShapeFlat.data(), lhsStrideFlat.data(),
                               lhsShapeFlat.size(), &lhsCosize) != TESSERA_LAYOUT_OK)
    return false;
  Tree scaledRhsStride;
  if (!scaledTree(rhsStride, lhsCosize, scaledRhsStride))
    return false;
  std::vector<int64_t> scaledRhsStrideFlat;
  flattenLeaves(scaledRhsStride, scaledRhsStrideFlat);
  if (variant == TESSERA_LAYOUT_LOGICAL || variant == TESSERA_LAYOUT_ZIPPED) {
    outShape = makeGroup({lhsShape, rhsShape});
    outStride = makeGroup({lhsStride, scaledRhsStride});
    return true;
  }
  if (variant == TESSERA_LAYOUT_TILED) {
    std::vector<Tree> shapes{lhsShape}, strides{lhsStride};
    for (int64_t value : rhsShapeFlat)
      shapes.push_back(Tree{value, {}});
    for (int64_t value : scaledRhsStrideFlat)
      strides.push_back(Tree{value, {}});
    outShape = makeGroup(std::move(shapes));
    outStride = makeGroup(std::move(strides));
    return true;
  }
  if (variant == TESSERA_LAYOUT_FLAT) {
    lhsShapeFlat.insert(lhsShapeFlat.end(), rhsShapeFlat.begin(), rhsShapeFlat.end());
    lhsStrideFlat.insert(lhsStrideFlat.end(), scaledRhsStrideFlat.begin(),
                         scaledRhsStrideFlat.end());
    outShape = flatLeavesTree(lhsShapeFlat);
    outStride = flatLeavesTree(lhsStrideFlat);
    return true;
  }
  std::vector<Tree> shapes, strides;
  shapes.reserve(lhsShapeFlat.size());
  strides.reserve(lhsStrideFlat.size());
  for (size_t i = 0; i < lhsShapeFlat.size(); ++i) {
    const bool raked = variant == TESSERA_LAYOUT_RAKED;
    shapes.push_back(makeGroup({Tree{raked ? rhsShapeFlat[i] : lhsShapeFlat[i], {}},
                                Tree{raked ? lhsShapeFlat[i] : rhsShapeFlat[i], {}}}));
    strides.push_back(makeGroup({Tree{raked ? scaledRhsStrideFlat[i] : lhsStrideFlat[i], {}},
                                 Tree{raked ? lhsStrideFlat[i] : scaledRhsStrideFlat[i], {}}}));
  }
  outShape = makeGroup(std::move(shapes));
  outStride = makeGroup(std::move(strides));
  return true;
}

bool divideResult(const Tree &shape, const Tree &stride, const Tree &tilerShape,
                  int variant, Tree &outShape, Tree &outStride) {
  if (variant < TESSERA_LAYOUT_LOGICAL || variant > TESSERA_LAYOUT_FLAT)
    return false;
  std::vector<int64_t> sourceShape, sourceStride, tileShape;
  flattenLeaves(shape, sourceShape);
  flattenLeaves(stride, sourceStride);
  flattenLeaves(tilerShape, tileShape);
  if (sourceShape.size() != tileShape.size())
    return false;
  std::vector<int64_t> quotientShape, quotientStride;
  quotientShape.reserve(sourceShape.size());
  quotientStride.reserve(sourceShape.size());
  for (size_t i = 0; i < sourceShape.size(); ++i) {
    if (sourceShape[i] % tileShape[i])
      return false;
    quotientShape.push_back(sourceShape[i] / tileShape[i]);
    if (!checkedMul(sourceStride[i], tileShape[i], quotientStride.emplace_back()))
      return false;
  }
  std::vector<int64_t> tileStride = sourceStride;
  if (variant == TESSERA_LAYOUT_LOGICAL) {
    std::vector<Tree> shapes, strides;
    for (size_t i = 0; i < tileShape.size(); ++i) {
      shapes.push_back(makeGroup({Tree{tileShape[i], {}}, Tree{quotientShape[i], {}}}));
      strides.push_back(makeGroup({Tree{tileStride[i], {}}, Tree{quotientStride[i], {}}}));
    }
    outShape = makeGroup(std::move(shapes));
    outStride = makeGroup(std::move(strides));
    return true;
  }
  if (variant == TESSERA_LAYOUT_ZIPPED) {
    outShape = makeGroup({flatLeavesTree(tileShape), flatLeavesTree(quotientShape)});
    outStride = makeGroup({flatLeavesTree(tileStride), flatLeavesTree(quotientStride)});
    return true;
  }
  if (variant == TESSERA_LAYOUT_TILED) {
    std::vector<Tree> shapes{flatLeavesTree(tileShape)}, strides{flatLeavesTree(tileStride)};
    for (int64_t value : quotientShape)
      shapes.push_back(Tree{value, {}});
    for (int64_t value : quotientStride)
      strides.push_back(Tree{value, {}});
    outShape = makeGroup(std::move(shapes));
    outStride = makeGroup(std::move(strides));
    return true;
  }
  tileShape.insert(tileShape.end(), quotientShape.begin(), quotientShape.end());
  tileStride.insert(tileStride.end(), quotientStride.begin(), quotientStride.end());
  outShape = flatLeavesTree(tileShape);
  outStride = flatLeavesTree(tileStride);
  return true;
}

bool inverseFlatLayout(const std::vector<int64_t> &shape,
                       const std::vector<int64_t> &stride,
                       std::vector<int64_t> &inverseShape,
                       std::vector<int64_t> &inverseStride) {
  struct Mode { int64_t extent, physical, logical; };
  std::vector<Mode> modes;
  int64_t logical = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] <= 0 || stride[i] < 0)
      return false;
    modes.push_back({shape[i], stride[i], logical});
    if (!checkedMul(logical, shape[i], logical))
      return false;
  }
  std::sort(modes.begin(), modes.end(), [](const Mode &lhs, const Mode &rhs) {
    return lhs.physical < rhs.physical;
  });
  int64_t expected = 1;
  for (const Mode &mode : modes) {
    if (mode.physical != expected)
      return false;
    if (!checkedMul(expected, mode.extent, expected))
      return false;
    inverseShape.push_back(mode.extent);
    inverseStride.push_back(mode.logical);
  }
  return true;
}

bool logicalToPhysical(const std::vector<int64_t> &shape,
                       const std::vector<int64_t> &stride, int64_t logical,
                       int64_t &physical) {
  if (logical < 0)
    return false;
  std::vector<int64_t> coord(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    // A CuTe layout is affine beyond its declared shape.  Retain all remaining
    // quotient in the slowest mode rather than treating it as an out-of-bounds
    // access; composition relies on this for non-compact tilers.
    if (i + 1 == shape.size()) {
      coord[i] = logical;
      logical = 0;
    } else {
      coord[i] = logical % shape[i];
      logical /= shape[i];
    }
  }
  physical = 0;
  for (size_t i = 0; i < stride.size(); ++i) {
    if (coord[i] && stride[i] > (INT64_MAX - physical) / coord[i])
      return false;
    physical += coord[i] * stride[i];
  }
  return true;
}

bool product(const std::vector<int64_t> &values, int64_t &out) {
  out = 1;
  for (int64_t value : values)
    if (!checkedMul(out, value, out))
      return false;
  return true;
}

int64_t gcd64(int64_t lhs, int64_t rhs) {
  while (rhs) {
    int64_t next = lhs % rhs;
    lhs = rhs;
    rhs = next;
  }
  return lhs < 0 ? -lhs : lhs;
}

// Replace each B leaf by the factors needed to keep A(B(c)) affine.  The
// factor period is the first A mixed-radix boundary crossed by that B stride.
bool composeTree(const Tree &bShape, const Tree &bStride,
                 const std::vector<int64_t> &aShape,
                 const std::vector<int64_t> &aStride,
                 const std::vector<int64_t> &aRadix, Tree &outShape,
                 Tree &outStride) {
  if (bShape.isLeaf()) {
    int64_t remaining = bShape.value, step = bStride.value;
    std::vector<int64_t> factors, strides;
    if (remaining == -1) {
      if (step < 0)
        return false;
      // Split the statically knowable prefix at every outer radix boundary;
      // retain one explicit dynamic tail once the address function is affine.
      while (true) {
        int64_t factor = 1;
        for (size_t i = 1; i < aRadix.size(); ++i) {
          const int64_t period = aRadix[i] / gcd64(aRadix[i], step);
          if (period > 1 && (factor == 1 || period < factor))
            factor = period;
        }
        int64_t physical = 0;
        if (!logicalToPhysical(aShape, aStride, step, physical))
          return false;
        if (factor == 1) {
          factors.push_back(-1);
          strides.push_back(physical);
          break;
        }
        factors.push_back(factor);
        strides.push_back(physical);
        if (!checkedMul(step, factor, step))
          return false;
      }
      outShape = factors.size() == 1 ? Tree{factors[0], {}} : makeFlatTree(factors);
      outStride = strides.size() == 1 ? Tree{strides[0], {}} : makeFlatTree(strides);
      return true;
    }
    if (remaining <= 0 || step < 0)
      return false;
    while (remaining > 1) {
      int64_t factor = remaining;
      for (size_t i = 1; i < aRadix.size(); ++i) {
        const int64_t period = aRadix[i] / gcd64(aRadix[i], step);
        if (period > 1 && period < factor)
          factor = period;
      }
      if (remaining % factor)
        return false;
      int64_t physical = 0;
      if (!logicalToPhysical(aShape, aStride, step, physical))
        return false;
      factors.push_back(factor);
      strides.push_back(physical);
      remaining /= factor;
      if (!checkedMul(step, factor, step))
        return false;
    }
    if (factors.empty()) {
      factors.push_back(1);
      strides.push_back(0);
    }
    if (factors.size() == 1) {
      outShape = Tree{factors[0], {}};
      outStride = Tree{strides[0], {}};
    } else {
      outShape = makeFlatTree(factors);
      outStride = makeFlatTree(strides);
    }
    return true;
  }
  outShape = Tree{};
  outStride = Tree{};
  outShape.children.resize(bShape.children.size());
  outStride.children.resize(bStride.children.size());
  for (size_t i = 0; i < bShape.children.size(); ++i)
    if (!composeTree(bShape.children[i], bStride.children[i], aShape, aStride,
                     aRadix, outShape.children[i], outStride.children[i]))
      return false;
  return true;
}

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

int tessera_layout_rank2_index_plan_v1(
    int order, TesseraLayoutRank2IndexPlanV1 *result) {
  if (!result)
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  tessera::layout::Rank2Order nativeOrder;
  if (order == TESSERA_LAYOUT_ROW_MAJOR)
    nativeOrder = tessera::layout::Rank2Order::RowMajor;
  else if (order == TESSERA_LAYOUT_COLUMN_MAJOR)
    nativeOrder = tessera::layout::Rank2Order::ColumnMajor;
  else
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  const tessera::layout::Rank2IndexPlan plan =
      tessera::layout::rank2IndexPlan(nativeOrder);
  result->major_coordinate = plan.majorCoordinate;
  result->minor_coordinate = plan.minorCoordinate;
  return TESSERA_LAYOUT_OK;
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

int tessera_layout_factorizes_v1(
    const int64_t *readShape, const int64_t *readStride, size_t readRank,
    const int64_t *partitionShape, const int64_t *partitionStride,
    size_t partitionRank, int64_t enumerationLimit,
    TesseraLayoutFactorizationV1 *result) {
  if (!readShape || !readStride || !partitionShape || !partitionStride ||
      !result || readRank == 0 || partitionRank == 0 || enumerationLimit <= 0)
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  int64_t readSize = 0, partitionSize = 0;
  int status = tessera_layout_size_v1(readShape, readRank, &readSize);
  if (status != TESSERA_LAYOUT_OK)
    return status;
  status = tessera_layout_size_v1(partitionShape, partitionRank, &partitionSize);
  if (status != TESSERA_LAYOUT_OK)
    return status;
  status = tessera_layout_cosize_v1(readShape, readStride, readRank,
                                     &result->read_cosize);
  if (status != TESSERA_LAYOUT_OK)
    return status;
  status = tessera_layout_cosize_v1(partitionShape, partitionStride,
                                     partitionRank,
                                     &result->partition_cosize);
  if (status != TESSERA_LAYOUT_OK)
    return status;
  result->factorizes = 0;

  // A compact bijection has image [0,size), irrespective of mode order.  This
  // proves FORGE-scale layouts without enumerating their 62 GB logical image.
  std::vector<size_t> order(partitionRank);
  for (size_t i = 0; i < partitionRank; ++i)
    order[i] = i;
  std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
    return partitionStride[lhs] < partitionStride[rhs];
  });
  int64_t expected = 1;
  bool compact = true;
  for (size_t mode : order) {
    if (partitionStride[mode] != expected ||
        !checkedMul(expected, partitionShape[mode], expected)) {
      compact = false;
      break;
    }
  }
  if (compact) {
    result->factorizes = result->read_cosize <= partitionSize;
    return TESSERA_LAYOUT_OK;
  }

  if (readSize > enumerationLimit || partitionSize > enumerationLimit)
    return TESSERA_LAYOUT_DYNAMIC_UNRESOLVED;
  std::set<int64_t> partitionImage;
  std::vector<int64_t> coordinate(partitionRank);
  for (int64_t logical = 0; logical < partitionSize; ++logical) {
    status = tessera_layout_idx2crd_v1(partitionShape, partitionRank, logical,
                                        coordinate.data(), coordinate.size());
    if (status != TESSERA_LAYOUT_OK)
      return status;
    int64_t physical = 0;
    status = tessera_layout_crd2idx_v1(partitionShape, partitionStride,
                                        coordinate.data(), partitionRank,
                                        &physical);
    if (status != TESSERA_LAYOUT_OK)
      return status;
    partitionImage.insert(physical);
  }
  coordinate.resize(readRank);
  for (int64_t logical = 0; logical < readSize; ++logical) {
    status = tessera_layout_idx2crd_v1(readShape, readRank, logical,
                                        coordinate.data(), coordinate.size());
    if (status != TESSERA_LAYOUT_OK)
      return status;
    int64_t physical = 0;
    status = tessera_layout_crd2idx_v1(readShape, readStride,
                                        coordinate.data(), readRank, &physical);
    if (status != TESSERA_LAYOUT_OK)
      return status;
    if (!partitionImage.count(physical))
      return TESSERA_LAYOUT_OK;
  }
  result->factorizes = 1;
  return TESSERA_LAYOUT_OK;
}

int tessera_layout_residency_v1(
    const int64_t *shape, const int64_t *stride, size_t rank,
    int64_t elementBytes, int64_t capacityBytes,
    TesseraLayoutResidencyV1 *result) {
  if (!shape || !stride || !result || rank == 0 || elementBytes <= 0 ||
      capacityBytes < 0)
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  int status = tessera_layout_cosize_v1(shape, stride, rank,
                                         &result->elements);
  if (status != TESSERA_LAYOUT_OK)
    return status;
  if (result->elements > INT64_MAX / elementBytes)
    return TESSERA_LAYOUT_OVERFLOW;
  result->bytes = result->elements * elementBytes;
  result->capacity_bytes = capacityBytes;
  result->admitted = result->bytes <= capacityBytes;
  return TESSERA_LAYOUT_OK;
}

int tessera_layout_coalesce_v1(
    const TesseraLayoutNodeV1 *shapeNodes, size_t shapeCount,
    const TesseraLayoutNodeV1 *strideNodes, size_t strideCount,
    TesseraLayoutNodeV1 *outputShape, size_t outputShapeCapacity,
    TesseraLayoutNodeV1 *outputStride, size_t outputStrideCapacity,
    size_t *outputShapeCount, size_t *outputStrideCount) {
  if (!outputShapeCount || !outputStrideCount)
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  Tree shape, stride;
  if (!decodeLayout(shapeNodes, shapeCount, strideNodes, strideCount, shape,
                    stride))
    return TESSERA_LAYOUT_MALFORMED_SPEC;
  if (!validShapeLeaves(shape) || !validStrideLeaves(stride))
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  std::vector<int64_t> flatShape, flatStride;
  flattenLeaves(shape, flatShape);
  flattenLeaves(stride, flatStride);
  std::vector<int64_t> canonicalShape, canonicalStride;
  for (size_t i = 0; i < flatShape.size(); ++i) {
    // A dynamic residue is still a valid structured result.  It simply cannot
    // participate in an algebraic merge until a runtime scalar resolves it.
    if (flatShape[i] == -1 || flatStride[i] == -1) {
      canonicalShape.push_back(flatShape[i]);
      canonicalStride.push_back(flatStride[i]);
      continue;
    }
    if (flatShape[i] == 1)
      continue;
    int64_t contiguousStride = 0;
    if (!canonicalShape.empty() && canonicalShape.back() > 0 &&
        canonicalStride.back() >= 0 &&
        checkedMul(canonicalStride.back(), canonicalShape.back(),
                   contiguousStride) &&
        flatStride[i] == contiguousStride) {
      if (!checkedMul(canonicalShape.back(), flatShape[i],
                      canonicalShape.back()))
        return TESSERA_LAYOUT_OVERFLOW;
      continue;
    }
    canonicalShape.push_back(flatShape[i]);
    canonicalStride.push_back(flatStride[i]);
  }
  if (canonicalShape.empty()) {
    canonicalShape.push_back(1);
    canonicalStride.push_back(0);
  }
  const Tree resultShape = makeFlatTree(canonicalShape);
  const Tree resultStride = makeFlatTree(canonicalStride);
  if (!writeTrees(resultShape, resultStride, outputShape, outputShapeCapacity,
                  outputStride, outputStrideCapacity, outputShapeCount,
                  outputStrideCount))
    return TESSERA_LAYOUT_BUFFER_TOO_SMALL;
  return TESSERA_LAYOUT_OK;
}

int tessera_layout_right_inverse_v1(
    const TesseraLayoutNodeV1 *shapeNodes, size_t shapeCount,
    const TesseraLayoutNodeV1 *strideNodes, size_t strideCount,
    TesseraLayoutNodeV1 *outputShape, size_t outputShapeCapacity,
    TesseraLayoutNodeV1 *outputStride, size_t outputStrideCapacity,
    size_t *outputShapeCount, size_t *outputStrideCount) {
  if (!outputShapeCount || !outputStrideCount)
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  Tree shape, stride;
  if (!decodeLayout(shapeNodes, shapeCount, strideNodes, strideCount, shape,
                    stride))
    return TESSERA_LAYOUT_MALFORMED_SPEC;
  if (!staticPositive(shape))
    return TESSERA_LAYOUT_DYNAMIC_UNRESOLVED;
  std::vector<int64_t> flatShape, flatStride, inverseShape, inverseStride;
  flattenLeaves(shape, flatShape);
  flattenLeaves(stride, flatStride);
  if (!inverseFlatLayout(flatShape, flatStride, inverseShape, inverseStride))
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  if (!writeTrees(makeFlatTree(inverseShape), makeFlatTree(inverseStride),
                  outputShape, outputShapeCapacity, outputStride,
                  outputStrideCapacity, outputShapeCount, outputStrideCount))
    return TESSERA_LAYOUT_BUFFER_TOO_SMALL;
  return TESSERA_LAYOUT_OK;
}

int tessera_layout_left_inverse_v1(
    const TesseraLayoutNodeV1 *shapeNodes, size_t shapeCount,
    const TesseraLayoutNodeV1 *strideNodes, size_t strideCount,
    TesseraLayoutNodeV1 *outputShape, size_t outputShapeCapacity,
    TesseraLayoutNodeV1 *outputStride, size_t outputStrideCapacity,
    size_t *outputShapeCount, size_t *outputStrideCount) {
  return tessera_layout_right_inverse_v1(
      shapeNodes, shapeCount, strideNodes, strideCount, outputShape,
      outputShapeCapacity, outputStride, outputStrideCapacity, outputShapeCount,
      outputStrideCount);
}

int tessera_layout_complement_v1(
    const TesseraLayoutNodeV1 *shapeNodes, size_t shapeCount,
    const TesseraLayoutNodeV1 *strideNodes, size_t strideCount, int64_t cotarget,
    TesseraLayoutNodeV1 *outputShape, size_t outputShapeCapacity,
    TesseraLayoutNodeV1 *outputStride, size_t outputStrideCapacity,
    size_t *outputShapeCount, size_t *outputStrideCount) {
  if (!outputShapeCount || !outputStrideCount)
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  Tree shape, stride;
  if (!decodeLayout(shapeNodes, shapeCount, strideNodes, strideCount, shape,
                    stride))
    return TESSERA_LAYOUT_MALFORMED_SPEC;
  if (!staticPositive(shape))
    return TESSERA_LAYOUT_DYNAMIC_UNRESOLVED;
  std::vector<int64_t> flatShape, flatStride;
  flattenLeaves(shape, flatShape);
  flattenLeaves(stride, flatStride);
  struct Mode { int64_t extent, stride; };
  std::vector<Mode> modes;
  for (size_t i = 0; i < flatShape.size(); ++i) {
    if (flatStride[i] < 0)
      return TESSERA_LAYOUT_INVALID_ARGUMENT;
    modes.push_back({flatShape[i], flatStride[i]});
  }
  std::sort(modes.begin(), modes.end(), [](const Mode &lhs, const Mode &rhs) {
    return lhs.stride < rhs.stride;
  });
  int64_t span = 1;
  std::vector<int64_t> resultShape, resultStride;
  for (const Mode &mode : modes) {
    if (mode.stride < span || mode.stride % span)
      return TESSERA_LAYOUT_INVALID_ARGUMENT;
    const int64_t gap = mode.stride / span;
    if (gap > 1) {
      resultShape.push_back(gap);
      resultStride.push_back(span);
      if (!checkedMul(span, gap, span))
        return TESSERA_LAYOUT_OVERFLOW;
    }
    if (!checkedMul(span, mode.extent, span))
      return TESSERA_LAYOUT_OVERFLOW;
  }
  if (cotarget <= 0) {
    int64_t cosize = 0;
    const int status = tessera_layout_cosize_v1(flatShape.data(), flatStride.data(),
                                                 flatShape.size(), &cosize);
    if (status != TESSERA_LAYOUT_OK)
      return status;
    cotarget = cosize;
  }
  if (cotarget <= 0)
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  if (cotarget > INT64_MAX - (span - 1))
    return TESSERA_LAYOUT_OVERFLOW;
  const int64_t tail = (cotarget + span - 1) / span;
  resultShape.push_back(tail);
  resultStride.push_back(span);
  if (!writeTrees(makeFlatTree(resultShape), makeFlatTree(resultStride),
                  outputShape, outputShapeCapacity, outputStride,
                  outputStrideCapacity, outputShapeCount, outputStrideCount))
    return TESSERA_LAYOUT_BUFFER_TOO_SMALL;
  return TESSERA_LAYOUT_OK;
}

int tessera_layout_product_v1(
    const TesseraLayoutNodeV1 *lhsShapeNodes, size_t lhsShapeCount,
    const TesseraLayoutNodeV1 *lhsStrideNodes, size_t lhsStrideCount,
    const TesseraLayoutNodeV1 *rhsShapeNodes, size_t rhsShapeCount,
    const TesseraLayoutNodeV1 *rhsStrideNodes, size_t rhsStrideCount,
    int variant, TesseraLayoutNodeV1 *outputShape, size_t outputShapeCapacity,
    TesseraLayoutNodeV1 *outputStride, size_t outputStrideCapacity,
    size_t *outputShapeCount, size_t *outputStrideCount) {
  if (!outputShapeCount || !outputStrideCount)
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  Tree lhsShape, lhsStride, rhsShape, rhsStride;
  if (!decodeLayout(lhsShapeNodes, lhsShapeCount, lhsStrideNodes, lhsStrideCount,
                    lhsShape, lhsStride) ||
      !decodeLayout(rhsShapeNodes, rhsShapeCount, rhsStrideNodes, rhsStrideCount,
                    rhsShape, rhsStride))
    return TESSERA_LAYOUT_MALFORMED_SPEC;
  if (!staticPositive(lhsShape) || !staticPositive(rhsShape))
    return TESSERA_LAYOUT_DYNAMIC_UNRESOLVED;
  std::vector<int64_t> lhsStrideFlat, rhsStrideFlat;
  flattenLeaves(lhsStride, lhsStrideFlat);
  flattenLeaves(rhsStride, rhsStrideFlat);
  if (std::any_of(lhsStrideFlat.begin(), lhsStrideFlat.end(),
                  [](int64_t value) { return value < 0; }) ||
      std::any_of(rhsStrideFlat.begin(), rhsStrideFlat.end(),
                  [](int64_t value) { return value < 0; }))
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  Tree resultShape, resultStride;
  if (!productResult(lhsShape, lhsStride, rhsShape, rhsStride, variant,
                     resultShape, resultStride))
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  if (!writeTrees(resultShape, resultStride, outputShape, outputShapeCapacity,
                  outputStride, outputStrideCapacity, outputShapeCount,
                  outputStrideCount))
    return TESSERA_LAYOUT_BUFFER_TOO_SMALL;
  return TESSERA_LAYOUT_OK;
}

int tessera_layout_divide_v1(
    const TesseraLayoutNodeV1 *shapeNodes, size_t shapeCount,
    const TesseraLayoutNodeV1 *strideNodes, size_t strideCount,
    const TesseraLayoutNodeV1 *tilerShapeNodes, size_t tilerShapeCount,
    const TesseraLayoutNodeV1 *tilerStrideNodes, size_t tilerStrideCount,
    int variant, TesseraLayoutNodeV1 *outputShape, size_t outputShapeCapacity,
    TesseraLayoutNodeV1 *outputStride, size_t outputStrideCapacity,
    size_t *outputShapeCount, size_t *outputStrideCount) {
  if (!outputShapeCount || !outputStrideCount)
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  Tree shape, stride, tilerShape, tilerStride;
  if (!decodeLayout(shapeNodes, shapeCount, strideNodes, strideCount, shape,
                    stride) ||
      !decodeLayout(tilerShapeNodes, tilerShapeCount, tilerStrideNodes,
                    tilerStrideCount, tilerShape, tilerStride))
    return TESSERA_LAYOUT_MALFORMED_SPEC;
  if (!staticPositive(shape) || !staticPositive(tilerShape))
    return TESSERA_LAYOUT_DYNAMIC_UNRESOLVED;
  std::vector<int64_t> sourceStride, tileShape, tileStride;
  flattenLeaves(stride, sourceStride);
  flattenLeaves(tilerShape, tileShape);
  flattenLeaves(tilerStride, tileStride);
  if (std::any_of(sourceStride.begin(), sourceStride.end(),
                  [](int64_t value) { return value < 0; }))
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  Tree resultShape, resultStride;
  if (compactColumnMajor(tileShape, tileStride)) {
    if (!divideResult(shape, stride, tilerShape, variant, resultShape,
                      resultStride))
      return TESSERA_LAYOUT_INVALID_ARGUMENT;
  } else {
    // A non-compact tiler is not a rectangular reshape.  Construct the
    // documented (tiler, complement(tiler, size(source))) coordinate map and
    // feed it through the same radix materializer as ordinary composition.
    if (variant != TESSERA_LAYOUT_LOGICAL)
      return TESSERA_LAYOUT_INVALID_ARGUMENT;
    std::vector<int64_t> sourceShape, complementShape, complementStride;
    flattenLeaves(shape, sourceShape);
    int64_t sourceSize = 0;
    if (!product(sourceShape, sourceSize) ||
        !complementFlat(tileShape, tileStride, sourceSize, complementShape,
                        complementStride))
      return TESSERA_LAYOUT_INVALID_ARGUMENT;
    const Tree innerShape = makeGroup({tilerShape, flatLeavesTree(complementShape)});
    const Tree innerStride = makeGroup({tilerStride, flatLeavesTree(complementStride)});
    std::vector<int64_t> sourceRadix(sourceShape.size(), 1);
    for (size_t i = 1; i < sourceShape.size(); ++i)
      if (!checkedMul(sourceRadix[i - 1], sourceShape[i - 1], sourceRadix[i]))
        return TESSERA_LAYOUT_OVERFLOW;
    if (!composeTree(innerShape, innerStride, sourceShape, sourceStride,
                     sourceRadix, resultShape, resultStride))
      return TESSERA_LAYOUT_INVALID_ARGUMENT;
  }
  if (!writeTrees(resultShape, resultStride, outputShape, outputShapeCapacity,
                  outputStride, outputStrideCapacity, outputShapeCount,
                  outputStrideCount))
    return TESSERA_LAYOUT_BUFFER_TOO_SMALL;
  return TESSERA_LAYOUT_OK;
}

int tessera_layout_slice_v1(
    const TesseraLayoutNodeV1 *shapeNodes, size_t shapeCount,
    const TesseraLayoutNodeV1 *strideNodes, size_t strideCount,
    const int64_t *coordinates, size_t coordinateCount,
    TesseraLayoutNodeV1 *outputShape, size_t outputShapeCapacity,
    TesseraLayoutNodeV1 *outputStride, size_t outputStrideCapacity,
    size_t *outputShapeCount, size_t *outputStrideCount,
    TesseraLayoutSliceV1 *result) {
  if (!outputShapeCount || !outputStrideCount || !result || !coordinates)
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  Tree shape, stride;
  if (!decodeLayout(shapeNodes, shapeCount, strideNodes, strideCount, shape,
                    stride))
    return TESSERA_LAYOUT_MALFORMED_SPEC;
  if (!validShapeLeaves(shape) || !validStrideLeaves(stride))
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  std::vector<int64_t> flatShape, flatStride, residualShape, residualStride;
  flattenLeaves(shape, flatShape);
  flattenLeaves(stride, flatStride);
  if (coordinateCount != flatShape.size())
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  int64_t offset = 0;
  for (size_t i = 0; i < flatShape.size(); ++i) {
    const int64_t coordinate = coordinates[i];
    if (coordinate == -1) {
      residualShape.push_back(flatShape[i]);
      residualStride.push_back(flatStride[i]);
      continue;
    }
    if (coordinate < 0 || flatShape[i] == -1 || flatStride[i] < 0 ||
        coordinate >= flatShape[i] ||
        (coordinate && flatStride[i] > (INT64_MAX - offset) / coordinate))
      return flatShape[i] == -1 ? TESSERA_LAYOUT_DYNAMIC_UNRESOLVED
                                 : TESSERA_LAYOUT_INVALID_ARGUMENT;
    offset += coordinate * flatStride[i];
  }
  if (residualShape.empty()) {
    residualShape.push_back(1);
    residualStride.push_back(0);
  }
  if (!writeTrees(flatLeavesTree(residualShape), flatLeavesTree(residualStride),
                  outputShape, outputShapeCapacity, outputStride,
                  outputStrideCapacity, outputShapeCount, outputStrideCount))
    return TESSERA_LAYOUT_BUFFER_TOO_SMALL;
  result->offset = offset;
  return TESSERA_LAYOUT_OK;
}

int tessera_layout_compose_v1(
    const TesseraLayoutNodeV1 *aShapeNodes, size_t aShapeCount,
    const TesseraLayoutNodeV1 *aStrideNodes, size_t aStrideCount,
    const TesseraLayoutNodeV1 *bShapeNodes, size_t bShapeCount,
    const TesseraLayoutNodeV1 *bStrideNodes, size_t bStrideCount,
    TesseraLayoutNodeV1 *outputShape, size_t outputShapeCapacity,
    TesseraLayoutNodeV1 *outputStride, size_t outputStrideCapacity,
    size_t *outputShapeCount, size_t *outputStrideCount) {
  if (!outputShapeCount || !outputStrideCount)
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  Tree aShapeTree, aStrideTree, bShapeTree, bStrideTree;
  if (!decodeLayout(aShapeNodes, aShapeCount, aStrideNodes, aStrideCount,
                    aShapeTree, aStrideTree) ||
      !decodeLayout(bShapeNodes, bShapeCount, bStrideNodes, bStrideCount,
                    bShapeTree, bStrideTree))
    return TESSERA_LAYOUT_MALFORMED_SPEC;
  if (!staticPositive(aShapeTree) || !validShapeLeaves(bShapeTree))
    return TESSERA_LAYOUT_DYNAMIC_UNRESOLVED;

  std::vector<int64_t> aShape, aStride, bShape, bStride;
  flattenLeaves(aShapeTree, aShape);
  flattenLeaves(aStrideTree, aStride);
  flattenLeaves(bShapeTree, bShape);
  flattenLeaves(bStrideTree, bStride);
  if (std::any_of(aStride.begin(), aStride.end(), [](int64_t s) { return s < 0; }) ||
      std::any_of(bStride.begin(), bStride.end(), [](int64_t s) { return s < 0; }))
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  std::vector<int64_t> radix(aShape.size(), 1);
  for (size_t i = 1; i < aShape.size(); ++i)
    if (!checkedMul(radix[i - 1], aShape[i - 1], radix[i]))
      return TESSERA_LAYOUT_OVERFLOW;

  Tree resultShape, resultStride;
  if (!composeTree(bShapeTree, bStrideTree, aShape, aStride, radix,
                   resultShape, resultStride))
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  std::vector<int64_t> resultFlatShape, resultFlatStride;
  flattenLeaves(resultShape, resultFlatShape);
  flattenLeaves(resultStride, resultFlatStride);
  // composeTree is an exact radix-factor construction.  Unlike the former
  // proof-table implementation it has no cardinality ceiling and can retain a
  // dynamic tail; a later runtime carrier resolves that tail without a second
  // algebra implementation.
  if (resultFlatShape.size() != resultFlatStride.size())
    return TESSERA_LAYOUT_INVALID_ARGUMENT;
  if (!writeTrees(resultShape, resultStride, outputShape, outputShapeCapacity,
                  outputStride, outputStrideCapacity, outputShapeCount,
                  outputStrideCount))
    return TESSERA_LAYOUT_BUFFER_TOO_SMALL;
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
