#pragma once

#include <cstdint>

namespace tessera::layout {

/// Physical rank-2 order shared by MLIR address materialization and native
/// kernels. A plan names which logical coordinate is multiplied by the
/// leading dimension, so consumers can inline scalar or SSA arithmetic while
/// retaining one order-mapping authority.
enum class Rank2Order : std::uint8_t { RowMajor, ColumnMajor };

struct Rank2IndexPlan {
  std::uint8_t majorCoordinate;
  std::uint8_t minorCoordinate;
};

constexpr Rank2IndexPlan rank2IndexPlan(Rank2Order order) noexcept {
  return order == Rank2Order::RowMajor ? Rank2IndexPlan{0, 1}
                                        : Rank2IndexPlan{1, 0};
}

constexpr std::int64_t linearIndex2D(std::int64_t row, std::int64_t column,
                                     std::int64_t leadingDimension,
                                     Rank2Order order) noexcept {
  const std::int64_t coordinates[2] = {row, column};
  const Rank2IndexPlan plan = rank2IndexPlan(order);
  return coordinates[plan.majorCoordinate] * leadingDimension +
         coordinates[plan.minorCoordinate];
}

template <Rank2Order Order>
constexpr std::int64_t linearIndex2D(std::int64_t row, std::int64_t column,
                                     std::int64_t leadingDimension) noexcept {
  return linearIndex2D(row, column, leadingDimension, Order);
}

} // namespace tessera::layout
