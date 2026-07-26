#ifndef TESSERA_ROCM_DIRECT_ATTENTION_H
#define TESSERA_ROCM_DIRECT_ATTENTION_H

#include "Tessera/Dialect/Tile/TileDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tessera_rocm {

LogicalResult
materializeROCMDirectAttention(tessera::tile::AttentionKernelOp kernel,
                               OpBuilder &builder);

LogicalResult materializeROCMDirectAttentionBackward(
    tessera::tile::AttentionBackwardKernelOp kernel, OpBuilder &builder);

} // namespace mlir::tessera_rocm

#endif
