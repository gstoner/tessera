
#include "Tessera/IR/TesseraOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/TilingInterface.h"

using namespace mlir;
using namespace tessera;

// TilingInterface conservative scaffolding for tessera ops.
//
// Status (2026-05-20):
//   * The ODS in ``TesseraOps.td`` declares ``TilingInterface::Trait``
//     on ``Tessera_MatmulOp`` and ``Tessera_Conv2DNHWCOp`` via
//     ``DeclareOpInterfaceMethods<TilingInterface>``.  Under MLIR 21
//     that declaration form does **not** auto-emit per-Op method
//     stubs the same way it did in MLIR ≤16 — methods like
//     ``getLoopIteratorTypes`` / ``getTiledImplementation`` /
//     ``getResultTilePosition`` need either an explicit method-name
//     list on ``DeclareOpInterfaceMethods<...>`` or an external
//     model interface implementation registered against the dialect.
//   * Until that switch lands, the dialect inherits MLIR's
//     default-failure implementations from
//     ``TilingInterface::Trait`` — every tile driver gets
//     ``failure()`` from these ops, which is the safe answer.
//   * Default state: ``TESSERA_ENABLE_TILING_INTERFACE`` is **off**.
//     A pre-MLIR-21 v1 implementation lived in this file behind
//     ``#if TESSERA_ENABLE_TILING_INTERFACE`` and was removed
//     2026-05-20 because (a) it depended on the auto-emitted method
//     declarations that MLIR 21 no longer produces and (b) the API
//     signatures themselves changed (FailureOr<TilingResult>,
//     out-param ``getResultTilePosition``).
//   * The full follow-up is tracked in
//     ``TilingInterface_NOTES.md``.
#ifndef TESSERA_ENABLE_TILING_INTERFACE
#  define TESSERA_ENABLE_TILING_INTERFACE 0
#endif

#if TESSERA_ENABLE_TILING_INTERFACE
// Intentionally empty until the ODS-side switch lands.  Re-defining
// the interface methods here against the current MLIR 21 signatures
// requires either:
//   * ``DeclareOpInterfaceMethods<TilingInterface, ["getLoop...", ...]>``
//     listing every method we mean to implement, or
//   * a separate ``ExternalModel<...>`` implementation registered
//     in ``TesseraDialect.cpp`` (preferred — keeps ODS lean).
// See ``TilingInterface_NOTES.md`` for the full plan.
#endif  // TESSERA_ENABLE_TILING_INTERFACE
