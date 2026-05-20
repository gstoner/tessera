
# TilingInterface — Status & Deferred Work

> **B3 (2026-05-20):** the previous "scaffolding with TODOs" warning is
> retired.  This document now describes the **actual** state of the
> TilingInterface integration in the tessera dialect plus the precise
> work that remains before the interface can drive real tiling
> decisions.

## What's in tree today

* **`TesseraOps.td`** declares `TilingInterface::Trait` on:
  - `Tessera_MatmulOp` (via `DeclareOpInterfaceMethods<TilingInterface>`)
  - `Tessera_Conv2DNHWCOp` (same)

  The trait inheritance is wired correctly — every `MatmulOp` /
  `Conv2DNHWCOp` instance reports `isa<TilingInterface>() == true`,
  so any pass that probes for the interface finds it.

* **`TesseraTiling.cpp`** is intentionally close-to-empty under the
  default build (`TESSERA_ENABLE_TILING_INTERFACE=0`).  The dialect
  inherits MLIR's default-failure implementations from
  `TilingInterface::Trait`, which is the safe answer for any tile
  driver: it cleanly tells the driver "this op can't be tiled by
  this interface yet, fall back to your non-tiled lowering path."

* **CMake** links `MLIRTilingInterface` into `TesseraIR` so the
  interface symbols are available at runtime.

## What's deferred (the v2 work)

Under MLIR ≤16, `DeclareOpInterfaceMethods<TilingInterface>` would
auto-emit per-Op method **declarations** that the C++ side would
define.  Under MLIR 21 that auto-emission no longer happens for
`TilingInterface` — its methods need either:

1. **Explicit method-name list** on `DeclareOpInterfaceMethods<...>` in
   the ODS, e.g.

   ```tablegen
   DeclareOpInterfaceMethods<TilingInterface, [
     "getLoopIteratorTypes",
     "getIterationDomain",
     "getTiledImplementation",
     "getResultTilePosition",
   ]>
   ```

2. **An external-model implementation** registered in
   `TesseraDialect.cpp` (preferred — keeps the ODS lean and lets us
   evolve the impl without rebuilding the dialect proper):

   ```cpp
   // In TesseraDialect.cpp ::initialize()
   MatmulOp::attachInterface<MatmulTilingModel>(*ctx);
   Conv2DNHWCOp::attachInterface<Conv2DNHWCTilingModel>(*ctx);
   ```

The **MLIR 21 method signatures** the v2 work must satisfy:

```cpp
SmallVector<utils::IteratorType> getLoopIteratorTypes();
SmallVector<Range>                getIterationDomain(OpBuilder &b);
FailureOr<TilingResult>           getTiledImplementation(
    OpBuilder &b,
    ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes);
LogicalResult                     getResultTilePosition(
    OpBuilder &b, unsigned resultNumber,
    ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes,
    SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes);
```

These differ from the pre-MLIR-17 signatures in three ways:
- `getTiledImplementation` returns `FailureOr<TilingResult>` (was
  `FailureOr<SmallVector<Operation *>>`).
- `getResultTilePosition` takes out-parameters for the result
  offsets/sizes and returns `LogicalResult` (was a `FailureOr`
  returning the offset list).
- `getLoopIteratorTypes` is mandatory (didn't exist in older MLIR).

## v2 scope (matmul-first, conv2d deferred further)

A focused v2 sprint can ship:

* **MatmulOp**: external-model impl with the v1 "conservative clone
  + annotation attribute" semantics from the original scaffold,
  ported to the new signatures.  Roughly 80 LOC + 1 lit fixture.

* **Conv2DNHWCOp**: per-op iteration domain (parallel over N/H/W/C,
  reduction over Kc/R/S), but `getTiledImplementation` returns
  `failure()` until the stride/pad-aware window reconstruction
  lands.  This stays honest about what's not yet built.

* **Lit fixture** under `tests/tessera-ir/transforms/tiling_interface/`
  driving `linalg::tileToScfForOp` (or similar tile driver) against
  a tessera.matmul to confirm the annotation attrs flow through.

* **CMake flip**: enable `TESSERA_ENABLE_TILING_INTERFACE=1` by
  default, with `-DTESSERA_DISABLE_TILING_INTERFACE=ON` as the
  opt-out for downstream consumers.

## Why this isn't urgent

No upstream pass in the tessera pipeline today consumes
`TilingInterface` directly — the tile-IR work goes through the
schedule/tile dialect lowering pipeline, not through the generic
linalg-style tile-and-fuse driver.  The interface declaration sits in
ODS so we can attach a concrete model when the first consumer (e.g., a
linalg-bridge pass) actually needs it.

The default-failure path keeps any opportunistic consumer (linalg
tile drivers, the MLIR `-test-tiling-interface` opt suite, downstream
forks) safe — they get a clean `failure()` instead of crashing on a
missing implementation.

## How to regenerate this view

```bash
# Confirm the trait is wired:
grep -n "DeclareOpInterfaceMethods<TilingInterface>" \
    src/compiler/ir/TesseraOps.td

# Confirm the default-off build is clean:
cmake --build build --target TesseraIR

# Confirm MLIR's default-failure path is what callers see today:
build/tools/tessera-opt/tessera-opt --help | grep -i tiling
# (no tessera-specific tiling pass — the only consumers would be
#  generic linalg-style drivers we haven't yet plumbed through.)
```
