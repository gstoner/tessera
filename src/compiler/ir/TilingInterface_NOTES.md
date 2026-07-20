
# TilingInterface — Status & Deferred Work

> **B3 v2 (2026-05-20):** the matmul TilingInterface implementation
> shipped against the current MLIR 23 signatures.  Conv2DNHWCOp has
> a real iteration domain + identity result-tile-position but its
> `getTiledImplementation` continues to return `failure()` until the
> stride/pad-aware window reconstruction lands (deferred v3 work).

## What's in tree today

### ODS — `TesseraOps.td`

Both ops declare `TilingInterface` via the **explicit method-list**
form (required by the MLIR 23 ODS generator — see "Why explicit method
list" below for the MLIR 23 transition that introduced this):

```tablegen
DeclareOpInterfaceMethods<TilingInterface, [
  "getLoopIteratorTypes",
  "getIterationDomain",
  "getTiledImplementation",
  "getResultTilePosition",
]>
```

The generated `TesseraOps.h.inc` now declares all four methods on the
`MatmulOp` and `Conv2DNHWCOp` C++ classes.

### C++ impl — `TesseraTiling.cpp`

The four MLIR 23 method signatures the impl satisfies are:

```cpp
SmallVector<utils::IteratorType> getLoopIteratorTypes();
SmallVector<Range>                getIterationDomain(OpBuilder &b);
FailureOr<TilingResult>           getTiledImplementation(
    OpBuilder &b,
    ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes);
LogicalResult                     getResultTilePosition(
    OpBuilder &b, unsigned resultNumber,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes);
```

* **MatmulOp** — full v1 implementation against MLIR 23 signatures:
  - `getLoopIteratorTypes()` → `{parallel, parallel}` (M, N).  The
    K-axis reduction lives inside the cloned op, not in the iteration
    domain.
  - `getIterationDomain(b)` → static-shape `Range[0, M) × [0, N)`.
  - `getTiledImplementation(b, offsets, sizes)` → clones the matmul
    with the canonical annotation attrs
    (`tessera.tiling_interface = "matmul_conservative_ranked_tensor"`,
    `tessera.tile_rank = 2`, `tessera.full_k = K`) and returns a
    populated `TilingResult`.  The cloned op keeps the full K
    reduction — the v1 semantics are "split M / N into tiles; keep
    K full inside the op."  Operand-tile extraction (`tensor.extract_slice`
    on LHS/RHS) is a v2 follow-up.
  - `getResultTilePosition(b, n, offsets, sizes, &resOff, &resSizes)` →
    identity (the matmul's only result tile is exactly the iteration
    tile).

* **Conv2DNHWCOp** — partial implementation:
  - `getLoopIteratorTypes()` → `{parallel, parallel, parallel, parallel}`
    over (N, H, W, C).
  - `getIterationDomain(b)` → static-shape four-axis Range.
  - `getResultTilePosition(...)` → identity.
  - `getTiledImplementation(...)` → `failure()` (see "Deferred work"
    below).

### Build flag

Default ON: `TESSERA_ENABLE_TILING_INTERFACE=1` ships in every build.
The opt-out remains `-DTESSERA_DISABLE_TILING_INTERFACE` for downstream
consumers that prefer the old default-failure trait impls.

### Driver-observable sentinels

The matmul `getTiledImplementation` stamps three annotation attrs on
the cloned op that a tile driver can FileCheck against:

| Attribute | Value | Purpose |
|---|---|---|
| `tessera.tiling_interface` | `"matmul_conservative_ranked_tensor"` | proves the interface ran |
| `tessera.tile_rank` | `2 : i64` | declares the per-tile rank |
| `tessera.full_k` | `K : index` | records the (kept-full) reduction extent |

## Why explicit method list

Under MLIR ≤16, `DeclareOpInterfaceMethods<Interface>` (no method
list) auto-emitted per-Op declarations for every method on the
interface.  In MLIR 23 that auto-emission was tightened — for
multi-method interfaces with default implementations like
`TilingInterface`, the ODS generator now requires an explicit method
list so it knows which decls the user intends to override (vs. which
should fall through to MLIR's default-failure trait impls).

The earlier B3-v1 attempt at flipping the build flag failed precisely
because the ODS without an explicit method list produced no per-Op
decls, so the out-of-line C++ defs couldn't bind.  The B3-v2 fix
(this commit) lists every method explicitly, which is the MLIR 23
canonical form.

## Deferred work (v3)

### Conv2DNHWCOp stride/pad-aware tiling

`Conv2DNHWCOp::getTiledImplementation` returns `failure()` because
producing a correct convolution-window slice on a tile requires
threading stride + pad metadata through the offset / size computation:

* Output tile starting at `(n0, h0, w0, c0)` of size `(N, H, W, C)`
  requires input window starting at
  `(n0, h0 * stride_h - pad_h, w0 * stride_w - pad_w, 0)` of size
  `(N, (H-1) * stride_h + R, (W-1) * stride_w + S, C_in)`.
* Negative-offset cases need either explicit `tensor.pad` insertion or
  bounds clamping on `tensor.extract_slice`.
* Dilations multiply through the filter R / S axes.

That's a focused ~150 LOC follow-up.  The iteration domain + result
tile position are already correct in B3-v2; the v3 work is the
operand-slice synthesis.

### MatmulOp operand-tile extraction

The B3-v2 matmul `getTiledImplementation` keeps K full inside the
cloned op rather than extracting LHS / RHS operand slices.  A v2-of-v2
sprint can promote this to true operand slicing:

* Extract `LHS[m0:m0+M, :]` via `tensor.extract_slice` (full K).
* Extract `RHS[:, n0:n0+N]` (full K).
* Clone the matmul on the slices, producing a result of shape `(M, N)`.

That's a useful intermediate step before K-loop tiling (which needs
either an explicit reduction over partial accumulators, or a
`linalg::ReduceOp` rewrite — both deferred).

## Why no urgent consumer

No upstream tessera pass in tree today drives `TilingInterface`
directly.  The tile-IR work routes through the schedule/tile dialect
lowering pipeline, not through generic linalg-style tile-and-fuse
drivers.  The interface implementation is here so:

1. **A linalg-bridge pass can attach to it** when the
   tessera ↔ linalg interop lane lights up.
2. **External MLIR tools** (`mlir-opt --transform-interpreter`,
   `--test-tiling-interface` when MLIR is built with test passes)
   can drive `tessera.matmul` through the same machinery used for
   `linalg.matmul`.
3. **Driver-observable sentinels** mean downstream forks can
   FileCheck whether the interface flowed without depending on
   tessera-internal lit infrastructure.

## How to verify the v2 wiring

```bash
# 1. ODS regen + library build:
cmake --build build --target TesseraIR

# 2. Confirm per-Op method decls appear in the generated header:
awk '/^class MatmulOp /,/^};$/' build/src/compiler/ir/TesseraOps.h.inc \
    | grep -E "getLoopIteratorTypes|getTiledImplementation|getIterationDomain|getResultTilePosition"

# 3. Run the Python structural guard:
pytest tests/unit/test_tiling_interface_matmul.py -v

# 4. Confirm tessera-opt still loads after the rebuild:
cmake --build build --target tessera-opt
```
