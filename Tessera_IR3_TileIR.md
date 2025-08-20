# Tessera IR Layer 3 — Tile IR (Blocks, Warps, Tensor Cores)
*(CUDA-style programming guide companion; normative unless stated otherwise)*

---

## 1. Scope
Tile IR encodes **per-GPU kernels**: loop nests, shared-memory tiling, warp-level fragments (WMMA/MMA), and deterministic reductions.

---

## 2. Execution Model
- **ttile.func**: a device kernel entry.
- **Threading**: implicit mapping to blocks/warps/threads chosen by the compiler.
- **Memory**: explicit shared/reg/HBM operations.
- **Barriers**: tile-scope synchronization only; no global barriers.

---

## 3. MLIR Example (custom `ttile` dialect)
### 3.1 Tiled Matmul (FP8→FP32 Accum)
```mlir
ttile.func @matmul_128x128x64(%A: memref<?x?xf8E4M3>, %B: memref<?x?xf8E4M3>, %C: memref<?x?xf16>) {
  %shA = ttile.shared.alloc : memref<128x64xf8E4M3, 3>
  %shB = ttile.shared.alloc : memref<64x128xf8E4M3, 3>
  %acc = ttile.acc.alloc : memref<128x128xf32, 3>

  ttile.for %k = 0 to %K step 64 {
    ttile.cp.async %A[%i,%k..%k+63] to %shA
    ttile.cp.async %B[%k..%k+63,%j] to %shB
    ttile.barrier

    // Warp-fragments & MMA
    %fragA = ttile.ldmatrix %shA : vector<16x16xf8E4M3>
    %fragB = ttile.ldmatrix %shB : vector<16x16xf8E4M3>
    %acc   = ttile.mma.sync %fragA, %fragB, %acc : f32
    ttile.barrier
  }

  // Epilogue: cast + store
  ttile.cast.store %acc to %C : f16
  ttile.return
}
```

### 3.2 Stable Softmax Tile
```mlir
ttile.func @softmax_stable(%X: memref<128x128xf32>, %Y: memref<128x128xf32>) {
  %m = ttile.rowreduce.max %X
  %Xs = ttile.sub %X, %m
  %E = ttile.exp %Xs
  %s = ttile.rowreduce.sum %E {stable = true}
  %Y = ttile.div %E, %s
  ttile.return
}
```

---

## 4. Deterministic Reductions
- Row/col reductions use fixed tree order within a tile.
- Cross-tile reductions are scheduled by Sched IR collectives.

---

## 5. Memory & Concurrency
- `ttile.cp.async` double-buffers shared memory tiles.
- Barriers ensure no read-after-write hazards.
- No atomics that break determinism.

---

## 6. Mapping Hints
- Chosen block/warp sizes are attached as attributes (informative).
- Tuning happens in Sched IR; Tile IR only reflects final choices.
