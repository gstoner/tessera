# 04_tile_ir.md
# TileIR (Normative)

## 1. Objectives
- Be a stable interchange for tile kernels across backends.

## 2. IR Structure
- Module → Functions → Blocks → Ops; SSA; attributes for tile bounds/halos.

## 3. Required Ops
- `tile.launch`, `tile.copy`, `tile.reduce`, `tile.comm` (collectives), etc.

## 4. Validity Rules
- No implicit aliasing across tiles; explicit region capabilities.