# Tessera Language Specification (Normative)

## 1. Lexical Elements
- Tokens, identifiers, literals. Grammar sketch in `tessera/compiler/tsr_grammar.ebnf`.

## 2. Types and Shapes
- **Tensor[Dims...]** with symbolic extents and constraints.
- Shape equations resolved at compile‑time; failures are diagnosable.

## 3. Tiles
- `tile {}` blocks declare compute regions with explicit boundaries and halos.
- **Placement** and **Affinity** directives influence scheduling but are deterministic.

## 4. Memory Model
- Tile‑local, device, and host memory spaces; coherency rules; async copies.

## 5. Concurrency & Sync
- Events, barriers within and across tiles; forward progress guarantees.

## 6. Intrinsics
- Minimal set for math, reductions, and comms; portable fallbacks defined.