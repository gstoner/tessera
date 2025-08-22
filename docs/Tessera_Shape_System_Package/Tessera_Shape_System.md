# Tessera Shape System

The Tessera Shape System extends the type system with tensor dimensions,
layouts, and distributed shards. Its goal is to catch shape mismatches at compile time,
reducing runtime errors and making programs safer and more predictable.

## Core Concepts

- **Dim**: Symbolic or concrete integer dimensions (B, N, D).
- **Shape**: Tuples of Dims, e.g. [B, N, D].
- **Layout**: Memory arrangement, e.g. row-major, tiled.
- **Shard**: Device mesh partitioning rules.
- **Constraints**: Equalities, divisibility, and bounds (e.g., N % 128 == 0).

---

## Example: Attention

```tessera
dim B, N, M, D

fn attention(q: Tensor[B, N, D],
             k: Tensor[B, M, D]) -> Tensor[B, N, M] {
  let s = q @ transpose(k, axes=(0, 2, 1))  // requires D == D
  return s * (1.0 / sqrt(D))
}
```

If K != D, compile-time error:

```
error[shape-mismatch]: matmul(Q[B,N,D], K^T[B,D,M]) requires inner dims equal.
  found: D and K
```
---
## Head-splitting with derived dims
```python 

dim B, T, H, Dh
let D = H * Dh

fn mha(q: Tensor[B,T,D], k: Tensor[B,T,D], v: Tensor[B,T,D]) -> Tensor[B,T,D]
where D = H * Dh {
  let qh = reshape(q, [B, T, H, Dh])
  let kh = reshape(k, [B, T, H, Dh])
  let vh = reshape(v, [B, T, H, Dh])
  let out_h = attention(qh, kh, vh)    // checks Dh alignment automatically
  return reshape(out_h, [B, T, D])
}
---

## Broadcasting

```tessera
Tensor[B, 1, D] + Tensor[B, N, D] -> Tensor[B, N, D]
```

Compiler inserts `broadcast(1→N)` or raises error if disallowed.

---

## Distributed Consistency

```tessera
mesh M(tp=8, dp=4)

let X: Tensor[B, N, D] @shard({B: dp, D: tp})
// compile-time checks:
//  B % dp == 0, D % tp == 0
```

---

## Sample Error Messages

**Mismatch:**

```
error[shape-mismatch]: matmul inner dimensions differ
  left:  q : Tensor[B=8, N=1024, D=128]
  right: kT: Tensor[B=8, D=256, M=1024]
  need:  D(left) == D(right)
```

**Illegal tile:**

```
error[tile-constraints]: schedule BM=128,BN=256 invalid for N=2304
  violated: N % 256 == 0
  suggestion: pad to 2304 → 2304+256
```

---

## IDE Integration

Python bindings can annotate symbolic dimensions:

```python
import tessera as ts

B, N, M, D = ts.sym("B N M D")

@ts.check_shapes
def attention(q: ts.Tensor[B, N, D], k: ts.Tensor[B, M, D]) -> ts.Tensor[B, N, M]:
    return ts.ops.matmul(q, ts.ops.transpose(k, (0, 2, 1))) * (1.0 / ts.sqrt(D))
```

---

# Summary

- Shapes become **first-class types** in Tessera.
- Compile-time verification avoids subtle runtime bugs.
- Errors are human-friendly with suggested fixes.
- Works across tiles, layouts, and distributed shards.

How to implement (compiler side)

	1.	Type checker pass (Graph IR)

	-Collect symbolic dims, build constraint graph, unify across ops.
	-Solve using an integer/affine solver (Presburger subset: equalities, divisibility, bounds).
	-Emit actionable diagnostics with source spans and suggested fixes.

	2.	Schedule feasibility pass

	-Combine shape constraints with schedule candidates; prune infeasible points before autotuning.
	-Optionally suggest padding rewrites and quantify overhead.

	3.	Tile verifier

	-Check fragment sizes, shared memory banking, vector alignment; produce concrete PTX/ROCDL contracts.

	4.	Runtime witnesses (optional)

	-When Dim? appears, insert a single runtime assert; record refinement in module metadata so later kernels see the narrowed shape.

	5.	IDE support

	-*.pyi stubs and a mypy/Pyright plugin to validate annotations and show operator result shapes in tooltips.
