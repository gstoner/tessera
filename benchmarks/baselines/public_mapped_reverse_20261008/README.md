# Public mapped reverse-scale gradients: gfx1201

Owner: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / LAYOUT-ALG-1.
Synchronization key: MAPPED-INVERSE-COTANGENT-20261008.

The compiler exports actual inverse output-cotangent Graph transposes as
distinct private program members before the scale adjoint reductions.
Native Schedule dispatch preserves the isolated Graph movement member; the
existing result-permutation consumer lowers it through Tile to a GPU
load/store kernel and ROCm native image. Buffer extents, requested gradient
order, output-seed lineage and first-write/last-read lifetimes are checked
before native execution. Python supplies frontend semantics and ABI marshalling.

## Numerical evidence

Tajasaurus, live gfx1201, passes 26 public reverse-map cases:

- independent RHS, shared LHS and shared RHS rows;
- nonleading and trailing mapped results;
- selected and reordered floating-scale gradients;
- nonuniform output cotangents and changed-seed warm reuse without compilation;
- retained output allocations across calls;
- nested output placement and N=129 across column-scale groups.

An independent float64 blockwise oracle checks both scale gradients.
The owning toolchain also passes 30 inverse export/Target/manifest tests,
including reencoded wrong seed lineage and prematurely shortened lifetimes.

## Timing scope

Run from the owning host with matching compiler tools and native provider:

```sh
PYTHONPATH=python:. python benchmarks/rocm/record_public_mapped_result.py \
  --mode reverse --output gfx1201.json
```

The packet checks correctness before and after timing, verifies the live
architecture, and records GPU inventory and source/tool/provider identities.
Captured member windows group repeated pure SSA kernels. They include device
graph dispatch and exclude capture, instantiation and copies.
Public timings include ordinary input preparation, checked cache admission,
native execution and copied outputs. The cold measurement also includes
frontend certification and compilation. These scopes must be compared separately.

The two output-axis cases record inverse-movement medians 0.002212/0.002268 ms,
LHS-scale reduction medians 0.288767/0.288894 ms and RHS-scale reduction
medians 3.986274/3.985425 ms. Warm public medians are 5.676308/5.681118 ms.
The shortest captured windows exceed 36 ms. These values characterize this
small serial reduction envelope; device/member and public times are separate
scopes, not a speedup comparison.

No speedup, default-route promotion, dynamic-shape closure or general AD
closure is claimed. Encoded/storage derivatives and sibling native consumers
remain follow-up work. SM120 attention JVP regression evidence does not prove
an SM120 inverse-cotangent scale package.
