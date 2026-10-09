# Public continuous scaled-product pipeline: gfx1201

Owner: E2E-REAL-6 / AD-RESIDUAL-EVAL-1 / FRONTEND-IR-MEDIUM-1.
Sync: CONTINUOUS-SCALED-PRODUCT-20261009. Dependent on PR911.

Recorder: `benchmarks/rocm/record_public_floating_scaled_product.py`.
Packet: [gfx1201.json](gfx1201.json).
Owning RX 9070 XT architecture is queried live and rejected unless gfx1201.
The packet binds compiler/provider, recorder and frontend/native source hashes.

## Proof and timing scope

504 public numerical checks cover direct and mapped ordinary primal/paired
JVP, four orientations, fifteen operand-sharing masks, two map levels, mixed
axes and nonleading outputs. Independent float64 oracles precede timing.
Warm calls with changed operands/seeds forbid compiler subprocesses.

Sixteen profiles cover four orientations, direct/nested mixed-axis input
frames, and primal/paired JVP. Seven warm rounds alternate operand/seed values
and check every output after timing. Warm primal medians are 1.27–1.71 ms;
paired JVP medians are 1.79–2.53 ms. Maximum absolute error is below 1e-7.

First-call latency includes frontend work and compilation/preparation, although
process caches may already be warm. Warm public latency includes host
checks/map packing, upload, dispatch, readback and output placement.
It is distinct from native events in
`benchmarks/baselines/continuous_scaled_product_20261009/gfx1201.json`.
No matched-control speedup, default promotion, dynamic-shape support or
sibling-architecture performance claim follows from this packet.

## Reproduction

Use the matching full LLVM/MLIR 23.1.1 compiler, owning gfx1201 runtime provider
and active validation environment; set TESSERA_OPT, TESSERA_ROCM_OPT,
TESSERA_ROCM_NATIVE_MOVEMENT_LIB and PYTHONPATH.

```sh
python benchmarks/rocm/record_public_floating_scaled_product.py --output benchmarks/baselines/public_continuous_scaled_product_20261009/gfx1201.json
```
