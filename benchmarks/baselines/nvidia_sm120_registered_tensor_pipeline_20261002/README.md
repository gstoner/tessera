# Registered SM120 canonical tensor pipeline

Owner W1.1; sync NVIDIA-W1.1-REGISTERED-TENSOR-2026-10-02.

The registered pipeline previously scheduled the inner K16 tensor step as an
entire kernel, losing ragged dimensions and fused epilogue operands. Early
whole-function semantic tiling replay now recovers the logical Graph contraction
before Graph prepasses. The ordinary verified Graph -> Schedule -> Tile pipeline
then owns native storage, typed fragments and accumulator lineage.
Graph-to-Schedule rejects isolated canonical K steps on every target.

## Exact-device evidence

Super-Bear RTX 5070, SM120. Twelve fp16/bf16, plain and bias/ReLU/residual
rows passed independent NumPy comparisons before and after resident timing.
Full-pipeline and checked Schedule-package PTX are byte-identical for every row;
the checked package executes those bytes. Executable functions also match after
identical MLIR canonicalization (constant placement/CSE differs beforehand).
414 focused producer, diagnostic, pass metadata and pipeline tests passed.
Adjacent Schedule/audit gates passed 135 tests; 49 owning-architecture gates
were skipped. Three generated views and the compiler-plan gate passed.
Graphify refresh was attempted but its CLI is unavailable on this WSL host.

| M x K x N | Storage | Fused | Device window ms | Host launch ms |
|---|---|---|---:|---:|
| 16x32x16 | fp16 | False | 0.010716 | 0.3717 |
| 16x32x16 | fp16 | True | 0.009094 | 0.4590 |
| 16x32x16 | bf16 | False | 0.011333 | 0.3470 |
| 16x32x16 | bf16 | True | 0.010384 | 0.4710 |
| 17x35x19 | fp16 | False | 0.010473 | 0.3608 |
| 17x35x19 | fp16 | True | 0.009888 | 0.4478 |
| 17x35x19 | bf16 | False | 0.010621 | 0.3789 |
| 17x35x19 | bf16 | True | 0.010120 | 0.4306 |
| 64x256x64 | fp16 | False | 0.011829 | 0.3561 |
| 64x256x64 | fp16 | True | 0.010585 | 0.4262 |
| 64x256x64 | bf16 | False | 0.007861 | 0.3236 |
| 64x256x64 | bf16 | True | 0.007954 | 0.4295 |

Device windows include C++/driver dispatch gaps; host wall includes transfer,
allocation and module lifecycle. Compilation stages are separate. No speedup
or isolated instruction-time claim is made.

## Remaining

Arbitrary/noncanonical producer graphs, dynamic generic loop recovery, other
activation exact-device proof and older SM hardware remain open. Sibling
backends need their own canonical-loop recovery; this guard adds no parity.
Historical packet source hashes remain historical; this packet records the
current sources and matching compiler.

## Reproduce

    source .build-sm120-w1-1/validation-env.sh
    .venv/bin/python benchmarks/nvidia/benchmark_canonical_tensor_replay.py --samples 5 --reps 200 --output benchmarks/baselines/nvidia_sm120_registered_tensor_pipeline_20261002/rtx5070.json
