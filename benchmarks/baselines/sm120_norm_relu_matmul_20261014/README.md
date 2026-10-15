# SM120 RMSNorm -> ReLU -> matmul native package

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Synchronization key: W11-RELU-PRODUCER-20261014.

`guarded.json` records the RTX 5070 UUID, live compute capability 12.0 and driver
610.88, source hashes and matching compiler/provider binary hashes. Its Git
revision names the merged base; the exact implementation is the explicitly
SHA-256-pinned working-tree snapshot. NVIDIA, ROCm, x86, EBM and Clifford
are enabled, matching the required CI compiler configuration. Earlier `sm120.json` and `ci_config.json` remain frozen at their source/binary pins; neither is relabeled as current proof. The final guard-only correction preserves every native program, image, PTX and Tile identity. No benchmark result is transferred to
B200, Hopper, ROCm, Apple or x86.

The ordinary textual frontend captures RMSNorm, ReLU and matmul. Native MLIR
outlines the actual SSA members and their two private f16/bf16 stores, then
lowers each through Schedule/Tile to NVIDIA Target IR and LLVM/PTX. Runtime
marshalling does not compute either intermediate. Public JIT and prepared
execution are checked against an independent f32 normalization, explicit
storage rounding, ReLU and matmul oracle before timing.

## Measurements

Five samples, 100 repetitions and 10 warmup calls per case. All values below
are medians. Device stage/program values are microseconds; host values are
milliseconds.

| M,K,N | Storage | RMSNorm event us | ReLU event us | Matmul event us | Program event us | Prepared host ms | Public JIT ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (7, 32, 9) | f16 | 8.619 | 9.104 | 8.857 | 24.741 | 0.241 | 0.631 |
| (7, 32, 9) | bf16 | 9.723 | 9.224 | 9.606 | 28.188 | 0.309 | 0.694 |
| (17, 64, 19) | f16 | 10.066 | 9.204 | 8.176 | 26.270 | 0.253 | 0.638 |
| (17, 64, 19) | bf16 | 11.161 | 9.857 | 9.616 | 29.593 | 0.261 | 0.638 |
| (129, 256, 33) | f16 | 9.239 | 42.967 | 10.011 | 53.477 | 0.308 | 0.702 |
| (129, 256, 33) | bf16 | 12.754 | 43.017 | 10.524 | 53.733 | 0.306 | 0.682 |

Stage groups time isolated repeated launches between CUDA events. The program
region times the complete three-member native sequence. These event regions
can include CPU enqueue gaps; they are not claimed as pure kernel durations.
Prepared/public wall times include their respective binding/admission, staging
and completion. Isolated stage sums need not equal the program measurement.
The public call uses a different owner of the shared native arena; the recorder
reacquires the prepared frame before profiling and keeps the stale-lease guard.

The long-row serial ReLU is about 43 us in this packet, compared with roughly
9–13 us for its neighboring stages. Coalesced activation mapping and captured
measurement remain follow-up work. This packet establishes an executable
route and measured bottleneck, not a speedup or complete W1.1 closure.

## Reproduce on Super-Bear WSL

```sh
source /home/angstorms/scratch/sm120-relu-env-20261014.sh
"$TESSERA_PYTHON" benchmarks/nvidia/benchmark_norm_relu_matmul.py \
  --samples 5 --repetitions 100 --warmup 10 \
  --output benchmarks/baselines/sm120_norm_relu_matmul_20261014/guarded.json
```

Validation: 28 new route and exact-device tests pass, including the public JIT,
both storage types, mutated Schedule policies, two private-buffer lifetimes,
changed inputs, retained outputs and bitwise nonfinite/zero checks. The observed
eager zero convention is explicit: f16 zero is positive, bf16 keeps its sign.

The CI-configured compiler passes 253 affected producer, bounded-chain, DAG and public-JIT regressions, 547 registry/audit gates and five-file mypy. The broad unit lane remains a publication gate.

Seven composed-descriptor mutation gates reject forged zero/NaN/schedule/storage/digest policies before native owner allocation. One shared checker binds packaged, composed, DAG/prepared and direct submission paths to the same numerical row contract.
