# Bounded native producer chains: RTX5070

Owner: W1.1 / E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Sync: SM120-BOUNDED-PRODUCER-CHAIN-20261009. Dependent on PR912.
Recorder: `benchmarks/nvidia/benchmark_bounded_producer_chain.py`.
Packet: [rtx5070.json](rtx5070.json).

## Compiler and execution proof

Native MLIR outlines actual RMSNorm/LayerNorm/softmax SSA chains, projects every
bounded result and supplies Schedule/Tile members, typed fragments and native
capacity/lifetime metadata. Python reads and binds that contract.
The v4 native plan retains original and dynamic Graph witnesses; outer portable
chain v3, single-producer native v2 and static native v3 remain compatible.

57 native/legacy partition, 331 focused registry/seal and 1,050 compiler/host-evidence checks pass. The affected legacy NVIDIA device lane passes all 133 cases.
32 RTX5070 public/portable tests cover FP16/BF16, two/three producers, every
nonempty M/N/K bound subset, fused final store, changed values, warm compiler
exclusion, fixed staging counts and retained outputs. Invalid bounds are
rejected before native context access.

## Timing domains

Twelve profiles reuse one bounded package over initial, capacity and singleton
frames. Seven public rounds use changed operands and check numerical agreement
after every round; compiler subprocesses are forbidden. Warm public medians
span 0.615–0.933 ms, including host checks/packing, uploads, complete native
execution, synchronization and readback.

Every separate resident stage is checked before timing and after each of seven
CUDA-event samples (128 repetitions, three warmups). Producer stage medians
span 8.43–14.82 us and matmul 8.42–9.82 us. Upload/readback/compilation are outside
those event samples. Stage events are not summed to claim a whole-program
device measurement. First public call includes compilation/preparation but
process caches may already be warm. Maximum public absolute error is 2.33e-5.

The packet records actual GPU UUID/driver, matching compiler and target tool,
both source-built CUDA runtime providers, source/recorder hashes, native plan,
images and descriptors. No A/B speedup, default promotion, arbitrary producer
support or sibling-device claim follows.

## Reproduction

Use the matching full LLVM/MLIR 23.1.1 compiler and actual SM120 CUDA host.
Set TESSERA_OPT, TESSERA_NVIDIA_OPT, TESSERA_NVIDIA_PTX_LAUNCH_LIB,
TESSERA_NVIDIA_GEMM_LIB and PYTHONPATH. Source scripts/_nvidia_env.sh.

```sh
python benchmarks/nvidia/benchmark_bounded_producer_chain.py --output benchmarks/baselines/sm120_bounded_producer_chain_20261009/rtx5070.json
```

The earlier device test failure exposed per-active-frame staging growth;
native maximum reservation repairs it. Recorder setup also found a missing
resident provider and incomplete dynamic stride/layout bindings. These were
fixed before the sealed packet was produced. Generic scaled batching/transpose
full-unit closure remains open.

The complete CI unit command returned 3 failures, 20,442 passes and 9,583 skips.
The new host-evidence declaration failure was corrected and its full gate
rerun passed; the two pre-existing scaled-matmul batching/transpose failures
remain open. This packet does not claim a green full suite.
