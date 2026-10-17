# SM120 static row graph replay experiment

Owner: E2E-REAL-6 / AD-RESIDUAL-EVAL-1 / FRONTEND-IR-MEDIUM-1.
Synchronization: SM120-NATIVE-ROW-GRAPH-20261016.

## Result

The unmerged explicit native graph candidate passes 50 owning RTX 5070 device gates with
zero skips. This includes FP16/BF16 host and resident full/dV-only gradients,
changed data, independent FP64 arithmetic, bitwise direct parity, delayed
producer writes, fresh root/stream rebinding, output retention, invalid ABI
recovery and isolated driver instantiation/launch failure retirement.

Two fresh-process eight-profile timing packets retain 80 samples per arm per
public/prepared domain, alternating AB/BA. They show no consistent speedup.
The experimental production APIs and flag were removed; direct submission and
the existing inline/prepass selector remain unchanged. The frozen patch retains
the complete candidate and tests for reproduction.
The first trial overlapped a CPU Graphify update; the recheck ran after that
process completed. Neither packet transfers evidence to another architecture.

## Boundaries

These are the same original frontend, native AD, Schedule/Tile and LLVM/PTX
members under both arms; only native submission changes. First capture is
excluded from warm timings. The independent oracle and bitwise checks run
outside every measured call. Public native_backward and prepared invoke wall
times are separate from native forward/row/backward and device interval times.
Device intervals exclude host staging and transfer nodes outside the kernels.

The final provider also passes the original adjacent suite: 330 passes and
36 opt-in skips, followed by those 36 explicitly enabled within final-device.txt.
The final 50-test gate uses the rebuilt provider; the earlier adjacent suite
preceded only an indentation edit in the native enqueue lambda.
Drift gates: 336 passes. Adjacent ABI/header/artifact: 23 passes, three skips.
The skipped ABI tests are not hardware evidence. Initial failed new-fixture
runs remain in WSL scratch; batch-size variable shadowing was fixed before
the final green device run. Driver faults are injected before GPU submission,
not a proof of CUDA context-loss recovery.

## Files

- record_graph_replay.py: measured recorder.
- graph-ab*.json.gz and graph-ab*.log: both complete timing trials and raw output.
- final-device.txt: final 50-test result, including existing fork warnings.
- drift.txt and abi-adjacent.txt: focused host WSL gates.
- manifest.json: identities, limitations and decision.

## Reproduction and dispatch profile

Apply experimental_graph_replay.patch.gz to a separate checkout of main6480571,
build the native provider with CUDA13.3, and run its14 graph device tests plus
existing36 opt-in JVP/VJP controls. record_graph_replay.py imports the test helper
included in that patch. Do not run this experimental recorder on unpatched main.
The 50-test proof belongs to the candidate, not a new production feature.

profile_dispatch.py and public/prepared profile text retain200-call diagnostic
profiles. The profiler includes native blocking and instrumentation overhead;
it identifies repeated Graph serialization/certificate/path work, not a speedup.
The self-contained profile_dispatch.py companion imports the same original test
helpers directly; measured_profile_dispatch.py.gz preserves the exact measured
script bytes. Profile replay must use the candidate patch.

Remaining: finish publication/generated-doc gates and continue the full five
compiler slices. Further dispatch optimization must preserve mutation checks
and original frontend/Graph authority.


Timing JSON files are compressed losslessly; gzip decoding preserves every raw
sample and recorded identity byte. No measured samples were dropped.

With the packet copied into a separate checkout of 6480571 and its matching
LLVM/MLIR 23.1.1 compiler and Python environment active, the candidate provider
and device gate can be reproduced as follows. Use the owning RTX 5070 WSL host:

```bash
gzip -dc benchmarks/baselines/sm120_native_row_graph_20261016/experimental_graph_replay.patch.gz | git apply
export CUDA_HOME=/usr/local/cuda-13.3
source scripts/_nvidia_env.sh
mkdir -p ~/scratch/sm120-row-graph-reproduction
export TESSERA_NVIDIA_PTX_LAUNCH_LIB=~/scratch/sm120-row-graph-reproduction/libtessera_nvidia_ptx_launch.so
g++ -std=c++17 -O3 -fPIC -shared -pthread -I"$CUDA_HOME/include" -Isrc/runtime/include \
  src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp \
  src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/attention_jvp_prepared.cpp \
  src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/matmul_prepared.cpp \
  -L/usr/lib/wsl/lib -lcuda -ldl -o "$TESSERA_NVIDIA_PTX_LAUNCH_LIB"
TESSERA_NVIDIA_DEVICE_PROOF=1 python -m pytest -q \
  tests/device/nvidia/test_native_row_graph_replay.py \
  tests/device/nvidia/test_prepared_attention_jvp.py \
  tests/device/nvidia/test_prepared_attention_vjp.py \
  tests/device/nvidia/test_prepared_attention_stream_isolation.py
python benchmarks/baselines/sm120_native_row_graph_20261016/record_graph_replay.py \
  --output ~/scratch/sm120-row-graph-reproduction/graph-ab.json
```

`TESSERA_OPT` and `TESSERA_NVIDIA_OPT` must name matching built tools before
running these commands. Run timings after other GPU work and validation stop.
