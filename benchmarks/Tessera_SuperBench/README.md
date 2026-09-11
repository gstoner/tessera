
# Tessera Benchmark Suite (SuperBench-style) — current compiler

This suite is a portable SuperBench-style harness for the current Tessera
compiler surface:

- **Executable compiler path:** GEMM+ReLU uses `@tessera.jit` through
  Graph IR -> Schedule IR -> Tile IR -> Target IR -> CPU.
- **Artifact-only compiler paths:** Conv2D and FlashAttention capture compiler
  artifacts and run NumPy timing/correctness references until native runtimes
  are promoted.
- **Shared telemetry:** each compiler-backed row emits `tessera.telemetry.v1`
  with compiler path, runtime status, artifact hashes, latency, and bottleneck
  labels.
- **Autotune artifacts:** GEMM can attach the current `tessera.autotune`
  schedule artifact through `--autotune`.
- **Portable collectives:** default configs use the `tessera.collectives` mock
  facade; CUDA/NCCL remains an opt-in hardware config.
- **System probes and reports:** GPU/NIC probes skip safely, Chrome Trace JSON
  is emitted by the runner, and `report_html.py` renders a roofline plus
  telemetry summary.

## Quick Start
```bash
# 1) Build C++ microbenches
cmake -S benches -B build && cmake --build build -j

# 2) Run the portable compiler smoke
~/venv/bin/python runner/bench_run.py --config configs/compiler_smoke.yaml --out out/compiler_smoke

# 3) Run the broader portable suite
~/venv/bin/python runner/bench_run.py --config configs/default.yaml --out out/default

# 4) Generate HTML with roofline and telemetry summary
~/venv/bin/python runner/report_html.py --results out/default/results.json --html out/default/report.html --peaks peaks/example_peaks.yaml

# Portable collective smoke via Tessera facade
~/venv/bin/python benches/distributed/collectives_torch.py --world_size 2 --backend tessera_mock --iters 10 --bytes 1048576

# Optional CUDA/NCCL hardware run
~/venv/bin/python benches/distributed/collectives_torch.py --world_size 2 --backend nccl --iters 100 --bytes 134217728

# View trace
# Open out/trace.json in https://ui.perfetto.dev or Chrome tracing.
```

## Status Labels

| Status | Meaning |
|---|---|
| `executable` | The benchmark measured a currently executable Tessera path. |
| `artifact_only` | Compiler artifacts were captured, but timing uses a reference implementation. |
| `mock` | Runtime-facing facade path is active without native hardware. |
| `backend_unavailable` | Requested native backend is not available in this environment. |

`configs/sm90_cuda.yaml` is a hardware-oriented overlay. It preserves NCCL and
large-shape attention/GEMM settings but should not be used as a CPU-only CI
gate.

## Retained and retired entry points

The sleep-based `attention_placeholder.py` is retired and exits with an error;
it must not produce latency or throughput rows. The three `_stub.py` files are
compatibility forwarding entry points to the current reference/artifact scripts.
This harness's attention/conv gaps do not imply missing native compiler families
elsewhere. See the [alignment review](../COMPILER_ALIGNMENT.md#additional-suite-review--2026-09-10).

### Native ANN packages

`configs/native_nvidia.yaml` and `configs/native_rocm.yaml` run
`benches/kernel/ann_native.py`. Set `TESSERA_OPT` to the owning host's compiler,
source its backend environment, then use the normal suite runner. The adapter
accepts explicit `--compiler`, `--rows`, `--width`, and `--activation` options.
This is a bounded two-affine ANN workload (2–64 rows, 2–8 features), not a
replacement for the GEMM, attention or distributed suites. Both original and
transformed packages are verified; there is no reference fallback. Latency
includes instrumentation, H2D, checked launch, synchronization and D2H. Receipts
count actual driver launch API calls, not profiler kernel records. No promotion.

The native configs also include a 64×8 square ANN and serial/cooperative SSD
at `(time=32, heads=2, states=8, width=4, chunk=8)`. `ssd_native.py` delegates to
the checked SSD recorder: its `latency_ms` is one resident checked host call;
`device_event_ms` is the median of seven 100-launch device-event windows per
call. These domains are not interchangeable, and compilation is excluded.

### Scheduled GEMM/attention and mixed captures

`matrix_native.py --backend nvidia --workload gemm|attention|mixed` lowers a
bounded input Graph once and packages its resulting Schedule/Tile artifact.
Input layouts come from the descriptor. GEMM is 32×64×48 fp16→fp32; CUDA
attention is fp32 at B=1, H=2, Sq=8, Sk=12, D=Dv=4. ROCm uses D=Dv=64 fp16.
The independent NumPy oracle runs on every checked call. Timing includes the
host bridge, validation and oracle; it is not isolated kernel latency.

`--nvtx` (CUDA only) compiles a small NVTX3 header-based marker helper with the
owning toolkit and system C compiler. Profile with `nsys --trace=cuda,nvtx`.
Each synchronous measured call names its run, Schedule artifact and native
image; compilation and initial warmups are outside ranges. Use
`benchmarks/attribute_mixed_cuda.py` with the packet and exported SQLite to
join each range's successful launch API to exactly one correlated kernel per
launch. Multiple launches per range are supported; overlapping/multi-thread
ranges and escaping asynchronous completion are refused. Warmups remain
unattributed. The flat mixed latency is a median across the recorded calls;
use per-workload rows for comparisons. No performance promotion.

Current ROCm GEMM/attention execution is blocked by a duplicate-attribute LLVM
assertion in GPUFuncOpLowering with the installed assertions-enabled compiler.
The native ROCm config exposes these failures; the older four workloads remain
validated. Do not interpret runner process exit alone as success: inspect rows.
