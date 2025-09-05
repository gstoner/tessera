
# Tessera Benchmark Suite (SuperBench-style) — v2

This extends v1 with:
- **Tessera IR-style kernels** (GEMM/conv/FlashAttn stubs) + **correctness checks** vs NumPy baselines.
- **GPU system probes** (PCIe/NVLink/NIC) with safe fallbacks (skips if tools not present).
- **Distributed collectives adapters** (PyTorch `torch.distributed` with NCCL/RCCL if available; Gloo otherwise).
- Built-in **trace hooks** (Chrome Trace JSON / Perfetto) and **NVTX** (if `nvtx` Python package is available or C++ NVTX headers are present).
- An HTML **roofline overlay** (pure SVG, no external libs) with device peak parameters from `peaks/*.yaml`.

## Quick Start
```bash
# 1) Build C++ microbenches
cmake -S benches -B build && cmake --build build -j

# 2) Run the suite
python3 runner/bench_run.py --config configs/default.yaml --out out

# 3) Generate HTML with roofline
python3 runner/report_html.py --results out/results.json --html out/report.html --peaks peaks/example_peaks.yaml

# (Optional) Distributed collectives (single-node local launch)
python3 benches/distributed/collectives_torch.py --world_size 2 --backend nccl --iters 100 --bytes 134217728

# View trace
# Open out/trace.json in https://ui.perfetto.dev or Chrome tracing.
```
