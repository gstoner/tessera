# Tessera Perf & Stability Toolkit

## Sweep & Report
```bash
# Build (choose target arch)
export TESSERA_TARGET_SM=90
python ext/setup.py build_ext --inplace

# Run sweeps (S x D x dropout x causal) and generate report
python tests/perf/model/bench_flashattn_sweep.py --seqs 128,256,512,1024,2048 --dims 64,128,256 --dropouts 0.0,0.1 --causals 0,1 --seed 1234
python scripts/make_roofline_report.py --csv runs/flashattn_sweep.csv --outdir runs

# Open:
# - runs/flashattn_sweep_report.md
# - runs/flashattn_sweep_report.html
```

## GPU Fryer-like Stress Test
```bash
# Duration and shape knobs
export TESSERA_FRYER_DURATION_S=60
export TESSERA_FRYER_S=1024
export TESSERA_FRYER_D=128
export TESSERA_FRYER_H=16
export TESSERA_FRYER_B=1
export TESSERA_FRYER_DROPOUT=0.1

pytest -q tests/system/test_gpu_fryer_like.py
```

## Deterministic Seeds
- All tiled attention benches accept `--seed` to control the Philox RNG used for dropout.
- System stress test honors `TESSERA_SEED` for PyTorch RNG.

## CI
- Nightly: perf gate (`gpu-nightly-perf-gate`) — micro-benches.
- Weekly: sweep+report (`weekly-sweep-report`) — uploads CSV/HTML/PNG.

## Memory / PCIe / Collectives Benches
```bash
# Device-to-device memcpy (bandwidth vs size)
python tests/perf/io/bench_mem_io.py --sizes_mb 64,128,256,512,1024

# Host↔Device over PCIe (pinned vs pageable)
python tests/perf/io/bench_pcie_io.py --sizes_mb 64,128,256,512,1024

# NCCL collectives (intra-node NVLINK or inter-node NIC, depending on setup)
torchrun --nproc_per_node=2 tests/perf/collectives/bench_nccl.py --sizes_mb 1,2,4,8,16,32,64,128,256 --dtype float16
```

## Weekly Matrix CI & Aggregation
- `weekly-sweep-report-matrix` runs on labels **gpu-sm80** and **gpu-sm90**, uploads `runs-<label>` artifacts.
- `aggregate-reports` builds a **single tabbed HTML** combining per-label reports.
