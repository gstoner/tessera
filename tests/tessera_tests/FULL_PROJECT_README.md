# Tessera Full Project (Kernels + Wrappers + Build + Tests + CI)

## Build (choose your arch; BF16 optional on sm80+)
```bash
pip install torch pytest pandas matplotlib pynvml
export TESSERA_TARGET_SM=90           # 80 | 90 | 100
export TESSERA_ENABLE_BF16=1          # enable BF16 WMMA on sm80+
python ext/setup.py build_ext --inplace
```

## Run tests & benches
```bash
pytest -q tests                # unit + functional + system (GPU fryer-like)
python tests/perf/micro/bench_flashattn_tiled.py --seed 1234
python tests/perf/model/bench_flashattn_sweep.py --seqs 128,256,512,1024,2048 --dims 64,128,256 --dropouts 0.0,0.1 --causals 0,1 --seed 1234
python scripts/make_roofline_report.py --csv runs/flashattn_sweep.csv --outdir runs
```

## CI (optional)
- Nightly perf gate: `ci/github/gpu_nightly_perf_gate.yaml`
- Weekly sweeps (single-runner): `ci/github/weekly_sweep_report.yaml`
- Weekly sweeps (matrix sm80+sm90) + aggregation: `ci/github/weekly_sweep_report_matrix.yaml`

## Notes
- Nsight Compute recipes: `scripts/ncu_recipes.md` and `scripts/profile_ncu.sh`
- Tiled attention kernels support **Philox dropout** with deterministic `seed`.
- I/O + NCCL benches write CSV/JSONL to `runs/` and charts via the report script.
