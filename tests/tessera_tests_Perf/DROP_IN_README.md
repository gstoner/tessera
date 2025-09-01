# Tessera Test & Bench Bundle (drop-in)

## How to use
1) Unzip at the root of your repository.
   - This will create/merge a `tests/` directory with:
     - unit, functional, system tests
     - micro/model perf benches
     - I/O and NCCL collectives benches
   - It will also add optional `scripts/` and `ci/github/` helpers.
2) (Optional) Keep the CI files if you want weekly sweep reports. Otherwise delete `ci/github/*.yaml`.

## Quick run
```bash
pip install torch pytest pandas matplotlib pynvml
pytest -q tests
# Perf sweeps + report
python tests/perf/model/bench_flashattn_sweep.py --seqs 128,256,512,1024,2048 --dims 64,128,256 --dropouts 0.0,0.1 --causals 0,1 --seed 1234
python scripts/make_roofline_report.py --csv runs/flashattn_sweep.csv --outdir runs
```

## Notes
- Benches write artifacts under `runs/`.
- The stress test uses NVTX ranges and optional NVML via `pynvml`.
- CI workflows assume self-hosted runners with GPU labels (e.g., `gpu-sm80`, `gpu-sm90`).
