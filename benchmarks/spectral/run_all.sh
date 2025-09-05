#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# 1D FFT + conv sweeps
python spectral_bench.py --ops fft1d,conv1d_fft,spectrum --sizes 1024,2048,4096,8192,16384 --device auto --repeats 50 --warmup 10

# 2D FFT + conv sweeps
python spectral_bench.py --ops fft2d,conv2d_fft,spectrum --sizes 256x256,512x512,1024x1024 --device auto --repeats 30 --warmup 10

# DCT-II sweep (NumPy CPU)
python spectral_bench.py --ops dct2 --sizes 1024,2048,4096 --device cpu --repeats 20 --warmup 5

# Build report
python spectral_report.py --results results/results.csv --outdir results/report

echo "Done. See results/report/index.html"
