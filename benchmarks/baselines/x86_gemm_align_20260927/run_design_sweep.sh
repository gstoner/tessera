#!/bin/bash
# Design sweep behind the X86-GEMM-ALIGN-1 choice: candidates in
# design_variants.cpp at B%64 in {0,16,32,48}, one fresh process each, under
# the fleet timing lock. Timing: CLOCK_MONOTONIC_RAW, median of 9 samples
# (a design-phase screen; the before/after table uses the TSC-witnessed probe).
# usage: run_design_sweep.sh <out.txt>
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=$1
bin=$(mktemp -d)/design_variants
cmd="g++ -O2 -mavx512f -mavx512bw -mavx512dq -mavx512vl -std=gnu++17 $here/design_variants.cpp -o $bin"
$cmd
{
  echo "# host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
  echo "# compiler=$(g++ --version | head -1)"
  echo "# compile: $cmd"
  echo "# design_variants.cpp sha256=$(sha256sum "$here/design_variants.cpp" | cut -d' ' -f1)"
  echo "# variants: v0 pre-fix kernel; v1 whole-B padded copy; v2 per-strip K x 16 panel;"
  echo "#   v3 8-strip panels, one accumulator; v7 8-strip K x 128 panel, 8 accumulators (shipped, M>1);"
  echo "#   v8 the same blocking read directly from B (shipped, M==1)"
} > "$out"
shapes=("16 16 16 200000" "32 32 32 50000" "1 256 256 20000" "2 256 256 10000" "4 256 256 5000"
        "8 256 256 3000" "1 1024 1024 200" "2 1024 1024 100" "4 1024 1024 50" "16 1024 1024 16"
        "64 1024 1024 4" "250 250 250 30" "256 256 256 30" "256 1000 256 8" "1024 256 256 8"
        "256 256 4096 2" "64 128 16384 4" "1024 1024 1024 1" "1 4096 4096 8" "2 4096 4096 4")
for s in "${shapes[@]}"; do
  set -- $s
  for off in 0 16 32 48; do
    for v in 0 1 2 3 7 8; do
      # v1 copies all of B on every call; skip it where B is 64 MiB
      if [ "$v" = 1 ] && [ $(( $2 * $3 )) -ge 16777216 ]; then continue; fi
      flock /tmp/tessera-timing.lock "$bin" $v $1 $2 $3 $off $4 8 >> "$out"
    done
  done
done
