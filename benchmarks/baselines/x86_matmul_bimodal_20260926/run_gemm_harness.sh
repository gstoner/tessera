#!/usr/bin/env bash
# X86-MATMUL-BIMODAL-1: build gemm_harness.cpp against this checkout's GEMM
# kernel source and run the B-offset x THP-request matrix, three fresh
# processes per setting, under the shared timing lock.
# usage: run_gemm_harness.sh <repo-root> <output.txt>
set -euo pipefail
repo=$(cd "$1" && pwd)
out=$2
here=$(cd "$(dirname "$0")" && pwd)
kernel=$repo/src/compiler/codegen/tessera_x86_backend/src/kernels/avx512_gemm_f32.cpp
bin=$(mktemp -d)/gemm_harness
cmd=(g++ -O2 -mavx512f -mavx512bw -mavx512dq -mavx512vl -std=gnu++17
     -I "$repo/src/compiler/layout_algebra/include" "$here/gemm_harness.cpp" "$kernel" -o "$bin")
"${cmd[@]}"
{
  echo "# host: $(hostname) | $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
  echo "# kernel: $(uname -r)"
  echo "# date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# source_commit: $(git -C "$repo" rev-parse HEAD)"
  echo "# compiler: $(g++ --version | head -1)"
  echo "# compile: ${cmd[*]}"
  echo "# gemm_harness.cpp sha256: $(sha256sum "$here/gemm_harness.cpp" | cut -d' ' -f1)"
  echo "# avx512_gemm_f32.cpp sha256: $(sha256sum "$kernel" | cut -d' ' -f1)"
  echo "# THP mode: $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
  echo "# args: offA=0 offB=<ob> offC=0 hugepage=<h>; three processes per setting"
  flock /tmp/tessera-timing.lock bash -c '
    for r in 1 2 3; do for ob in 0 16 32 48 64; do for h in 0 1; do
      echo -n "run=$r offB=$ob huge=$h "; "$0" 0 "$ob" 0 "$h"
    done; done; done' "$bin"
} > "$out"
