#!/bin/bash
# X86-GEMM-ALIGN-1: before/after sweep, B%64 in {0,16,32,48} x shapes, three
# fresh processes each, every process under the fleet timing lock.
# usage: run_probe.sh <before-tree> <after-tree> <out.jsonl> [processes]
set -euo pipefail
before=$1; after=$2; out=$3; procs=${4:-3}
here=$(cd "$(dirname "$0")" && pwd)
shapes=("256 256 256" "250 250 250" "64 1024 1024" "16 1024 1024" "1024 1024 1024"
        "256 1000 256" "32 32 32" "1 256 256" "1 1024 1024" "256 256 4096"
        "64 128 16384" "2 256 256" "4 256 256" "8 64 64" "2 1024 1024" "4 1024 1024"
        "4 512 512")
: > "$out"
for s in "${shapes[@]}"; do
  set -- $s
  for off in 0 16 32 48; do
    for p in $(seq 1 "$procs"); do
      flock /tmp/tessera-timing.lock python "$here/probe_gemm_align.py" \
        --before "$before" --after "$after" -M "$1" -N "$2" -K "$3" \
        --offset-b "$off" --tag "p$p" >> "$out"
    done
  done
done
