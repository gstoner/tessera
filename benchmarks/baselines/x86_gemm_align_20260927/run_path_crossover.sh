#!/bin/bash
# X86-GEMM-ALIGN-1 path-selection crossover: builds three single-kernel shared
# libraries from one checkout and times them against each other with
# probe_gemm_paths.py (TSC witness, one fresh process per cell, timing lock).
#   old    = the kernel at d8da67f7 (before X86-GEMM-ALIGN-1)
#   packed = this checkout's kernel with the packed path forced for every M > 1
#   direct = this checkout's kernel with the direct (unpacked) path forced
# The forcing edits the path-selection predicate only (see sed lines below).
# usage: [NKS="N K,N K,..."] [MS="1 2 ..."] run_path_crossover.sh <checkout> <out.jsonl> [processes]
set -euo pipefail
repo=$1; out=$2; procs=${3:-1}
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
flags="-O2 -fPIC -shared -std=gnu++17 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni -mavx512bf16 -mavx512vpopcntdq -mf16c"
inc="-I $repo/src/compiler/layout_algebra/include"
src=src/compiler/codegen/tessera_x86_backend/src/kernels/avx512_gemm_f32.cpp
git -C "$repo" show d8da67f7:$src > "$work/old.cpp"
sed 's/^constexpr bool kForcePath = false; *\/\/ PATH-PROBE.*$/constexpr bool kForcePath = true;/; s/^constexpr bool kForcedPacked = false; *\/\/ PATH-PROBE.*$/constexpr bool kForcedPacked = true;/' "$repo/$src" > "$work/packed.cpp"
sed 's/^constexpr bool kForcePath = false; *\/\/ PATH-PROBE.*$/constexpr bool kForcePath = true;/' "$repo/$src" > "$work/direct.cpp"
cp "$repo/$src" "$work/shipped.cpp"
for v in old packed direct shipped; do g++ $flags $inc "$work/$v.cpp" -o "$work/lib_$v.so"; done
grep -q "kForcePath = true" "$work/packed.cpp" && grep -q "kForcedPacked = true" "$work/packed.cpp"
grep -q "kForcePath = true" "$work/direct.cpp"
{
  echo "# host=$(hostname) commit=$(git -C "$repo" rev-parse HEAD) compiler=$(g++ --version | head -1)"
  echo "# flags: $flags"
  for v in old packed direct shipped; do echo "# $v sha256=$(sha256sum "$work/lib_$v.so" | cut -d' ' -f1)"; done
} > "$out.header"
: > "$out"
IFS=, read -ra nks <<< "${NKS:-64 64,256 256,1024 1024,256 4096,4096 256}"
for nk in "${nks[@]}"; do
  set -- $nk; n=$1; k=$2
  for m in ${MS:-1 2 3 4 6 8 12 16}; do
    for off in 0 16; do
      for p in $(seq 1 "$procs"); do
        flock /tmp/tessera-timing.lock python "$here/probe_gemm_paths.py" \
          --lib old="$work/lib_old.so" --lib packed="$work/lib_packed.so" \
          --lib direct="$work/lib_direct.so" --lib shipped="$work/lib_shipped.so" \
          -M "$m" -N "$n" -K "$k" --offset-b "$off" --tag "p$p" >> "$out"
      done
    done
  done
done
