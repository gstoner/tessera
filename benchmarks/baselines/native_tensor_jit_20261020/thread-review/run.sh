#!/usr/bin/env bash
set -euo pipefail
source /home/angstorms/scratch/w11-program-pipeline-20261019/jit-env.sh
cd /home/angstorms/scratch/bounded-source-attention-20261019/worktree
task_root=/home/angstorms/scratch/w11-program-pipeline-20261019/jit-thread-abba
candidate_archive="$task_root/jit-candidate-source.tar.gz"
test ! -e "$candidate_archive"
tar czf "$candidate_archive" python/tessera/compiler/jit.py python/tessera/compiler/execution_matrix.py python/tessera/compiler/native_tensor_program.py python/tessera/runtime.py
restore_candidate() {
  tar xzf "$candidate_archive"
}
trap restore_candidate EXIT
arm() {
  local route="$1"
  local label="$2"
  local code=0
  python -m benchmarks.nvidia.record_native_tensor_jit --route "$route" --label "$label" --rounds 21 --repeats 16 --output "$task_root/$label.json" > "$task_root/$label.txt" 2>&1 || code=$?
  echo "$code" > "$task_root/$label.exit"
  tail -8 "$task_root/$label.txt"
  return "$code"
}
arm native jit-abba-native-a1
tar xzf "$task_root/ordinary-jit-control-source.tar.gz"
arm legacy jit-abba-legacy-b1
arm legacy jit-abba-legacy-b2
restore_candidate
arm native jit-abba-native-a2
