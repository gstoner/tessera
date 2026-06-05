#!/usr/bin/env bash
# Tessera sanitizer smoke driver.
#
# Builds the C++ runtime + collectives smoke binaries with each
# requested sanitizer and runs them.  Designed to catch the classes
# of bug that prompted the 2026-05-19 static-review findings:
#
#   * ASAN  — heap/stack use-after-free (e.g., the pre-fix
#             tsrShutdown-then-handle-deref scenario), heap overflows,
#             leaks.  Pair with UBSan via -fsanitize=address,undefined.
#   * TSAN  — data races (e.g., the PerfettoTraceWriter race we hit
#             when first dropping the global submit mutex).
#   * UBSan — signed overflow, strict-aliasing UB, null deref.
#
# Usage:
#   scripts/run_sanitizers.sh                  # all three
#   scripts/run_sanitizers.sh asan             # just ASAN+UBSan
#   scripts/run_sanitizers.sh tsan
#   scripts/run_sanitizers.sh ubsan
#
# Each sanitizer uses a separate build directory
# (``build-${name}``) so configurations don't collide.  The smoke
# binary is the same one ``tests/unit/test_collective_runtime_lifecycle.py``
# drives in the regular Python sweep.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

cmake_name() {
  case "$1" in
    asan)  echo "address" ;;
    tsan)  echo "thread" ;;
    ubsan) echo "undefined" ;;
    *)     echo "" ;;
  esac
}

if (( $# > 0 )); then
  REQUESTED="$*"
else
  REQUESTED="asan tsan ubsan"
fi

# Each smoke binary covers a different C-side surface:
#
#   tessera-collective-runtime-smoke (src/collectives/...) — drives the
#     collective `tessera_*` entry points (qos_limit_set, submit_chunk_async,
#     shutdown_runtime, trace_write).  Catches PerfettoTraceWriter races
#     under TSAN, lifetime issues in the shared_ptr-based runtime slot.
#
#   tessera-runtime-abi-smoke (src/runtime/...) — drives the device
#     runtime `tsr*` C ABI (tsrInit/tsrMalloc/tsrCreateStream/tsrFree/
#     tsrShutdown).  Catches handle-lifetime corruption under ASAN
#     (the use-after-free shape the live-handle ratchet prevents),
#     counter-accounting races under TSAN, arithmetic UB under UBSAN.
#
# Both are built per sanitizer flavor and run sequentially so a
# single regression in either surface fails the lane.
declare -a SMOKES=(
  "tessera-collective-runtime-smoke|src/collectives/tools/tessera-collective-runtime-smoke/tessera-collective-runtime-smoke"
  "tessera-runtime-abi-smoke|src/runtime/tessera-runtime-abi-smoke"
)

run_one() {
  local label="$1"
  local cmake_val
  cmake_val="$(cmake_name "$label")"
  if [[ -z "$cmake_val" ]]; then
    echo "error: unknown sanitizer $label (expected asan/tsan/ubsan)" >&2
    return 2
  fi
  local build_dir="$REPO_ROOT/build-$label"
  echo
  echo "=========================================================="
  echo " Tessera sanitizer build: $label (CMake: $cmake_val)"
  echo "=========================================================="

  # Honor an existing LLVM_DIR/MLIR_DIR if the caller set one; else probe
  # the canonical Homebrew ``llvm`` keg (22.x — the current MLIR build
  # pin), falling back to the legacy ``llvm@21`` keg if that's all that's
  # installed.
  local llvm_flags=()
  local llvm_prefix=""
  if [[ -d "/opt/homebrew/opt/llvm/lib/cmake/llvm" ]]; then
    llvm_prefix="/opt/homebrew/opt/llvm"
  elif [[ -d "/opt/homebrew/opt/llvm@21/lib/cmake/llvm" ]]; then
    llvm_prefix="/opt/homebrew/opt/llvm@21"
  fi
  if [[ -n "${LLVM_DIR:-}" ]]; then
    llvm_flags+=(-DLLVM_DIR="$LLVM_DIR")
  elif [[ -n "$llvm_prefix" ]]; then
    llvm_flags+=(-DLLVM_DIR="$llvm_prefix/lib/cmake/llvm")
  fi
  if [[ -n "${MLIR_DIR:-}" ]]; then
    llvm_flags+=(-DMLIR_DIR="$MLIR_DIR")
  elif [[ -n "$llvm_prefix" ]]; then
    llvm_flags+=(-DMLIR_DIR="$llvm_prefix/lib/cmake/mlir")
  fi

  cmake -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DTESSERA_ENABLE_SANITIZERS="$cmake_val" \
    -DTESSERA_BUILD_APPLE_BACKEND=OFF \
    -DTESSERA_BUILD_EXAMPLES=OFF \
    "${llvm_flags[@]}" \
    >/dev/null
  # Build every smoke target.
  local targets=()
  for entry in "${SMOKES[@]}"; do
    targets+=("${entry%%|*}")
  done
  cmake --build "$build_dir" --target "${targets[@]}" -j

  # macOS ASAN doesn't ship a leak detector; the linux runtime does.
  local asan_opts="halt_on_error=1:abort_on_error=1:print_stacktrace=1"
  if [[ "$(uname -s)" == "Linux" ]]; then
    asan_opts="$asan_opts:detect_leaks=1"
  fi

  # Run each smoke binary under the sanitizer runtime.  A failure
  # in any binary fails the lane.
  local lane_rc=0
  for entry in "${SMOKES[@]}"; do
    local target="${entry%%|*}"
    local rel="${entry##*|}"
    local binary="$build_dir/$rel"
    if [[ ! -x "$binary" ]]; then
      echo "error: smoke binary not found at $binary" >&2
      lane_rc=2
      continue
    fi
    echo
    echo "--- running $target under $label ---"
    if ! ASAN_OPTIONS="$asan_opts" \
         TSAN_OPTIONS="halt_on_error=1:print_stacktrace=1:second_deadlock_stack=1" \
         UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1" \
         "$binary"; then
      echo "error: $target failed under $label" >&2
      lane_rc=1
    fi
  done
  return $lane_rc
}

FAILED=()
for label in $REQUESTED; do
  if ! run_one "$label"; then
    FAILED+=("$label")
  fi
done

echo
if (( ${#FAILED[@]} > 0 )); then
  echo "FAILED sanitizers: ${FAILED[*]}" >&2
  exit 1
fi
echo "[sanitizer-smoke] all OK: $REQUESTED"
