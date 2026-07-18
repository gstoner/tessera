#!/usr/bin/env bash
# APPLE-CI-1: local exact-device proof gate for the Apple M1 Max / Apple7.
#
# This command runs directly on the backend Mac. It does not register or use a
# GitHub self-hosted runner. Publish the sealed evidence packet to the PR with
# --publish-dir after the local gate succeeds.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PUBLISH_DIR=""
if [ "${1:-}" = "--publish-dir" ]; then
  PUBLISH_DIR="${2:-}"
  shift 2
fi
if [ "$#" -ne 0 ]; then
  echo "usage: $0 [--publish-dir PATH]" >&2
  exit 2
fi

LLVM23_PREFIX="${TESSERA_LLVM23_PREFIX:-/opt/homebrew/llvm-23.1.0-rc1}"
RUN_ID="${TESSERA_APPLE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)}"
RUN_ROOT="${TESSERA_APPLE_RUN_ROOT:-/private/tmp/tessera-apple-metal4-$RUN_ID}"
REPORT_DIR="${TESSERA_APPLE_REPORT_DIR:-$ROOT/artifacts/apple-release/$RUN_ID/metal4}"
BUILD_DIR="$RUN_ROOT/build"
VENV_DIR="$RUN_ROOT/venv"
PYTHON="$VENV_DIR/bin/python"
LOCK_DIR="${TESSERA_APPLE_LOCK_DIR:-/private/tmp/tessera-apple-metal4-release.lock}"

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "Apple Metal 4 release proof requires a Darwin arm64 host" >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Apple Metal 4 release proof requires a clean tested commit" >&2
  exit 2
fi
for required in llvm-config mlir-opt clang clang++; do
  if [ ! -x "$LLVM23_PREFIX/bin/$required" ]; then
    echo "missing LLVM/MLIR 23 tool: $LLVM23_PREFIX/bin/$required" >&2
    exit 2
  fi
done
if [ ! -f "$LLVM23_PREFIX/lib/libmlir_c_runner_utils.dylib" ]; then
  echo "missing LLVM/MLIR 23 runner utils" >&2
  exit 2
fi
if [ "$("$LLVM23_PREFIX/bin/llvm-config" --version | cut -d. -f1)" != 23 ]; then
  echo "Apple release proof requires LLVM major 23" >&2
  exit 2
fi
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "another Apple Metal 4 proof owns $LOCK_DIR" >&2
  exit 3
fi

mkdir -p "$REPORT_DIR" "$RUN_ROOT/cache"
SOURCE_COMMIT="$(git rev-parse HEAD)"
printf '%s\n' "$SOURCE_COMMIT" > "$REPORT_DIR/source-commit.txt"
printf 'status=running\ncommit=%s\n' "$SOURCE_COMMIT" > "$REPORT_DIR/status.txt"

finish() {
  rc=$?
  trap - EXIT
  if [ "$rc" -ne 0 ]; then
    printf 'status=failed\ncommit=%s\nexit_code=%s\n' \
      "$SOURCE_COMMIT" "$rc" > "$REPORT_DIR/status.txt"
  fi
  rmdir "$LOCK_DIR" 2>/dev/null || true
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

{
  printf '%s\n' '[sw_vers]'
  sw_vers
  printf '%s\n' '[system_profiler SPDisplaysDataType]'
  system_profiler SPDisplaysDataType
  printf '%s\n' '[xcodebuild]'
  xcodebuild -version 2>&1 || printf '%s\n' \
    'unavailable: full Xcode is not selected; Command Line Tools are sufficient'
  printf '%s\n' '[macOS SDK]'
  xcrun --sdk macosx --show-sdk-version
  printf '%s\n' '[metal compiler]'
  xcrun --find metal 2>&1 || printf '%s\n' \
    'unavailable: offline metal CLI is not required by runtime source compilation'
  printf '%s\n' '[llvm-config]'
  "$LLVM23_PREFIX/bin/llvm-config" --version
  printf '%s\n' '[python]'
  python3 --version
  printf '%s\n' '[commit]'
  printf '%s\n' "$SOURCE_COMMIT"
} > "$REPORT_DIR/machine-identity.txt"

python3 scripts/capture_apple_metal4_environment.py \
  --output "$REPORT_DIR/metal4-environment.json"
python3 scripts/validate_apple_metal4_evidence.py environment \
  "$REPORT_DIR/metal4-environment.json"

python3 -m venv "$VENV_DIR"
PYTHON="$PYTHON" bash scripts/install_test_deps.sh
"$PYTHON" -m pip install -e . --no-deps

LLVM_LIT="$VENV_DIR/bin/lit"
if [ ! -x "$LLVM_LIT" ]; then
  echo "Apple Metal 4 release proof requires lit in its fresh environment" >&2
  exit 2
fi
cmake -S . -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$LLVM23_PREFIX/bin/clang" \
  -DCMAKE_CXX_COMPILER="$LLVM23_PREFIX/bin/clang++" \
  -DLLVM_DIR="$LLVM23_PREFIX/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM23_PREFIX/lib/cmake/mlir" \
  -DLLVM_EXTERNAL_LIT="$LLVM_LIT" \
  -DTESSERA_BUILD_APPLE_BACKEND=ON \
  -DTESSERA_BUILD_NVIDIA_BACKEND=OFF \
  -DTESSERA_BUILD_ROCM_BACKEND=OFF \
  -DTESSERA_BUILD_EXAMPLES=ON
cmake --build "$BUILD_DIR" --target \
  tessera-opt tessera-translate-mlir tessera_jit TesseraAppleRuntime
cp "$BUILD_DIR/CMakeCache.txt" "$REPORT_DIR/apple-cmake-cache.txt"
cp "$BUILD_DIR/.ninja_log" "$REPORT_DIR/ninja-log.txt"

export PYTHONPATH="$ROOT/python:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TESSERA_CACHE_DIR="$RUN_ROOT/cache"
export TESSERA_OPT="$BUILD_DIR/tools/tessera-opt/tessera-opt"
export TESSERA_OPT_BIN="$TESSERA_OPT"
export TESSERA_JIT_LIB="$BUILD_DIR/tools/tessera-jit/libtessera_jit.dylib"
export TESSERA_MLIR_C_RUNNER_UTILS="$LLVM23_PREFIX/lib/libmlir_c_runner_utils.dylib"

"$PYTHON" -c 'import json; from tessera import runtime as r; from tessera.compiler.apple_route_selector import live_apple_device_tag; caps=r.apple_gpu_metal4_caps(); assert r.DeviceTensor.is_metal(); assert caps.get("available"); device=live_apple_device_tag(); assert "unknown" not in device; print(json.dumps({"device":device,"metal4":caps},sort_keys=True))' \
  > "$REPORT_DIR/metal4-capabilities.json"
"$PYTHON" scripts/validate_apple_metal4_evidence.py capabilities \
  "$REPORT_DIR/metal4-capabilities.json"

for run in 1 2; do
  "$PYTHON" -m pytest tests/unit -q -n 0 \
    -m "metal4 and hardware_apple_gpu and not performance" \
    --durations=100 \
    --junitxml="$REPORT_DIR/metal4-correctness-$run.xml"
  "$PYTHON" scripts/validate_apple_metal4_evidence.py junit \
    "$REPORT_DIR/metal4-correctness-$run.xml"
done

for run in 1 2; do
  "$PYTHON" benchmarks/apple_gpu/benchmark_route_characterization.py \
    --ops matmul --matmul-shapes 64x64x64 256x256x256 \
    --reps 10 --trials 5 \
    --output "$REPORT_DIR/metal4-routes-$run.json"
  "$PYTHON" scripts/validate_apple_metal4_evidence.py route-report \
    "$REPORT_DIR/metal4-routes-$run.json"
done
"$PYTHON" benchmarks/apple_gpu/select_stable_gemm_routes.py \
  "$REPORT_DIR/metal4-routes-1.json" \
  "$REPORT_DIR/metal4-routes-2.json" \
  --output "$REPORT_DIR/metal4-stable-ledger.json"
"$PYTHON" scripts/validate_apple_metal4_evidence.py ledger \
  "$REPORT_DIR/metal4-stable-ledger.json"
printf 'status=success\ncommit=%s\nexit_code=0\n' "$SOURCE_COMMIT" > "$REPORT_DIR/status.txt"
"$PYTHON" scripts/validate_apple_metal4_evidence.py seal "$REPORT_DIR"

if [ -n "$PUBLISH_DIR" ]; then
  mkdir -p "$PUBLISH_DIR"
  cp -R "$REPORT_DIR/." "$PUBLISH_DIR/"
  echo "published Apple Metal 4 proof packet: $PUBLISH_DIR"
else
  echo "retained Apple Metal 4 proof packet: $REPORT_DIR"
fi
