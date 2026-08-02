#!/usr/bin/env bash
# X86-3: local exact-device correctness gate for Intel AMX-INT8.
#
# The proof bundle is retained locally. This project does not execute private
# hardware proofs through GitHub-hosted or self-hosted runners.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ "$#" -ne 0 ]; then
  echo "usage: $0" >&2
  exit 2
fi

PYTHON="${PYTHON:-}"
LOCAL_VENV="${TESSERA_X86_AMX_VENV:-$ROOT/.venv}"
if [ -z "$PYTHON" ]; then
  if [ -x "$LOCAL_VENV/bin/python" ]; then
    PYTHON="$LOCAL_VENV/bin/python"
  else
    PYTHON=python3
  fi
fi
RUN_ID="${TESSERA_X86_AMX_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)}"
REPORT_DIR="${TESSERA_X86_AMX_REPORT_DIR:-$ROOT/artifacts/x86-amx-release/$RUN_ID/device}"
LOCK_FILE="${TESSERA_X86_AMX_LOCK_FILE:-/tmp/tessera-x86-amx-release.lock}"

mkdir -p "$REPORT_DIR"
command -v flock >/dev/null 2>&1 || {
  echo "AMX release gate requires flock for host concurrency control" >&2
  exit 2
}
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another AMX exact-device proof owns $LOCK_FILE" >&2
  exit 3
fi
status_record() {
  rc=$1
  trap - EXIT
  if [ "$rc" -eq 0 ]; then
    status=success
  else
    status=failed
  fi
  printf 'status=%s\ncommit=%s\nexit_code=%s\n' \
    "$status" "$(git rev-parse HEAD)" "$rc" >"$REPORT_DIR/status.txt"
  exit "$rc"
}
trap 'status_record $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ]; then
  echo "AMX release gate requires a native Linux x86_64 host" >&2
  exit 2
fi
if ! grep -qm1 '^vendor_id[[:space:]]*: GenuineIntel$' /proc/cpuinfo; then
  echo "AMX release gate requires an Intel processor identity" >&2
  exit 2
fi
if ! grep -qm1 '\bamx_tile\b' /proc/cpuinfo || \
   ! grep -qm1 '\bamx_int8\b' /proc/cpuinfo; then
  echo "AMX release gate requires CPUID AMX-TILE and AMX-INT8" >&2
  exit 2
fi
if ! "$PYTHON" -c 'import numpy, pytest, xdist' >/dev/null 2>&1; then
  PYTHON=python3 bash scripts/install_test_deps.sh --venv
  PYTHON="$LOCAL_VENV/bin/python"
fi
CXX="${CXX:-c++}"
command -v "$CXX" >/dev/null 2>&1 || {
  echo "AMX release gate requires an AMX-capable C++ compiler; set CXX" >&2
  exit 2
}
export CXX
export PYTHONPATH="$ROOT/python:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

{
  uname -a
  lscpu
  grep -m3 -E '^(vendor_id|model name|flags)[[:space:]]*:' /proc/cpuinfo
  "$CXX" --version
  "$PYTHON" --version
  git rev-parse HEAD
} >"$REPORT_DIR/machine-identity.txt"
cat >"$REPORT_DIR/README.md" <<EOF
# Intel AMX exact-device proof bundle

Commit: $(git rev-parse HEAD)
Target: Intel AMX-TILE + AMX-INT8 with Linux tile-state permission

This directory is one local exact-device correctness bundle. Attach or
summarize it in the coordinating PR; it is not GitHub-hosted or
self-hosted-runner evidence. A successful result does not replace the separate
AMX performance packet required to close X86-3.
EOF

# Collection must be non-empty. Pytest exits 5 when this selector owns no tests.
"$PYTHON" -m pytest tests/device/x86 -m hardware_amx --collect-only -q \
  >"$REPORT_DIR/collection.txt"

# Repeat exact-device correctness to catch process/permission instability. The
# native invocation is child-process isolated inside the test, and xdist stays
# disabled so a fault cannot be mistaken for a worker crash or rerun.
"$PYTHON" -m pytest tests/device/x86 \
  -m "hardware_amx and not performance" -q -n 0 --durations=20 \
  --junitxml="$REPORT_DIR/device-correctness-1.xml"
"$PYTHON" -m pytest tests/device/x86 \
  -m "hardware_amx and not performance" -q -n 0 --durations=20 \
  --junitxml="$REPORT_DIR/device-correctness-2.xml"
