#!/usr/bin/env bash
# Host-free HSACO image proof: compiler/device bitcode only, no GPU driver.
set -euo pipefail

version="10.0.0rc4"
wheel_name="rocm_sdk_core-$version-py3-none-linux_x86_64.whl"
sha256="930c00c36aa67fd0b5fc5b59bc7078acecde07e6e27d97da56db2c9059d8551b"
url="https://rocm.prereleases.amd.com/whl-multi-arch/$wheel_name"
root="${TESSERA_CI_ROCM_ROOT:-${RUNNER_TEMP:-/tmp}/tessera-rocm-$version}"
archive="${TESSERA_CI_ROCM_ARCHIVE:-$root/$wheel_name}"
mkdir -p "$root"
if [[ ! -f "$archive" ]]; then
  curl --fail --location --retry 3 --silent --show-error --output "$archive" "$url"
fi
echo "$sha256  $archive" | sha256sum --check --status || {
  echo "::error ::ROCm $version core wheel SHA256 mismatch" >&2
  exit 1
}
# Wheel installation retains executable modes and native launcher placement.
# Keep it outside the project Python environment and the machine's /opt/rocm.
python -m pip install --no-deps --no-compile --upgrade --target "$root/python" "$archive"
sdk="$root/python/_rocm_sdk_core"
# The wheel uses lib/llvm; native MLIR's toolkit serializer and AMD clang also
# consume the standard llvm/ and amdgcn/bitcode paths. Both name the same bytes.
ln -sfn lib/llvm "$sdk/llvm"
ln -sfn lib/llvm/amdgcn "$sdk/amdgcn"
export ROCM_PATH="$sdk"
export HIP_PATH="$sdk"
export TESSERA_ROCM_CLANG="$sdk/lib/llvm/bin/amdclang++"
for path in lib/llvm/bin/amdclang++ lib/llvm/bin/ld.lld amdgcn/bitcode/ocml.bc amdgcn/bitcode/ockl.bc; do
  if [[ ! -e "$sdk/$path" ]]; then
    echo "::error ::ROCm core wheel lacks $path" >&2
    exit 1
  fi
done
python - "$sdk" "$archive" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sdk, archive = map(Path, sys.argv[1:])
manifest = json.loads((sdk / "share/therock/therock_manifest.json").read_text())
if (manifest["rocm_package_version"] != "10.0.0rc4" or
        manifest["the_rock_commit"] != "16adc4d875fd4f65ea23c7c84e1c66706fde3047"):
    raise SystemExit("ROCm wheel source manifest differs from the pinned SDK")
clang = sdk / "lib/llvm/bin/amdclang++"
clang_version = subprocess.check_output([str(clang), "--version"], text=True)
if "8f497e0992fb7513f7f78a6f6b6f1056c375e961" not in clang_version:
    raise SystemExit("AMD LLVM compiler differs from the pinned SDK")
record = {
    "version": manifest["rocm_package_version"],
    "the_rock_commit": manifest["the_rock_commit"],
    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    "rocm_path": str(sdk),
    "amd_clang_version": clang_version.strip(),
    "bitcode_sha256": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((sdk / "amdgcn/bitcode").glob("*.bc"))
    },
    "scope": "host-free native image packaging; no device execution",
}
output = Path("ci-toolchain/rocm-image-sdk.json")
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    echo "ROCM_PATH=$ROCM_PATH"
    echo "HIP_PATH=$HIP_PATH"
    echo "TESSERA_ROCM_CLANG=$TESSERA_ROCM_CLANG"
  } >> "$GITHUB_ENV"
fi
if [[ -n "${GITHUB_PATH:-}" ]]; then
  echo "$sdk/lib/llvm/bin" >> "$GITHUB_PATH"
fi
if [[ -n "${GITHUB_ACTIONS:-}" && -z "${TESSERA_CI_ROCM_ARCHIVE:-}" ]]; then
  rm -f "$archive"
fi
echo "Pinned ROCm image SDK $version at $sdk"
