#!/usr/bin/env bash
# Host-free SM120 image packaging: pinned assembler/libdevice, without a driver.
set -euo pipefail

version="13.3.73"
release="13.3.1"
nvcc_sha256="2ff9f9954060794a1c5134a933ccb45bec723d866b2629dadfe4a1a313f21068"
nvvm_sha256="206b1ab4979c09b5c32f8bf907c42bc9e16cd7454cf6036f524c45a58d060f93"
root="${TESSERA_CI_CUDA_ROOT:-${RUNNER_TEMP:-/tmp}/tessera-cuda-$version}"
sdk="$root/sdk"
mkdir -p "$sdk"
install_component() {
  local component="$1" digest="$2" required="$3"
  local name="$component-linux-x86_64-$version-archive"
  local archive="$root/$name.tar.xz"
  local url="https://developer.download.nvidia.com/compute/cuda/redist/$component/linux-x86_64/$name.tar.xz"
  if [[ ! -f "$archive" ]]; then
    curl --fail --location --retry 3 --silent --show-error --output "$archive" "$url"
  fi
  echo "$digest  $archive" | sha256sum --check --status || {
    echo "::error ::CUDA $version $component SHA256 mismatch" >&2
    exit 1
  }
  if [[ ! -f "$sdk/$required" ]]; then
    tar -xJf "$archive" -C "$sdk" --strip-components=1
  fi
}
install_component cuda_nvcc "$nvcc_sha256" bin/ptxas
install_component libnvvm "$nvvm_sha256" nvvm/libdevice/libdevice.10.bc
for path in bin/ptxas nvvm/libdevice/libdevice.10.bc; do
  if [[ ! -f "$sdk/$path" ]]; then
    echo "::error ::CUDA $version image SDK lacks $path" >&2
    exit 1
  fi
done
if ! "$sdk/bin/ptxas" --version | grep -Fq "V$version"; then
  echo "::error ::CUDA image assembler does not report $version" >&2
  exit 1
fi
export CUDA_HOME="$sdk"
export PTXAS="$sdk/bin/ptxas"
python - "$sdk" "$nvcc_sha256" "$nvvm_sha256" "$release" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sdk=Path(sys.argv[1])
record={
    "release":sys.argv[4],
    "component_version":"13.3.73",
    "archive_sha256":{"cuda_nvcc":sys.argv[2],"libnvvm":sys.argv[3]},
    "manifest_url":"https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.3.1.json",
    "cuda_home":str(sdk),
    "ptxas_version":subprocess.check_output([str(sdk/"bin/ptxas"),"--version"],text=True).strip(),
    "files_sha256":{name:hashlib.sha256((sdk/name).read_bytes()).hexdigest()
                    for name in ("bin/ptxas","nvvm/libdevice/libdevice.10.bc")},
    "scope":"host-free SM120 PTX assembly and native image packaging; no GPU execution",
}
output=Path("ci-toolchain/cuda-image-sdk.json")
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
PY
if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    echo "CUDA_HOME=$CUDA_HOME"
    echo "PTXAS=$PTXAS"
  } >> "$GITHUB_ENV"
fi
if [[ -n "${GITHUB_PATH:-}" ]]; then
  echo "$sdk/bin" >> "$GITHUB_PATH"
fi
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  rm -f "$root/cuda_nvcc-linux-x86_64-$version-archive.tar.xz"
  rm -f "$root/libnvvm-linux-x86_64-$version-archive.tar.xz"
fi
echo "Pinned CUDA image SDK $version at $sdk"
