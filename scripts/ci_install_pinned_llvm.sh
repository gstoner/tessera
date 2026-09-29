#!/usr/bin/env bash
# Install the exact LLVM/MLIR fleet pin for hosted compiler-proof lanes.
# The apt.llvm.org LLVM 23 repository is rolling; it supplied 23.1.2 while
# Tessera's fleet and CMake pin remained at 23.1.1. Use the official release
# archive and verify its SHA256 before extracting or executing any binary.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
pin="$(sed -n 's/^set(TESSERA_REQUIRED_LLVM_VERSION[[:space:]]*"\([0-9.]*\)".*/\1/p' "$repo_root/cmake/TesseraToolchainPins.cmake")"
version="23.1.1"
sha256="832aeb58d105de1cabc7b982dd2c65de0610f7377df48ae8fc2dd8e97420a15c"
if [[ "$pin" != "$version" ]]; then
  echo "::error ::LLVM archive pin $version disagrees with fleet pin $pin; update the official archive and digest together" >&2
  exit 1
fi

root="${TESSERA_CI_LLVM_ROOT:-${RUNNER_TEMP:-/tmp}/tessera-llvm-$version}"
prefix="$root/LLVM-$version-Linux-X64"
archive="${TESSERA_CI_LLVM_ARCHIVE:-$root/LLVM-$version-Linux-X64.tar.xz}"
url="https://github.com/llvm/llvm-project/releases/download/llvmorg-$version/LLVM-$version-Linux-X64.tar.xz"
mkdir -p "$root"
# Hosted runners carry large unrelated SDKs. Make space for the complete
# official MLIR development archive before extraction; never touch fleet hosts.
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  sudo rm -rf /usr/local/lib/android /usr/share/dotnet /opt/ghc /usr/local/share/powershell
fi
if [[ ! -f "$archive" ]]; then
  curl --fail --location --retry 3 --silent --show-error --output "$archive" "$url"
fi
echo "$sha256  $archive" | sha256sum --check --status || {
  echo "::error ::LLVM $version release archive SHA256 mismatch" >&2
  exit 1
}
if [[ ! -x "$prefix/bin/mlir-opt" ]]; then
  mkdir -p "$prefix"
  tar -xJf "$archive" -C "$prefix" --strip-components=1
fi
if [[ -n "${GITHUB_ACTIONS:-}" && -z "${TESSERA_CI_LLVM_ARCHIVE:-}" ]]; then
  rm -f "$archive"
fi
# LLVM's official Linux release uses ICU 70. Ubuntu 24/26 runners ship newer
# ICU, so carry the exact Ubuntu 22 library beside this isolated toolchain.
icu_archive="$root/libicu70_70.1-2_amd64.deb"
icu_sha256="58a154f6307289813da2276f900498ef536ae7c0522d2cf31a3c3c5cf62dfd9a"
icu_lib="$root/icu70/usr/lib/x86_64-linux-gnu"
if [[ ! -f "$icu_lib/libicui18n.so.70" ]]; then
  curl --fail --location --retry 3 --silent --show-error --output "$icu_archive" \
    "https://archive.ubuntu.com/ubuntu/pool/main/i/icu/libicu70_70.1-2_amd64.deb"
  echo "$icu_sha256  $icu_archive" | sha256sum --check --status || {
    echo "::error ::LLVM $version ICU runtime SHA256 mismatch" >&2
    exit 1
  }
  dpkg-deb -x "$icu_archive" "$root/icu70"
  rm -f "$icu_archive"
fi
export LD_LIBRARY_PATH="$icu_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for path in bin/llvm-config bin/mlir-opt bin/ld.lld lib/cmake/llvm/LLVMConfig.cmake lib/cmake/mlir/MLIRConfig.cmake; do
  if [[ ! -e "$prefix/$path" ]]; then
    echo "::error ::LLVM $version release bundle lacks $path" >&2
    exit 1
  fi
done
if [[ "$("$prefix/bin/llvm-config" --version)" != "$version" ]] ||
   ! "$prefix/bin/mlir-opt" --version | grep -Fq "LLVM version $version" ||
   ! "$prefix/bin/ld.lld" --version | grep -Fq "LLD $version"; then
  echo "::error ::LLVM/MLIR release bundle does not report $version" >&2
  exit 1
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    echo "TESSERA_CI_LLVM_PREFIX=$prefix"
    echo "LLVM_DIR=$prefix/lib/cmake/llvm"
    echo "MLIR_DIR=$prefix/lib/cmake/mlir"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  } >> "$GITHUB_ENV"
fi
if [[ -n "${GITHUB_PATH:-}" ]]; then
  echo "$prefix/bin" >> "$GITHUB_PATH"
fi
echo "Pinned LLVM/MLIR $version at $prefix"
