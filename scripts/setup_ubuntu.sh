#!/usr/bin/env bash
# =============================================================================
# Tessera — Ubuntu developer setup
#
# Provisions the toolchain Tessera needs to build / lint / lit-test / unit-test
# on Linux, alongside the existing macOS Homebrew flow (which is unchanged — see
# CLAUDE.md "Local Toolchain"). Target environment:
#
#   * A supported Ubuntu release
#   * LLVM/MLIR 23 from apt.llvm.org
#   * TheRock ROCm 7.14 at /opt/rocm/core
#   * A project-local Python venv (.venv) with the lean dependency set
#
# This script needs root for the apt steps; it calls `sudo` itself, so run it
# as a normal user:
#
#   bash scripts/setup_ubuntu.sh                 # full setup
#   bash scripts/setup_ubuntu.sh --no-llvm       # skip the LLVM apt install
#   bash scripts/setup_ubuntu.sh --no-python     # skip the venv
#   bash scripts/setup_ubuntu.sh --configure     # also run the cmake configure
#
# It is idempotent: re-running is safe.
# =============================================================================
set -euo pipefail

LLVM_VERSION=23
ROCM_REQUIRED="7.14"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLVM_PREFIX="/usr/lib/llvm-${LLVM_VERSION}"

DO_LLVM=1 DO_PYTHON=1 DO_APT=1 DO_CONFIGURE=0
for arg in "$@"; do
  case "$arg" in
    --no-llvm)    DO_LLVM=0 ;;
    --no-python)  DO_PYTHON=0 ;;
    --no-apt)     DO_APT=0 ;;
    --configure)  DO_CONFIGURE=1 ;;
    -h|--help)    grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown arg: $arg (try --help)" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

SUDO=""
if [[ $EUID -ne 0 ]]; then
  command -v sudo >/dev/null 2>&1 || die "need root or sudo for the apt steps"
  SUDO="sudo"
fi

# ---------------------------------------------------------------------------
say "Checking OS"
# ---------------------------------------------------------------------------
. /etc/os-release || die "cannot read /etc/os-release"
CODENAME="${VERSION_CODENAME:-noble}"
echo "  ${PRETTY_NAME:-unknown}  (codename: ${CODENAME})"
[[ "$ID" == "ubuntu" ]] || warn "not Ubuntu — apt package names assume Debian/Ubuntu"

configure_llvm_repo() {
  local keyring=/etc/apt/keyrings/apt.llvm.org.gpg
  local list=/etc/apt/sources.list.d/llvm-${LLVM_VERSION}.list
  local versioned_suite=llvm-toolchain-${CODENAME}-${LLVM_VERSION}
  local snapshot_suite=llvm-toolchain-${CODENAME}
  local suite=$versioned_suite

  # apt.llvm.org publishes the development major through the unversioned
  # snapshot suite. Once that major branches, it moves to the versioned suite.
  # Probe both so setup remains valid across that transition and across Ubuntu
  # releases (for example Resolute currently serves LLVM 23 as the snapshot).
  if ! wget -q --spider \
      "https://apt.llvm.org/${CODENAME}/dists/${versioned_suite}/Release"; then
    suite=$snapshot_suite
    wget -q --spider \
      "https://apt.llvm.org/${CODENAME}/dists/${suite}/Release" \
      || die "apt.llvm.org has neither ${versioned_suite} nor ${snapshot_suite}"
  fi

  $SUDO install -d -m 0755 /etc/apt/keyrings
  if [[ ! -s "$keyring" ]]; then
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
      | $SUDO gpg --dearmor -o "$keyring"
  fi
  echo "deb [signed-by=${keyring}] https://apt.llvm.org/${CODENAME}/ ${suite} main" \
    | $SUDO tee "$list" >/dev/null
  echo "  apt suite: ${suite}"
}

# Repair/configure the LLVM source before the first apt update. This matters
# when a previous failed attempt left an invalid source in llvm-23.list.
if [[ $DO_LLVM -eq 1 && ! -x "${LLVM_PREFIX}/bin/mlir-tblgen" ]]; then
  say "Configuring LLVM/MLIR ${LLVM_VERSION} apt source (codename: ${CODENAME})"
  configure_llvm_repo
fi

# ---------------------------------------------------------------------------
if [[ $DO_APT -eq 1 ]]; then
  say "Installing base build dependencies (apt)"
  $SUDO apt-get update -y
  $SUDO apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git pkg-config \
    python3 python3-venv python3-dev python3-pip \
    zlib1g-dev libzstd-dev libtinfo-dev ca-certificates wget gnupg lsb-release
else
  warn "skipping base apt deps (--no-apt)"
fi

# ---------------------------------------------------------------------------
if [[ $DO_LLVM -eq 1 ]]; then
  say "Installing LLVM/MLIR ${LLVM_VERSION} from apt.llvm.org (codename: ${CODENAME})"
  if [[ -x "${LLVM_PREFIX}/bin/mlir-tblgen" ]]; then
    echo "  ${LLVM_PREFIX} already has MLIR — skipping repo add"
  else
    $SUDO apt-get update -y
  fi
  # Use one matched upstream LLVM/MLIR major for the whole compiler build.
  $SUDO apt-get install -y \
    llvm-${LLVM_VERSION} llvm-${LLVM_VERSION}-dev llvm-${LLVM_VERSION}-tools \
    libmlir-${LLVM_VERSION}-dev mlir-${LLVM_VERSION}-tools \
    libpolly-${LLVM_VERSION}-dev \
    clang-${LLVM_VERSION} lld-${LLVM_VERSION}

  # Sanity: MLIR CMake config + tools must exist.
  [[ -d "${LLVM_PREFIX}/lib/cmake/mlir" ]] || die "MLIRConfig.cmake missing under ${LLVM_PREFIX}"
  "${LLVM_PREFIX}/bin/llvm-config" --version
  "${LLVM_PREFIX}/bin/mlir-tblgen" --version | head -1
  "${LLVM_PREFIX}/bin/FileCheck" --version | head -1 || true
else
  warn "skipping LLVM install (--no-llvm)"
fi

# ---------------------------------------------------------------------------
say "Checking ROCm (the AMD backend's pinned toolchain — pin: ${ROCM_REQUIRED})"
# ---------------------------------------------------------------------------
# This script does NOT install ROCm — that is a large, machine-specific AMD-repo
# operation (see https://rocm.docs.amd.com). It only verifies what is present.
if command -v hipcc >/dev/null 2>&1 || [[ -x /opt/rocm/bin/hipcc ]]; then
  HIPCC="$(command -v hipcc || echo /opt/rocm/bin/hipcc)"
  HIPVER="$("$HIPCC" --version 2>/dev/null | grep -i 'HIP version' | head -1 || true)"
  echo "  hipcc: ${HIPCC}"
  echo "  ${HIPVER:-<version unknown>}"
  ROCM_DIR="$(ls -d /opt/rocm-${ROCM_REQUIRED} /opt/rocm 2>/dev/null | head -1 || true)"
  echo "  ROCm dir: ${ROCM_DIR:-<not found>}"
  # GPU detection: rocminfo is authoritative (native /dev/kfd OR WSL /dev/dxg).
  GPU_ARCH=""
  if command -v rocminfo >/dev/null 2>&1; then
    GPU_ARCH="$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | sort -u | paste -sd, -)"
  fi
  if [[ -n "$GPU_ARCH" ]]; then
    echo "  GPU arch(es): ${GPU_ARCH}"
    echo "  validate the toolchain for your GPU with:"
    echo "    python scripts/validate_hipcc_compile.py --hipcc ${HIPCC} --arch ${GPU_ARCH%%,*}"
  elif [[ -e /dev/dxg ]]; then
    warn "WSL /dev/dxg present but rocminfo found no GPU agent — check the WSL ROCm runtime."
  elif [[ ! -e /dev/kfd ]]; then
    warn "no GPU agent (no /dev/kfd, no /dev/dxg). The ROCm backend builds as Target IR /"
    warn "lit artifacts regardless; this is fine for development."
  fi
  warn "Note: Tessera's ROCm *runtime* execution path (launch bridge) is still gated"
  warn "(Phase H) even where silicon + hipcc are present — IR/artifact + lit only today."
else
  warn "hipcc not found. Install ROCm ${ROCM_REQUIRED} per https://rocm.docs.amd.com"
  warn "(e.g. the amdgpu-install package). The CPU/x86 + LLVM build works without it;"
  warn "the ROCm backend needs hipcc to compile-validate kernels."
fi

# ---------------------------------------------------------------------------
if [[ $DO_PYTHON -eq 1 ]]; then
  say "Creating Python venv (.venv) with the lean dependency set"
  # Lean set: runtime numerics + dev tooling. Torch/transformers are reference
  # vocabularies only (Decision #23) and intentionally omitted; add them by hand
  # if a specific reference test needs them.
  python3 -m venv "${REPO_ROOT}/.venv"
  # shellcheck disable=SC1091
  . "${REPO_ROOT}/.venv/bin/activate"
  # Runtime numerics + test/lint/type tooling — delegated to the single source
  # of truth so this script and install_test_deps.sh never drift (the latter
  # also carries pytest-timeout / pytest-xdist and the numpy<2.2 mypy-ratchet
  # cap, and verifies the install).
  PYTHON=python bash "${REPO_ROOT}/scripts/install_test_deps.sh"
else
  warn "skipping Python venv (--no-python)"
fi

# ---------------------------------------------------------------------------
CONFIGURE_CMD=$(cat <<EOF
cmake -S "${REPO_ROOT}" -B "${REPO_ROOT}/build" -G Ninja \\
  -DLLVM_DIR=${LLVM_PREFIX}/lib/cmake/llvm \\
  -DMLIR_DIR=${LLVM_PREFIX}/lib/cmake/mlir \\
  -DTESSERA_ENABLE_HIP=ON \\
  -DTESSERA_BUILD_ROCM_BACKEND=ON \\
  -DCMAKE_HIP_COMPILER=/opt/rocm/core/lib/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm/core
ninja -C "${REPO_ROOT}/build" tessera-opt
EOF
)

if [[ $DO_CONFIGURE -eq 1 ]]; then
  say "Configuring the C++ build (ROCm + LLVM ${LLVM_VERSION})"
  eval "$CONFIGURE_CMD"
else
  say "Setup complete. Configure + build the C++ tree with:"
  echo
  echo "${CONFIGURE_CMD}"
fi

cat <<EOF

Next steps
----------
  source ${REPO_ROOT}/.venv/bin/activate
  export PYTHONPATH=${REPO_ROOT}/python

  # Python unit tests (no C++ build needed):
  python -m pytest tests/unit/ -m \
    "not slow and not performance and not hardware_apple_gpu and not hardware_nvidia and not hardware_rocm"

  # MLIR lit tests (after building tessera-opt):
  python -m lit tests/tessera-ir/ -v

  # Generated-doc drift gate:
  bash scripts/check_generated_docs.sh

  # Validate AMDGCN intrinsics against the installed ROCm hipcc:
  python scripts/validate_hipcc_compile.py --hipcc /opt/rocm/bin/hipcc

See docs/GETTING_STARTED.md for the cross-platform build matrix (Linux + macOS).
EOF
