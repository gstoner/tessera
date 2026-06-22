#!/usr/bin/env bash
# =============================================================================
# Tessera — Ubuntu 24.04 (noble) developer setup
#
# Provisions the toolchain Tessera needs to build / lint / lit-test / unit-test
# on Linux, alongside the existing macOS Homebrew flow (which is unchanged — see
# CLAUDE.md "Local Toolchain"). Target environment:
#
#   * Ubuntu 24.04 LTS (noble)
#   * LLVM/MLIR 22 from apt.llvm.org  (ROCm's bundled LLVM has no MLIR)
#   * ROCm 7.2.4 at /opt/rocm  (the AMD ROCm backend's pinned toolchain)
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

LLVM_VERSION=22
ROCM_REQUIRED="7.2.4"
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
    # Repo signing key (idempotent).
    KEYRING=/etc/apt/keyrings/apt.llvm.org.gpg
    $SUDO install -d -m 0755 /etc/apt/keyrings
    if [[ ! -s "$KEYRING" ]]; then
      wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
        | $SUDO gpg --dearmor -o "$KEYRING"
    fi
    LIST=/etc/apt/sources.list.d/llvm-${LLVM_VERSION}.list
    echo "deb [signed-by=${KEYRING}] http://apt.llvm.org/${CODENAME}/ llvm-toolchain-${CODENAME}-${LLVM_VERSION} main" \
      | $SUDO tee "$LIST" >/dev/null
    $SUDO apt-get update -y
  fi
  # MLIR dev libs + tools are the part ROCm's LLVM lacks.
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
  python -m pip install --upgrade pip
  # numpy is capped <2.2: numpy>=2.2 ships PEP 695 `type` statements + stricter
  # reduction overloads in its bundled stubs that break the mypy ratchet under
  # the project's python_version=3.10 type-check target (the macOS env pins the
  # same band). Bump only alongside a baseline refresh.
  python -m pip install \
    "numpy>=2.0,<2.2" scipy ml_dtypes pyyaml click rich tqdm \
    pytest pytest-cov mypy ruff black isort flake8 lit
  python -c "import numpy, scipy, ml_dtypes; print('  python deps OK:', numpy.__version__)"
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
  -DCMAKE_PREFIX_PATH=/opt/rocm
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
  python -m pytest tests/unit/ -m "not slow"

  # MLIR lit tests (after building tessera-opt):
  python -m lit tests/tessera-ir/ -v

  # Generated-doc drift gate:
  bash scripts/check_generated_docs.sh

  # Validate ROCm 7.2.4 AMDGCN intrinsics against installed hipcc:
  python scripts/validate_hipcc_compile.py --hipcc /opt/rocm/bin/hipcc

See docs/GETTING_STARTED.md for the cross-platform build matrix (Linux + macOS).
EOF
