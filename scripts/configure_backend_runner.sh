#!/usr/bin/env bash
# Register one known physical backend host as a dedicated GitHub Actions runner.
#
# The registration token is obtained through the authenticated gh CLI unless
# GITHUB_RUNNER_REGISTRATION_TOKEN is supplied.  It is never echoed or saved.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/configure_backend_runner.sh PROFILE [--install-user-service]

Profiles:
  nvidia-rtx5070ti-sm120       RTX 5070 Ti, CUDA sm_120
  rocm-strix-halo-gfx1151      Ryzen AI Max+ 395 / Radeon 8060S, ROCm gfx1151
  apple-m1max-apple7           Apple M1 Max, Metal Apple7

The caller needs authenticated `gh` access to the repository and write access
to $HOME/actions-runner.  --install-user-service is supported on Linux only;
it needs loginctl lingering to survive logout.
EOF
}

[ "$#" -ge 1 ] || { usage >&2; exit 2; }
PROFILE="$1"
shift
INSTALL_USER_SERVICE=false
while [ "$#" -gt 0 ]; do
  case "$1" in
    --install-user-service) INSTALL_USER_SERVICE=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

REPOSITORY="${TESSERA_RUNNER_REPOSITORY:-gstoner/tessera}"
case "$PROFILE" in
  nvidia-rtx5070ti-sm120)
    RUNNER_NAME="tessera-nvidia-rtx5070ti-sm120"
    LABELS="nvidia-rtx5070ti-sm120,nvidia-sm120,nvidia"
    command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 2; }
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | \
      grep -Eq 'RTX 5070 Ti.*, *12\.0' || {
        echo "profile requires RTX 5070 Ti with compute capability 12.0" >&2; exit 2;
      }
    ;;
  rocm-strix-halo-gfx1151)
    RUNNER_NAME="tessera-rocm-strix-halo-gfx1151"
    LABELS="rocm-strix-halo-gfx1151,rocm-gfx1151,rocm"
    command -v rocminfo >/dev/null || { echo "rocminfo is required" >&2; exit 2; }
    rocminfo 2>/dev/null | grep -q 'gfx1151' || {
      echo "profile requires a gfx1151 ROCm device" >&2; exit 2;
    }
    ;;
  apple-m1max-apple7)
    RUNNER_NAME="tessera-apple-m1max-apple7"
    LABELS="apple-m1max-apple7,apple7,apple-gpu"
    [ "$(uname -s)" = Darwin ] || { echo "profile requires macOS" >&2; exit 2; }
    sysctl -n machdep.cpu.brand_string | grep -q 'Apple M1 Max' || {
      echo "profile requires an Apple M1 Max host" >&2; exit 2;
    }
    ;;
  *) echo "unknown profile: $PROFILE" >&2; usage >&2; exit 2 ;;
esac

case "$(uname -s):$(uname -m)" in
  Linux:x86_64) PLATFORM="linux-x64" ;;
  Darwin:arm64) PLATFORM="osx-arm64" ;;
  *) echo "unsupported runner platform: $(uname -s) $(uname -m)" >&2; exit 2 ;;
esac

RUNNER_ROOT="${TESSERA_RUNNER_ROOT:-$HOME/actions-runner}"
RUNNER_DIR="$RUNNER_ROOT/$RUNNER_NAME"
mkdir -p "$RUNNER_DIR"

if [ ! -x "$RUNNER_DIR/config.sh" ]; then
  command -v gh >/dev/null || { echo "gh is required to download the runner" >&2; exit 2; }
  VERSION="${GITHUB_ACTIONS_RUNNER_VERSION:-$(gh api repos/actions/runner/releases/latest --jq .tag_name)}"
  ARCHIVE="actions-runner-${PLATFORM}-${VERSION#v}.tar.gz"
  curl --fail --location --retry 3 \
    "https://github.com/actions/runner/releases/download/${VERSION}/${ARCHIVE}" \
    --output "$RUNNER_DIR/$ARCHIVE"
  tar -xzf "$RUNNER_DIR/$ARCHIVE" -C "$RUNNER_DIR"
  rm -f "$RUNNER_DIR/$ARCHIVE"
fi

if [ ! -f "$RUNNER_DIR/.runner" ]; then
  TOKEN="${GITHUB_RUNNER_REGISTRATION_TOKEN:-}"
  if [ -z "$TOKEN" ]; then
    command -v gh >/dev/null || { echo "gh is required to obtain a registration token" >&2; exit 2; }
    TOKEN="$(gh api -X POST "repos/$REPOSITORY/actions/runners/registration-token" --jq .token)"
  fi
  "$RUNNER_DIR/config.sh" --unattended --replace \
    --url "https://github.com/$REPOSITORY" --token "$TOKEN" \
    --name "$RUNNER_NAME" --labels "$LABELS" --work _work
  unset TOKEN
fi

if "$INSTALL_USER_SERVICE"; then
  [ "$(uname -s)" = Linux ] || {
    echo "use the documented launchd service procedure on macOS" >&2; exit 2;
  }
  UNIT_DIR="$HOME/.config/systemd/user"
  mkdir -p "$UNIT_DIR"
  cat >"$UNIT_DIR/${RUNNER_NAME}.service" <<EOF
[Unit]
Description=Tessera GitHub Actions runner ($RUNNER_NAME)
After=network-online.target

[Service]
ExecStart=$RUNNER_DIR/run.sh
WorkingDirectory=$RUNNER_DIR
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable --now "${RUNNER_NAME}.service"
  if ! loginctl show-user "$(id -u)" -p Linger --value 2>/dev/null | grep -qx yes; then
    echo "WARNING: enable lingering for persistence: sudo loginctl enable-linger $(id -un)" >&2
  fi
fi

printf 'Registered %s with exact label %s\n' "$RUNNER_NAME" "${LABELS%%,*}"
