#!/usr/bin/env bash
# Shared ROCm toolkit resolver for Tessera's device lanes. Source it BEFORE
# launching python/pytest; it exports ROCM_PATH, HIP_PATH, PATH and
# LD_LIBRARY_PATH for a detected ROCm install.
#
#   source "$(dirname "$0")/_rocm_env.sh"
#   "${TESSERA_PY}" -m pytest tests/unit/ -q
#
# Why this exists
# ---------------
# TheRock / packaged ROCm installs under /opt/rocm/core but exports nothing
# system-wide: ROCM_PATH, PATH and LD_LIBRARY_PATH come from an *interactive*
# .bashrc, which a non-interactive shell (an `ssh <host> <cmd>`, a CI step, a
# bare `pytest`) never sources. Two things then break, and neither says why:
#
#   1. MLIR's ROCDL `gpu-module-to-binary` serializer finds `ld.lld` through
#      ROCM_PATH. Without it every compiled ROCm lane dies at hsaco
#      serialization with `error: lld invocation failed` — which reads like a
#      compiler bug, not a missing variable. (~1.5k spurious failures in a bare
#      `pytest tests/unit/` sweep on the Strix Halo box, root-caused 2026-08-23.)
#   2. libamdhip64's transitive deps (comgr, hsa-runtime) resolve through the
#      dynamic loader. `ldconfig` on that box knows only a stale ROCm 5
#      libamdhip64.so.5, so without LD_LIBRARY_PATH they bind against the wrong
#      ROCm and a concentrated device sweep SEGFAULTS partway through.
#
# LD_LIBRARY_PATH is read by glibc's loader at process startup ONLY. That is why
# this is a shell script sourced before launch and NOT a pytest conftest hook:
# a conftest setting os.environ cannot repair its own already-running process,
# and re-exec'ing from inside pytest loses the output stream.
#
#   3. TESSERA_ROCM_CHIP selects the arch every compiled lane is built FOR, and
#      it used to default to gfx1151 with nothing on the host correcting it. On
#      the gfx1201 box a full sweep then compiles gfx1151 images, fails to load
#      them, and reports ~1444 failures saying `no usable AMD GPU for cached
#      module (..., 'gfx1151')`. Because that refusal names an arch which is
#      not the host's, the conftest hook correctly declines to soften it into a
#      skip — so a label error is indistinguishable from a broken branch until
#      you read the text. (Cost two full validate runs, 2026-09-19.)
#
# Detection is by capability (does this root actually have ld.lld? which arch
# does the device report?) rather than by hardcoded path or a default, and an
# already-exported ROCM_PATH / TESSERA_ROCM_CHIP is respected. A fleet with two
# ROCm architectures cannot have a correct default, and "remember to export it"
# is documentation, not a mechanism. On a host with no ROCm at all (Mac /
# NVIDIA boxes) this is a silent no-op — it must NOT fabricate a device, so
# those lanes still skip honestly (repo Decision #26).

_tessera_rocm_lld_dir() {
  # Echo the directory holding ld.lld under a toolkit root, if any.
  # TheRock mirrors the LLVM bin at lib/llvm/bin; a classic install uses llvm/bin.
  local root="${1:-}" rel
  [ -n "${root}" ] || return 1
  for rel in lib/llvm/bin llvm/bin; do
    if [ -x "${root}/${rel}/ld.lld" ]; then
      printf '%s\n' "${root}/${rel}"
      return 0
    fi
  done
  return 1
}

_tessera_rocm_resolve_root() {
  local candidate
  for candidate in "${ROCM_PATH:-}" /opt/rocm/core /opt/rocm; do
    [ -n "${candidate}" ] || continue
    if _tessera_rocm_lld_dir "${candidate}" >/dev/null; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

_tessera_rocm_detect_chip() {
  # Echo the launch architecture of the first real GPU agent, or fail.
  # `rocm_agent_enumerator` also lists `gfx000` for the CPU agent; that is not
  # a launch target and must never be selected.
  local enum out gfx
  for enum in "${TESSERA_ROCM_ROOT}/bin/rocm_agent_enumerator" \
              "${TESSERA_ROCM_ROOT}/bin/rocminfo"; do
    [ -x "${enum}" ] || continue
    out="$("${enum}" 2>/dev/null)" || continue
    gfx="$(printf '%s\n' "${out}" \
             | grep -oE 'gfx[0-9a-f]+' \
             | grep -v '^gfx000$' \
             | head -n 1)"
    if [ -n "${gfx}" ]; then
      printf '%s\n' "${gfx}"
      return 0
    fi
  done
  return 1
}

_tessera_rocm_prepend() {
  # Prepend $2 to the os.pathsep-style var named $1 unless already present.
  local var="${1}" entry="${2}" current
  [ -d "${entry}" ] || return 0
  eval "current=\${${var}:-}"
  case ":${current}:" in
    *":${entry}:"*) return 0 ;;
  esac
  if [ -n "${current}" ]; then
    export "${var}=${entry}:${current}"
  else
    export "${var}=${entry}"
  fi
}

if TESSERA_ROCM_ROOT="$(_tessera_rocm_resolve_root)"; then
  export TESSERA_ROCM_ROOT
  export ROCM_PATH="${TESSERA_ROCM_ROOT}"
  export HIP_PATH="${TESSERA_ROCM_ROOT}"

  # PATH: the toolkit's own bin (rocm_agent_enumerator, amdclang) and the LLVM
  # bin that carries ld.lld — same order the box's interactive shell uses.
  _tessera_rocm_lld_bin="$(_tessera_rocm_lld_dir "${TESSERA_ROCM_ROOT}")"
  _tessera_rocm_prepend PATH "${_tessera_rocm_lld_bin}"
  _tessera_rocm_prepend PATH "${TESSERA_ROCM_ROOT}/bin"

  # LD_LIBRARY_PATH: <root>/lib first, then any sibling extras*/lib (WSL's
  # librocdxg lives there). Prepending in reverse yields that final order.
  _tessera_rocm_parent="$(dirname "${TESSERA_ROCM_ROOT}")"
  for _tessera_rocm_extra in "${_tessera_rocm_parent}"/extras*/lib; do
    _tessera_rocm_prepend LD_LIBRARY_PATH "${_tessera_rocm_extra}"
  done
  unset _tessera_rocm_parent
  _tessera_rocm_prepend LD_LIBRARY_PATH "${TESSERA_ROCM_ROOT}/lib"
  unset _tessera_rocm_extra _tessera_rocm_lld_bin

  # WSL2 exposes the GPU as /dev/dxg (there is no /dev/kfd); HSA needs this to
  # enumerate the device. Harmless on native Linux.
  case "$(uname -r)" in
    *microsoft*|*WSL*|*wsl*) export HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}" ;;
  esac

  # TESSERA_ROCM_CHIP: the arch every compiled lane is built FOR. It used to
  # default to gfx1151 with nothing on the host correcting it, so on the
  # gfx1201 box a full sweep compiled gfx1151 images, failed to load them, and
  # reported ~1444 failures reading `no usable AMD GPU for cached module
  # (..., 'gfx1151')` -- a label error that looks exactly like a broken branch.
  # The refusal names an arch that is not the host's, so the conftest hook
  # correctly declines to soften it to a skip, and the run is unusable.
  #
  # Detect it from the device for the same reason the toolkit root is detected
  # by capability: a fleet with two ROCm architectures cannot have a correct
  # default, and "remember to export it" is not a mechanism. An explicit
  # setting always wins -- a test that pins a foreign arch on purpose must
  # still be able to.
  if [ -z "${TESSERA_ROCM_CHIP:-}" ]; then
    if _tessera_rocm_detected_chip="$(_tessera_rocm_detect_chip)"; then
      export TESSERA_ROCM_CHIP="${_tessera_rocm_detected_chip}"
    fi
    unset _tessera_rocm_detected_chip
  fi
fi
