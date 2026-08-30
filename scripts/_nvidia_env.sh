#!/usr/bin/env bash
# Shared NVIDIA/CUDA resolver for Tessera's device lanes. Source it BEFORE
# launching python/pytest; it puts the driver shim and the CUDA toolkit on PATH
# and LD_LIBRARY_PATH.
#
#   source "$(dirname "$0")/_nvidia_env.sh"
#   "${TESSERA_PY}" -m pytest tests/device/nvidia/ -q
#
# Why this exists
# ---------------
# The NVIDIA device gate is `tests/_support/nvidia.py::nvidia_cuda_host_ready`,
# which requires `shutil.which("nvidia-smi")`. Under WSL2 the driver shim lives
# in /usr/lib/wsl/lib -- a directory the interactive .bashrc adds and a
# non-interactive shell (an `ssh <host> <cmd>`, a CI step, a bare `pytest`)
# never sees.
#
# The failure mode is a GREEN RUN, not an error. Measured on The-Super-Bear
# 2026-08-30: `pytest tests/device/nvidia/` reported **454 passed, 395 skipped,
# exit 0** while executing zero GPU work -- every skip read "host WSL CUDA
# device/toolchain unavailable". The GPU, /dev/dxg and CUDA 13.3 were all
# present and healthy the entire time; only PATH was wrong. Reporting that run
# as sm_120 evidence would have asserted a hardware result that never happened,
# which is the exact trap CLAUDE.md's Working Rules name ("a missing device
# *skips* rather than errors").
#
# Once the PATH was fixed the suite ran for real and surfaced 80 genuine
# failures that had been invisible behind the skip, two of which were compiler
# defects. So this is not hygiene: the silent skip was actively hiding bugs.
#
# This is the NVIDIA twin of _rocm_env.sh, and it follows the same contract:
# detect by capability, respect anything already exported, and be a SILENT
# NO-OP on a host with no NVIDIA GPU so Mac/ROCm boxes still skip honestly
# (Decision #26) instead of having a device fabricated for them.
#
# Every expansion is nounset-safe (${VAR:-}). A release or CI script sourcing
# this under `set -u` would otherwise abort on the first unset CUDA_HOME or
# LD_LIBRARY_PATH instead of detecting the toolkit.

# Driver shim (WSL2 puts nvidia-smi and libcuda.so here; native Linux does not
# have this directory at all, and does not need it).
if [ -d /usr/lib/wsl/lib ]; then
  case ":${PATH:-}:" in
    *":/usr/lib/wsl/lib:"*) ;;
    *) PATH="/usr/lib/wsl/lib:${PATH}"; export PATH ;;
  esac
  case ":${LD_LIBRARY_PATH:-}:" in
    *":/usr/lib/wsl/lib:"*) ;;
    *) LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
       export LD_LIBRARY_PATH ;;
  esac
fi

# CUDA toolkit. An already-exported CUDA_HOME wins; otherwise prefer the
# versioned symlink over a bare guess.
if [ -z "${CUDA_HOME:-}" ]; then
  for _tessera_cuda_root in /usr/local/cuda /opt/cuda; do
    if [ -x "${_tessera_cuda_root}/bin/nvcc" ]; then
      CUDA_HOME="${_tessera_cuda_root}"
      export CUDA_HOME
      break
    fi
  done
  unset _tessera_cuda_root
fi
if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
  case ":${PATH:-}:" in
    *":${CUDA_HOME}/bin:"*) ;;
    *) PATH="${CUDA_HOME}/bin:${PATH}"; export PATH ;;
  esac
  if [ -d "${CUDA_HOME}/lib64" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":${CUDA_HOME}/lib64:"*) ;;
      *) LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
         export LD_LIBRARY_PATH ;;
    esac
  fi
fi
