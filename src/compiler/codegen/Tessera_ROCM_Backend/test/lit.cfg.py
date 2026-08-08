import os
from pathlib import Path

import lit.formats

config.name = "tessera-rocm"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = ['.mlir']

# Make the LLVM tools the fixtures invoke (FileCheck, not, count) resolvable
# from RUN lines.  The site config sets ``config.llvm_tools_dir``; prepend it to
# PATH.  Self-contained on purpose — this does NOT depend on ``lit.llvm`` (which
# some standalone ``lit`` installs lack), only on the stdlib + the tools dir.
if getattr(config, "environment", None) is None:
    config.environment = dict(os.environ)
# Standalone lit may pre-create a minimal environment that omits the active
# ROCm alternatives prefix. Preserve the owning WSL toolchain selection so
# gpu-module-to-binary finds amdgcn/bitcode instead of guessing /opt/rocm.
for name in ("ROCM_PATH", "HIP_PATH"):
    if os.environ.get(name):
        config.environment[name] = os.environ[name]

# Final HSACO packaging requires AMD's device-library bitcode.  The host-free
# LLVM/MLIR lane intentionally installs only the linker, so keep Target/ROCDL
# coverage active there and enable binary fixtures only for a real ROCm SDK.
_rocm_path = config.environment.get("ROCM_PATH", "")
_device_lib_root = Path(_rocm_path) / "amdgcn" / "bitcode"
if _rocm_path and _device_lib_root.is_dir():
    config.available_features.add("rocm-device-libs")
_tools = getattr(config, "llvm_tools_dir", "") or ""
_path = config.environment.get("PATH", os.environ.get("PATH", ""))
if _tools:
    config.environment["PATH"] = os.pathsep.join([_tools, _path])
else:
    config.environment.setdefault("PATH", _path)
