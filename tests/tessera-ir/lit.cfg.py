
# -*- Python -*-
import os, shutil, subprocess, lit.formats
config.name = "Tessera-IR v0.3.1"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = ['.mlir']
config.environment['PATH'] = os.environ.get('PATH', '')
config.substitutions.append(('tessera-opt', 'tessera-opt -allow-unregistered-dialect'))

# Probe tessera-opt for optional backends so per-target fixtures can
# REQUIRE the right feature. We only mark the feature as available if
# the corresponding pipeline alias is actually registered in this build.
def _opt_help_contains(needle: str) -> bool:
    binary = shutil.which("tessera-opt")
    if binary is None:
        return False
    try:
        out = subprocess.run([binary, "--help"], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:
        return False
    return needle in out

if _opt_help_contains("tessera-lower-to-metalium"):
    config.available_features.add("tessera-metalium-backend")
if _opt_help_contains("tessera-lower-to-rocm"):
    config.available_features.add("tessera-rocm-backend")
