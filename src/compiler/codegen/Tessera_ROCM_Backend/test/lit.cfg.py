import os

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
_tools = getattr(config, "llvm_tools_dir", "") or ""
_path = config.environment.get("PATH", os.environ.get("PATH", ""))
if _tools:
    config.environment["PATH"] = os.pathsep.join([_tools, _path])
else:
    config.environment.setdefault("PATH", _path)
