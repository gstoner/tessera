import os

import lit.formats

config.name = "tessera-nvidia"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]

if getattr(config, "environment", None) is None:
    config.environment = dict(os.environ)
tools = getattr(config, "llvm_tools_dir", "") or ""
path = config.environment.get("PATH", os.environ.get("PATH", ""))
if tools:
    config.environment["PATH"] = os.pathsep.join([tools, path])
else:
    config.environment.setdefault("PATH", path)
