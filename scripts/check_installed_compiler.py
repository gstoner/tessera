#!/usr/bin/env python3
"""Exercise installed compiler drivers from a relocated prefix, outside the checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def check(prefix: Path) -> dict[str, object]:
    prefix = prefix.resolve()
    with tempfile.TemporaryDirectory(prefix="tessera-installed-") as directory:
        root = Path(directory)
        relocated = root / "prefix"
        shutil.copytree(prefix, relocated, symlinks=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith(
            ("TESSERA_", "PYTHONPATH", "LD_", "DYLD_"))}
        env["PATH"] = "/usr/bin:/bin"
        results = {}
        for name in ("tessera-opt", "tessera-translate-mlir"):
            executable = relocated / "bin" / name
            version = subprocess.run([str(executable), "--version"], cwd=root,
                                     env=env, text=True, capture_output=True,
                                     check=True, timeout=30).stdout
            results[name] = {"sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                             "version": version.strip()}
        source = 'module { llvm.func @identity(%x: i32) -> i32 { llvm.return %x : i32 } }'
        optimized = subprocess.run([str(relocated / "bin/tessera-opt"), "--canonicalize"],
                                   input=source, cwd=root, env=env, text=True,
                                   capture_output=True, check=True, timeout=30).stdout
        translated = subprocess.run([str(relocated / "bin/tessera-translate-mlir"),
                                     "--mlir-to-llvmir"], input=optimized, cwd=root,
                                    env=env, text=True, capture_output=True,
                                    check=True, timeout=30).stdout
        if "define i32 @identity" not in translated or "ret i32" not in translated:
            raise RuntimeError("installed drivers did not produce the identity LLVM function")
        return {"drivers": results, "relocated": True, "loader_overrides": False,
                "llvm_translation": True, "device_execution": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(check(args.prefix), indent=2) + "\n")


if __name__ == "__main__":
    main()
