"""Precompile targeted profiling artifacts before attaching Nsight."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import importlib.util
    from tessera.compiler.emit import nvidia_cuda as nv
    spec = importlib.util.spec_from_file_location(
        "targeted", ROOT / "benchmarks/nvidia/profile_test5_transport_serving.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.main()
    args.output.write_text(json.dumps({
        "moe": nv._moe_artifact,
        "resident_ops": nv._resident_ops_artifact,
    }) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
