"""``tessera-translate`` — inter-IR translation CLI.

Python-side counterpart to ``tools/tessera-translate/`` (see README
there for the long-term plan).  Today this CLI exposes the
file-format export surface from :mod:`tessera.aot` (S14) under a
single stable entry point::

    tessera-translate stablehlo  --in artifact.zip --out model.mlir
    tessera-translate gguf       --in artifact.zip --out model.gguf
    tessera-translate safetensors --in artifact.zip --out model.st

When the C++ ``tessera-translate`` MLIR binary lands (gated on
``tessera-opt`` building against MLIR 21), it will install under
``tessera-translate-mlir`` so this Python entry point keeps the
short name.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


def _cmd_stablehlo(args: argparse.Namespace) -> int:
    from tessera import aot
    artifact = aot.load(args.input)
    text = aot.stablehlo_export(artifact)
    Path(args.output).write_text(text, encoding="utf-8")
    print(f"wrote StableHLO: {args.output} ({len(text)} bytes)")
    return 0


def _cmd_gguf(args: argparse.Namespace) -> int:
    from tessera import aot
    artifact = aot.load(args.input)
    out_path = aot.gguf_export(artifact, args.output)
    print(f"wrote GGUF: {out_path}")
    return 0


def _cmd_safetensors(args: argparse.Namespace) -> int:
    from tessera import aot
    # SafeTensors export takes a state mapping; for the CLI we load
    # the artifact's recorded params dict.
    artifact = aot.load(args.input)
    state = getattr(artifact, "params", None)
    if state is None:
        # AOTArtifact may not expose `params` directly; the artifact
        # zip should carry a `params.json` we can decode.
        params_path = Path(args.input) / "params.json"
        if params_path.exists():
            state = json.loads(params_path.read_text(encoding="utf-8"))
        else:
            print(
                f"error: no params found inside {args.input}; "
                f"safetensors export needs a state mapping",
                file=sys.stderr,
            )
            return 2
    out_path = aot.safetensors_export(state, args.output)
    print(f"wrote SafeTensors: {out_path}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    """Print a one-shot summary of an AOT artifact."""
    from tessera import aot
    artifact = aot.load(args.input)
    fields = {
        "target":  getattr(artifact, "target", None),
        "graph_ir_present":  bool(getattr(artifact, "graph_ir", None)),
        "tile_ir_present":   bool(getattr(artifact, "tile_ir", None)),
        "metadata":          getattr(artifact, "metadata", None),
    }
    print(json.dumps(fields, indent=2, default=str))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tessera-translate",
        description="Translate a Tessera AOT artifact to an external IR format.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_st = sub.add_parser("stablehlo", help="Export to StableHLO MLIR text")
    p_st.add_argument("--in", dest="input", required=True)
    p_st.add_argument("--out", dest="output", required=True)
    p_st.set_defaults(func=_cmd_stablehlo)

    p_gguf = sub.add_parser("gguf", help="Export to GGUF binary")
    p_gguf.add_argument("--in", dest="input", required=True)
    p_gguf.add_argument("--out", dest="output", required=True)
    p_gguf.set_defaults(func=_cmd_gguf)

    p_safe = sub.add_parser("safetensors", help="Export model params to SafeTensors")
    p_safe.add_argument("--in", dest="input", required=True)
    p_safe.add_argument("--out", dest="output", required=True)
    p_safe.set_defaults(func=_cmd_safetensors)

    p_info = sub.add_parser("info", help="Print AOT artifact summary")
    p_info.add_argument("--in", dest="input", required=True)
    p_info.set_defaults(func=_cmd_info)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":    # pragma: no cover
    raise SystemExit(main())
