"""P1 (2026-06-09) — `apple_runtime_ops.inc` drift gate (Python-only).

The C++ Tile→Apple GPU pass consumes a generated X-macro table of the
runtime envelope. This test regenerates the table from
``apple_gpu_envelope.py`` and fails if the committed ``.inc`` is stale —
no C++ build needed (the build-time pass behavior is locked separately by
``test_apple_gpu_tile_pass_status_matches_envelope``).
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "scripts" / "generate_apple_runtime_ops_table.py"
INC = (REPO_ROOT / "src" / "compiler" / "codegen" / "Tessera_Apple_Backend"
       / "include" / "Tessera" / "Target" / "Apple" / "apple_runtime_ops.inc")


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_apple_runtime_ops", GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_inc_matches_generator_output():
    gen = _load_generator()
    assert INC.is_file(), f"missing {INC}; run `python {GENERATOR.name}`"
    assert INC.read_text() == gen.generate(), (
        f"{INC} is stale; regenerate with `python scripts/{GENERATOR.name}`")


def test_inc_op_set_equals_envelope():
    from tessera.compiler.apple_gpu_envelope import _APPLE_GPU_RUNTIME_OPS

    ops = set(re.findall(r'TESSERA_APPLE_GPU_RUNTIME_OP\("([^"]+)"\)', INC.read_text()))
    assert ops == set(_APPLE_GPU_RUNTIME_OPS)
