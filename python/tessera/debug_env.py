"""Environment-driven IR dump helpers.

When ``TESSERA_DEBUG_IR`` is set, IR snapshots are written for every JIT
artifact at the configured stages. Use this for "kernel ran, results wrong,
what now?" workflows — see
``docs/guides/Tessera_Debugging_Tools_Guide.md``.

Recognized environment variables
--------------------------------

``TESSERA_DEBUG_IR``
    Comma-separated list of stages to dump. Valid: ``graph``, ``schedule``,
    ``tile``, ``target``, or ``all``. Empty/unset disables dumping.
    Aliases: ``graph-ir`` / ``graph_ir`` accepted for compatibility with
    ``tessera-mlir --emit``. Whitespace is ignored.

``TESSERA_DEBUG_DUMP_DIR``
    Directory to write dumps into. Required when ``TESSERA_DEBUG_IR`` is
    non-empty; created on demand. Files are written as
    ``<symbol>.<stage>.mlir``; the ``<symbol>`` defaults to ``_jit`` if not
    supplied.

Example
-------

::

    TESSERA_DEBUG_IR=graph,schedule \\
    TESSERA_DEBUG_DUMP_DIR=/tmp/tessera-ir \\
    python my_script.py

After the run, ``/tmp/tessera-ir/`` contains files like
``my_jit_fn.graph.mlir`` and ``my_jit_fn.schedule.mlir`` — one per JIT
artifact × stage.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, Optional

# Canonical stage names used in dump filenames + env-var values.
_VALID_STAGES: Final = frozenset({"graph", "schedule", "tile", "target"})

# Allow ``-ir`` / ``_ir`` suffixes (matches `tessera-mlir --emit=graph-ir`).
_STAGE_ALIASES: Final = {
    "graph-ir": "graph",
    "graph_ir": "graph",
    "schedule-ir": "schedule",
    "schedule_ir": "schedule",
    "tile-ir": "tile",
    "tile_ir": "tile",
    "target-ir": "target",
    "target_ir": "target",
    "all": None,  # sentinel — expand to every stage
}


def _normalize(token: str) -> str | None:
    t = token.strip().lower()
    if not t:
        return None
    if t in _VALID_STAGES:
        return t
    if t in _STAGE_ALIASES:
        return _STAGE_ALIASES[t]
    return None  # unknown — silently dropped (don't crash user code)


def parse_debug_ir(value: str | None = None) -> frozenset[str]:
    """Parse ``TESSERA_DEBUG_IR`` (or the supplied string) into a set of stages.

    Returns an empty frozenset when the variable is unset or empty. ``all``
    expands to every valid stage.
    """
    raw = value if value is not None else os.environ.get("TESSERA_DEBUG_IR", "")
    if not raw:
        return frozenset()
    selected: set[str] = set()
    for token in raw.split(","):
        norm = _normalize(token)
        if norm is None:
            # 'all' or unknown
            if token.strip().lower() == "all":
                return frozenset(_VALID_STAGES)
            continue
        selected.add(norm)
    return frozenset(selected)


def dump_dir(value: str | None = None) -> Path | None:
    """Return the configured dump directory, creating it on demand.

    Returns ``None`` when ``TESSERA_DEBUG_DUMP_DIR`` is unset.
    """
    raw = value if value is not None else os.environ.get("TESSERA_DEBUG_DUMP_DIR")
    if not raw:
        return None
    p = Path(raw).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def should_dump(stage: Optional[str] = None) -> bool:
    """True when ``TESSERA_DEBUG_IR`` requests dumping. Pass ``stage`` to
    test a specific stage; omit to test "any dumping at all".
    """
    stages = parse_debug_ir()
    if not stages:
        return False
    if stage is None:
        return True
    return stage in stages


def dump_ir(
    stage: str,
    mlir: str,
    *,
    symbol: str = "_jit",
    directory: Path | None = None,
) -> Path | None:
    """Write ``mlir`` to ``<directory>/<symbol>.<stage>.mlir`` if requested.

    Returns the file path written, or ``None`` if dumping was not requested
    or the directory was not configured.

    Skips empty IR strings — they're not informative and create false-positive
    "the dump worked" signals.
    """
    if stage not in _VALID_STAGES:
        raise ValueError(
            f"Unknown IR stage {stage!r}; valid: {sorted(_VALID_STAGES)}"
        )
    if not mlir:
        return None
    if not should_dump(stage):
        return None
    target_dir = directory if directory is not None else dump_dir()
    if target_dir is None:
        return None
    safe_symbol = "".join(c if c.isalnum() or c in "._-" else "_" for c in symbol)
    path = target_dir / f"{safe_symbol}.{stage}.mlir"
    path.write_text(mlir)
    return path


def dump_artifact(
    symbol: str,
    *,
    graph_ir: str = "",
    schedule_ir: str = "",
    tile_ir: str = "",
    target_ir: str = "",
    directory: Path | None = None,
) -> dict[str, Path]:
    """Convenience: dump every requested stage for one JIT artifact.

    Returns a ``{stage: path}`` map for the files actually written. Stages
    not in ``TESSERA_DEBUG_IR`` are skipped silently.
    """
    written: dict[str, Path] = {}
    for stage, ir in (
        ("graph", graph_ir),
        ("schedule", schedule_ir),
        ("tile", tile_ir),
        ("target", target_ir),
    ):
        path = dump_ir(stage, ir, symbol=symbol, directory=directory)
        if path is not None:
            written[stage] = path
    return written


__all__ = [
    "parse_debug_ir",
    "dump_dir",
    "should_dump",
    "dump_ir",
    "dump_artifact",
]
