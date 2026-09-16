"""Every native GPU storage consumer replays the packager's own arena pipeline.

`build_native_gpu_storage` gained `--tessera-expand-lowp-conversions` in
100a2980 while six consumers kept a hand-copied four-pass replay, so every
resident heap, SSD, ANN, exception-heap, public-result and gradient-sum package
was refused on both device boxes with "disagrees with native replay"
(2026-09-15). The pipeline is spelled once, in `native_gpu_storage.ARENA_PIPELINE`.
"""
from pathlib import Path
import re

from tessera.compiler import native_gpu_storage

COMPILER = Path(native_gpu_storage.__file__).resolve().parent
TOKEN = re.compile(r"""['"]--tessera-tile-buffer-arena['"]""")


def test_arena_pipeline_is_spelled_once():
    offenders = sorted(
        str(path.relative_to(COMPILER))
        for path in COMPILER.rglob("*.py")
        if path.name != "native_gpu_storage.py" and TOKEN.search(path.read_text(encoding="utf-8", errors="replace"))
    )
    assert offenders == [], f"replay the packager pipeline via replay_arena_ir, not a copy: {offenders}"


def test_packager_pipeline_expands_lowp_before_canonicalize():
    passes = native_gpu_storage.ARENA_PIPELINE
    assert passes.index("--tessera-tile-buffer-arena") < passes.index("--tessera-expand-lowp-conversions") < passes.index("--canonicalize")


def test_replay_uses_the_packager_pipeline(monkeypatch):
    calls = []
    monkeypatch.setattr(native_gpu_storage, "_run", lambda compiler, *args, source: calls.append((compiler, args, source)) or "ir")
    assert native_gpu_storage.replay_arena_ir(Path("/opt/tessera-opt"), "module {}") == "ir"
    assert calls == [(Path("/opt/tessera-opt"), native_gpu_storage.ARENA_PIPELINE, "module {}")]
