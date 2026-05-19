"""M2-Step-5 — Python object-model Tile IR equivalence harness.

Closes the M2 follow-up that was previously gated on ``tessera-opt``
building against MLIR 21.  This test locks the *Python side* of
the equivalence claim: for each shipped canonical-program driver,
the Python-emitted IR text is byte-identical to a checked-in
golden file, and the canonical's ``hash_ir_text`` digest matches.

When ``tessera-opt`` lands a working build, the missing piece is a
FileCheck pass that lowers the same canonical through the C++ MLIR
pipeline and asserts the same op shape.  Until then this guard
catches any silent Python-side IR drift — which is the loud
half of the equivalence contract.

Drift-fix workflow: when a canonical's emitted IR legitimately
changes, regenerate the golden with::

    PYTHONPATH=python venv/bin/python tests/unit/canonical_golden_ir/regenerate.py

(see :func:`_regenerate_goldens` below; the regenerator is a
plain Python helper rather than a CLI to stay self-contained).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from tessera.compiler.canonical import shipped_programs
from tessera.compiler.compile_report import hash_ir_text


GOLDEN_DIR = Path(__file__).parent / "canonical_golden_ir"

# Fixed shape arguments for every canonical that exposes a free
# ``_ir_text(*shape)`` function.  ``rotor_sandwich_norm`` builds IR
# via @clifford_jit decoration (the IR text is empty on non-Darwin
# hosts when the artifact isn't materialized) and is intentionally
# excluded from the byte-identity harness — its hash is still
# guarded by the CompileReport plan_hash stability gate
# (`test_compile_report_stability_gate.py`).
_SHAPES_FOR_GOLDEN: dict[str, tuple[int, ...]] = {
    "matmul_softmax_matmul":              (8, 8, 8),
    "conv2d_norm_activation":             (2, 4, 8, 8, 8, 3, 3),
    "kv_cache_append_prune_read":         (2, 4, 8, 1, 4),
    "decode_init_inner_loop_self_verify": (8, 8, 8, 8),
    "rotor_sandwich_ebt_tiny":            (8, 8, 8, 8),
}


def _ir_text_for(program_id: str) -> str:
    """Invoke the canonical's ``_ir_text`` with its locked shape."""
    args = _SHAPES_FOR_GOLDEN[program_id]
    mod = importlib.import_module(f"tessera.compiler.canonical.{program_id}")
    return mod._ir_text(*args)


# ─────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────

def test_golden_dir_exists_and_has_one_file_per_canonical() -> None:
    assert GOLDEN_DIR.is_dir(), f"missing golden dir: {GOLDEN_DIR}"
    actual = {p.stem for p in GOLDEN_DIR.glob("*.ir")}
    expected = set(_SHAPES_FOR_GOLDEN)
    assert actual == expected, (
        f"golden directory drift: expected {sorted(expected)}, "
        f"got {sorted(actual)}"
    )


@pytest.mark.parametrize("program_id", sorted(_SHAPES_FOR_GOLDEN))
def test_canonical_ir_text_matches_golden(program_id: str) -> None:
    """Byte-identical lock on the Python-emitted Tile-IR-shaped text.

    Any change in the canonical's ``_ir_text(...)`` output fails this
    test loudly — which is what we want: silent IR drift is the
    leading cause of "the canonical report digests changed and we
    didn't notice"."""
    emitted = _ir_text_for(program_id)
    golden = (GOLDEN_DIR / f"{program_id}.ir").read_text(encoding="utf-8")
    assert emitted == golden, (
        f"{program_id}: Python-emitted IR text drifted from golden.\n"
        f"If this is intentional, regenerate goldens via "
        f"tests/unit/canonical_golden_ir/regenerate.py"
    )


@pytest.mark.parametrize("program_id", sorted(_SHAPES_FOR_GOLDEN))
def test_canonical_ir_hash_is_deterministic(program_id: str) -> None:
    """Two invocations of ``_ir_text`` with the same shape produce
    the same hash.  This is the stability contract that
    CompileReport.ir_hashes relies on."""
    text_a = _ir_text_for(program_id)
    text_b = _ir_text_for(program_id)
    assert text_a == text_b
    assert hash_ir_text(text_a) == hash_ir_text(text_b)


def test_every_shipped_canonical_with_ir_text_is_in_the_golden_set() -> None:
    """If a new shipped canonical exposes ``_ir_text(...)``, this
    harness must cover it.  Catches the failure mode where someone
    adds a 7th canonical and forgets the golden file."""
    for program in shipped_programs():
        pid = program.program_id
        mod = importlib.import_module(f"tessera.compiler.canonical.{pid}")
        if hasattr(mod, "_ir_text"):
            assert pid in _SHAPES_FOR_GOLDEN, (
                f"canonical {pid!r} exposes _ir_text() but is not "
                f"covered by the M2-Step-5 golden harness.  Either "
                f"add a golden file + shape tuple to _SHAPES_FOR_GOLDEN, "
                f"or document why this canonical opts out."
            )


def test_canonicals_emit_required_op_set() -> None:
    """Structural check — each canonical's IR text must mention the
    op set that defines its identity.  Catches the regression where
    someone accidentally turns ``softmax`` into ``softmax_safe`` or
    drops a ``conv2d_nhwc`` step."""
    expected_ops = {
        "matmul_softmax_matmul":              ("matmul", "softmax"),
        "conv2d_norm_activation":             ("conv2d_nhwc", "layer_norm", "gelu"),
        "kv_cache_append_prune_read":         ("kv_cache",),
        "decode_init_inner_loop_self_verify": ("decode_init", "inner_step", "self_verify"),
        "rotor_sandwich_ebt_tiny":            ("rotor_sandwich", "ebt_tiny"),
    }
    for pid, ops in expected_ops.items():
        text = _ir_text_for(pid)
        for op in ops:
            assert op in text, (
                f"{pid}: expected op {op!r} in canonical IR text; "
                f"got:\n{text}"
            )


# ─────────────────────────────────────────────────────────────────
# Regeneration helper (not a test — invoked by hand when goldens drift)
# ─────────────────────────────────────────────────────────────────

def _regenerate_goldens() -> None:    # pragma: no cover
    """Rewrite every golden file from the current canonicals.

    Call from a Python REPL after legitimate IR shape changes::

        from tests.unit.test_canonical_ir_equivalence import _regenerate_goldens
        _regenerate_goldens()
    """
    for pid in _SHAPES_FOR_GOLDEN:
        text = _ir_text_for(pid)
        path = GOLDEN_DIR / f"{pid}.ir"
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}: {len(text)} bytes")
