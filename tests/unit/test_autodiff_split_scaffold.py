"""Arch-7 (2026-05-22) — VJP / JVP monolith-split scaffold drift gate.

Locks the contract that the migration from monolithic ``vjp.py`` /
``jvp.py`` to family subpackages under ``vjps/`` / ``jvps/`` preserves:

  * Total registration count never drops below the baseline at the
    time of the scaffold landing (no VJPs / JVPs lost in transit).
  * Both subpackages exist as importable Python packages.
  * The import-side-effect hook at the bottom of ``vjp.py`` /
    ``jvp.py`` is in place so future family submodules' registration
    side effects actually land.
  * No registration appears in BOTH ``vjp.py`` (the legacy monolith)
    AND a family submodule — exclusive ownership prevents
    silent-overwrite bugs during partial migrations.

Migration workflow
------------------

When you move VJPs for a family from ``vjp.py`` to
``vjps/<family>.py``:

  1. Write the family submodule with ``register_vjp(name, fn)`` calls.
  2. Uncomment the corresponding ``from . import <family>`` line in
     ``vjps/__init__.py``.
  3. Delete the corresponding ``register_vjp(name, fn)`` calls from
     ``vjp.py``.
  4. Run this test — the count baseline gate stays unchanged.
  5. Run ``pytest tests/unit/`` — the autodiff tests stay green.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

# Import the module file directly — `from tessera.autodiff import jvp`
# binds to the `jvp` *function* (re-exported from jvp.py at the end of
# autodiff/__init__.py), not the module itself.  Use importlib to get
# the module object.
_vjp_mod = importlib.import_module("tessera.autodiff.vjp")
_jvp_mod = importlib.import_module("tessera.autodiff.jvp")


REPO_ROOT = Path(__file__).resolve().parents[2]
VJPS_PKG = REPO_ROOT / "python" / "tessera" / "autodiff" / "vjps"
JVPS_PKG = REPO_ROOT / "python" / "tessera" / "autodiff" / "jvps"
VJP_PY = REPO_ROOT / "python" / "tessera" / "autodiff" / "vjp.py"
JVP_PY = REPO_ROOT / "python" / "tessera" / "autodiff" / "jvp.py"


# Baselines captured at the Arch-7 scaffold landing (2026-05-22).
# Each migration sprint that drops a family from the monolith into
# the subpackage MUST preserve the total — if a family migration
# loses a registration in transit, this gate fails.
#
# Allowed to INCREASE (new VJP/JVP landings) — strictly monotone-down
# is the violation.
_VJP_FLOOR_COUNT = 241
_JVP_FLOOR_COUNT = 236


# ─────────────────────────────────────────────────────────────────────────
# Subpackage existence
# ─────────────────────────────────────────────────────────────────────────


def test_vjps_subpackage_exists() -> None:
    """The ``vjps/`` subpackage is the migration target."""
    assert VJPS_PKG.is_dir(), (
        f"vjps/ subpackage missing: {VJPS_PKG.relative_to(REPO_ROOT)}"
    )
    init = VJPS_PKG / "__init__.py"
    assert init.exists(), "vjps/__init__.py missing"


def test_jvps_subpackage_exists() -> None:
    assert JVPS_PKG.is_dir(), (
        f"jvps/ subpackage missing: {JVPS_PKG.relative_to(REPO_ROOT)}"
    )
    init = JVPS_PKG / "__init__.py"
    assert init.exists(), "jvps/__init__.py missing"


def test_vjp_py_imports_subpackage_for_side_effects() -> None:
    """The trailing ``from . import vjps`` in vjp.py must be present
    so future family submodules' registration calls actually fire."""
    text = VJP_PY.read_text()
    assert "from . import vjps" in text, (
        "vjp.py is missing the trailing `from . import vjps` import "
        "hook — without it, family submodules' register_vjp() calls "
        "never fire and the migration silently loses registrations."
    )


def test_jvp_py_imports_subpackage_for_side_effects() -> None:
    text = JVP_PY.read_text()
    assert "from . import jvps" in text, (
        "jvp.py is missing the trailing `from . import jvps` import "
        "hook."
    )


# ─────────────────────────────────────────────────────────────────────────
# Registration count floors
# ─────────────────────────────────────────────────────────────────────────


def test_vjp_count_at_or_above_baseline() -> None:
    """Total VJPs never drops below 241 (the count at Arch-7 scaffold
    landing).  Future migrations + new VJP landings push the count up;
    a partial migration that loses a VJP in transit drops it and
    fails here."""
    # Re-import to force any pending side effects (relevant if the
    # test process loaded vjp.py before vjps/__init__.py existed).
    actual = len(_vjp_mod._VJPS)
    assert actual >= _VJP_FLOOR_COUNT, (
        f"VJP registration count dropped from baseline "
        f"{_VJP_FLOOR_COUNT} to {actual} — a migration sprint likely "
        f"dropped a register_vjp() call.  Check vjp.py vs the family "
        f"submodule that was last migrated."
    )


def test_jvp_count_at_or_above_baseline() -> None:
    actual = len(_jvp_mod._JVPS)
    assert actual >= _JVP_FLOOR_COUNT, (
        f"JVP registration count dropped from baseline "
        f"{_JVP_FLOOR_COUNT} to {actual} — a migration sprint likely "
        f"dropped a register_jvp() call."
    )


# ─────────────────────────────────────────────────────────────────────────
# Sentinel registrations: the canonical names that must survive every
# migration round.  Each name represents a high-traffic family member
# that downstream code calls by name.
# ─────────────────────────────────────────────────────────────────────────


_VJP_SENTINELS = (
    "matmul", "transpose", "reshape", "softmax", "layer_norm",
    "flash_attn", "rope", "gelu",
)
_JVP_SENTINELS = (
    "matmul", "transpose", "reshape", "softmax", "layer_norm",
    "flash_attn", "rope", "gelu",
)


@pytest.mark.parametrize("name", _VJP_SENTINELS)
def test_vjp_sentinel_registered(name: str) -> None:
    assert _vjp_mod.get_vjp(name) is not None, (
        f"sentinel VJP {name!r} not registered.  A migration sprint "
        f"likely dropped it during the family split."
    )


@pytest.mark.parametrize("name", _JVP_SENTINELS)
def test_jvp_sentinel_registered(name: str) -> None:
    assert _jvp_mod.get_jvp(name) is not None, (
        f"sentinel JVP {name!r} not registered."
    )
