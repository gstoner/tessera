"""Audit-as-data coverage gate for the tessera.train package.

Step-4 decision (2026-06-16): ``tessera.train`` is surfaced to audit-as-data via
this **drift-gated coverage test**, deliberately NOT via a generated dashboard /
``SurfaceEntry`` row. Rationale:

* It gives the same anti-drift guarantee the rest of the repo has — a new model
  or skill cannot land untested / undocumented.
* It is **firewall-safe**: this test lives on the audit side, references the
  package by filesystem path, and greps test sources as text. Nothing here makes
  ``tessera.train`` import the compiler audit/registry machinery — the firewall
  (``test_train_agent_native_firewall.py``) only forbids that reverse arrow.
* It honors PithTrain principle 3 (no implicit indirection): this is pure audit
  data, never a construction-time registry — nothing builds models through it.
* A full generated dashboard is the wrong shape — train models are tested
  library classes, not standalone smoke-command scripts, and the package's
  "compact" principle argues against adding fleet machinery for four files.

If train ever grows runnable CLIs with smoke commands, promoting this to a
``surface_manifest`` entry consumed by ``surface_status`` is the natural next
step.
"""

from __future__ import annotations

import importlib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _REPO_ROOT / "python" / "tessera" / "train" / "models"
_SKILLS_DIR = _REPO_ROOT / "python" / "tessera" / "train" / "skills"
_TEST_DIR = Path(__file__).resolve().parent


def _model_modules() -> list[str]:
    return sorted(
        p.stem
        for p in _MODELS_DIR.glob("*.py")
        if p.stem != "__init__"
    )


def _train_test_sources() -> str:
    """Concatenated text of every train test except this manifest file."""
    chunks = []
    for p in sorted(_TEST_DIR.glob("test_train_*.py")):
        if p.name == Path(__file__).name:
            continue
        chunks.append(p.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def test_every_model_module_has_test_coverage():
    """Each model module exposes at least one public class referenced by a
    ``tests/unit/test_train_*.py`` source — model <-> test drift gate."""
    modules = _model_modules()
    assert modules, "no model modules discovered under train/models/"

    test_text = _train_test_sources()
    uncovered = []
    for name in modules:
        mod = importlib.import_module(f"tessera.train.models.{name}")
        public = getattr(mod, "__all__", None) or [
            n for n in dir(mod) if not n.startswith("_")
        ]
        classes = [n for n in public if n[:1].isupper()]
        if not any(cls in test_text for cls in classes):
            uncovered.append((name, classes))
    assert not uncovered, (
        "train model modules with no class referenced in any "
        f"test_train_*.py: {uncovered}"
    )


def test_every_skill_has_playbook_and_script():
    """Each skill directory ships a SKILL.md playbook and a runnable script —
    the agent-skill contract from train/__init__.py (principle 4)."""
    skill_dirs = sorted(p for p in _SKILLS_DIR.iterdir() if p.is_dir())
    assert skill_dirs, "no skills discovered under train/skills/"

    problems = []
    for d in skill_dirs:
        if not (d / "SKILL.md").is_file():
            problems.append(f"{d.name}: missing SKILL.md")
        scripts = list((d / "scripts").glob("*.py")) if (d / "scripts").is_dir() else []
        if not scripts:
            problems.append(f"{d.name}: no runnable script under scripts/")
    assert not problems, f"train skill contract violations: {problems}"
