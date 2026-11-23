"""LEAN-PILOT-1 gate: the Lean 4 pilot proofs and the Tessera text they model.

Two kinds of check, so a missing toolchain can never read as a proof:

* **Host-independent (always run).** Every pilot theorem file names the Tessera
  source it models with ``tessera-anchor: <path> :: <text>`` lines; each anchor
  must still occur verbatim in that file, so a refactor that moves or changes
  the modeled line fails here instead of leaving a proof about code that no
  longer exists (Decision #29: the proof's consumer is this gate). The files
  must also state what they do NOT cover, contain no ``sorry``/``admit``/
  ``axiom``, and the Lean project must stay out of every CMake build.
* **Toolchain-gated.** ``lake build`` and the axiom audit run only when
  ``lake`` is on PATH *and* Mathlib's prebuilt cache is present; otherwise the
  test skips with the reason. A skip means "this host did not check the
  proofs", never that they passed.

Plan: docs/audit/compiler/LEAN_VALIDATION_PILOT_PLAN.md.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_LEAN = _REPO / "formal" / "lean"
_THEOREMS = {
    "VJPExp.lean": "vjpExp_eq_pullback",
    "PagedKVSlotMap.lean": "physical_injective",
    "SoftmaxShift.lean": "softmax_shift",
}
_ANCHOR = re.compile(r"^tessera-anchor:\s*(?P<path>\S+)\s*::\s*(?P<text>.+?)\s*$", re.M)
_ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def _theorem_files() -> list[Path]:
    return [_LEAN / "TesseraFormal" / name for name in _THEOREMS]


def test_pilot_has_exactly_the_three_planned_theorem_files() -> None:
    present = sorted(p.name for p in (_LEAN / "TesseraFormal").glob("*.lean"))
    assert present == sorted(_THEOREMS), (
        "LEAN-PILOT-1 is scoped to three theorems; widen the plan before adding "
        f"files. present={present}"
    )
    root = (_LEAN / "TesseraFormal.lean").read_text(encoding="utf-8")
    for name in _THEOREMS:
        assert f"import TesseraFormal.{name[:-5]}" in root, f"{name} not built by the root"


@pytest.mark.parametrize("path", _theorem_files(), ids=lambda p: p.name)
def test_theorem_file_anchors_still_match_tessera_source(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    anchors = list(_ANCHOR.finditer(text))
    assert anchors, f"{path.name}: no `tessera-anchor:` lines naming the modeled source"
    for m in anchors:
        source = _REPO / m.group("path")
        assert source.is_file(), f"{path.name}: anchored file {m.group('path')} is gone"
        body = source.read_text(encoding="utf-8")
        # A literal `\n` in an anchor spans lines, so a formula can be tied to
        # the function that owns it rather than to any identical line.
        anchor = m.group("text").replace("\\n", "\n")
        hits = body.count(anchor)
        assert hits == 1, (
            f"{path.name}: anchor must match exactly once in {m.group('path')}, "
            f"found {hits}: {anchor!r} -- re-check the proof against the code, "
            "or widen the anchor until it is unique"
        )


@pytest.mark.parametrize("path", _theorem_files(), ids=lambda p: p.name)
def test_theorem_file_states_its_limits_and_has_no_escape_hatch(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert "What this does NOT cover:" in text, f"{path.name}: missing non-claims"
    assert re.search(r"generated\s+(?:\w+\s+)?code", text), (
        f"{path.name}: must disclaim generated code")
    assert re.search(r"IEEE|[Mm]achine integers", text), (
        f"{path.name}: must say the proof is not about float/fixed-width arithmetic")
    code = re.sub(r"/-.*?-/", "", text, flags=re.S)
    code = re.sub(r"--.*", "", code)
    for word in ("sorry", "admit", "axiom", "native_decide"):
        assert not re.search(rf"\b{word}\b", code), f"{path.name}: uses `{word}`"
    assert f"theorem {_THEOREMS[path.name]}" in code


def test_lean_project_is_excluded_from_cmake() -> None:
    files = [_REPO / "CMakeLists.txt"]
    for top in ("src", "tests", "tools", "cmake"):
        files += (_REPO / top).rglob("CMakeLists.txt")
        files += (_REPO / top).rglob("*.cmake")
    offenders = [
        str(p.relative_to(_REPO)) for p in files
        if re.search(r"\bformal\b", p.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert not offenders, f"formal/ must not be a CMake target: {offenders}"


def _require_lean_toolchain() -> str:
    lake = shutil.which("lake")
    if lake is None:
        pytest.skip("lake not on PATH: this host cannot check the LEAN-PILOT-1 proofs "
                    "(install elan; see formal/lean/README.md)")
    if not (_LEAN / ".lake" / "packages" / "mathlib").is_dir():
        pytest.skip("Mathlib not fetched under formal/lean/.lake: run `lake exe cache get` "
                    "there first (the test will not download ~5 GB on its own)")
    return lake


def test_lake_build_checks_every_pilot_theorem() -> None:
    lake = _require_lean_toolchain()
    run = subprocess.run([lake, "build"], cwd=_LEAN, capture_output=True, text=True,
                         timeout=1800, check=False)
    assert run.returncode == 0, f"lake build failed:\n{run.stdout[-4000:]}\n{run.stderr[-4000:]}"
    assert "declaration uses 'sorry'" not in run.stdout + run.stderr


def test_pilot_theorems_use_only_standard_axioms() -> None:
    lake = _require_lean_toolchain()
    run = subprocess.run([lake, "env", "lean", "scripts/AxiomAudit.lean"], cwd=_LEAN,
                         capture_output=True, text=True, timeout=600, check=False)
    assert run.returncode == 0, f"axiom audit failed:\n{run.stdout}\n{run.stderr}"
    lines = [ln for ln in run.stdout.splitlines() if ln.startswith("'TesseraFormal.")]
    assert len(lines) == 9, f"expected 9 audited theorems, got:\n{run.stdout}"
    for line in lines:
        m = re.search(r"depends on axioms: \[(.*)\]", line)
        used = {a.strip() for a in m.group(1).split(",")} if m else set()
        assert used <= _ALLOWED_AXIOMS, f"non-standard axiom: {line}"
