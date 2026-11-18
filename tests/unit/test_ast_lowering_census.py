"""The AST lowering census plugin must keep measuring what it claims.

``scripts/ast_lowering_census.py`` attributes each AST build by reading
``jit._decorate``'s locals and frame names in ``jit.py``. A rename there
would not fail the census; it would turn every reason into ``unexplained``
or ``reason_probe_failed`` and every reader into the wrong frame. This runs
the plugin over a known file and checks that it attributes, not merely runs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET = "tests/unit/test_lazy_ast_emission.py"


def _run_census(out: Path) -> list[dict]:
    env = dict(os.environ)
    env["TESSERA_AST_CENSUS_DIR"] = str(out)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "python"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "-p", "no:xdist", "-p", "scripts.ast_lowering_census", TARGET],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    records = []
    for path in sorted(out.glob("census-*.jsonl")):
        records += [json.loads(line) for line in path.read_text().splitlines() if line]
    return records


def test_the_census_attributes_every_build_it_records(tmp_path: Path) -> None:
    records = _run_census(tmp_path)
    kinds = {r["kind"] for r in records}
    assert "census_error" not in kinds, [r for r in records if r["kind"] == "census_error"]
    decorations = [r for r in records if r["kind"] == "decoration"]
    assert any(r["deferred"] for r in decorations)
    eager = [r for r in decorations if not r["deferred"]]
    assert eager, "the target file decorates bodies that must emit eagerly"
    reasons = [reason for r in eager for reason in r["eager_reasons"]]
    assert not [x for x in reasons if x == "unexplained" or x.startswith("reason_probe_failed")], reasons
    # The file's eager cases name these readers (see test_lazy_ast_emission).
    assert "apple_gpu:body_may_need_tracer" in reasons
    assert all(r["eager_reasons"] for r in eager), "every eager decoration names why"
    assert "differentiation_request" in reasons

    lowers = [r for r in records if r["kind"] == "lower"]
    assert lowers and all(r["route"] in {"ast", "trace"} for r in lowers)
    assert all(r["why_ast"] for r in lowers if r["route"] == "ast")
    # A deferred emission names the field that forced it, and the reader is
    # outside the emission machinery (never the census's own wrapper).
    materialized = [r for r in records if r["kind"] == "materialize"]
    assert materialized and all(r["field"] for r in materialized)
    assert not [r for r in records if str(r.get("reader", "")).startswith("ast_lowering_census.py")]

    # One record per emission actually run: four racing first readers run it
    # once, and reads after a failure rethrow the cached error without
    # rerunning it.
    def materialized_in(test_name: str) -> list[dict]:
        return [r for r in materialized if r["test"].endswith("::" + test_name)]

    concurrent = materialized_in("test_concurrent_first_readers_all_get_the_module")
    assert [r["outcome"] for r in concurrent] == ["ok"], concurrent
    failed = materialized_in("test_an_emission_failure_reaches_every_reader")
    assert [r["outcome"] for r in failed] == ["failed:TesseraJitError"], failed
    assert failed[0]["field"] == "_decoration_compile"


def test_the_census_refuses_to_record_nowhere(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k != "TESSERA_AST_CENSUS_DIR"}
    env["PYTHONPATH"] = str(REPO_ROOT / "python")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "-p", "scripts.ast_lowering_census", "--collect-only", TARGET],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode != 0
    assert "TESSERA_AST_CENSUS_DIR" in proc.stdout + proc.stderr
