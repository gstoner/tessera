"""Static guards for ``.github/workflows/validate.yml``.

The workflow is the single source of truth for the required-checks
contract documented in ``.github/BRANCH_PROTECTION.md``.  These tests
lock the structural shape of the workflow so a rename or accidental
deletion of a required lane fails at PR time instead of at merge time.

This is intentionally a *static* check — it loads the YAML but does
not invoke any GitHub Actions infrastructure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "validate.yml"
BRANCH_PROTECTION_DOC = REPO_ROOT / ".github" / "BRANCH_PROTECTION.md"


# Required lanes (must all be inputs to the validate-required aggregator).
REQUIRED_LANES = ("lint", "unit", "audit", "build")
OPTIONAL_LANES = ("lit", "sanitizer")
AGGREGATOR_JOB = "validate-required"


def _load_workflow() -> dict:
    assert WORKFLOW.is_file(), f"missing CI workflow: {WORKFLOW}"
    with WORKFLOW.open() as f:
        return yaml.safe_load(f)


class TestWorkflowStructure:
    def test_yaml_parses(self) -> None:
        wf = _load_workflow()
        assert isinstance(wf, dict)
        assert "jobs" in wf

    def test_every_required_lane_exists(self) -> None:
        wf = _load_workflow()
        for lane in REQUIRED_LANES:
            assert lane in wf["jobs"], (
                f"required lane {lane!r} is missing from validate.yml"
            )

    def test_optional_lanes_exist(self) -> None:
        wf = _load_workflow()
        for lane in OPTIONAL_LANES:
            assert lane in wf["jobs"], (
                f"optional lane {lane!r} is missing from validate.yml"
            )

    def test_aggregator_exists_and_needs_required_lanes(self) -> None:
        wf = _load_workflow()
        assert AGGREGATOR_JOB in wf["jobs"], (
            f"aggregator job {AGGREGATOR_JOB!r} is missing — "
            f"branch protection depends on it"
        )
        agg = wf["jobs"][AGGREGATOR_JOB]
        needs = agg.get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        for lane in REQUIRED_LANES:
            assert lane in needs, (
                f"{AGGREGATOR_JOB} must list {lane!r} in `needs:` so "
                f"branch protection blocks on it"
            )

    def test_aggregator_runs_always(self) -> None:
        """``if: always()`` keeps a skipped lane from spoofing success."""

        wf = _load_workflow()
        agg = wf["jobs"][AGGREGATOR_JOB]
        # YAML key `if` is parsed as a Python bool when value is
        # `true`/`false`, but `always()` is parsed as a string.
        condition = agg.get("if")
        assert isinstance(condition, str) and "always()" in condition, (
            f"{AGGREGATOR_JOB} must use `if: always()` so a skipped "
            f"required lane is treated as a failure"
        )

    def test_aggregator_verifies_each_required_lane_result(self) -> None:
        """The bash verification step must read every required lane's
        ``needs.<lane>.result`` so skips are treated as failures."""

        wf = _load_workflow()
        agg = wf["jobs"][AGGREGATOR_JOB]
        steps = agg.get("steps", [])
        # Find the step that does the verification (any run: script).
        script_text = "\n".join(s.get("run", "") for s in steps if "run" in s)
        for lane in REQUIRED_LANES:
            assert f"needs.{lane}.result" in script_text, (
                f"aggregator's verification step must read "
                f"`needs.{lane}.result`; current script:\n{script_text}"
            )

    def test_optional_lanes_are_gated(self) -> None:
        """Opt-in lanes must NOT run on every push — they require an
        explicit label, manual dispatch, or push-to-main."""

        wf = _load_workflow()
        for lane in OPTIONAL_LANES:
            job = wf["jobs"][lane]
            condition = job.get("if", "")
            assert isinstance(condition, str) and condition.strip(), (
                f"optional lane {lane!r} must declare an `if:` gate so "
                f"it doesn't run on every PR — currently runs unconditionally"
            )
            # Must reference at least one of the documented triggers.
            tokens = (
                "workflow_dispatch",
                "labels",
                "refs/heads/main",
                "refs/heads/master",
            )
            assert any(t in condition for t in tokens), (
                f"optional lane {lane!r} gate doesn't reference any of "
                f"the documented triggers {tokens!r}; current `if:` is:\n"
                f"  {condition}"
            )

    def test_lit_lane_builds_both_mlir_binaries(self) -> None:
        """The lit lane is responsible for both MLIR-bearing binaries.

        ``tessera-opt`` runs FileCheck against the in-tree MLIR
        fixtures; ``tessera-translate-mlir`` does the MLIR ↔ LLVM
        IR + SPIR-V round-trips.  Both depend on MLIR/LLVM 21.  If
        either target drops out of the lane, the matching unit
        coverage (``test_tessera_opt_build.py`` /
        ``test_cli_translate.py``) loses its execution-side proof
        in CI even when the local ``scripts/validate.sh`` keeps it.
        """

        wf = _load_workflow()
        lit = wf["jobs"]["lit"]
        steps = lit.get("steps", [])
        # Find the cmake-build step.
        build_text = ""
        for step in steps:
            run = step.get("run", "")
            if "cmake --build" in run:
                build_text += run
        assert build_text, "lit lane is missing a `cmake --build` step"
        for target in ("tessera-opt", "tessera-translate-mlir"):
            assert target in build_text, (
                f"lit lane's cmake build step must include "
                f"--target {target} so its proof tests have something "
                f"to FileCheck against; current build text:\n{build_text}"
            )

    def test_lit_lane_runs_proof_tests_for_both_binaries(self) -> None:
        """After building the two MLIR binaries the lit lane must
        invoke the matching unit proof tests so a build that links
        but produces a broken binary still fails CI."""

        wf = _load_workflow()
        lit = wf["jobs"]["lit"]
        steps = lit.get("steps", [])
        run_text = "\n".join(s.get("run", "") for s in steps)
        for test_file in (
            "test_cli_translate.py",
            "test_tessera_opt_build.py",
        ):
            assert test_file in run_text, (
                f"lit lane must invoke {test_file} so the cmake build "
                f"is verified end-to-end (build + execution + "
                f"FileCheck); current `run:` steps:\n{run_text}"
            )


class TestBranchProtectionDoc:
    def test_doc_exists(self) -> None:
        assert BRANCH_PROTECTION_DOC.is_file(), (
            f"missing {BRANCH_PROTECTION_DOC}"
        )

    def test_doc_names_aggregator_job(self) -> None:
        text = BRANCH_PROTECTION_DOC.read_text(encoding="utf-8")
        assert AGGREGATOR_JOB in text, (
            f"{BRANCH_PROTECTION_DOC.name} must mention "
            f"the aggregator job {AGGREGATOR_JOB!r}"
        )

    def test_doc_lists_required_lanes(self) -> None:
        text = BRANCH_PROTECTION_DOC.read_text(encoding="utf-8")
        for lane in REQUIRED_LANES:
            assert lane in text, (
                f"{BRANCH_PROTECTION_DOC.name} must mention the "
                f"required lane {lane!r}"
            )


class TestWorkflowEnv:
    """Lock the env contract so a refactor doesn't silently drop a key
    the lanes depend on."""

    def test_env_unbuffered_python(self) -> None:
        wf = _load_workflow()
        env = wf.get("env", {})
        assert env.get("PYTHONUNBUFFERED") == "1"

    def test_env_pip_quiet(self) -> None:
        wf = _load_workflow()
        env = wf.get("env", {})
        assert env.get("PIP_DISABLE_PIP_VERSION_CHECK") == "1"
