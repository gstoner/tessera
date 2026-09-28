"""Static guards for ``.github/workflows/validate.yml``.

The workflow is the single source of truth for the required-checks
contract documented in ``.github/BRANCH_PROTECTION.md``.  These tests
lock the structural shape of the workflow so a rename or accidental
deletion of a required lane fails at PR time instead of at merge time.

This is intentionally a *static* check — it loads the YAML but does
not invoke any GitHub Actions infrastructure.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "validate.yml"
BRANCH_PROTECTION_DOC = REPO_ROOT / ".github" / "BRANCH_PROTECTION.md"


# Required lanes (must all be inputs to the validate-required aggregator).
REQUIRED_LANES = ("lint", "unit", "audit")
OPTIONAL_LANES = ("lit", "sanitizer", "rocm-serialize")
#: Lanes that configure a CMake build against LLVM/MLIR 23.
MLIR_LANES = ("lit", "rocm-serialize", "sanitizer")
AGGREGATOR_JOB = "validate-required"


def _code(run: str) -> str:
    """A step's shell text with whole-line comments removed."""

    return "\n".join(
        line for line in run.splitlines() if not line.strip().startswith("#")
    )


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

    def test_runtime_and_collectives_are_local_only(self) -> None:
        """Runtime/collectives validation must not depend on hosted apt mirrors."""

        wf = _load_workflow()
        assert "build" not in wf["jobs"]
        validate = (REPO_ROOT / "scripts" / "validate.sh").read_text(
            encoding="utf-8"
        )
        assert "Standalone CPU runtime build and tests" in validate
        assert "Collectives runtime compile check" in validate

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
        IR + SPIR-V round-trips.  Both depend on MLIR/LLVM 23.  If
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

    def test_lit_lane_configures_every_fixture_backend(self) -> None:
        """The lit lane ends with ``check_lit_fleet_union.py`` over its one
        report, so every active fixture must be able to run on this lane.
        Six fixtures ``REQUIRES: tessera-ebm`` / ``tessera-clifford``, which
        lit derives from passes ``tessera-opt`` registers only when those
        backends are configured ON. Without them the union gate fails with
        every test passing (first seen 2026-09-28, run 36386976430, once the
        lane stopped skipping on a toolchain mismatch)."""

        wf = _load_workflow()
        steps = wf["jobs"]["lit"].get("steps", [])
        configure = "\n".join(
            s.get("run", "") for s in steps if "cmake -S" in s.get("run", "")
        )
        assert configure, "lit lane is missing its `cmake -S` configure step"
        for flag in (
            "-DTESSERA_BUILD_EBM_BACKEND=ON",
            "-DTESSERA_BUILD_CLIFFORD_BACKEND=ON",
        ):
            assert flag in configure, (
                f"lit lane configure must pass {flag} or the fleet-union gate "
                f"cannot cover the fixtures that require it"
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

    def test_lit_lane_runs_scheduled_attention_proofs(self) -> None:
        """Compiler-backed attention proofs must execute with ``tessera-opt``.

        The ordinary unit lane deliberately does not build LLVM/MLIR tools.
        Keep these exact-artifact tests in the lane that owns the tool instead
        of allowing them to become permanent unit-lane skips.
        """

        wf = _load_workflow()
        lit = wf["jobs"]["lit"]
        steps = lit.get("steps", [])
        proof_step = next(
            (step for step in steps if step.get("name") == "Run MLIR tool proof tests"),
            None,
        )
        assert proof_step is not None, "lit lane is missing its MLIR proof step"
        assert "TESSERA_OPT" in proof_step.get("env", {}), (
            "scheduled-attention proofs require an explicit tessera-opt path"
        )
        run_text = proof_step.get("run", "")
        for test_name in (
            "test_attention_lowers_through_one_content_addressed_tile_artifact",
            "test_attention_modifiers_survive_the_shared_recurrence",
            "test_attention_schedule_policy_tampering_fails_closed",
        ):
            assert test_name in run_text, (
                f"lit lane must execute {test_name}; current proof step:\n{run_text}"
            )

    @pytest.mark.parametrize("lane", MLIR_LANES)
    def test_mlir_lanes_resolve_a_23_1_series_toolchain_or_fail(self, lane: str) -> None:
        """A rolling apt.llvm.org patch is recorded, never a reason to skip.

        Owner decision 2026-09-27 (sync ``FOUNDATION-BATCH-2-2026-09-27``):
        hosted CI accepts any 23.1.x patch, records the exact version it ran,
        and FAILS when none is available. The previous shape of this test
        required the lane to set ``mlir=false`` and skip on patch drift --
        which is exactly how the lit and rocm-serialize lanes went green on
        main having built and tested nothing.
        """

        wf = _load_workflow()
        steps = wf["jobs"][lane].get("steps", [])
        run_text = "\n".join(step.get("run", "") for step in steps)
        assert "scripts/ci_resolve_llvm.sh" in run_text, (
            f"{lane} must resolve (record or fail) its LLVM/MLIR through "
            "scripts/ci_resolve_llvm.sh"
        )
        assert "--manifest" in run_text, f"{lane} must record a toolchain manifest"
        resolve_at = next(
            i for i, step in enumerate(steps)
            if "scripts/ci_resolve_llvm.sh" in step.get("run", "")
        )
        build_at = next(
            (i for i, step in enumerate(steps)
             if "cmake -S" in _code(step.get("run", ""))
             or "bash scripts/run_sanitizers.sh" in _code(step.get("run", ""))),
            None,
        )
        assert build_at is not None and resolve_at < build_at, (
            f"{lane} must resolve the toolchain before it configures"
        )
        opted_in = "-DTESSERA_LLVM_PIN_MODE=minor" in _code(run_text) or any(
            (step.get("env") or {}).get("TESSERA_LLVM_PIN_MODE") == "minor"
            for step in steps
        )
        assert opted_in, (
            f"{lane} must opt into the CI-only 23.1.x tolerance explicitly "
            "(-DTESSERA_LLVM_PIN_MODE=minor); without it CMake's exact fleet "
            "pin fails configure on every rolling patch"
        )

    def test_rocm_compiler_suite_is_local_only(self) -> None:
        """The ROCm compiler suite is too heavy for hosted runners.

        The lane was removed from CI 2026-08-19 at the repo owner's
        direction (apt LLVM/MLIR 23 install + from-scratch
        ``tessera-rocm-opt`` build, ~25min). The coverage moved to
        ``scripts/validate.sh``, which runs ``check-tessera-rocm`` on the
        primary box when the build tree has the ROCm backend configured.
        """

        wf = _load_workflow()
        assert "rocm-compiler" not in wf["jobs"], (
            "the ROCm compiler suite is local-only — do not reintroduce the "
            "CI lane without repo-owner direction"
        )
        validate = (REPO_ROOT / "scripts" / "validate.sh").read_text(
            encoding="utf-8"
        )
        assert "check-tessera-rocm" in validate, (
            "scripts/validate.sh must own the local ROCm backend suite "
            "(check-tessera-rocm) now that the CI lane is gone"
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


# ---------------------------------------------------------------------------
# No lane may report success having skipped its build/test on a toolchain
# mismatch (sync FOUNDATION-BATCH-2-2026-09-27). Until 2026-09-27 the lit and
# rocm-serialize lanes printed `::warning ... skipping`, set a step output that
# gated every later step off, and exited 0; the sanitizer lane installed no
# LLVM/MLIR at all. These tests pin the structural shape that made that
# possible AND execute the resolver and CMake pin logic against faked
# toolchains, so the rule is checked by behaviour, not by substring alone.
# ---------------------------------------------------------------------------

WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
RESOLVER = REPO_ROOT / "scripts" / "ci_resolve_llvm.sh"
REQUIRE_EXECUTED = REPO_ROOT / "scripts" / "ci_require_executed.py"
PINS = REPO_ROOT / "cmake" / "TesseraToolchainPins.cmake"

#: Workflow steps allowed to report success without having proven anything,
#: each with the reason. Adding an entry is a reviewed decision, not a default.
ADVISORY_SUCCESS_ALLOWLIST = {
    # pylint runs unconfigured purely to surface findings; ruff + mypy gate.
    ("pylint.yml", "--exit-zero"): "advisory lint; ruff + mypy ratchet gate",
    # Hosted runners have no Metal/ROCm/CUPTI device: the lane snapshots
    # provider availability into an uploaded status JSON. A green check here
    # is NOT a profiler proof (listed in the 2026-09-27 CI audit; owner call
    # whether an unavailable provider should fail the label-triggered lane).
    ("profiler-native-proofs.yml", "--allow-unavailable"): (
        "provider-status snapshot; the artifact records `unavailable`"
    ),
}
_ADVISORY_TOKENS = ("--exit-zero", "--allow-unavailable", "continue-on-error")


def _all_workflows() -> dict[str, dict]:
    result = {}
    for path in sorted(WORKFLOWS_DIR.glob("*.yml")):
        with path.open() as f:
            result[path.name] = yaml.safe_load(f)
    return result


def _bash() -> str:
    found = shutil.which("bash")
    if found is None or sys.platform == "win32":
        pytest.skip("needs a POSIX bash to execute the CI resolver")
    return found


class TestNoSilentToolchainSkip:
    def test_no_step_is_gated_off_by_an_earlier_step_output(self) -> None:
        """`if: steps.<id>.outputs...` is the skip-and-succeed switch.

        Every hollow lane had the same shape: a dependency step decided the
        toolchain was unsuitable, wrote an output, and every later step was
        gated off it -- so the job ran nothing and reported success. A lane
        that cannot run must fail in the step that found out.
        """

        offenders = []
        for name, wf in _all_workflows().items():
            for job_id, job in (wf.get("jobs") or {}).items():
                for step in job.get("steps", []) or []:
                    cond = str(step.get("if", ""))
                    if "steps." in cond and ".outputs." in cond:
                        offenders.append(f"{name}:{job_id}:{step.get('name')} if: {cond}")
        assert not offenders, (
            "steps gated on an earlier step's output can turn a lane into a "
            "green no-op:\n" + "\n".join(offenders)
        )

    def test_toolchain_install_failure_is_not_downgraded_to_a_warning(self) -> None:
        offenders = []
        for name, wf in _all_workflows().items():
            for job_id, job in (wf.get("jobs") or {}).items():
                for step in job.get("steps", []) or []:
                    run = step.get("run", "")
                    for line in run.splitlines():
                        stripped = line.strip()
                        if stripped.startswith("#"):
                            continue
                        if "|| echo" in stripped and "::warning" in stripped:
                            offenders.append(f"{name}:{job_id}: {stripped}")
                        if "::warning" in stripped and "skip" in stripped.lower():
                            offenders.append(f"{name}:{job_id}: {stripped}")
        assert not offenders, (
            "a lane downgrades a failure to a ::warning and continues:\n"
            + "\n".join(offenders)
        )

    @pytest.mark.parametrize("lane", MLIR_LANES)
    def test_mlir_lane_has_no_early_success_exit(self, lane: str) -> None:
        wf = _load_workflow()
        for step in wf["jobs"][lane].get("steps", []):
            run = step.get("run", "")
            for line in run.splitlines():
                code = line.split("#", 1)[0]
                assert "exit 0" not in code, (
                    f"{lane} step {step.get('name')!r} can exit 0 early: {line.strip()}"
                )

    @pytest.mark.parametrize("lane", ("lit", "rocm-serialize"))
    def test_proof_steps_fail_when_every_test_skipped(self, lane: str) -> None:
        wf = _load_workflow()
        run_text = "\n".join(
            step.get("run", "") for step in wf["jobs"][lane].get("steps", [])
        )
        assert "--junitxml=" in run_text and "scripts/ci_require_executed.py" in run_text, (
            f"{lane}'s pytest proof must be checked by ci_require_executed.py: "
            "its tests skipif the tool is missing, so a broken build is a green "
            "all-skipped run otherwise"
        )

    @pytest.mark.parametrize("lane", MLIR_LANES)
    def test_lane_uploads_its_toolchain_record_even_on_failure(self, lane: str) -> None:
        wf = _load_workflow()
        uploads = [
            step for step in wf["jobs"][lane].get("steps", [])
            if str(step.get("uses", "")).startswith("actions/upload-artifact")
            and "ci-toolchain" in str(step.get("with", {}).get("path", ""))
        ]
        assert uploads, f"{lane} must upload its ci-toolchain/ record"
        assert all("always()" in str(step.get("if", "")) for step in uploads)

    def test_sanitizer_lane_installs_llvm_mlir(self) -> None:
        wf = _load_workflow()
        run_text = "\n".join(
            step.get("run", "") for step in wf["jobs"]["sanitizer"].get("steps", [])
        )
        for package in ("llvm-23-dev", "libmlir-23-dev", "mlir-23-tools", "clang-23"):
            assert package in run_text, (
                f"the sanitizer lane configures the top-level CMakeLists, which "
                f"runs find_package(LLVM/MLIR); it must install {package}"
            )
        san_text = (REPO_ROOT / "scripts" / "run_sanitizers.sh").read_text(encoding="utf-8")
        assert "TESSERA_LLVM_PIN_MODE" in san_text

    def test_advisory_success_is_explicitly_allowlisted(self) -> None:
        found = set()
        for name, wf in _all_workflows().items():
            for job in (wf.get("jobs") or {}).values():
                if "continue-on-error" in job:
                    found.add((name, "continue-on-error"))
                for step in job.get("steps", []) or []:
                    if "continue-on-error" in step:
                        found.add((name, "continue-on-error"))
                    run = step.get("run", "")
                    for token in _ADVISORY_TOKENS:
                        if token in run:
                            found.add((name, token))
        unlisted = sorted(found - set(ADVISORY_SUCCESS_ALLOWLIST))
        assert not unlisted, (
            "a workflow reports success without proving anything and is not "
            f"in ADVISORY_SUCCESS_ALLOWLIST: {unlisted}"
        )
        stale = sorted(set(ADVISORY_SUCCESS_ALLOWLIST) - found)
        assert not stale, f"stale ADVISORY_SUCCESS_ALLOWLIST entries: {stale}"


class TestFleetPinStaysExact:
    """The CI tolerance must not leak into the fleet gate."""

    def test_fleet_pin_is_a_full_version_and_default_mode_is_exact(self) -> None:
        text = PINS.read_text(encoding="utf-8")
        import re

        pin = re.search(r'set\(TESSERA_REQUIRED_LLVM_VERSION\s+"([0-9.]+)"', text)
        assert pin and re.fullmatch(r"\d+\.\d+\.\d+", pin.group(1))
        mode = re.search(r'set\(TESSERA_LLVM_PIN_MODE\s+"([a-z]+)"', text)
        assert mode and mode.group(1) == "exact"

    def test_only_hosted_ci_opts_into_the_minor_tolerance(self) -> None:
        """No fleet build entry point may pass the CI tolerance."""

        offenders = []
        for path in (REPO_ROOT / "scripts").glob("*.sh"):
            text = _code(path.read_text(encoding="utf-8"))
            if "TESSERA_LLVM_PIN_MODE=minor" in text:
                offenders.append(str(path.relative_to(REPO_ROOT)))
        for path in (REPO_ROOT / "CMakeLists.txt", REPO_ROOT / "CMakePresets.json"):
            if path.is_file() and "TESSERA_LLVM_PIN_MODE=minor" in path.read_text(
                encoding="utf-8"
            ):
                offenders.append(str(path.relative_to(REPO_ROOT)))
        assert not offenders, f"fleet entry points pass the CI-only tolerance: {offenders}"


def _fake_prefix(root: Path, llvm: str, mlir: str, lld: str | None, cmake: bool = True) -> Path:
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    if llvm:
        (bin_dir / "llvm-config").write_text(f"#!/bin/sh\necho {llvm}\n")
    if mlir:
        (bin_dir / "mlir-opt").write_text(
            f"#!/bin/sh\necho 'LLVM (http://llvm.org/):'\necho '  LLVM version {mlir}'\n"
        )
    if lld:
        (bin_dir / "ld.lld").write_text(f"#!/bin/sh\necho 'Ubuntu LLD {lld} (compatible with GNU linkers)'\n")
    for tool in bin_dir.iterdir():
        tool.chmod(0o755)
    if cmake:
        for pkg, cfg in (("llvm", "LLVMConfig.cmake"), ("mlir", "MLIRConfig.cmake")):
            (root / "lib" / "cmake" / pkg).mkdir(parents=True, exist_ok=True)
            (root / "lib" / "cmake" / pkg / cfg).write_text("")
    return root


class TestResolverBehaviour:
    """Execute scripts/ci_resolve_llvm.sh against faked toolchains."""

    def _run(self, tmp_path: Path, prefix: Path, *extra: str) -> tuple[int, str, dict[str, str], str]:
        gh_out = tmp_path / "gh_output"
        summary = tmp_path / "summary.md"
        manifest = tmp_path / "manifest.json"
        env = dict(os.environ)
        env.update(
            TESSERA_CI_LLVM_PREFIX=str(prefix),
            GITHUB_OUTPUT=str(gh_out),
            GITHUB_STEP_SUMMARY=str(summary),
        )
        proc = subprocess.run(
            [_bash(), str(RESOLVER), "--lane", "test", "--manifest", str(manifest), *extra],
            capture_output=True, text=True, env=env, timeout=60,
        )
        outputs = {}
        if gh_out.exists():
            for line in gh_out.read_text().splitlines():
                key, _, value = line.partition("=")
                outputs[key] = value
        return proc.returncode, proc.stdout + proc.stderr, outputs, (
            summary.read_text() if summary.exists() else ""
        )

    def _pin(self) -> str:
        import re

        return re.search(
            r'set\(TESSERA_REQUIRED_LLVM_VERSION\s+"([0-9.]+)"', PINS.read_text()
        ).group(1)

    def test_exact_pin_passes_and_is_recorded(self, tmp_path: Path) -> None:
        pin = self._pin()
        rc, log, outputs, summary = self._run(
            tmp_path, _fake_prefix(tmp_path / "p", pin, pin, pin), "--require-lld"
        )
        assert rc == 0, log
        assert outputs["llvm_version"] == pin and outputs["pin_match"] == "exact"
        assert pin in summary
        import json

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["llvm_version"] == pin and manifest["mlir_version"] == pin

    def test_newer_patch_in_series_passes_and_records_the_real_version(self, tmp_path: Path) -> None:
        pin = self._pin()
        major_minor = pin.rsplit(".", 1)[0]
        newer = f"{major_minor}.{int(pin.rsplit('.', 1)[1]) + 1}"
        rc, log, outputs, summary = self._run(
            tmp_path, _fake_prefix(tmp_path / "p", f"{newer}~++20260926", newer, newer)
        )
        assert rc == 0, log
        assert outputs["llvm_version"] == newer and outputs["pin_match"] == "series"
        assert newer in summary and "::notice" in log

    @pytest.mark.parametrize(
        "llvm, mlir, lld, cmake, require_lld, why",
        [
            ("", "", None, False, False, "no runnable llvm-config"),
            ("NEXT_MINOR", "NEXT_MINOR", None, True, False, "outside the accepted"),
            ("NEXT_MAJOR", "NEXT_MAJOR", None, True, False, "outside the accepted"),
            ("NEWER", "PIN", None, True, False, "mixed LLVM/MLIR pair"),
            ("NEWER", "NEWER", None, True, True, "requires ld.lld"),
            ("NEWER", "NEWER", None, False, False, "LLVMConfig.cmake missing"),
        ],
    )
    def test_unusable_toolchain_fails_rather_than_skips(
        self, tmp_path: Path, llvm, mlir, lld, cmake, require_lld, why
    ) -> None:
        pin = self._pin()
        major, minor, patch = (int(x) for x in pin.split("."))
        names = {
            "": "",
            "PIN": pin,
            "NEWER": f"{major}.{minor}.{patch + 1}",
            "NEXT_MINOR": f"{major}.{minor + 1}.0",
            "NEXT_MAJOR": f"{major + 1}.1.0",
        }
        prefix = _fake_prefix(tmp_path / "p", names[llvm], names[mlir], lld, cmake=cmake)
        extra = ("--require-lld",) if require_lld else ()
        rc, log, outputs, summary = self._run(tmp_path, prefix, *extra)
        assert rc != 0, f"resolver accepted an unusable toolchain:\n{log}"
        assert why in log and "::error" in log
        assert "FAILED" in summary
        assert "llvm_version" not in outputs


class TestCMakePinModes:
    """Run tessera_pin_llvm() in CMake script mode against faked versions."""

    def _configure(self, tmp_path: Path, mode: str | None, llvm: str, mlir: str) -> tuple[int, str]:
        cmake = shutil.which("cmake")
        if cmake is None:
            pytest.skip("cmake not installed")
        script = tmp_path / "pin.cmake"
        script.write_text(
            f'set(CMAKE_BINARY_DIR "{tmp_path.as_posix()}")\n'
            f'include("{PINS.as_posix()}")\n'
            f'set(LLVM_PACKAGE_VERSION "{llvm}")\n'
            f'set(MLIR_VERSION "{mlir}")\n'
            "tessera_pin_llvm(${TESSERA_REQUIRED_LLVM_VERSION})\n"
        )
        args = [cmake]
        if mode is not None:
            args.append(f"-DTESSERA_LLVM_PIN_MODE={mode}")
        args += ["-P", str(script)]
        proc = subprocess.run(args, capture_output=True, text=True, timeout=60)
        return proc.returncode, proc.stdout + proc.stderr

    def _versions(self) -> tuple[str, str, str, str]:
        import re

        pin = re.search(
            r'set\(TESSERA_REQUIRED_LLVM_VERSION\s+"([0-9.]+)"', PINS.read_text()
        ).group(1)
        major, minor, patch = (int(x) for x in pin.split("."))
        return pin, f"{major}.{minor}.{patch + 1}", f"{major}.{minor + 1}.0", f"{major + 1}.1.0"

    @pytest.mark.parametrize("mode", (None, "exact"))
    def test_fleet_mode_is_exact(self, tmp_path: Path, mode) -> None:
        pin, newer, _, _ = self._versions()
        assert self._configure(tmp_path, mode, pin, pin)[0] == 0
        rc, log = self._configure(tmp_path, mode, newer, newer)
        assert rc != 0 and "pins LLVM/MLIR" in log

    def test_minor_mode_accepts_a_matched_newer_patch_and_records_it(self, tmp_path: Path) -> None:
        pin, newer, _, _ = self._versions()
        rc, log = self._configure(tmp_path, "minor", newer, newer)
        assert rc == 0, log
        assert "not the fleet pin" in log
        record = (tmp_path / "tessera_llvm_pin.txt").read_text()
        assert f"llvm={newer}" in record and "mode=minor" in record

    def test_minor_mode_still_rejects_a_mixed_pair_and_other_series(self, tmp_path: Path) -> None:
        pin, newer, next_minor, next_major = self._versions()
        assert self._configure(tmp_path, "minor", newer, pin)[0] != 0
        assert self._configure(tmp_path, "minor", next_minor, next_minor)[0] != 0
        assert self._configure(tmp_path, "minor", next_major, next_major)[0] != 0

    def test_unknown_mode_is_refused(self, tmp_path: Path) -> None:
        pin, _, _, _ = self._versions()
        rc, log = self._configure(tmp_path, "any", pin, pin)
        assert rc != 0 and "TESSERA_LLVM_PIN_MODE" in log


class TestRequireExecuted:
    def _junit(self, tmp_path: Path, cases: str) -> Path:
        path = tmp_path / "junit.xml"
        path.write_text(f'<testsuites><testsuite name="s">{cases}</testsuite></testsuites>')
        return path

    def _run(self, path: Path, *allow: str) -> int:
        args = [sys.executable, str(REQUIRE_EXECUTED), str(path)]
        for token in allow:
            args += ["--allow-skip", token]
        return subprocess.run(args, capture_output=True, text=True, timeout=60).returncode

    def test_all_skipped_run_fails(self, tmp_path: Path) -> None:
        path = self._junit(
            tmp_path,
            '<testcase classname="t" name="a"><skipped message="tessera-opt not built"/></testcase>',
        )
        assert self._run(path) != 0

    def test_unexpected_skip_fails_even_beside_a_pass(self, tmp_path: Path) -> None:
        path = self._junit(
            tmp_path,
            '<testcase classname="t" name="a"/>'
            '<testcase classname="t" name="b"><skipped message="no ld.lld on this host"/></testcase>',
        )
        assert self._run(path, "no ROCm device bitcode") != 0

    def test_named_skip_beside_a_pass_is_accepted(self, tmp_path: Path) -> None:
        path = self._junit(
            tmp_path,
            '<testcase classname="t" name="a"/>'
            '<testcase classname="t" name="b"><skipped message="no ROCm device bitcode (amdgcn/bitcode) on this host"/></testcase>',
        )
        assert self._run(path, "no ROCm device bitcode") == 0

    def test_missing_results_file_fails(self, tmp_path: Path) -> None:
        assert self._run(tmp_path / "absent.xml") != 0
