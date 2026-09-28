# Branch protection — required CI checks

Tessera's `Validate` workflow (`.github/workflows/validate.yml`) is
split into 6 lanes plus one aggregator job. The aggregator
(`validate-required`) is the single status check we wire into branch
protection — it succeeds iff every required lane succeeds.

## Required status checks (configure once, in GitHub UI)

Settings → Branches → branch protection rule for `main` → "Require
status checks to pass before merging":

| Check                | Source            | Why required                    |
|----------------------|-------------------|---------------------------------|
| `validate-required`  | `validate.yml`    | Fans in lint / unit / audit — one check, three lanes. |

Selecting just `validate-required` is sufficient. Each underlying
lane (`lint (ruff + mypy ratchet)`, `unit (pytest -m "not slow")`, and
`audit (drift + claim_lint + examples)`) is still reported
individually in the PR Checks tab so contributors can see which lane failed
without expanding the aggregator log.

## Opt-in lanes (NOT required for merge)

| Check                                       | Trigger                                          |
|---------------------------------------------|--------------------------------------------------|
| `lit (MLIR FileCheck — opt-in)`             | PR label `lit-smoke` · manual dispatch · push to main |
| `sanitizer (asan / tsan / ubsan — opt-in)`  | PR label `sanitizer-smoke` · manual dispatch     |
| `rocm hsaco serialization (host-free — opt-in)` | PR label `lit-smoke` · manual dispatch · push to main |

Apply the labels from the PR's right-side sidebar.

### Opt-in means "runs when triggered", never "may skip when triggered"

Once an opt-in lane runs, it runs for real or it fails. The three lanes that
build against LLVM/MLIR (`lit`, `rocm-serialize`, `sanitizer`) follow one
toolchain rule (owner decision 2026-09-27, sync
`FOUNDATION-BATCH-2-2026-09-27`):

* **Hosted CI accepts any LLVM/MLIR 23.1.x patch.** apt.llvm.org is a rolling
  source and cannot be held at the fleet's exact pin, so the lanes configure
  with `-DTESSERA_LLVM_PIN_MODE=minor`. That mode still rejects a mixed
  LLVM/MLIR pair and any other major.minor.
* **The exact version is recorded**: `scripts/ci_resolve_llvm.sh` writes it to
  the job summary and to `ci-toolchain/*.json`, and configure writes
  `tessera_llvm_pin.txt`; each lane uploads `ci-toolchain/` as the
  `ci-toolchain-<lane>-<sha>-<attempt>` artifact (also on failure). A
  `pin_match=series` result is not a fleet-comparable measurement.
* **No usable 23.1.x fails the lane.** There is no `::warning … skipping`
  path. The pytest proof steps also fail when their tests skip for a missing
  tool (`scripts/ci_require_executed.py`); the only allowed skip is
  `rocm-serialize`'s OCML case, which needs AMD device bitcode a stock runner
  lacks.

The fleet boxes keep the **exact** pin (`TESSERA_REQUIRED_LLVM_VERSION` in
`cmake/TesseraToolchainPins.cmake`, default `TESSERA_LLVM_PIN_MODE=exact`);
nothing outside hosted CI passes `minor`. Before 2026-09-27 the `lit` and
`rocm-serialize` lanes compared apt's 23.1.2 against the exact 23.1.1 pin,
printed a warning, skipped configure/build/test and reported success (push run
36347063229 on main), and the `sanitizer` lane installed no LLVM/MLIR at all
(its last real run, 35609056270, failed at `find_package(MLIR)`).
`tests/unit/test_ci_workflow.py` (`TestNoSilentToolchainSkip`,
`TestResolverBehaviour`, `TestCMakePinModes`) gates the pattern in every
workflow.

Two workflows still report success without proving anything, deliberately
and allow-listed in that test: `pylint.yml` (`--exit-zero`, advisory — ruff +
mypy gate) and `profiler-native-proofs.yml` (`--allow-unavailable`: hosted
runners have no Metal/ROCm/CUPTI device, so a green check there is a
provider-status snapshot recorded in its artifact, **not** a profiler proof).

## Apple Metal 4 promotion

Apple exact-device promotion is a local backend-host proof, never a registered
GitHub self-hosted runner. Run `scripts/run_apple_metal4_release_gate.sh` on the
named Metal 4 Mac and publish its sealed packet under
`docs/audit/evidence/apple/metal4/` in the coordinating PR. The ordinary
required `validate-required` fan-in remains portable: its unit and audit lanes
verify the pushed packet's schema, hashes, commit provenance, two clean
correctness reports, paired device/end-to-end evidence, fresh LLVM/MLIR 23
cache, and explicit power/thermal/GPU-contention availability. Metal 3 is a
non-blocking compatibility surface.

## Configuration via GitHub CLI

```sh
gh api -X PUT \
  "repos/tessera-ai/tessera/branches/main/protection" \
  -f required_status_checks.strict=true \
  -F 'required_status_checks.checks[]={"context":"validate-required"}' \
  -f required_pull_request_reviews.required_approving_review_count=1 \
  -f enforce_admins=false
```

(Adjust the repo slug and reviewer count to match your governance
policy.)

## Lane-by-lane wall-clock budget

| Lane         | Wall-clock target | Notes |
|--------------|------------------:|-------|
| lint         | ~30s              | ruff + mypy ratchet (defends 0). |
| unit         | ~2min             | `pytest -m "not slow"`, ~4300 tests. |
| audit        | ~10s              | support_table drift + claim_lint + examples audit. |
| lit          | ~10min            | LLVM/MLIR 23.1.x install + tessera-opt build + lit; fails if no 23.1.x. |
| sanitizer    | ~15min per matrix | LLVM/MLIR 23.1.x install; asan + tsan + ubsan run in parallel. |
| rocm-serialize | ~15min          | LLVM/MLIR 23.1.x + lld-23 install + HIP-less `tessera-rocm-opt` build + hsaco proof; fails if no 23.1.x. |

The standalone C++ runtime and collectives compile-check are intentionally
local-only. Run `scripts/validate.sh` on the owning host; it builds and tests
the standalone CPU runtime and compiles the collectives execution unit without
making pull-request approval depend on GitHub-hosted apt mirrors.

The ROCm compiler suite is likewise local-only (removed from CI 2026-08-19:
an apt LLVM/MLIR 23 install plus a from-scratch `tessera-rocm-opt` build is
~25min — too heavy for hosted runners). `scripts/validate.sh` runs
`check-tessera-rocm` when the build tree has the ROCm backend configured
(`-DTESSERA_BUILD_ROCM_BACKEND=ON`); run it on the primary box before merging
ROCm backend changes. This is the ONLY automated coverage for
`src/compiler/codegen/Tessera_ROCM_Backend/test/rocm/` — `check-tessera` does
not include that suite and `lit tests/tessera-ir/` runs a different one
through a different driver, so skipping it lets a ROCm backend fixture
regression reach main unnoticed.

`rocm hsaco serialization` proves the compiled ROCm lane still emits an
AMDGPU code object. It needs **no GPU and no ROCm install** — serialization
is compile-time work that shells out to `ld.lld` — so it runs on a stock
hosted runner and closes the blind spot that let PR #619's total serializer
outage go unnoticed. It does NOT prove the object runs or is numerically
correct; that evidence needs the real gfx1151 device.

The lit + sanitizer lanes are intentionally off the critical path so a
contributor doesn't have to wait 15+ minutes on every PR.

## How "required" interacts with `if:` filters

`validate-required` uses `if: always()` and pulls `needs.<lane>.result`
explicitly so a *skipped* required lane (which would normally pass
GitHub's default status check logic) is treated as a failure. The three named
lanes must all report `success`.

## Adding a new required lane

1. Add the job to `validate.yml`.
2. Append it to the `needs:` list on the `validate-required` job.
3. Append a `"${{ needs.<job>.result }}"` line to the `required`
   array in the verification step.
4. Open a PR; once it lands, no branch-protection change is needed —
   the aggregator already covers the new lane.
