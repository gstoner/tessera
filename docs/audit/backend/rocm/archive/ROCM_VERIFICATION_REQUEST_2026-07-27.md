---
last_updated: 2026-07-27
audit_role: plan
plan_state: closed
scope: ROCm-host verification of the tessera-opt build-capability change
sync_key: TESSERA-OPT-BUILD-CAPABILITY-2026-07-27
owner: ROCm backend owner (Strix Halo gfx1151 host)
---

# ROCm verification request — tessera-opt build-capability change

Run this on the **Strix Halo / gfx1151** box after PR #469 merges to `main`.

Everything below was authored and verified on an Apple M1 Max, which has **no
ROCm build**. That means three specific claims in that PR are unverified on a
real ROCm host, and one of them touches the ROCm registration path directly.
This file lists exactly what to run and what the answer should be, so the
result is a check rather than an exploration.

**Nothing here is expected to require code changes.** If any step disagrees
with its stated expectation, that is a finding — record it in `todo.md` under
the sync key above and do not paper over it.

---

## Why this needs a ROCm host

Three things cannot be checked on a Mac:

1. **The lean-driver configure error is now a hard failure.** `tessera-opt`'s
   leanness used to be re-derived in C++ from
   `(ROCM || NVIDIA) && !CORE_TESSERA_IR`. It is now one explicit CMake intent
   (`TESSERA_OPT_LEAN_ARTIFACT_DRIVER`), and combining it with any feature
   outside `{core-tessera-ir, nvidia-backend, rocm-backend}` **fails
   configuration**. A ROCm box is the only place the real lean/full ROCm
   branches get exercised.
2. **171 of 178 local test failures are ROCm tests** that cannot run without a
   ROCm build. Their failure mode on the Mac is
   `Unknown command line argument '--generate-rocm-*'`. They are *expected* to
   pass on your host — that is the whole point of step 3.
3. **Two streaming-attention lit fixtures** need `--rocm-*` passes that no Mac
   build registers.

---

## Step 0 — build

Per `CLAUDE.md`, the ROCm configure on Ubuntu 24.04:

```bash
cmake -S . -B build-rocm -G Ninja \
  -DLLVM_DIR=/usr/lib/llvm-23/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-23/lib/cmake/mlir \
  -DTESSERA_ENABLE_HIP=ON -DTESSERA_BUILD_ROCM_BACKEND=ON \
  -DCMAKE_PREFIX_PATH=/opt/rocm/core
```

```bash
ninja -C build-rocm tessera-opt
```

**Expected:** configures and builds clean. `TESSERA_ENABLE_HIP=ON` makes this a
*full* driver, not the lean artifact one, so the new validation must stay quiet.

---

## Step 1 — the build identity actually reports ROCm

```bash
./build-rocm/tools/tessera-opt/tessera-opt --tessera-build-info
```

**Expected:** `build profile: full`, and `features:` contains
`rocm-backend core-tessera-ir`. This flag is new; if it prints nothing, the
binary predates the change and the rest of this file will not apply.

---

## Step 2 — the leanness contract fails loudly, and only when it should

This is the finding the change exists to prevent: a lean artifact driver that
silently compiles out every non-NVIDIA/ROCm registration while still linking
the libraries. It is now a configure error.

**2a — must FAIL to configure** (lean ROCm + a feature the lean arm cannot
register). No real toolchain, so this is the lean branch:

```bash
cmake -S . -B /tmp/tessera-lean-conflict -G Ninja \
  -DLLVM_DIR=/usr/lib/llvm-23/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-23/lib/cmake/mlir \
  -DTESSERA_BUILD_ROCM_BACKEND=ON -DTESSERA_BUILD_APPLE_BACKEND=ON
```

**Expected:** `FATAL_ERROR` naming `apple-backend` as a conflicting feature and
offering both resolutions. **A successful configure here is a regression** — it
means the guard is not seeing linked features on this platform.

*(If `TESSERA_BUILD_APPLE_BACKEND=ON` is rejected earlier on Linux for an
unrelated reason, substitute any other optional feature that does build there —
e.g. `-DTESSERA_BUILD_TPP=ON` or the neighbors/solvers toggles — the guard is
feature-agnostic.)*

**2b — must SUCCEED** (lean ROCm alone, the legitimate artifact driver):

```bash
cmake -S . -B /tmp/tessera-lean-ok -G Ninja \
  -DLLVM_DIR=/usr/lib/llvm-23/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-23/lib/cmake/mlir \
  -DTESSERA_BUILD_ROCM_BACKEND=ON
ninja -C /tmp/tessera-lean-ok tessera-opt
/tmp/tessera-lean-ok/tools/tessera-opt/tessera-opt --tessera-build-info
```

**Expected:** configures, builds, and reports
`build profile: lean-artifact-driver`. This is the case that must keep working —
the guard must not have made the legitimate lean build impossible.

---

## Step 3 — the 171 ROCm test failures should disappear

On the Mac these fail with `Unknown command line argument`. With a ROCm build
they should pass.

```bash
export TESSERA_OPT=$PWD/build-rocm/tools/tessera-opt/tessera-opt
export TESSERA_OPT_PATH=$TESSERA_OPT
export TESSERA_OPT_BIN=$TESSERA_OPT
python3 -m pytest tests/unit/ -m "not slow" -q -p no:randomly --tb=short
```

**Expected:** the ROCm files below pass (or skip for a *stated* reason such as
"no /dev/kfd"). What must **not** appear is
`Unknown command line argument '--generate-rocm-*'`.

Baseline for comparison — these 35 files account for 171 failures on the Mac:

```
test_rocm_activation_rope_codegen      test_rocm_arch_fragment_compiler
test_rocm_attn_bias_codegen            test_rocm_binary_loss_compiled
test_rocm_deltanet_compiled            test_rocm_dequant_gemm_compiled
test_rocm_dspark_draft_block_compiled  test_rocm_elementwise_p2_compiled
test_rocm_fft_compiled                 test_rocm_flash_attn_pipeline_routing
test_rocm_fp8_cpp_python_consistency   test_rocm_fpquant_compiled
test_rocm_fused_epilogue_codegen       test_rocm_gemm_pipeline_routing
test_rocm_linalg_compiled              test_rocm_linear_attn_codegen
test_rocm_logit_softcap_codegen        test_rocm_loss_compiled
test_rocm_lu_qr_compiled               test_rocm_matmul_front_end_glue
test_rocm_mla_decode_step_compiled     test_rocm_moe_compiled
test_rocm_norm_codegen                 test_rocm_optimizer_compiled
test_rocm_predicate_compiled           test_rocm_sliding_window_codegen
test_rocm_softmax_codegen              test_rocm_sparse_attn_compiled
test_rocm_sparse_compiled              test_rocm_ssm_backward_compiled
test_rocm_state_space_compiled         test_rocm_svd_compiled
test_rocm_target_wmma_lowering         test_training_loss_adamw_compiled
test_training_loss_sgd_compiled
```

**Note the three env vars.** The test tree currently uses **eight** different
names to select this binary (`TESSERA_OPT` ×202, `TESSERA_OPT_BIN` ×16,
`TESSERA_OPT_CPP` ×10, `TESSERA_OPT_PATH` ×9, and four more), and 52 files
hardcode `build/tools/tessera-opt/tessera-opt` and ignore all of them. Exporting
three covers most of it. If a test still picks up the wrong binary, that is the
tracked cleanup, not a new bug.

---

## Step 4 — nine migrated files still behave with a real ROCm build

These moved onto the shared capability-aware helper
(`tests/_support/compiler_tool.py`). On the Mac they went from 32 failures to 0
by **skipping**; on your host they should go to 0 by **passing**. A skip here
would mean the helper is over-skipping against a build that does register the
passes — that *is* a finding.

```bash
python3 -m pytest \
  tests/unit/test_rocm_{argreduce,binary,bitwise,compare,logical,reduce,scan,unary,where}_compiled.py \
  -q -rs
```

**Expected:** passes, with no skip reason mentioning "does not register".

---

## Step 5 — the two unverified lit fixtures

Flagged as unverified when they landed; they need `--rocm-*` passes absent from
every Mac build.

Use the `lit` console script, not `python3 -m lit` — `lit` is a package with no
`__main__`, so the module form fails. `scripts/validate.sh` drives it via
`TESSERA_OPT_BIN`; match that:

```bash
export TESSERA_OPT_BIN=$PWD/build-rocm/tools/tessera-opt/tessera-opt
lit tests/tessera-ir/phase3/streaming_attention_modifiers_rocm.mlir -v
lit tests/tessera-ir/phase3/streaming_attention_rank4.mlir -v
```

While you are there, the three siblings that have never run on hardware either:

```bash
lit tests/tessera-ir/phase3/ -v --filter streaming_attention
```

Or through the repo's own gate, which resolves `lit` the same way:

```bash
TESSERA_VALIDATE_LIT=1 bash scripts/validate.sh
```

**Expected:** pass. CI's lit lane is opt-in and skips, so these have never had a
real run.

---

## Step 6 — the emit pipeline change (ROCDL lane)

`tessera-emit-rocdl` and `tessera-emit-nvvm` now share one spine string,
register only when the core IR is linked, and report a build failure through the
pass registry's error handler instead of `report_fatal_error`. The previous
convenience wrapper took a `void` builder, so a failed build would have installed
a **silently empty** pipeline — success reported for a translation never
performed (Decision #21).

```bash
# The alias is registered in a full ROCm build:
./build-rocm/tools/tessera-opt/tessera-opt --help | grep tessera-emit-rocdl

# And rejects options rather than silently ignoring them:
echo 'module {}' | ./build-rocm/tools/tessera-opt/tessera-opt - \
  --pass-pipeline='builtin.module(tessera-emit-rocdl{bogus=1})'
```

**Expected:** the alias is listed; the second command prints
`error: this pipeline takes no options` and exits non-zero.

Then the real thing — a tessera kernel through to ROCDL:

```bash
python3 -m pytest tests/unit/test_rocm_mlir_to_hsaco.py -q
```

**Expected:** passes. This is the in-process MLIR→hsaco lane (Stage L3), which
only means anything with a real HIP toolchain.

---

## Step 7 — registry/evidence agreement

While on the box, confirm the sealed gfx1151 packet still validates against the
merged tree:

```bash
python3 -m tessera.compiler.e2e_fleet validate \
  docs/audit/evidence/e2e_spine/rocm_gfx1151/gfx1151
bash scripts/check_generated_docs.sh
```

**Expected:** validates; 24 generated docs in sync. (`packet_pending` in
`FLEET_REGISTRATIONS` is only the no-packet *fallback* text — with a sealed
packet the dashboard computes `release_ready` per family. I misread this as a
registry/evidence disagreement earlier; it is not one.)

---

## What to record

Add one row to `docs/audit/backend/rocm/todo.md` under sync key
`TESSERA-OPT-BUILD-CAPABILITY-2026-07-27` stating:

- ROCm host + ROCm version + `--tessera-build-info` output;
- step 2a/2b outcomes (the guard is the load-bearing claim);
- the pytest failure count vs the 178/12761 Apple baseline, and whether any
  `Unknown command line argument` survived;
- the lit results from step 5, which close a genuinely open item.

If steps 3–5 come back clean, the ROCm half of
`tests/_support/compiler_tool.py`'s value is proven and the remaining ~43
unmigrated files become mechanical rather than speculative.

---

## Known-failing, not your problem

`tests/unit/test_apple_legacy_retune_benchmark.py::test_strict_retune_ledger_admits_on_its_exact_live_apple_host`
is `hardware_apple_gpu`-gated and will skip on your host. It fails on the Mac
because the Apple runtime source hash changed and the retained route ledger
correctly refuses stale evidence — owed re-measurement under APPLE-RETUNE-1, not
a ROCm concern.
