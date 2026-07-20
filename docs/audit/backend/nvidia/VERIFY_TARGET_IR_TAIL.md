---
last_updated: 2026-07-13
audit_role: reference
---

# Verify: NVIDIA Target IR tail + emit cleanups (branch `nvidia-target-ir-typed-ops`)

Cross-platform verification checklist for the three commits on this branch:

| Commit | Theme |
|--------|-------|
| `44aee141` | Target IR tail **increment 1** — typed NVIDIA Target IR dialect (10 registered ops, Decision #19) |
| `87483dde` | Target IR tail **increment 2** — real `nvvm.mma.sync` for the m16n8k16 f16 fragment contract (NVVM-verifier-validated) + honest marker fallback |
| `8f60ca91` | **emit cleanups** — public `emit/__init__` façade + `measure_latency` timing docstring |

The NVIDIA Target IR changes are **hardware-free** (MLIR + NVVM dialect only — no CUDA
toolkit, no GPU). The lit fixtures and Python tests below run on **any** box that
builds the NVIDIA backend. On-device execution (sm_120) is only exercised by the
pre-existing `test_nvidia_plugin.py` / `test_conformance_execute_compare_nvidia.py`
emit-path tests — not by these commits — and is called out per platform.

> **Where the fixtures live:** `src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia/`
> — `nvidia_target_ir_typed.mlir` (increment 1) and `nvidia_mma_sync_to_nvvm.mlir`
> (increment 2) are the two new ones; the rest are the pre-existing NVIDIA suite.

---

## ✅ ROCm box (gfx1151, Ubuntu 24.04) — VERIFIED 2026-07-11

Ran on the Strix Halo dev box. Results:

- **NVIDIA Target IR lit fixtures: 9/9 PASS** (typed round-trip, real `nvvm.mma.sync`
  + marker fallback, sm120/hopper/blackwell tile→target, both →NVVM contract
  pipelines, both TMEM gate diagnostics).
- **Python target_ir + emit subsystem: 81 PASS** (`test_target_ir`,
  `test_target_ir_contract`, `test_kernel_emitter`, `test_kernel_cache`,
  `test_candidate_arbiter`, `test_arbiter_autotune`).
- **ROCm hardware numerical fixtures: 32 PASS** on real gfx1151 (`matmul`/rocm WMMA
  GEMM + `flash_attn`/rocm — the complete-cell fixtures that use the runtime-symbol
  lane; device reported `gfx1151`).

Reproduce (from repo root, with the LLVM-23 apt toolchain):

```bash
# One-time: build tessera-opt with the (hardware-free) NVIDIA backend enabled.
cmake -S . -B build -DTESSERA_BUILD_NVIDIA_BACKEND=ON        # additive; keeps ROCm
ninja -C build tessera-opt

export PYTHONPATH=python
OPT=build/tools/tessera-opt/tessera-opt
FC=/usr/lib/llvm-23/bin/FileCheck
T=src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia

# [1] the two NEW Target IR fixtures (registration + real nvvm.mma.sync)
$OPT $T/nvidia_target_ir_typed.mlir | $FC $T/nvidia_target_ir_typed.mlir
$OPT --lower-tessera-nvidia-to-nvvm $T/nvidia_mma_sync_to_nvvm.mlir | $FC $T/nvidia_mma_sync_to_nvvm.mlir

# [2] Python
python3 -m pytest -q tests/unit/test_target_ir.py tests/unit/test_target_ir_contract.py \
  tests/unit/test_kernel_emitter.py tests/unit/test_kernel_cache.py \
  tests/unit/test_candidate_arbiter.py tests/unit/test_arbiter_autotune.py

# [3] ROCm hardware complete-cell fixtures (needs live gfx1151 + /dev/kfd)
python3 -m pytest -q tests/unit/test_rocm_wmma_runtime_symbol.py \
  tests/unit/test_rocm_flash_attn_runtime_symbol.py
```

> **Note on the other 3 rocm complete cells** (`softmax`, `conv2d`, `kv_cache_read`):
> their fixtures need the in-process Stage-L pipeline (`convert-scf-to-cf`,
> `convert-gpu-to-rocdl`, …), which the **lean** `tessera-opt` build on this box does
> not register (`TESSERA_HAVE_CORE_TESSERA_IR` off). They fail here on a build-config
> limitation, **not** a numerical/registry finding. To exercise them, build
> `tessera-opt` with the CORE IR passes registered.

---

## 🍎 Apple system (macOS, Apple Silicon) — VERIFIED 2026-07-11

Ran on macOS / Apple Silicon (Homebrew LLVM/MLIR 23, Python 3.14.6, no GPU
needed for the Target IR path). Results:

- **Build:** `tessera-opt` configured + built clean with
  `-DTESSERA_BUILD_NVIDIA_BACKEND=ON -DTESSERA_BUILD_APPLE_BACKEND=ON`
  (`CUDA support: OFF`, hardware-free NVIDIA Target IR tools only). Only
  deprecation warnings (`OpTy::create`), no errors.
- **NVIDIA Target IR lit fixtures: 7/7 PASS** — the two NEW ones
  (`nvidia_target_ir_typed` typed round-trip, `nvidia_mma_sync_to_nvvm` real
  `nvvm.mma.sync` lowering) plus the 5 pre-existing (sm120/hopper/blackwell
  tile→nvidia + hopper/blackwell →NVVM contract pipelines).
- **Python target_ir + emit + conformance: 109 PASS** (`test_target_ir`,
  `test_target_ir_contract`, `test_kernel_emitter`, `test_kernel_cache`,
  `test_candidate_arbiter`, `test_arbiter_autotune`, `test_conformance_evaluator`,
  `test_conformance_complete_cells_proven`) — **0 failures**.
- **Darwin-only corroboration: PASS** —
  `test_complete_cells_are_evaluator_corroborated_on_darwin` ran (not skipped):
  Apple CPU/GPU complete cells re-derived to HARDWARE_VERIFIED.

The NVIDIA Target IR dialect builds on macOS (Homebrew LLVM/MLIR 23, no GPU needed).
Run from repo root:

```bash
# Build tessera-opt with the NVIDIA backend on, against Homebrew LLVM/MLIR 23.
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/opt/homebrew/opt/llvm@23/lib/cmake/llvm \
  -DMLIR_DIR=/opt/homebrew/opt/llvm@23/lib/cmake/mlir \
  -DTESSERA_CPU_ONLY=ON -DTESSERA_BUILD_APPLE_BACKEND=ON \
  -DTESSERA_BUILD_NVIDIA_BACKEND=ON
ninja -C build tessera-opt

export PYTHONPATH=python
OPT=build/tools/tessera-opt/tessera-opt
FC=/opt/homebrew/opt/llvm@23/bin/FileCheck        # Homebrew FileCheck path
T=src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia

# [1] the two NEW Target IR fixtures — expect PASS (hardware-free)
$OPT $T/nvidia_target_ir_typed.mlir | $FC $T/nvidia_target_ir_typed.mlir
$OPT --lower-tessera-nvidia-to-nvvm $T/nvidia_mma_sync_to_nvvm.mlir | $FC $T/nvidia_mma_sync_to_nvvm.mlir

# [1b] full pre-existing NVIDIA suite — expect PASS
for f in sm120_tile_to_nvidia:sm=120 hopper_tile_to_nvidia:sm=90 blackwell_tile_to_nvidia:sm=100; do
  name=${f%%:*}; arg=${f##*:}
  $OPT --allow-unregistered-dialect --lower-tile-to-nvidia=$arg $T/$name.mlir | $FC $T/$name.mlir && echo "PASS $name"
done
$OPT --allow-unregistered-dialect --tessera-lower-to-hopper    $T/hopper_to_nvvm_contract.mlir    | $FC $T/hopper_to_nvvm_contract.mlir
$OPT --allow-unregistered-dialect --tessera-lower-to-blackwell $T/blackwell_to_nvvm_contract.mlir | $FC $T/blackwell_to_nvvm_contract.mlir

# [2] Python (target_ir + emit) — expect PASS. Apple GPU complete-cell corroboration
#     (the Evaluator's Darwin lane) also runs here:
python3 -m pytest -q tests/unit/test_target_ir.py tests/unit/test_target_ir_contract.py \
  tests/unit/test_kernel_emitter.py tests/unit/test_kernel_cache.py \
  tests/unit/test_candidate_arbiter.py tests/unit/test_arbiter_autotune.py \
  tests/unit/test_conformance_evaluator.py tests/unit/test_conformance_complete_cells_proven.py
```

**Expected:** all lit fixtures PASS; Python PASS including the Darwin-only
`test_complete_cells_are_evaluator_corroborated_on_darwin` (Apple CPU/GPU complete
cells re-derived to HARDWARE_VERIFIED). Record pass/fail counts back here.

---

## ✅ NVIDIA box (RTX 5070 Ti, sm_120 / consumer Blackwell) — VERIFIED 2026-07-13

Build with CUDA + the NVIDIA backend. The lit fixtures verify the same as above;
additionally the pre-existing emit-path tests **execute on sm_120**.

Results on the local WSL CUDA environment:

- **Build: PASS** — CUDA 13.3.33; NVIDIA backend enabled; `tessera-opt` built.
- **Target IR fixtures: 2/2 PASS** — typed Target IR round-trip and real
  `nvvm.mma.sync` lowering checked by FileCheck/NVVM structural verification.
- **On-device emit path: 68 PASS, 1 skipped** in 18.16s — RTX 5070 Ti
  (sm_120), driver 610.62. The skipped case was skip-clean.

During this verification, `tessera-opt` exposed a duplicate legacy registration
symbol that hid the typed `tessera_nvidia` dialect and the NVVM lowering pass. The
legacy pipeline now has distinct registration symbols; `tessera-opt` registers both
the typed Target IR and legacy pipeline without collision.

```bash
cmake -S . -B build -DTESSERA_ENABLE_CUDA=ON \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -DTESSERA_BUILD_NVIDIA_BACKEND=ON
ninja -C build tessera-opt

export PYTHONPATH=python
OPT=build/tools/tessera-opt/tessera-opt
FC=$(command -v FileCheck || echo /usr/lib/llvm-23/bin/FileCheck)
T=src/compiler/codegen/tessera_gpu_backend_NVIDIA/test/nvidia

# [1] the two NEW Target IR fixtures — expect PASS (structural; no GPU needed)
$OPT $T/nvidia_target_ir_typed.mlir | $FC $T/nvidia_target_ir_typed.mlir
$OPT --lower-tessera-nvidia-to-nvvm $T/nvidia_mma_sync_to_nvvm.mlir | $FC $T/nvidia_mma_sync_to_nvvm.mlir

# [2] on-device execution (sm_120) — PRE-EXISTING emit-path tests, not these commits,
#     but the natural on-hardware check while the box is up:
python3 -m pytest -q tests/unit/test_nvidia_plugin.py \
  tests/unit/test_conformance_execute_compare_nvidia.py \
  tests/unit/test_nvidia_perf_ratchet.py
```

**Actual:** both lit fixtures PASS. `test_nvidia_plugin.py` etc. execute the
`mma.sync` GEMM / flash-attention lanes on sm_120 and compare to the numpy reference;
the selected suite produced **68 PASS, 1 skipped**.

> **Increment-2 scope reminder:** the real `nvvm.mma.sync` lowering is validated by
> the NVVM verifier (structural), and is fed by a hand-written fragment-typed fixture.
> The `tile.mma` → fragment decomposition that would drive it from the full pipeline,
> and on-device execution *of the MLIR path* (vs. the Python emit path that carries
> execution today), remain follow-on work — see NVIDIA_AUDIT.md "Next Work" #6.
