# Frontend residual and NVIDIA fragment proof — 2026-09-28

Owners: FRONTEND-IR-MEDIUM-1, AD-RESIDUAL-EVAL-1 and W1.1.
Sync: `FRONTEND-RESIDUAL-FRAGMENT-2026-09-28`.

## Public frontend and saved-input ownership

The public residual `x*x*x - theta` exposed missing Tracer arithmetic and an AST
fallback in persistent-tape compilation. Tensor `+`, `-` and `*` now use existing
canonical operation hooks. Public persistent-tape compilation requires the
tracer-owned Graph; unsupported source refuses before packaging.

The recorder compiles the paired MLIR AD products through native GPU storage,
serializes/reloads both images and checked ABIs, captures resident inputs, then
overwrites the caller's inputs. Eleven backward calls still match the analytic
VJP. Backward after frame closure refuses. Width 17, f32: 136 bytes of saved
input snapshots; zero additional exported residual tensor bytes. This does not
prove general residual-layout or alias support.

- Super-Bear, RTX 5070 sm_120: median 0.190 ms.
- Princess-Luna, gfx1151: median 0.400 ms.

These are synchronized host-wall backward calls including output allocation,
native launch and synchronization; readback, compile and capture are excluded.
They are diagnostics, with no device-time, comparative speedup or promotion
claim. Each JSON records compiler/LLVM/image-binding/product-lineage identities,
ABIs and all samples. The host compilers were built from merged #877, with the
x86/ROCm attention compiler slice additionally present on Princess-Luna. Python
frontend changes in this slice were applied on both hosts.

Reproduce with `PYTHONPATH=python:.` and the owning backend environment script:
`python benchmarks/autodiff/record_frontend_residual_tape.py --backend nvidia --chip sm_120 --compiler build/tools/tessera-opt/tessera-opt --output /tmp/residual.json`
(use `--backend rocm --chip gfx1151` on Princess-Luna).

## NVIDIA typed accumulator

Super-Bear: existing nine typed-fragment tests passed. Four new exact-device
rows prove zero, one, two and four 16-wide K panels against NumPy. Multi-panel
rows explicitly distinguish accumulated output from the final panel alone.
The fixture now advances by 16 with matching physical leading dimensions of
64; its FileCheck lowering gate passed. This validates the existing SCF type
conversion through LLVM/NVPTX, without inventing another materializer.

W1.1 remains landing: the two tensor-valued TileIRLoweringPass producers still
need native tensor-to-fragment materialization. This pointer-backed proof does
not migrate them or certify the legacy WGMMA compatibility route.

## Validation and limits

Frontend/tape/registry sweep on WSL: 103 passed, 32 skipped. The initial focused
frontend/tape set passed 30 tests. Apple and x86 residual execution were not
measured in this GPU slice. Broader raising, masks, dynamic residual layouts,
async ownership and paired general-solver frontend integration remain open.
