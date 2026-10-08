# Public native gfx1201 scale-VJP integration

Owner: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization: SCALED-PUBLIC-TRANSPOSE-2026-10-07.

Ordinary @jit reverse requests now use the family-owned native scale-transpose
plugin. Traced typed Graph retains ROCm architecture and derivative intent.
Native paired AD -> Schedule -> Tile -> Target -> ROCDL/LLVM -> HSACO feeds
the checked HIP program owner. Python binds inputs and immutable packages;
it performs no production numerical differentiation or GPU launch sequence.

72 owning gfx1201 cases pass independent float64 scale-adjoint numerics and
changed-cotangent replay. They cover scalar calls plus one/two leading maps,
three batching policies, KN/NK storage and lhs/rhs/paired/reordered scale
gradients. Compiler subprocesses are forbidden during warm replay.
Physical execution certificates validate actual gfx1201 execution.
Matrix-code and discrete-scale derivative maps remain rejected.

397 shared map/plugin/op/dtype/diagnostic/pass gates pass.
11 exact RTX 5070 NVFP4 map regressions pass; this validates existing NVIDIA
map parity, not NVIDIA scale transpose execution. Original loader, argument
binding and target-metadata failures remain recorded.

48 one/two-map public timing rows pass correctness before and after timing.
Warm public wall-time medians range 0.901–11.067 ms. They include frontend
checks, immutable package/ABI binding, uploads, native sequence and readback.
Compare the separately labeled native/prepared windows in
../rocm_native_scaled_vjp_20261007/README.md; these are not isolated ISA timings
or a speedup/selector promotion.

The actual repaired compiler and its linked layout library were delivered to
Tajasaurus scratch; identity.json and compiler-identity receipt bind the bytes.
TESSERA_ROCM_NATIVE_MOVEMENT_LIB explicitly selects the existing checked HIP
owner when the isolated compiler path is used. This is not a new local C++
rebuild on Tajasaurus. Device tests retain a pytest timeout-plugin warning.

Reproduce with the matching compiler and owner:
pytest -q tests/device/rocm/test_public_scaled_vjp.py
python benchmarks/rocm/benchmark_native_scaled_vjp.py --public --output FILE

Open: dynamic/nonleading/deeper/mixed maps, general composed and storage AD,
initial serial-reduction performance optimization, sibling reverse execution,
generic batching/transpose closure, fresh full-unit green and publication.
