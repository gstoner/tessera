# Encoded scaled residuals — gfx1201

Owner: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Synchronization key: ENCODED-SCALED-RESIDUAL-20261014.

Named shape: M17/K256/N19, KN E4M3FN matrix storage, row-major F32 gate/output.
FP32 scales use K128/N128 groups; E8M0 exponents use K32/N1 groups.
Native Graph AD retains encoded primal coefficients and differentiates only
requested continuous FP32 scales/gates. E8M0 is gate-only; encoded matrices
and exponent bytes are not differentiable storage. Original policy/bytes
flow through verified Schedule/Tile/Target/LLVM/HSACO program members.

The differentiated public owner now executes native primal, public JVP and
native reverse. Twelve compiler/contract tests cover six native product
constructions plus host/native rejection of coded-storage gradients.
Four exact RX 9070 XT/gfx1201 cases cover full, gate-only and reordered roles,
changed raw matrices/scales, both warm JVP outputs, compiler-subprocess
refusal, original Graph/request identity and retained result generations.
The existing eight continuous FP32 cases also pass with the serial default.

Shared assessment: 3483 fully provisioned WSL host AD/registry cases and 47
RTX5070 bounded saved-LSE/resident reverse cases pass after rebuilding.
The standalone-multiply review regression has six real frontend red/green
cases on gfx1151/gfx1201. Native multiplication remains a private program
member; existing standalone scalar/tensor routing is preserved.

Fresh final-source WSL non-slow suite: 26,072 passed, 7,568 skipped, 874 deselected and 11 warnings; no failures.

## Timing

Measured serial / wave / wave / serial, alone on Tajasaurus. Every packet
records actual GPU identity, HIP version, compiler/provider hashes and changed
source/manifest/capability pins. All four source/binary maps match. Primal and
JVP program/image identities match across arms; reverse identities match
within each schedule. Independent FP64/finite-difference comparisons precede
and follow every completed-call/native-program/grouped-member timing domain.
Maximum absolute error: 3.222383341494606e-06.

| Scale format / schedule | Completed primal ms | Completed JVP ms | Completed VJP ms | Reverse program events us |
| --- | ---: | ---: | ---: | ---: |
| fp32 / serial | 1.952–2.091 | 1.314–1.322 | 5.861–5.921 | 3887.507–3906.186 |
| fp32 / wave | 1.682–1.715 | 1.101–1.133 | 1.693–1.707 | 167.012–167.637 |
| e8m0 / serial | 1.896–1.959 | 1.141–1.238 | 1.576–1.592 | 111.965–112.058 |
| e8m0 / wave | 1.898–1.899 | 1.227–1.254 | 1.553–1.592 | 111.179–111.188 |

Interleaved program events include host dispatch gaps; they are not pure
kernel time. Grouped member windows use 2048 repeats and exclude host copies,
capture and instantiation. Captured member samples remain a separate
diagnostic schedule. Public medians include frontend binding, preparation,
upload/launch/readback and independent outputs.

The explicit wave option passes all four encoded cases. It reduces the named
FP32-scale reverse event/completed-call time in this paired characterization.
The eight continuous FP32 cases retain the existing separate-admission
refusal for general wave scheduling. The serial default is unchanged.
No generic wave/default promotion, FP4 proof, global AD closure or sibling
physical schedule claim is made.

Recorder: benchmarks/rocm/record_native_encoded_scaled_gate.py.
Packets: run1.json, wave-run1.json, wave-run2.json, run2.json.
Explicit option: --scale-vjp-schedule wave_per_scale_element.
Remaining: FP4 encoded coefficients, dynamic/mapped/transpose/general AD,
broader wave admission, native reverse ownership/binding overhead and
gfx1151/Apple/x86/SM120 encoded-gate consumers.

Validation receipts: host-full-unit.txt, host-ci-audit.txt and sm120-parity.txt.
