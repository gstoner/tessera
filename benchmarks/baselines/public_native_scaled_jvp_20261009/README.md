# Public native scaled JVP — GFX1201

Owning items: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Synchronization key: PUBLIC-NATIVE-JVP-GFX1201-20261009; shared integration
key PUBLIC-NATIVE-JVP-20261009.

The public autodiff JVP entry invokes typed continuous FP32 scaled products
through Graph AD, native Schedule/Tile, ROCm Target/LLVM, HSACO and checked HIP
ownership. The caller Graph and differentiation request remain unchanged.
Both operand transpose flags, direct/leading-nested maps, non-leading result
placement and all/one-matrix/scale-only tangent activity are covered.
Inactive tangents are explicit None; the independent FP64 oracle supplies zero
directions for those inputs. Warm calls forbid compiler subprocesses and retain
independent old outputs. All 36 exact-device tests pass with a fresh provider.

Two fresh processes record the same 36 profiles. Each checks primal and tangent
before timing and after every public/event/captured-member sample.
The native program timed here is extracted from the actual public child package;
no alternate kernel is substituted. Both packets pin 23 source files, the
matching LLVM/MLIR 23.1.1 tessera-opt binary and both freshly built HIP providers.
The live device is RX 9070 XT, gfx1201; the packet records the active HIP device
index, name and UUID.

| Timing domain | Run 1 range | Run 2 range |
| --- | ---: | ---: |
| Completed warm public call | 1.065–2.533 ms | 1.094–2.643 ms |
| Interleaved native program events | 4.527–25.124 us | 4.072–24.881 us |
| Smallest captured member event window | 3.170 ms | 3.291 ms |

These ranges span different profiles. Public time includes request/binding,
native preparation, upload, launch, completed readback and independent outputs.
Interleaved event time includes device work and host dispatch gaps over 128
repetitions, excluding preparation/update/readback; the native value is already
per invocation. Captured member timing groups 2048 repetitions per pure-SSA
member, excluding capture/instantiation/copies, and is a distinct diagnostic
schedule. Its values must not be substituted for ordinary program timing.

Maximum absolute error in both processes: 8.189881861575543e-8.
The public overhead remains open; these packets do not demonstrate a speedup,
default-route promotion, generic scaled_matmul closure or quantized AD parity.

Recorder: benchmarks/rocm/record_public_native_scaled_jvp.py.
Packets: run1.json and run2.json. Fresh-provider test log: device-tests.txt.
Test: tests/device/rocm/test_public_native_scaled_jvp_transform.py.

This GFX1201 proof does not establish gfx1151, NVIDIA, Apple or x86 parity.
