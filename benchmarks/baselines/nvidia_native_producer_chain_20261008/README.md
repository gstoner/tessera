# Static native SM120 producer-chain characterization

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: SM120-NATIVE-PRODUCER-CHAIN-2026-10-08.

RTX 5070 execution of an ordinary traced RMSNorm → softmax → matmul chain.
Graph outlining and Schedule/Tile member generation are compiler-owned.
A checked C++ owner executes producers using alternating bounded scratch,
then typed-fragment matmul on the same stream. Completion precedes retirement.

Four profiles cover FP16/BF16, M/K/N=(17,35,19) and (64,256,64).
Independent float64 arithmetic rounds each producer to the physical storage.
All profiles pass before and after timing; maximum absolute error is
7.798298611305654e-6.

Separate resident CUDA-event stage medians span:
RMSNorm 9.176–9.927 us; softmax 9.885–45.538 us; matmul 9.224–9.737 us.
Prepared whole-program host-wall medians span 0.2234–0.2898 ms, including
copies and synchronization. These domains are not interchangeable; the
stage medians are not summed into a claimed whole-program device time.

packet.json records GPU UUID, driver, compiler hash, source fingerprints,
native image/descriptor/plan digests and every sample. No A/B performance
gain, fusion, selector promotion or cross-architecture parity is claimed.

The combined public JIT, partition and portable replay lane passes 167 tests.
Four ordinary public JIT/replay cases pass across both storage types and
RHS orientations. Replay and warm calls forbid compiler subprocesses.
Broader producer composition, dynamic-chain capacity, whole-program device
timing, fresh full-suite closure and focused PR delivery remain open.


## Bounded runtime investigation

The checked append-producer CUDA ABI passes 14 independent FP16/BF16
capacity tests across every nonempty M/N/K subset. Capacity, smaller and
singleton active shapes agree numerically; host arena statistics remain
unchanged and over-bound inputs are rejected. The compiler's dynamic
multi-producer admission guard remains intact. This runtime investigation
does not establish compiler-generated dynamic-chain manifests or public
dynamic-chain support.
Evidence: runtime-capacity-proof.log.


## Three-producer scratch alternation and epilogues

LayerNorm → RMSNorm → softmax → matmul has 16 owning-device cases across
FP16/BF16, row/column RHS, ragged/dense shapes and plain/fused epilogues.
Each verifies ordinary JIT, repeated warm execution, serialized replay and
resident execution. The fused contract performs bias, ReLU and residual
before the final FP16 store. Numerical oracles round every producer to storage.
No compiler subprocess or eager callable is allowed during warm/replay proof.

two_and_three_producer_packet.json contains eight before/after correctness
profiles, independent stage CUDA-event samples and complete prepared-program
host-wall samples. Three-producer host-wall medians span 0.2342–0.2936 ms.
This is characterization, not matched A/B speedup or whole-program device time.
Dynamic compiler admission remains gated; no v4 manifest is admitted.


## Expanded regression result

The expanded owning-device lane produced 184 passes and one new boundary-test
fixture failure (positional dataclass arguments in the wrong order). The fixture
was corrected to named arguments; both legacy chain-certificate refusal cases
pass in the focused rerun. The initial failure and repair logs are retained.
This does not claim a fresh full-suite green result. Full WSL validation remains
in progress; generic scaled batching/transpose states remain open.
