# Fragment retirement boundary: attribution experiment

Owner ROCM-MXFP4-W4A8-1; sync ROCM-FOLDED-NATIVE-OPT-2026-10-02.

K16 boundaries plus a no-crossing boundary after each folded-scale fragment store.
This changes compiler scheduling, not numerical ordering or workgroup fences.
The candidate was removed from the active implementation. It did not resolve
the register pressure or establish a meaningful gain over the existing native
schedule. These are historical measured candidate images, not current defaults.

Authored Graph/Schedule/Tile native package numerics pass the owning gfx1201
gate. Production outputs match the HIP control bitwise and independently
sampled reference values before timing. Seven alternating device-clock/event
trials, three rotating resident copies; retained marker windows exceed 5 ms
and agree with their HIP-event witness within 5%. Timing includes host dispatch
gaps. Public launch transfer/allocation/module overhead is recorded separately.

| M | N | K | Native us/launch | HIP us/launch | Native/HIP | Native VGPR |
|---|---|---|---|---|---|---|
| 256 | 4096 | 5120 | 73.32 | 69.45 | 1.0557 | 176 |
| 256 | 8192 | 5120 | 146.63 | 142.94 | 1.0258 | 176 |
| 256 | 16384 | 5120 | 283.32 | 264.44 | 1.0714 | 176 |
HIP control uses 123 VGPRs. Both routes use 25,600 bytes LDS and no spills.
The original native route used 177 VGPRs. K16 issue boundaries leave 177;
adding fragment retirement gives 176. This does not attribute the peak live
register set or close the per-column gap. No profiler counters or Radiance
comparison; timing across different processes is not a paired before/after
speedup claim.

The recorder now grows short marker windows to the required duration and
retains rejected short spans as diagnostics; it normalizes each admitted
window by its own actual launch count. It still rejects clock disagreement.

Next: inspect compiler register lifetimes rather than infer their location
from the kernel's peak resource count. Folded image keys remain shape-dependent.

## Reconstruct candidate

candidate.patch is a narrow source patch against the restored native producer.
It adds the K16 and fragment-store scheduling constraints and metadata.
Rebuild tessera-opt, run the folded compiler/device tests, then run
benchmarks/rocm/record_gfx1201_folded_native_package.py with the matching
compiler and LLVM tools. This is an attribution candidate, not a route promotion.

device-tests.txt records 67 passed, including the initial wide-scale cases.
wide-recovery-tests.txt separately records the strengthened three cases:
finite recovery despite overflowing combined scale, nonzero bf16 recovery
despite underflowing combined scale, and positive zero from a zero partial.
Both native and HIP packages agree with the independent wide oracle.

Final restored native schedule: 67 passed, no warnings (restored-device-tests.txt).
Host audit/navigation/diagnostic/pass gates: 311 passed.
