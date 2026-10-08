# K16 scheduling boundary: attribution experiment

Owner ROCM-MXFP4-W4A8-1; sync ROCM-FOLDED-NATIVE-OPT-2026-10-02.

One VALU/SALU-only crossing boundary between each K16 load set and its MMA group.
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
| 256 | 4096 | 5120 | 73.13 | 69.14 | 1.0578 | 177 |
| 256 | 8192 | 5120 | 147.70 | 142.85 | 1.0340 | 177 |
| 256 | 16384 | 5120 | 283.61 | 264.20 | 1.0735 | 177 |
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
