# Checked asynchronous derivative tickets — 2026-09-08

Owner: AD-RESIDUAL-EVAL-1 / W2.4a; sync key IR-NATIVE-FOUNDATION-1.

The recorder runs two checked derivative generations on separate streams,
verifies independent status allocations and refuses output access before a
successful host wait/poll. It then injects a nonzero status after device
completion to prove failure refusal and safe release. This last case is fault
injection at the status-reader boundary, not a naturally failing kernel guard.

NVIDIA evidence belongs to RTX 5070 / SM120; ROCm evidence belongs to Radeon
8060S / gfx1151. No Apple/x86 device proof, overlap measurement or performance
promotion is implied. Allocation and release retain synchronous operations;
tracked reader scopes and dynamic returned shapes remain separate work.

Reproduce on each owning WSL host with:

```sh
python benchmarks/record_checked_derivatives.py --backend nvidia \
  --compiler build/tools/tessera-opt/tessera-opt --output packet.json
# Use --backend rocm on Princess-Luna.
```

Packets retain compiler, source and recorder SHA-256 identities.
