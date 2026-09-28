# GA/EBM route receipts — 2026-09-27

EVIDENCE-PACKET-1, sync `EVIDENCE-PACKET-1-2026-09-27`. One record per fleet
host from `benchmarks/record_ga_ebm_route_receipts.py`. Each record holds the
`clifford_core`, `energy_core` and `visual_complex_core` default sweeps. Every
row carries `route` and `route_receipts` (`tessera._route_receipts`) for its
timed span. Each host ran from a clean worktree at the commit in its `host`
block.

| File | Host | Recorded on |
|---|---|---|
| `mac_m1max.json` | Mac M1 Max, macOS 27 | the Apple GPU runtime lane |
| `princess_luna.json` | Princess-Luna, Zen 5, WSL2 | the x86 AVX-512 lane; gfx1151 present |
| `tajasarus.json` | Tajasarus, Zen 5, WSL2 | the x86 AVX-512 lane; gfx1201 present |
| `super_bear.json` | The-Super-Bear, Zen 2, WSL2 | no x86 lane; sm_120 present |

These are attribution receipts, not performance evidence. The latencies are
host-wall composition timings, and no row can promote. Readers re-derive each
row's route with `validate_receipt_summary`
(`tests/unit/test_route_receipts.py`).
