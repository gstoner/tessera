# gfx1201 W8A8 current paired follow-up — 2026-10-01

This packet remeasures the production scheduled Tile route against AITER on
Tajasaurus (RX 9070 XT, gfx1201), using current source commit
58b848ccbc7682db03d3b1e350a5421ded56984d and a freshly built tessera-opt
(SHA-256 recorded in w8a8_paired.json). The checkout was clean when the packet
was generated. Seven device-clock windows of at least 6 ms were collected;
HIP-event timing cross-checks the device-clock marker. Every arm passed the
benchmark numerical oracle before timing.

| M×N×K | Production Tessera NK | AITER | Tessera/AITER | Result |
|---|---:|---:|---:|---|
| 200×8192×1024 | 38.82 µs | 42.67 µs | 0.910× | Tessera 9.0% faster |
| 200×2048×2048 | 20.46 µs | 23.70 µs | 0.863× | Tessera 13.7% faster |
| 1024×3072×1536 | 71.40 µs | 69.97 µs | 1.020× | Tessera 2.0% slower |

The first two rows close the previously measured M=200 regressions for these
shapes under the current source and exact gfx1201 device. The K=1536 ragged
case remains slightly slower than AITER. These three rows do not close the
wider short-K/ragged-K envelope and do not justify selector promotion. The
kn layout arm is included as an attribution control and is not the selected
production route.

Raw timings, source and compiler hashes, device metadata, clock windows, and
per-arm numerical errors are in [w8a8_paired.json](w8a8_paired.json).

## Expanded Tessera layout sweep

**Device:** Tajasarus, gfx1201  **Source:** `58b848ccbc7682db03d3b1e350a5421ded56984d`  **Compiler SHA-256:** `97f0ecb808c0232e7932f005773076e74cb5246a0f5e0e991001c5e072b925e2`

The exact-device run checks the Tessera KN and NK weight-layout arms against the fp64 numerical reference before timing, then records seven device-clock windows with HIP-event cross-checks. The sweep covers short K=1024/2048, K=1536, and M=127/200/255/1024.

| M×N×K | KN median | NK median | KN/NK | max relative oracle error |
|---|---:|---:|---:|---:|
| 127×8192×1024 | 180.24 µs | 24.98 µs | 7.21× | 8.38e-08 |
| 200×4096×1024 | 181.89 µs | 23.68 µs | 7.68× | 8.16e-08 |
| 1024×4096×1024 | 173.03 µs | 65.55 µs | 2.64× | 1.16e-07 |
| 200×2048×2048 | 380.61 µs | 21.00 µs | 18.12× | 7.40e-08 |
| 255×2048×2048 | 356.40 µs | 22.04 µs | 16.17× | 8.87e-08 |
| 127×3072×1536 | 280.69 µs | 16.64 µs | 16.86× | 7.35e-08 |
| 255×3072×1536 | 256.62 µs | 25.81 µs | 9.94× | 9.10e-08 |
| 1024×3072×1536 | 200.06 µs | 70.80 µs | 2.83× | 9.04e-08 |

The production NK layout outperformed the KN attribution arm on all eight rows. The comparison is between Tessera layouts; this packet has no AITER arm because the Tajasaurus benchmark environment lacks Triton. It does not close AITER-relative performance for the wider envelope or authorize a selector change.

The earlier paired AITER packet remains the comparator for its three exact shapes (200×8192×1024, 200×2048×2048, and 1024×3072×1536): [w8a8_paired.json](w8a8_paired.json). Those rows were measured in that separate run and are not presented as paired samples against this sweep.

Raw route sweep: [w8a8_short_ragged_envelope.json](w8a8_short_ragged_envelope.json).
