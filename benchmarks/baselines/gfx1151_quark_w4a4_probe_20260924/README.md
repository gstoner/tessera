# Quark W4A4 probe on gfx1151

The same scalar W4A4 probe sealed for gfx1201 in
[`gfx1201_quark_w4a4_probe_20260923`](../gfx1201_quark_w4a4_probe_20260923/README.md),
packaged for and launched on Princess-Luna's Radeon 8060S (`gfx1151`,
RDNA 3.5). The kernel is scalar HIP with no WMMA, so nothing in it is
RDNA4-specific; this packet is gfx1151's **own** exact-device run, not a
transfer of the gfx1201 result.

Both opt-in BF16 cases from the pinned projection-slice reference (gate and
down rows) and the ragged K=64 scale-cancellation case matched their host
references bit-exactly. [`evidence.json`](evidence.json) binds the generator,
the device and unit fixtures, the shared reference packet, and each launched
HSACO digest. HIP builds embed build-specific data, so a recompilation need
not reproduce these payload digests.

Scope is unchanged from the gfx1201 packet: a bounded numerical probe, not a
Quark checkpoint converter, a production W4A4 lowering, or a performance
claim. E8M0 codes 0 and 255 remain refused.
