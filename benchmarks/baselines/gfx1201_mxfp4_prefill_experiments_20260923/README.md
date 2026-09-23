# GFX1201 folded-prefill epilogue, residency, and BN128 experiments

Owner `ROCM-MXFP4-W4A8-1 / IKF-1`; sync
`GFX1201-MXFP4-PREFILL-EXPERIMENTS-2026-09-23`. The
[matched packet](evidence.json) comes from the Tajasarus RX 9070 XT/gfx1201
with pinned Radiance revision `dfdfa383` and fragment-order weights. Each
shape passed an independent sampled exact K32 reference and full bitwise BF16
agreement across the exact route, ordinary and safe expanded-folded routes,
manual BN128/TN4 expanded route, packed-permute route, and Radiance. Eleven
interleaved HIP-event trials of twelve iterations each measured kernels, not
host launch or model latency:

| M×N×K | Expanded TN2 | Safe TN2 | Expanded TN4 | Packed TN2 | Radiance |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256×5120×8704 | 170.5 µs | 164.9 µs | 179.4 µs | 180.4 µs | 138.4 µs |
| 1024×17408×5120 | 1158.2 µs | 1146.3 µs | 1104.6 µs | 1165.2 µs | 887.4 µs |

The safe-scale ABI has a separate identifier and host-array certificate for
the FP32 activation scales and E8M0 row references. The runtime recomputes
that certificate before a host-array launch; altered buffers are refused.
Tajasarus tests proved BF16 output for ordinary and cancellation-edge scales
and checked refusal of changed activation bytes. Removing the rare FP64
fallback reduces the selected symbol from 4,195 to 2,724 instructions, but
raises VGPRs 109→140 and yields only small timing gains. Its certificate is
not a device-resident producer contract; no automatic selection is admitted.

Expanded TN2 is 5.5% faster than packed TN2 on the smaller shape, but their
wide-shape medians differ by only 0.6%, well below a credible selector margin
given sample variation. Model-level weight-byte accounting is decisive for
whether both representations can remain resident: an illustrative model of
32 matrices with N=17408, K=5120 requires 1,515,749,376 bytes packed
including scale planes and another 2,852,683,776 bytes for expanded weights
and row references. Keeping packed decode while adding expanded prefill needs
the full 2.85 GB extra, not merely the difference between layouts. The
packet supplies no real free-memory budget (`available_extra_bytes=0`), so
the planner refuses this illustrative trial. It excludes activations,
fragmentation, code objects, and other model weights; it is not a whole-model
VRAM measurement.

BN128/TN4 doubles WMMA per CTA (32→64) and raises VGPRs 109→181 and LDS
25,600→30,720 bytes. It passes ragged-N device output at N=144. The
source-derived A staging request halves when BN doubles: 178→89 MB on the
small shape, 1426→713 MB on wide N. These are requested bytes, not measured
DRAM bytes. The small shape has only 40 BN128 CTAs versus 64 device CUs,
plausibly contributing to its regression; wide N has 544 CTAs and improves
4.6% versus TN2. Neither scheduling nor occupancy is attributed from static
ISA counts alone. The wide BN128 route still trails Radiance by 1.25×.

The packet's `source_revision` is the merged-main base `fb392b53`; its source
hashes bind the changed generators and benchmark in the isolated device
worktree. These candidates stay manual. Next, prove an upstream producer's
device-resident scale certificate and obtain a real model memory budget;
then compare selected-symbol load-to-wait placement and controlled A-stage
scheduling changes using matched inputs. Do not promote from these timings.
