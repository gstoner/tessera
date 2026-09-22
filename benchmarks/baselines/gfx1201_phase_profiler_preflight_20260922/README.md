# gfx1201 ISA-preserving phase-probe preflight: refused

Tajasarus (RX 9070 XT, `gfx1201`) can execute the production folded HSACO,
but the external rocprofv3 PC-sampling path is unavailable: `/dev/kfd` is
absent. The recorder therefore does not launch a profiler collection and
marks phase attribution and promotion ineligible. A plain `rocprofv3
--kernel-trace` attempt ran the selected test but produced no trace artifact;
that is not a validated phase probe. No counters or PC samples from this host
are treated as production evidence.

The existing inline trace still changes the emitted program (64 versus 32
FP8 WMMAs, 24 versus four barrier instructions). The next profiler attempt
must first restore a usable profiler device/API, then capture the *same*
production image, validate sample-to-ISA mapping, cross-CU clock consistency,
clock-read cost, and overhead before any phase fraction can train a selector.
The [packet](evidence.json) binds this refusal to the exact host and recorder.
