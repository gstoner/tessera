# gfx1151 ROCm shape-free cache recheck — 2026-10-02

Princess-Luna exact gfx1151 checks passed for shape-independent native image
reuse in softmax, reduction, and scheduled attention. The tests verify that
changed runtime shapes and caller symbols reuse one compiled image and entry,
while package-specific shape guards remain distinct and outputs match their
independent references. This closes only these tested envelopes. Timings were
not collected in this packet; gfx1201 scheduled matmul timings are recorded in
[the warmed cache packet](../rocm_gfx1201_shape_key_20261002/README.md).

[Fingerprinted result](exact_device_recheck.json) ·
[pytest transcript](exact_device_pytest.txt).
