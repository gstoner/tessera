# Bounded ROCm version-query reuse

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync ROCM-VERSION-METADATA-2026-10-07.

Repeated compiler/driver --version calls now use a bounded 32-entry cache
keyed by supplied/resolved executable paths, device/inode/size/mtime/ctime
and PATH/loader/ROCm environment. Replacement, rewrite, symlink and environment
changes re-query; failed queries are not cached. Device-library discovery,
content hashing and native image identities retain their existing authorities.

49 host WSL checks pass; 22 ROCm device cases appropriately skip on NVIDIA.
40 scalar owning gfx1201 primal/JVP regressions pass with the cache candidate.
All 16 real-package A/B profiles retain identical compiler/toolchain
fingerprints. Alternating five-sample trials prime version metadata outside
timing, then compare an isolated empty metadata dictionary against retained
metadata. Both arms perform the same native Graph/member compilation.

Subprocess count drops from 7 to 5. Uncached/cached median ratios span
1.1012–1.1458, median 1.1205. Package wall medians span 156.17–184.62 ms
uncached and 138.60–164.75 ms cached. This is a metadata-work comparison,
not public warm JIT latency or kernel performance. Raw samples and actual
native compiler/source fingerprints are retained.

The owning profile covers static scalar/shared-RHS/independent/shared-LHS
FP8 and MXFP8 with both B orientations. gfx1151, other ROCm families and
wider toolchain/layout/AD envelopes still require their own validation.
No selector promotion, broad family closure or sibling execution is claimed.
