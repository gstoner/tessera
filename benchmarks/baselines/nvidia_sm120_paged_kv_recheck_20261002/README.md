# SM120 paged-KV diagnostic recheck — 2026-10-02

Recorded on Super-Bear: NVIDIA GeForce RTX 5070, GPU UUID
`GPU-cba12639-821a-7a10-4cd3-f918f9c0a545`, driver 610.88, compute capability
12.0. The recorder compared compiler-owned Graph → Schedule → Tile → PTX
paged-KV gather with the legacy staged CUDA gather for 128 boundary tokens,
512 ragged tokens, and 2048 ragged tokens, with permuted physical pages.

All six candidate/case correctness checks passed. The deliberately short run
used three samples, ten device-event repetitions, three end-to-end repetitions,
and two warmups. All six candidate rows failed the 4% two-cohort repeatability
gate. Keep these timings diagnostic; they do not change the existing retained
paged-KV disposition or its selector.

The packet records device-event kernel timing separately from end-to-end
launch/allocation/copy timing, plus compile/cache and resource data.

[Raw packet](paged_kv_recheck.json)
