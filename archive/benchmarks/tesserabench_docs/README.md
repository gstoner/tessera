# TesseraBench design documents (archived 2026-09-17)

Eight documents (`tesserabench_doc1.md` … `tesserabench_doc8.md`) describing a
benchmarking framework called **TesseraBench**: a `TesseraBenchCore` class, a
`tesserabench` package and CLI, `tesserabench.ci` / `.regression` / `.reports` /
`.server` / `.production` modules, executive dashboards, multi-cloud deployment,
NVL72-scale distributed sweeps, a Dockerfile and an "enterprise" roadmap.

**None of that exists.** Checked against the tree on 2026-09-17: of the 26 class
names the documents present as real, 23 appear nowhere in `python/`, `benchmarks/`,
`tools/` or `src/`; of 14 module paths, 13 do not exist; there is no
`tesserabench` package and no `tesserabench` command. The documents are a
design sketch written in the voice of a shipped product, and `docs/benchmarks/README.md`
had been calling them "the official benchmarking and performance validation
framework" since 2026-07-14.

They are kept here for the same reason `../matrix_multiplication/` is: useful
historical context for what a benchmark framework *could* be, and not a
description of what runs. What runs is inventoried in
[`benchmarks/README.md`](../../../benchmarks/README.md); the live status per surface
is generated at `docs/audit/generated/surface_status.md`. Anyone reviving an
idea from these documents lands it as code beside a consumer (Decision #29),
not by editing the prose.
