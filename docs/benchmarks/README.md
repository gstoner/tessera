---
status: Informative
classification: Informative
authority: benchmark surface index
last_updated: 2026-09-17
---

# Benchmarks — where the real material is

This directory used to hold eight "TesseraBench" documents describing a
benchmarking framework (`TesseraBenchCore`, a `tesserabench` package and CLI,
CI/regression/reporting/production modules, an enterprise roadmap) and called
it "the official benchmarking and performance validation framework". A check
on 2026-09-17 found that 23 of the 26 classes and 13 of the 14 module paths
those documents name do not exist, nor does the package or the command. They
were a design sketch in the voice of a shipped product; they now live at
[`archive/benchmarks/tesserabench_docs/`](../../archive/benchmarks/tesserabench_docs/README.md)
with that check recorded.

What exists, and where it is described:

| Surface | Where | What it is |
|---|---|---|
| Benchmark runners and recorders | [`benchmarks/README.md`](../../benchmarks/README.md) | The inventory of active, proxy, hardware-gated and archived benchmark families, with the quick-check commands that run on a CPU-only host |
| Row schema | [`benchmarks/common/artifact_schema.py`](../../benchmarks/common/artifact_schema.py) | `compiler_path` / `runtime_status` / `execution_kind` on every row, so a reference number can never read as a native one (Decision #12 and its 2026-08-30 amendment) |
| Sealed evidence | [`benchmarks/baselines/`](../../benchmarks/baselines/) and [`benchmarks/e2e_spine/`](../../benchmarks/e2e_spine/) | Hash-sealed packets and ledgers recorded on the owning device; read through the generated dashboards, never copied into prose (Decision #26) |
| Live status | [`docs/audit/generated/surface_status.md`](../audit/generated/surface_status.md), [`docs/audit/generated/e2e_fleet.md`](../audit/generated/e2e_fleet.md), [`docs/audit/generated/runtime_execution_matrix.md`](../audit/generated/runtime_execution_matrix.md) | Generated, drift-gated; the primary evidence for what is proven on which hardware |
| Roofline tooling | [`tools/roofline_tools/`](../../tools/roofline_tools/) | Reads the stable JSON schema that `benchmarks/run_all.py` emits |
| Performance gates | [`benchmarks/perf_gate.py`](../../benchmarks/perf_gate.py) | The ratchet the device lanes are held to |

Two standing rules from `CLAUDE.md` that apply to every benchmark number in this
repository: a row must say which lowering **route** produced it and what its
**latency source** was (device clock and wall clock are not comparable), and no
number promotes anything without exact-device proof on the hardware it names —
WSL2 timings on the ROCm and NVIDIA boxes do not promote.
