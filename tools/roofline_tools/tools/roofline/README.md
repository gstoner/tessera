# tprof-roofline v2 — Compute + Communication rooflines

This extends the base tool with:
- **Comm rooflines & overlays**: NVLink/PCIe/NIC link bands (GB/s, right Y-axis) + per-link dots from CommQ or Perfetto.
- **Auto-peaks**: `tprof_roofline/peaks_auto.py` tries `tprof peaks --json` and emits YAML (fallback template provided).
- **Classifications export**: CSV/JSON of per-kernel class (memory/compute-bound) + `Δlog10(OI-knee)` distance.
- **Multi-device views**: `cli_v2.py multi` produces a tabbed HTML with side-by-side device charts.
- **Nsight Compute ingestion**: `--fmt nsight` heuristic parser for common columns (`dram__bytes.*`, `flop_count_*`, `Duration`).

## One device (CSV, Perfetto, or Nsight)
```bash
# CSV
python3 tools/roofline/cli_v2.py one --peaks tools/roofline/peaks/sm90_with_links.yaml   --input tools/roofline/examples/nsight_min.csv --fmt nsight --dtype bf16_tensor --outdir out/

# Perfetto (compute + comm)
python3 tools/roofline/cli_v2.py one --peaks tools/roofline/peaks/sm90_with_links.yaml   --input tools/roofline/examples/trace_perfetto_mixed.json --fmt perfetto --dtype bf16_tensor --outdir out/   --export-csv out/classify.csv --export-json out/classify.json
```

## Multi-device tabs
```bash
python3 tools/roofline/cli_v2.py multi --pairs '[
  {"peaks":"tools/roofline/peaks/sm90_with_links.yaml","input":"tools/roofline/examples/trace_perfetto_mixed.json","fmt":"perfetto","dtype":"bf16_tensor"},
  {"peaks":"tools/roofline/peaks/sm90_with_links.yaml","input":"tools/roofline/examples/nsight_min.csv","fmt":"nsight","dtype":"fp32"}
]' --outdir out_multi/
```

## Peaks auto-generation
```bash
python3 -m tools.roofline.tprof_roofline.peaks_auto tools/roofline/peaks/auto.yaml
```

### Perfetto schema (what to emit from your exporter)
- **Compute tile events**
  ```json
  {"type":"compute","name":"<tile>","flops":<FLOPs>,"dram_bytes":<bytes>,"dur_us":<microseconds>,"dtype_key":"bf16_tensor"}
  ```
- **CommQ events**
  ```json
  {"type":"comm","name":"<chunk>","bytes":<payload_bytes>,"dur_us":<microseconds>,"link":"NVLink4|PCIe|NIC"}
  ```
Links rendered as throughput bands are defined in the peaks YAML under `links:`.

## Outputs
- `roofline_comm.png` — compute roofline + right-axis link bands and comm points
- `roofline_report.html` — chart + tables
- (optional) `classify.csv` / `classify.json` — per-kernel classification export
- `roofline_multi.html` — tabbed, multi-device, side-by-side figures
