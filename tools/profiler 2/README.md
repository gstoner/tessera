# tools/profiler — Tessera Profiler (v4)

New:
- **`tprof peaks print`** subcommand to echo device peak FLOPs & HBM GB/s from YAML.
- Still includes: Perfetto exporter, Chrome trace, HTML roofline/hot-ops report, `tprof_view.py`, and `--peaks/--arch` wiring.

## Examples
```bash
# Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# Generate traces + report (auto peaks from YAML)
./build/tools/profiler/tprof   --demo-out demo.trace.json   --perfetto-out demo.perfetto.json   --report-out demo.report.html   --peaks tools/profiler/scripts/peaks_sample.yaml   --arch sm90

# Print peaks for CI logs
./build/tools/profiler/tprof peaks print --peaks tools/profiler/scripts/peaks_sample.yaml --arch sm90

# View report locally
python3 tools/profiler/scripts/tprof_view.py --root . --file demo.report.html
```

YAML shape:
```yaml
devices:
  sm90: { peak_flops: 2.0e14, hbm_gbs: 3000 }
# ...or...
sm90: { peak_flops: 2.0e14, hbm_gbs: 3000 }
```
