# tools/profiler - Tessera Profiler

Features:
- Chrome trace and Perfetto-compatible Trace Event JSON export.
- HTML report generation with hot-op and roofline views.
- `tprof peaks print` for device peak FLOP/s and HBM GB/s from YAML.
- `--peaks/--arch` wiring for reports.

## Examples

```bash
# Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# Generate traces and a report using device peaks from YAML.
./build/tools/profiler/tprof \
  --demo-out demo.trace.json \
  --perfetto-out demo.perfetto.json \
  --report-out demo.report.html \
  --peaks tools/profiler/scripts/peaks_sample.yaml \
  --arch sm90

# Print peaks for CI logs.
./build/tools/profiler/tprof peaks print --peaks tools/profiler/scripts/peaks_sample.yaml --arch sm90

# View report locally.
python3 tools/profiler/scripts/tprof_view.py --root . --file demo.report.html
```

YAML shape:

```yaml
devices:
  sm90: { peak_flops: 2.0e14, hbm_gbs: 3000 }
# or:
sm90: { peak_flops: 2.0e14, hbm_gbs: 3000 }
```
