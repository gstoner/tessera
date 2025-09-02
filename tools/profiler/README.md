# tools/profiler — Tessera Profiler (drop-in skeleton, Perfetto + report)

New in this version:
- **Perfetto exporter**: `tprof --perfetto-out demo.perfetto.json` (Trace Event JSON, Perfetto-compatible).
- **HTML report stub**: `tools/profiler/scripts/tprof_report.py` builds a hot-ops table and a roofline-ish scatter.

Quick build:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/tools/profiler/tprof --demo-out demo.trace.json --perfetto-out demo.perfetto.json
python3 tools/profiler/scripts/tprof_report.py --in demo.trace.json --out demo.report.html --peak-flops 2.0e14 --hbm-gbs 3000
```
Open `demo.report.html` in your browser.
