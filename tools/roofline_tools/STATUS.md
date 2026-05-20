# Status: `runnable`

Tracked by `python/tessera/compiler/tools_manifest.py`.

## Current state

`tools/roofline_tools` is runnable through the tools surface audit.  The
audit command runs the bundled Nsight Compute CSV example and writes its
HTML, PNG, and JSON artifacts under `/tmp/tessera_roofline_tools_audit`.

The previous import failure is closed: `tprof_roofline.model` now exports
`analyze(kernels, device, dtype_key="fp32")`, which returns a
`RooflineResult` consumed by the `one` and `multi` CLI modes.

## Historical note

Before the 2026-05-19 tools repair pass, `cli_v2.py` imported
`DevicePeaks, analyze` but `model.py` did not define `analyze`, so the
CLI failed before argparse could show `--help`.  Keep this note as a
breadcrumb for future audits, but treat the project as active.
