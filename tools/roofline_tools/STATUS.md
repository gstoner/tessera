# Status: `broken`

Tracked by `python/tessera/compiler/tools_manifest.py`.

## Why

`tools/roofline_tools/tools/roofline/cli_v2.py:4` imports:

```python
from tprof_roofline.model import DevicePeaks, analyze
```

but the bundled `tprof_roofline/model.py` does not export `analyze`.
The import therefore fails immediately at module load:

```
ImportError: cannot import name 'analyze' from 'tprof_roofline.model'
```

The CLI is otherwise structurally complete (it routes between
`one`/`multi` modes, ingests Nsight CSV + Perfetto JSON traces, and
emits HTML reports).

## Path forward

Either:

1. Add the missing `analyze` function to
   `tools/roofline_tools/tools/roofline/tprof_roofline/model.py`
   (likely the analysis entry point that was hoisted out and never
   re-exported).
2. Or rewrite the CLI import to use whatever the model module
   currently exposes (the existing `DevicePeaks` works; just the
   analysis function is missing).

Once the import is restored, the row promotes to `runnable` and the
audit gates kick in.
