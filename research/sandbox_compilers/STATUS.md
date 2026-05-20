# Status: `broken`

Tracked by `python/tessera/compiler/research_manifest.py`.

## Why

`tilec/driver.py` carries a `SyntaxError` at line 35.  The body of
the `elif args.backend == "cpu":` branch is indented at column 12
instead of column 8, which throws off the rest of the chain:

```python
    elif args.backend == "cpu":
            from .backends import codegen_cpu          # col 12 — wrong
            cfile = codegen_cpu.emit(...)              # col 12 — wrong
            ...

        elif args.backend == "tessera":                # col 8 — wrong (no matching if)
            ...
```

Python parses the first over-indented line as the suite for the
`elif`, then sees `elif args.backend == "tessera":` at column 8 and
fails because there is no preceding `if`/`elif` at that indent.

The whole `tilec` module is unusable until that block is reformatted.

## Path forward

Fix the indentation so the `cpu` branch body sits at column 8 like
the other branches, then either:

1. Promote the row to `compile_only` (the driver emits C / CPU /
   Tessera-MLIR artifacts but doesn't execute them).
2. Or add a runnable smoke that emits to a temp dir and asserts the
   expected output file lands.

Until the indentation fix lands, this directory ships in the `broken`
state and the audit calls it out explicitly.
