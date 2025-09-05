
# Contributing

1. Add a new benchmark under `benches/<tier>/`:
   - Provide either a C++ binary (print a single JSON line) or a Python module that prints JSON.
2. Register it in a YAML config under `configs/` with `id`, `runner`, `path`, and `args`.
3. Ensure the output follows the **Metric Definitions** schema.
4. Add a short docstring at the top of the file explaining purpose and metrics.
5. Include skip behavior if the capability isn't available.

CI hooks and examples will be expanded over time.
