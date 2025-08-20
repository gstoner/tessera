# tessera-runtime (Python)

Thin ctypes binding (with mock fallback) for the Tessera Runtime & ABI.
Matches the API in `Tessera_Runtime_ABI_Spec.md` and the C header `tessera_runtime.h`.

## Install (editable)
```bash
pip install -e .
```

## Build wheel
```bash
pip wheel . -w dist
```

## Using a real or mock shared library
The stub looks for a shared library in this order:
- `TESSERA_RUNTIME_LIB` environment variable
- `libtessera.so` / `libtessera.dylib` / `tessera.dll` on the default loader path

To point at the mock you built with CMake:
```bash
export TESSERA_RUNTIME_LIB=/path/to/build/libtessera.so
python -c "import tessera_runtime as ts; print(ts.get_version())"
```
