# Tessera Packaging Artifacts

## pkg-config
- `tessera.pc` — use during development:
  ```bash
  export PKG_CONFIG_PATH=$(pwd)
  pkg-config --cflags --libs tessera
  ```

## Python package
A wheel-friendly package skeleton is under `tessera_runtime_py/`.

### Build & install
```bash
cd tessera_runtime_py
pip install -e .              # editable
# Or build a wheel
pip wheel . -w dist
pip install dist/tessera_runtime-0.1.0-py3-none-any.whl
```
