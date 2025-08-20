# Tessera Runtime Mock — Build & Run

This directory contains a **mock** implementation of the Tessera Runtime & ABI
to allow building and linking host applications before the real backend exists.

## Files
- `tessera_runtime.h` — public header
- `tessera_runtime_mock.c` — minimal shared library (`libtessera`) returning success
- `demo.c` — tiny end-to-end example using the API
- `CMakeLists.txt` — build script

## Build
```bash
mkdir build && cd build
cmake ..
cmake --build . -j
```

## Run
```bash
./tessera_demo
# Output:
# tessera_demo: OK
```

## Python Integration
The provided `tessera_runtime.py` looks for `libtessera.so`/`.dylib`/`.dll`.
If you set:
```bash
export TESSERA_RUNTIME_LIB=./build/libtessera.so
```
the Python stub will call into this mock library.
