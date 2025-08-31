# Apply Instructions

Copy `tessera/runtime` into your repo:

```
<repo-root>/tessera/runtime/...
```

Top-level CMake:
```cmake
add_subdirectory(tessera/runtime)
```

Build standalone:
```bash
mkdir -p build && cd build
cmake -S ../tessera/runtime -B . -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
ctest --output-on-failure
```

Optional stubs:
```bash
cmake -S ../tessera/runtime -B . -DTESSERA_ENABLE_CUDA=ON -DTESSERA_ENABLE_HIP=ON
```
