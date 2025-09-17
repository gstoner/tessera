# Tessera Scaling & Resilience Extensions (v1)

This drop-in package adds **scaling-aware** and **fault-tolerant** capabilities to Tessera:
- Optimizer state sharding (ZeRO-style) as first-class IR semantics
- Recomputation / checkpoint scheduling in Schedule IR
- Resilience primitives (restart, replay, and health events)
- System-aware deployment manifest export (parallel strategy, sharding maps)
- Pass pipeline alias: `-tessera-sr-pipeline`

## Layout
```
.
├── CMakeLists.txt
├── cmake/
├── docs/
├── include/tessera/
├── ir/
├── lib/passes/
├── tests/
└── tools/tessera-opt-sr/
```
## Quick build (example)
```bash
mkdir -p build && cd build
cmake .. -DTESSERA_WITH_SR=ON
cmake --build . -j
# Run tests (requires llvm-lit in PATH)
llvm-lit ../tests -v
```