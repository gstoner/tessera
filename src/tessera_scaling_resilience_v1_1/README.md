# Tessera Scaling & Resilience Extensions (v1.1)

**What’s new vs v1.0:**
- Passes wired for inclusion in **existing `tessera-opt`** (no extra binary).
- **ODS expanded** with custom parsers/printers and **C++ verifiers**.
- **Deployment Manifest Exporter** walks Mesh/Collectives-like IR and writes a concrete JSON schema.

## Integrate into `tessera-opt`

1. Add this to your `tessera-opt` registration TU (or create the provided file):
```
#include "tessera/sr/Passes.h"
#include "tessera/sr/Dialect.h"

int register_tessera_sr() {
  mlir::tessera::sr::registerDialect();
  mlir::tessera::sr::registerPasses();
  return 0;
}
static int _ = register_tessera_sr();
```

2. Link the SR objects in your `tessera-opt` CMake:
```
# add_subdirectory to this package or add sources directly
add_subdirectory(external/tessera_scaling_resilience_v1_1)

target_link_libraries(tessera-opt PRIVATE TesseraSR DialectSR)
```

3. Run passes within your normal pipelines:
```
tessera-opt -tessera-optimizer-shard -tessera-insert-recompute             -tessera-resilience-restart -tessera-export-deployment-manifest input.mlir
```

## New DIrectory Layout
```
include/tessera/sr/        # Public headers
lib/sr/dialect/            # ODS-generated + C++ implementations
lib/sr/passes/             # Pass implementations
tests/sr/                  # FileCheck tests
docs/                      # Spec parts with MERGE markers
CMakeLists.txt             # Adds DialectSR & TesseraSR libs
```