#!/bin/bash
# Tessera test script

set -e

echo "🧪 Running Tessera tests..."

# Python tests
echo "Running Python tests..."
"${PYTHON:-python3}" -m pytest tests/unit -v

if [ "${TESSERA_RUN_PERFORMANCE_TESTS:-0}" = "1" ]; then
    echo "Running performance tests..."
    "${PYTHON:-python3}" -m pytest tests/performance -v
fi

# C++ tests (if build directory exists)
if [ -d "build" ]; then
    echo "Running C++ tests..."
    cd build
    ctest --parallel $(sysctl -n hw.ncpu)
    cd ..
fi

echo "✅ All tests passed!"
