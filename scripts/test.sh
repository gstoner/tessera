#!/bin/bash
# Tessera test script

set -e

echo "🧪 Running Tessera tests..."

# Python tests
echo "Running Python tests..."
"${PYTHON:-python3}" -m pytest tests/unit -v

# C++ tests (if build directory exists)
if [ -d "build" ]; then
    echo "Running C++ tests..."
    cd build
    ctest --parallel $(sysctl -n hw.ncpu)
    cd ..
fi

echo "✅ All tests passed!"
