#!/bin/bash
# Tessera build script

set -e

echo "🔨 Building Tessera..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. \
    -DTESSERA_ENABLE_CUDA=ON \
    -DTESSERA_BUILD_TESTS=ON \
    -DTESSERA_BUILD_EXAMPLES=ON \
    -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(sysctl -n hw.ncpu)

echo "✅ Build completed successfully!"
