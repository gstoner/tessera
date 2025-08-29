#!/bin/bash
# Code formatting script

echo "🎨 Formatting code..."

# Format Python code
black python/ tests/ examples/
isort python/ tests/ examples/

# Format C++ code (if clang-format is available)
if command -v clang-format >/dev/null 2>&1; then
    find src/ -name "*.cpp" -o -name "*.h" | xargs clang-format -i
fi

echo "✅ Code formatting completed!"
