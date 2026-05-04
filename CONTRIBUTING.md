# Contributing to Tessera

Thank you for your interest in contributing to Tessera!

## Development Setup

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (optional)
- LLVM/MLIR 21+ (Homebrew `llvm@21` on macOS)
- CMake 3.20+

### Setup Instructions
```bash
# Clone the repository
git clone https://github.com/tessera-ai/tessera.git
cd tessera

# Install dependencies
pip install -r requirements.txt

# Configure the C++/MLIR build with Homebrew LLVM/MLIR 21 on macOS
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/llvm@21 \
  -DLLVM_DIR=/opt/homebrew/opt/llvm@21/lib/cmake/llvm \
  -DMLIR_DIR=/opt/homebrew/opt/llvm@21/lib/cmake/mlir

# Build the project
cmake --build build
```

## Coding Standards

### Python Style
- Follow PEP 8
- Use type hints for all public APIs
- Write comprehensive docstrings

### C++ Style  
- Follow LLVM coding standards
- Use descriptive variable names
- Include comprehensive comments

## Testing

All contributions must include tests:

```bash
# Run Python tests
python -m pytest tests/

# Run C++ tests
cd build && ctest
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors.

## Getting Help

- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: General questions
- Discord: Real-time community chat

Thank you for contributing! 🚀
