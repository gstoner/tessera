# Tessera Project Structure

This document outlines the organization of the Tessera project.

## Directory Structure

```
tessera/
├── README.md                    # Project overview and setup
├── LICENSE                      # Apache 2.0 license
├── CONTRIBUTING.md              # Contribution guidelines
├── CMakeLists.txt              # Build configuration
├── pyproject.toml              # Python project metadata
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore patterns
│
├── docs/                       # Documentation
├── src/                        # Source code (MLIR dialects, runtime)
├── python/                     # Python frontend package
├── examples/                   # Usage examples and tutorials
├── tests/                      # Test suite
├── benchmarks/                 # Performance benchmarks
├── tools/                      # Development tools
├── scripts/                    # Build and utility scripts
└── cmake/                      # CMake modules
```

## Key Components

### Source Code (`src/`)
- **MLIR Dialects**: Graph IR, Schedule IR, Target IR
- **Runtime**: C++ execution engine and CUDA kernels
- **Compiler**: Optimization passes and code generation

### Python Package (`python/tessera/`)
- **Core**: Tensor, Module, and fundamental abstractions
- **NN**: Neural network layers and operations
- **Compiler**: Python to MLIR compilation
- **Runtime**: Python interface to C++ runtime

### Documentation (`docs/`)
- **Architecture**: System design and implementation
- **API**: Complete API reference for all languages
- **Tutorials**: Step-by-step learning materials

This structure supports both research and production use cases while maintaining clear separation of concerns.
