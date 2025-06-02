# OpenAlgebra: High-Performance Sparse Linear Algebra Library

[![Build Status](https://github.com/llamasearchai/openalgebra/workflows/CI/badge.svg)](https://github.com/llamasearchai/openalgebra/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OpenAlgebra is a high-performance sparse linear algebra library written in Rust and C++. It provides efficient sparse matrix operations, iterative solvers, and tensor computations with optional GPU acceleration.

## Features

### Core Linear Algebra
- **Sparse Matrix Operations**: COO, CSR, and CSC matrix formats
- **Iterative Solvers**: Conjugate Gradient, GMRES, BiCGSTAB
- **Direct Solvers**: Multifrontal and supernodal factorization
- **Preconditioners**: AMG, ILU, and custom preconditioners
- **Tensor Operations**: Sparse tensor computations and manipulations

### Performance Optimization
- **GPU Acceleration**: CUDA support for accelerated computations
- **Parallel Processing**: OpenMP and MPI support for distributed computing
- **Memory Efficiency**: Optimized memory layouts for sparse data structures
- **Hardware Adaptation**: CPU-specific optimizations and vectorization

### Programming Interface
- **Rust Library**: Native Rust API with zero-cost abstractions
- **C++ Interface**: Full C++ API for integration with existing codebases
- **Python Bindings**: Python interface for rapid prototyping

## Quick Start

### Prerequisites

```bash
# System requirements
- Ubuntu 20.04+ / CentOS 8+ / macOS 11+
- CUDA 11.8+ (optional, for GPU acceleration)
- CMake 3.18+
- GCC 9+ / Clang 10+
- Rust 1.70+
```

### Build from Source

```bash
# Clone repository
git clone https://github.com/llamasearchai/OpenAlgebra.git
cd OpenAlgebra

# Build Rust library
cargo build --release

# Build C++ library
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Basic Usage (Rust)

```rust
use openalgebra::sparse::{SparseMatrix, COOMatrix};
use openalgebra::solvers::IterativeSolver;

fn main() {
    // Create a sparse matrix in COO format
    let mut matrix = COOMatrix::<f64>::new(1000, 1000);
    
    // Add non-zero elements
    for i in 0..1000 {
        matrix.insert(i, i, 2.0);
        if i > 0 {
            matrix.insert(i, i-1, -1.0);
        }
        if i < 999 {
            matrix.insert(i, i+1, -1.0);
        }
    }
    
    // Convert to CSR format for efficient operations
    let csr_matrix = matrix.to_csr();
    
    // Solve linear system Ax = b
    let b = vec![1.0; 1000];
    let mut x = vec![0.0; 1000];
    
    let solver = openalgebra::solvers::ConjugateGradient::new();
    solver.solve(&csr_matrix, &b, &mut x);
    
    println!("Solution computed successfully");
}
```

### Basic Usage (C++)

```cpp
#include <openalgebra/sparse_matrix.hpp>
#include <openalgebra/solvers.hpp>

int main() {
    // Create a sparse matrix
    openalgebra::SparseMatrix<double> matrix(1000, 1000);
    
    // Build a tridiagonal matrix
    for (int i = 0; i < 1000; ++i) {
        matrix.insert(i, i, 2.0);
        if (i > 0) matrix.insert(i, i-1, -1.0);
        if (i < 999) matrix.insert(i, i+1, -1.0);
    }
    
    // Solve linear system
    std::vector<double> b(1000, 1.0);
    std::vector<double> x(1000, 0.0);
    
    openalgebra::ConjugateGradient<double> solver;
    solver.solve(matrix, b, x);
    
    std::cout << "Solution computed successfully" << std::endl;
    return 0;
}
```

## Installation

### Rust Package

```bash
# Add to Cargo.toml
[dependencies]
openalgebra = "1.0.0"

# Install from source
cargo install --path .
```

### C++ Library

```bash
# Configure build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DOPENALGEBRA_ENABLE_CUDA=ON \
  -DOPENALGEBRA_ENABLE_MPI=ON

# Build and install
make -j$(nproc) && sudo make install
```

### Python Package

```bash
# Install with pip
pip install openalgebra

# Build from source
pip install .
```

## Documentation

- **API Reference**: Generated documentation available after build
- **Examples**: See `examples/` directory for usage examples
- **Build Instructions**: Detailed build instructions in `docs/`

## Testing

```bash
# Run Rust tests
cargo test

# Run C++ tests
cd build && ctest

# Run Python tests
pytest tests/
```

## Performance

OpenAlgebra is optimized for high-performance computing:

- **Memory Efficiency**: Optimized sparse data structures minimize memory usage
- **Vectorization**: Automatic vectorization for modern CPU architectures
- **GPU Acceleration**: CUDA kernels for supported operations
- **Parallel Algorithms**: Multi-threaded implementations of core algorithms

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines.

### Development Setup

```bash
# Set up development environment
git clone https://github.com/llamasearchai/OpenAlgebra.git
cd OpenAlgebra

# Install development dependencies
cargo build
mkdir build && cd build && cmake .. && make

# Run tests
cargo test
ctest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For technical support and questions:
- **Issues**: [GitHub Issues](https://github.com/llamasearchai/OpenAlgebra/issues)
- **Discussions**: [GitHub Discussions](https://github.com/llamasearchai/OpenAlgebra/discussions)

---

**OpenAlgebra** - High-Performance Sparse Linear Algebra in Rust and C++ 