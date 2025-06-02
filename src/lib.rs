/*!
# OpenAlgebra: High-Performance Sparse Linear Algebra Library

OpenAlgebra is a high-performance sparse linear algebra library written in Rust and C++. 
It provides efficient sparse matrix operations, iterative solvers, and tensor computations 
with optional GPU acceleration.

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

## Quick Start

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*/

pub mod sparse;
pub mod solvers;
pub mod tensor;
pub mod preconditioners;
pub mod utils;
pub mod api;
pub mod agents;

#[cfg(feature = "gpu-acceleration")]
pub mod cuda;

#[cfg(feature = "mpi")]
pub mod distributed;

pub use sparse::{SparseMatrix, COOMatrix, CSRMatrix, CSCMatrix};
pub use solvers::{IterativeSolver, ConjugateGradient, GMRES, BiCGSTAB};
pub use tensor::{SparseTensor, DenseTensor};

/// Main result type for the library
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Error types for OpenAlgebra
#[derive(Debug, thiserror::Error)]
pub enum OpenAlgebraError {
    #[error("Matrix dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },
    
    #[error("Solver convergence failed after {iterations} iterations")]
    ConvergenceFailure { iterations: usize },
    
    #[error("Invalid matrix format: {0}")]
    InvalidFormat(String),
    
    #[error("GPU operation failed: {0}")]
    #[cfg(feature = "gpu-acceleration")]
    GpuError(String),
    
    #[error("MPI operation failed: {0}")]
    #[cfg(feature = "mpi")]
    MpiError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Initialize the OpenAlgebra library
pub fn init() -> Result<()> {
    #[cfg(feature = "gpu-acceleration")]
    {
        cuda::init_cuda()?;
    }
    
    #[cfg(feature = "mpi")]
    {
        distributed::init_mpi()?;
    }
    
    println!("OpenAlgebra v{} initialized", VERSION);
    Ok(())
}

/// Performance configuration
pub struct Config {
    pub num_threads: Option<usize>,
    pub gpu_enabled: bool,
    pub memory_limit: Option<usize>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_threads: None, // Use all available cores
            gpu_enabled: cfg!(feature = "gpu-acceleration"),
            memory_limit: None, // No limit
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_init() {
        assert!(init().is_ok());
    }

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
        assert_eq!(VERSION, "1.0.0");
    }
    
    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.gpu_enabled, cfg!(feature = "gpu-acceleration"));
    }
} 