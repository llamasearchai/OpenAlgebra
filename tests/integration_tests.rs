/*!
# OpenAlgebra Integration Tests

Comprehensive integration tests for the OpenAlgebra library.
Tests the interaction between different components and real-world scenarios.
*/

use openalgebra::{
    sparse::{COOMatrix, CSRMatrix, SparseMatrix},
    solvers::{ConjugateGradient, GMRES, BiCGSTAB, IterativeSolver},
    tensor::{SparseTensor, DenseTensor, Tensor},
    preconditioners::{ILUPreconditioner, JacobiPreconditioner, Preconditioner},
    init, Config,
};
use approx::assert_relative_eq;
use std::collections::HashMap;

#[test]
fn test_library_initialization() {
    assert!(init().is_ok());
}

#[test]
fn test_sparse_matrix_workflow() {
    // Create a 5x5 tridiagonal matrix
    let mut coo = COOMatrix::<f64>::new(5, 5);
    
    // Fill tridiagonal structure: -1, 2, -1
    for i in 0..5 {
        coo.insert(i, i, 2.0);
        if i > 0 {
            coo.insert(i, i - 1, -1.0);
        }
        if i < 4 {
            coo.insert(i, i + 1, -1.0);
        }
    }
    
    // Convert to CSR
    let csr = coo.to_csr();
    assert_eq!(csr.rows(), 5);
    assert_eq!(csr.cols(), 5);
    assert_eq!(csr.nnz(), 13); // 5 diagonal + 4 upper + 4 lower
    
    // Test matrix-vector multiplication
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = csr.matvec(&x);
    
    // Expected result for tridiagonal matrix
    let expected = vec![0.0, 0.0, 0.0, 0.0, 0.0]; // Ax = 0 for this specific case
    for i in 0..5 {
        assert_relative_eq!(y[i], expected[i], epsilon = 1e-10);
    }
}

#[test]
fn test_solver_convergence() {
    // Create a symmetric positive definite matrix (suitable for CG)
    let mut coo = COOMatrix::<f64>::new(10, 10);
    
    // Create a diagonally dominant matrix
    for i in 0..10 {
        coo.insert(i, i, 10.0); // Strong diagonal
        if i > 0 {
            coo.insert(i, i - 1, -1.0);
        }
        if i < 9 {
            coo.insert(i, i + 1, -1.0);
        }
    }
    
    let matrix = coo.to_csr();
    let b = vec![1.0; 10]; // Right-hand side
    let mut x = vec![0.0; 10]; // Initial guess
    
    // Test Conjugate Gradient
    let cg_solver = ConjugateGradient::new();
    let info = cg_solver.solve(&matrix, &b, &mut x);
    
    assert!(info.converged);
    assert!(info.iterations < 100);
    assert!(info.residual_norm < 1e-6);
    
    // Verify solution by computing residual
    let residual = matrix.matvec(&x);
    for i in 0..10 {
        assert_relative_eq!(residual[i], b[i], epsilon = 1e-5);
    }
}

#[test]
fn test_preconditioned_solver() {
    // Create a test matrix
    let mut coo = COOMatrix::<f64>::new(20, 20);
    
    for i in 0..20 {
        coo.insert(i, i, 4.0);
        if i > 0 {
            coo.insert(i, i - 1, -1.0);
        }
        if i < 19 {
            coo.insert(i, i + 1, -1.0);
        }
        if i > 1 {
            coo.insert(i, i - 2, -0.5);
        }
        if i < 18 {
            coo.insert(i, i + 2, -0.5);
        }
    }
    
    let matrix = coo.to_csr();
    let b = vec![1.0; 20];
    
    // Test without preconditioner
    let mut x1 = vec![0.0; 20];
    let solver = ConjugateGradient::new();
    let info1 = solver.solve(&matrix, &b, &mut x1);
    
    // Test with Jacobi preconditioner
    let precond = JacobiPreconditioner::new(&matrix);
    let mut x2 = vec![0.0; 20];
    let solver_precond = ConjugateGradient::with_preconditioner(Box::new(precond));
    let info2 = solver_precond.solve(&matrix, &b, &mut x2);
    
    // Preconditioned solver should converge faster
    assert!(info2.iterations <= info1.iterations);
    assert!(info1.converged && info2.converged);
    
    // Both should give similar solutions
    for i in 0..20 {
        assert_relative_eq!(x1[i], x2[i], epsilon = 1e-8);
    }
}

#[test]
fn test_multiple_solvers_consistency() {
    // Create a non-symmetric matrix for testing different solvers
    let mut coo = COOMatrix::<f64>::new(15, 15);
    
    for i in 0..15 {
        coo.insert(i, i, 5.0);
        if i > 0 {
            coo.insert(i, i - 1, -2.0);
        }
        if i < 14 {
            coo.insert(i, i + 1, -1.0);
        }
    }
    
    let matrix = coo.to_csr();
    let b = vec![1.0; 15];
    
    // Solve with GMRES
    let mut x_gmres = vec![0.0; 15];
    let gmres_solver = GMRES::new(10);
    let info_gmres = gmres_solver.solve(&matrix, &b, &mut x_gmres);
    
    // Solve with BiCGSTAB
    let mut x_bicgstab = vec![0.0; 15];
    let bicgstab_solver = BiCGSTAB::new();
    let info_bicgstab = bicgstab_solver.solve(&matrix, &b, &mut x_bicgstab);
    
    assert!(info_gmres.converged);
    assert!(info_bicgstab.converged);
    
    // Solutions should be similar
    for i in 0..15 {
        assert_relative_eq!(x_gmres[i], x_bicgstab[i], epsilon = 1e-6);
    }
}

#[test]
fn test_tensor_operations() {
    let shape = vec![3, 4, 5];
    
    // Create sparse tensor
    let mut sparse_tensor = SparseTensor::<f64>::new(shape.clone());
    
    // Add some values
    sparse_tensor.set(&[0, 1, 2], 1.5);
    sparse_tensor.set(&[1, 2, 3], 2.5);
    sparse_tensor.set(&[2, 0, 1], 3.5);
    
    assert_eq!(sparse_tensor.nnz(), 3);
    assert_eq!(sparse_tensor.shape(), &shape);
    
    // Test retrieval
    assert_relative_eq!(sparse_tensor.get(&[0, 1, 2]), 1.5, epsilon = 1e-10);
    assert_relative_eq!(sparse_tensor.get(&[1, 2, 3]), 2.5, epsilon = 1e-10);
    assert_relative_eq!(sparse_tensor.get(&[2, 0, 1]), 3.5, epsilon = 1e-10);
    assert_relative_eq!(sparse_tensor.get(&[0, 0, 0]), 0.0, epsilon = 1e-10);
    
    // Convert to dense
    let dense_tensor = sparse_tensor.to_dense();
    assert_eq!(dense_tensor.shape(), &shape);
    
    // Verify values in dense tensor
    assert_relative_eq!(dense_tensor.get(&[0, 1, 2]), 1.5, epsilon = 1e-10);
    assert_relative_eq!(dense_tensor.get(&[1, 2, 3]), 2.5, epsilon = 1e-10);
    assert_relative_eq!(dense_tensor.get(&[2, 0, 1]), 3.5, epsilon = 1e-10);
}

#[test]
fn test_large_scale_problem() {
    let size = 100;
    
    // Create a larger test problem
    let mut coo = COOMatrix::<f64>::new(size, size);
    
    // Create a 2D Laplacian-like matrix
    for i in 0..size {
        coo.insert(i, i, 4.0);
        if i > 0 {
            coo.insert(i, i - 1, -1.0);
        }
        if i < size - 1 {
            coo.insert(i, i + 1, -1.0);
        }
        if i >= 10 {
            coo.insert(i, i - 10, -1.0);
        }
        if i < size - 10 {
            coo.insert(i, i + 10, -1.0);
        }
    }
    
    let matrix = coo.to_csr();
    let b = vec![1.0; size];
    let mut x = vec![0.0; size];
    
    // Use ILU preconditioned CG
    let ilu_precond = ILUPreconditioner::new(&matrix, 0, 1e-8);
    let solver = ConjugateGradient::with_preconditioner(Box::new(ilu_precond));
    let info = solver.solve(&matrix, &b, &mut x);
    
    assert!(info.converged);
    assert!(info.residual_norm < 1e-6);
    
    // Verify solution quality
    let residual = matrix.matvec(&x);
    let mut residual_norm = 0.0;
    for i in 0..size {
        residual_norm += (residual[i] - b[i]).powi(2);
    }
    residual_norm = residual_norm.sqrt();
    
    assert!(residual_norm < 1e-5);
}

#[test]
fn test_matrix_properties() {
    let mut coo = COOMatrix::<f64>::new(4, 4);
    
    // Create a symmetric matrix
    coo.insert(0, 0, 2.0);
    coo.insert(0, 1, 1.0);
    coo.insert(1, 0, 1.0);
    coo.insert(1, 1, 3.0);
    coo.insert(1, 2, 0.5);
    coo.insert(2, 1, 0.5);
    coo.insert(2, 2, 2.5);
    coo.insert(2, 3, 1.5);
    coo.insert(3, 2, 1.5);
    coo.insert(3, 3, 4.0);
    
    let matrix = coo.to_csr();
    
    // Test transpose
    let matrix_t = matrix.transpose();
    assert_eq!(matrix_t.rows(), matrix.cols());
    assert_eq!(matrix_t.cols(), matrix.rows());
    
    // For symmetric matrix, A = A^T
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y1 = matrix.matvec(&x);
    let y2 = matrix_t.matvec(&x);
    
    for i in 0..4 {
        assert_relative_eq!(y1[i], y2[i], epsilon = 1e-10);
    }
}

#[test]
fn test_error_handling() {
    // Test dimension mismatch
    let matrix = COOMatrix::<f64>::new(5, 5).to_csr();
    let b = vec![1.0; 3]; // Wrong size
    let mut x = vec![0.0; 5];
    
    let solver = ConjugateGradient::new();
    let result = solver.solve(&matrix, &b, &mut x);
    
    // Should handle dimension mismatch gracefully
    assert!(!result.converged || result.iterations == 0);
}

#[test]
fn test_memory_efficiency() {
    // Test that sparse matrices use memory efficiently
    let size = 1000;
    let mut coo = COOMatrix::<f64>::new(size, size);
    
    // Add only diagonal elements (very sparse)
    for i in 0..size {
        coo.insert(i, i, 1.0);
    }
    
    let csr = coo.to_csr();
    
    // CSR should only store non-zero elements
    assert_eq!(csr.nnz(), size);
    assert_eq!(csr.values.len(), size);
    assert_eq!(csr.col_indices.len(), size);
    assert_eq!(csr.row_ptr.len(), size + 1);
}

#[test]
fn test_parallel_operations() {
    use rayon::prelude::*;
    
    let size = 500;
    let mut coo = COOMatrix::<f64>::new(size, size);
    
    // Create matrix in parallel
    (0..size).into_par_iter().for_each(|i| {
        // Note: This is just testing the concept; actual parallel insertion
        // would need synchronization
    });
    
    // Fill matrix sequentially for this test
    for i in 0..size {
        coo.insert(i, i, 2.0);
        if i > 0 {
            coo.insert(i, i - 1, -1.0);
        }
        if i < size - 1 {
            coo.insert(i, i + 1, -1.0);
        }
    }
    
    let matrix = csr.to_csr();
    let x = vec![1.0; size];
    
    // Test parallel matrix-vector multiplication
    let y: Vec<f64> = (0..size)
        .into_par_iter()
        .map(|i| {
            let mut sum = 0.0;
            for j in matrix.row_ptr[i]..matrix.row_ptr[i + 1] {
                sum += matrix.values[j] * x[matrix.col_indices[j]];
            }
            sum
        })
        .collect();
    
    // Compare with sequential version
    let y_seq = matrix.matvec(&x);
    
    for i in 0..size {
        assert_relative_eq!(y[i], y_seq[i], epsilon = 1e-10);
    }
}

#[test]
fn test_configuration() {
    let config = Config::default();
    assert_eq!(config.gpu_enabled, cfg!(feature = "gpu-acceleration"));
    
    let custom_config = Config {
        num_threads: Some(4),
        gpu_enabled: false,
        memory_limit: Some(1024 * 1024 * 1024), // 1GB
    };
    
    assert_eq!(custom_config.num_threads, Some(4));
    assert!(!custom_config.gpu_enabled);
    assert_eq!(custom_config.memory_limit, Some(1024 * 1024 * 1024));
}

#[test]
fn test_real_world_scenario() {
    // Simulate a finite difference discretization of a 2D Poisson equation
    let n = 10; // Grid size (n x n)
    let size = n * n;
    
    let mut coo = COOMatrix::<f64>::new(size, size);
    
    // Fill the matrix for 2D finite differences
    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            
            // Central point
            coo.insert(idx, idx, 4.0);
            
            // Neighbors
            if i > 0 {
                coo.insert(idx, (i - 1) * n + j, -1.0);
            }
            if i < n - 1 {
                coo.insert(idx, (i + 1) * n + j, -1.0);
            }
            if j > 0 {
                coo.insert(idx, i * n + (j - 1), -1.0);
            }
            if j < n - 1 {
                coo.insert(idx, i * n + (j + 1), -1.0);
            }
        }
    }
    
    let matrix = coo.to_csr();
    let b = vec![1.0; size]; // Right-hand side
    let mut x = vec![0.0; size]; // Solution vector
    
    // Solve with preconditioned CG
    let jacobi_precond = JacobiPreconditioner::new(&matrix);
    let solver = ConjugateGradient::with_preconditioner(Box::new(jacobi_precond));
    let info = solver.solve(&matrix, &b, &mut x);
    
    assert!(info.converged);
    assert!(info.residual_norm < 1e-6);
    
    // Verify the solution makes physical sense (no NaN or infinite values)
    for &val in &x {
        assert!(val.is_finite());
        assert!(!val.is_nan());
    }
} 