/*!
# Comprehensive Integration Tests for OpenAlgebra

This test suite validates the complete functionality of the OpenAlgebra library,
including sparse matrix operations, iterative solvers, tensor computations,
API endpoints, and AI agent integration.
*/

use openalgebra::{
    sparse::{COOMatrix, CSRMatrix, CSCMatrix, SparseMatrix},
    solvers::{ConjugateGradient, GMRES, BiCGSTAB, IterativeSolver, SolverInfo},
    tensor::{SparseTensor, DenseTensor, Tensor},
    preconditioners::{ILUPreconditioner, JacobiPreconditioner, Preconditioner},
    utils::{Timer, Profiler, MemoryInfo, Logger, LogLevel},
    api::{AppState, CreateMatrixRequest, MatrixEntry, SolveRequest},
    agents::{MathAgent, AgentConfig, MathProblem, ProblemType},
    Result,
};
use axum_test::TestServer;
use serde_json::json;
use std::collections::HashMap;
use tokio_test;

/// Test sparse matrix creation and basic operations
#[tokio::test]
async fn test_sparse_matrix_operations() -> Result<()> {
    // Test COO matrix creation
    let mut coo = COOMatrix::<f64>::new(5, 5);
    
    // Create a simple tridiagonal matrix
    for i in 0..5 {
        coo.insert(i, i, 2.0);
        if i > 0 {
            coo.insert(i, i-1, -1.0);
        }
        if i < 4 {
            coo.insert(i, i+1, -1.0);
        }
    }
    
    assert_eq!(coo.nnz(), 13);
    assert_eq!(coo.rows(), 5);
    assert_eq!(coo.cols(), 5);
    
    // Test conversion to CSR
    let csr = coo.to_csr();
    assert_eq!(csr.nnz(), 13);
    assert_eq!(csr.rows(), 5);
    assert_eq!(csr.cols(), 5);
    
    // Test matrix-vector multiplication
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = csr.matvec(&x)?;
    
    // Expected result for tridiagonal matrix
    let expected = vec![1.0, 0.0, 1.0, 2.0, 3.0];
    for (i, (&actual, &exp)) in y.iter().zip(expected.iter()).enumerate() {
        assert!((actual - exp).abs() < 1e-10, 
                "Mismatch at index {}: {} != {}", i, actual, exp);
    }
    
    // Test transpose
    let csr_t = csr.transpose();
    assert_eq!(csr_t.rows(), 5);
    assert_eq!(csr_t.cols(), 5);
    assert_eq!(csr_t.nnz(), 13);
    
    // Test CSC conversion
    let csc = csr.to_csc();
    assert_eq!(csc.nnz(), 13);
    assert_eq!(csc.rows(), 5);
    assert_eq!(csc.cols(), 5);
    
    Ok(())
}

/// Test iterative solvers with different matrix types
#[tokio::test]
async fn test_iterative_solvers() -> Result<()> {
    // Create a symmetric positive definite matrix (5-point stencil)
    let mut coo = COOMatrix::<f64>::new(9, 9);
    
    // 3x3 grid, 5-point stencil
    for i in 0..3 {
        for j in 0..3 {
            let idx = i * 3 + j;
            coo.insert(idx, idx, 4.0); // Center
            
            if i > 0 { coo.insert(idx, (i-1)*3 + j, -1.0); } // North
            if i < 2 { coo.insert(idx, (i+1)*3 + j, -1.0); } // South
            if j > 0 { coo.insert(idx, i*3 + (j-1), -1.0); } // West
            if j < 2 { coo.insert(idx, i*3 + (j+1), -1.0); } // East
        }
    }
    
    let matrix = coo.to_csr();
    let b = vec![1.0; 9];
    let mut x = vec![0.0; 9];
    
    // Test Conjugate Gradient solver
    let cg_solver = ConjugateGradient::new();
    let cg_info = cg_solver.solve(&matrix, &b, &mut x, 1e-8, 1000)?;
    
    assert!(cg_info.converged, "CG solver did not converge");
    assert!(cg_info.iterations < 100, "CG took too many iterations: {}", cg_info.iterations);
    assert!(cg_info.residual_norm < 1e-8, "CG residual too large: {}", cg_info.residual_norm);
    
    // Verify solution by computing residual
    let ax = matrix.matvec(&x)?;
    let residual: f64 = ax.iter().zip(b.iter())
        .map(|(&ax_i, &b_i)| (ax_i - b_i).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(residual < 1e-6, "Solution residual too large: {}", residual);
    
    // Test GMRES solver
    x.fill(0.0);
    let gmres_solver = GMRES::new(30); // restart = 30
    let gmres_info = gmres_solver.solve(&matrix, &b, &mut x, 1e-8, 1000)?;
    
    assert!(gmres_info.converged, "GMRES solver did not converge");
    assert!(gmres_info.residual_norm < 1e-8, "GMRES residual too large: {}", gmres_info.residual_norm);
    
    // Test BiCGSTAB solver
    x.fill(0.0);
    let bicgstab_solver = BiCGSTAB::new();
    let bicgstab_info = bicgstab_solver.solve(&matrix, &b, &mut x, 1e-8, 1000)?;
    
    assert!(bicgstab_info.converged, "BiCGSTAB solver did not converge");
    assert!(bicgstab_info.residual_norm < 1e-8, "BiCGSTAB residual too large: {}", bicgstab_info.residual_norm);
    
    Ok(())
}

/// Test preconditioners with iterative solvers
#[tokio::test]
async fn test_preconditioned_solvers() -> Result<()> {
    // Create a larger matrix for preconditioning test
    let mut coo = COOMatrix::<f64>::new(25, 25);
    
    // 5x5 grid, 5-point stencil
    for i in 0..5 {
        for j in 0..5 {
            let idx = i * 5 + j;
            coo.insert(idx, idx, 4.0);
            
            if i > 0 { coo.insert(idx, (i-1)*5 + j, -1.0); }
            if i < 4 { coo.insert(idx, (i+1)*5 + j, -1.0); }
            if j > 0 { coo.insert(idx, i*5 + (j-1), -1.0); }
            if j < 4 { coo.insert(idx, i*5 + (j+1), -1.0); }
        }
    }
    
    let matrix = coo.to_csr();
    let b = vec![1.0; 25];
    
    // Test with Jacobi preconditioner
    let jacobi_precond = JacobiPreconditioner::new(&matrix)?;
    let mut x = vec![0.0; 25];
    
    let cg_solver = ConjugateGradient::new();
    let info_precond = cg_solver.solve_preconditioned(&matrix, &b, &mut x, &jacobi_precond, 1e-8, 1000)?;
    
    assert!(info_precond.converged, "Preconditioned CG did not converge");
    
    // Test with ILU preconditioner
    let ilu_precond = ILUPreconditioner::new(&matrix, 0, 1e-6)?; // level 0, drop tolerance 1e-6
    x.fill(0.0);
    
    let info_ilu = cg_solver.solve_preconditioned(&matrix, &b, &mut x, &ilu_precond, 1e-8, 1000)?;
    
    assert!(info_ilu.converged, "ILU preconditioned CG did not converge");
    
    // ILU should generally converge faster than Jacobi
    println!("Jacobi iterations: {}, ILU iterations: {}", 
             info_precond.iterations, info_ilu.iterations);
    
    Ok(())
}

/// Test tensor operations
#[tokio::test]
async fn test_tensor_operations() -> Result<()> {
    // Test sparse tensor creation
    let shape = vec![3, 3, 3];
    let mut sparse_tensor = SparseTensor::<f64>::new(shape.clone());
    
    // Add some entries
    sparse_tensor.set(&[0, 0, 0], 1.0)?;
    sparse_tensor.set(&[1, 1, 1], 2.0)?;
    sparse_tensor.set(&[2, 2, 2], 3.0)?;
    
    assert_eq!(sparse_tensor.nnz(), 3);
    assert_eq!(sparse_tensor.shape(), &shape);
    
    // Test dense tensor creation
    let data = vec![1.0; 27]; // 3x3x3 = 27 elements
    let dense_tensor = DenseTensor::new(shape.clone(), data)?;
    
    assert_eq!(dense_tensor.shape(), &shape);
    assert_eq!(dense_tensor.nnz(), 27);
    
    // Test tensor conversion
    let dense_from_sparse = sparse_tensor.to_dense();
    assert_eq!(dense_from_sparse.shape(), &shape);
    
    let sparse_from_dense = dense_tensor.to_sparse(1e-10);
    assert_eq!(sparse_from_dense.shape(), &shape);
    assert_eq!(sparse_from_dense.nnz(), 27); // All elements are 1.0, above threshold
    
    // Test tensor operations
    let mut result = SparseTensor::<f64>::new(shape.clone());
    sparse_tensor.add(&sparse_tensor, &mut result)?;
    
    assert_eq!(result.get(&[0, 0, 0])?, 2.0); // 1.0 + 1.0
    assert_eq!(result.get(&[1, 1, 1])?, 4.0); // 2.0 + 2.0
    assert_eq!(result.get(&[2, 2, 2])?, 6.0); // 3.0 + 3.0
    
    Ok(())
}

/// Test API endpoints
#[tokio::test]
async fn test_api_endpoints() -> Result<()> {
    let app_state = AppState::new();
    let app = openalgebra::api::create_router().with_state(app_state);
    let server = TestServer::new(app)?;
    
    // Test health endpoint
    let response = server.get("/health").await;
    assert_eq!(response.status_code(), 200);
    
    let health_data: serde_json::Value = response.json();
    assert_eq!(health_data["status"], "healthy");
    assert_eq!(health_data["version"], "1.0.0");
    
    // Test matrix creation
    let matrix_request = CreateMatrixRequest {
        name: "test_matrix".to_string(),
        format: "csr".to_string(),
        rows: 3,
        cols: 3,
        entries: vec![
            MatrixEntry { row: 0, col: 0, value: 2.0 },
            MatrixEntry { row: 0, col: 1, value: -1.0 },
            MatrixEntry { row: 1, col: 0, value: -1.0 },
            MatrixEntry { row: 1, col: 1, value: 2.0 },
            MatrixEntry { row: 1, col: 2, value: -1.0 },
            MatrixEntry { row: 2, col: 1, value: -1.0 },
            MatrixEntry { row: 2, col: 2, value: 2.0 },
        ],
    };
    
    let response = server
        .post("/matrices")
        .json(&matrix_request)
        .await;
    assert_eq!(response.status_code(), 201);
    
    // Test matrix info retrieval
    let response = server.get("/matrices/test_matrix").await;
    assert_eq!(response.status_code(), 200);
    
    let matrix_info: serde_json::Value = response.json();
    assert_eq!(matrix_info["name"], "test_matrix");
    assert_eq!(matrix_info["rows"], 3);
    assert_eq!(matrix_info["cols"], 3);
    assert_eq!(matrix_info["nnz"], 7);
    
    // Test linear system solving
    let solve_request = SolveRequest {
        matrix_name: "test_matrix".to_string(),
        b: vec![1.0, 1.0, 1.0],
        solver: "cg".to_string(),
        tolerance: Some(1e-8),
        max_iterations: Some(1000),
    };
    
    let response = server
        .post("/matrices/test_matrix/solve")
        .json(&solve_request)
        .await;
    assert_eq!(response.status_code(), 200);
    
    let solve_response: serde_json::Value = response.json();
    assert_eq!(solve_response["converged"], true);
    assert!(solve_response["iterations"].as_u64().unwrap() < 100);
    
    // Test matrix list
    let response = server.get("/matrices").await;
    assert_eq!(response.status_code(), 200);
    
    let matrices: serde_json::Value = response.json();
    assert!(matrices.as_array().unwrap().len() >= 1);
    
    Ok(())
}

/// Test AI agent integration (if OpenAI API key is available)
#[tokio::test]
async fn test_ai_agent_integration() -> Result<()> {
    // Skip test if no API key is available
    if std::env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping AI agent test - no OpenAI API key");
        return Ok(());
    }
    
    let config = AgentConfig::default();
    let agent = MathAgent::new(config)?;
    
    // Create a test problem
    let mut numerical_properties = HashMap::new();
    numerical_properties.insert("condition_number".to_string(), 100.0);
    numerical_properties.insert("sparsity".to_string(), 0.1);
    
    let problem = MathProblem {
        problem_type: ProblemType::LinearSystem,
        description: "Solve a sparse linear system from finite element discretization".to_string(),
        constraints: vec!["Symmetric positive definite matrix".to_string()],
        objectives: vec!["Minimize solve time".to_string(), "Ensure numerical stability".to_string()],
        matrix_size: Some((1000, 1000)),
        sparsity_pattern: Some("5-point stencil".to_string()),
        numerical_properties,
    };
    
    // Test problem analysis
    let strategy = agent.analyze_problem(&problem).await?;
    
    assert!(!strategy.solver_type.is_empty());
    assert!(!strategy.reasoning.is_empty());
    
    // The AI should recommend CG for SPD matrices
    assert!(strategy.solver_type.to_lowercase().contains("cg") || 
            strategy.solver_type.to_lowercase().contains("conjugate"));
    
    // Test code generation
    let code = agent.generate_solution_code(&problem).await?;
    assert!(!code.is_empty());
    assert!(code.contains("openalgebra"));
    
    Ok(())
}

/// Test performance profiling and monitoring
#[tokio::test]
async fn test_performance_monitoring() -> Result<()> {
    let mut profiler = Profiler::new();
    let memory_info = MemoryInfo::new();
    let logger = Logger::new(LogLevel::Info);
    
    // Test timer functionality
    let mut timer = Timer::new();
    timer.start();
    
    // Simulate some work
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    
    let elapsed = timer.elapsed();
    assert!(elapsed > 0.0);
    assert!(elapsed < 1.0); // Should be much less than 1 second
    
    // Test profiler
    profiler.start_section("matrix_creation");
    let mut coo = COOMatrix::<f64>::new(100, 100);
    for i in 0..100 {
        coo.insert(i, i, 1.0);
    }
    profiler.end_section("matrix_creation");
    
    profiler.start_section("matrix_conversion");
    let _csr = coo.to_csr();
    profiler.end_section("matrix_conversion");
    
    let report = profiler.generate_report();
    assert!(report.contains("matrix_creation"));
    assert!(report.contains("matrix_conversion"));
    
    // Test memory monitoring
    let usage = memory_info.current_usage();
    assert!(usage > 0);
    
    let peak = memory_info.peak_usage();
    assert!(peak >= usage);
    
    // Test logging
    logger.info("Test log message");
    logger.warn("Test warning message");
    logger.error("Test error message");
    
    Ok(())
}

/// Test error handling and edge cases
#[tokio::test]
async fn test_error_handling() -> Result<()> {
    // Test dimension mismatch in matrix operations
    let mut coo1 = COOMatrix::<f64>::new(3, 3);
    let mut coo2 = COOMatrix::<f64>::new(4, 4);
    
    coo1.insert(0, 0, 1.0);
    coo2.insert(0, 0, 1.0);
    
    let csr1 = coo1.to_csr();
    let csr2 = coo2.to_csr();
    
    // This should fail due to dimension mismatch
    let result = csr1.add(&csr2);
    assert!(result.is_err());
    
    // Test invalid matrix-vector multiplication
    let x = vec![1.0, 2.0]; // Wrong size
    let result = csr1.matvec(&x);
    assert!(result.is_err());
    
    // Test solver with singular matrix
    let mut singular_coo = COOMatrix::<f64>::new(3, 3);
    singular_coo.insert(0, 0, 1.0);
    singular_coo.insert(1, 1, 1.0);
    // Row 2 is all zeros, making matrix singular
    
    let singular_csr = singular_coo.to_csr();
    let b = vec![1.0, 1.0, 1.0];
    let mut x = vec![0.0, 0.0, 0.0];
    
    let cg_solver = ConjugateGradient::new();
    let result = cg_solver.solve(&singular_csr, &b, &mut x, 1e-8, 10);
    
    // Should either fail or not converge
    if let Ok(info) = result {
        assert!(!info.converged || info.residual_norm > 1e-6);
    }
    
    // Test tensor with invalid indices
    let tensor = SparseTensor::<f64>::new(vec![3, 3]);
    let result = tensor.get(&[5, 5]); // Out of bounds
    assert!(result.is_err());
    
    Ok(())
}

/// Test concurrent operations and thread safety
#[tokio::test]
async fn test_concurrent_operations() -> Result<()> {
    use std::sync::Arc;
    use tokio::task;
    
    // Create a shared matrix
    let mut coo = COOMatrix::<f64>::new(100, 100);
    for i in 0..100 {
        coo.insert(i, i, 2.0);
        if i > 0 {
            coo.insert(i, i-1, -1.0);
        }
        if i < 99 {
            coo.insert(i, i+1, -1.0);
        }
    }
    let matrix = Arc::new(coo.to_csr());
    
    // Spawn multiple tasks performing matrix-vector multiplication
    let mut handles = Vec::new();
    
    for i in 0..10 {
        let matrix_clone = Arc::clone(&matrix);
        let handle = task::spawn(async move {
            let x = vec![1.0; 100];
            let y = matrix_clone.matvec(&x).unwrap();
            (i, y[0]) // Return task id and first element
        });
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    let mut results = Vec::new();
    for handle in handles {
        let result = handle.await?;
        results.push(result);
    }
    
    // All results should be the same
    assert_eq!(results.len(), 10);
    let expected_value = results[0].1;
    for (_, value) in results {
        assert!((value - expected_value).abs() < 1e-10);
    }
    
    Ok(())
}

/// Test memory efficiency and large matrix handling
#[tokio::test]
async fn test_large_matrix_operations() -> Result<()> {
    // Create a larger sparse matrix (but not too large for CI)
    let size = 1000;
    let mut coo = COOMatrix::<f64>::new(size, size);
    
    // Create a banded matrix
    for i in 0..size {
        coo.insert(i, i, 4.0);
        if i > 0 {
            coo.insert(i, i-1, -1.0);
        }
        if i < size-1 {
            coo.insert(i, i+1, -1.0);
        }
        if i > 1 {
            coo.insert(i, i-2, -0.5);
        }
        if i < size-2 {
            coo.insert(i, i+2, -0.5);
        }
    }
    
    let matrix = coo.to_csr();
    assert_eq!(matrix.rows(), size);
    assert_eq!(matrix.cols(), size);
    
    // Test matrix-vector multiplication performance
    let x = vec![1.0; size];
    let start = std::time::Instant::now();
    let y = matrix.matvec(&x)?;
    let matvec_time = start.elapsed();
    
    assert_eq!(y.len(), size);
    println!("Matrix-vector multiplication time: {:?}", matvec_time);
    
    // Test solver performance
    let b = vec![1.0; size];
    let mut x_solve = vec![0.0; size];
    
    let cg_solver = ConjugateGradient::new();
    let start = std::time::Instant::now();
    let info = cg_solver.solve(&matrix, &b, &mut x_solve, 1e-6, 1000)?;
    let solve_time = start.elapsed();
    
    assert!(info.converged, "Large matrix solve did not converge");
    println!("Large matrix solve time: {:?}, iterations: {}", solve_time, info.iterations);
    
    Ok(())
}

/// Integration test for the complete workflow
#[tokio::test]
async fn test_complete_workflow() -> Result<()> {
    // Initialize the library
    openalgebra::init()?;
    
    // Create a realistic problem (2D Poisson equation on a grid)
    let n = 10; // 10x10 grid
    let size = n * n;
    let mut coo = COOMatrix::<f64>::new(size, size);
    
    // 5-point stencil for 2D Laplacian
    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            coo.insert(idx, idx, 4.0);
            
            if i > 0 { coo.insert(idx, (i-1)*n + j, -1.0); }
            if i < n-1 { coo.insert(idx, (i+1)*n + j, -1.0); }
            if j > 0 { coo.insert(idx, i*n + (j-1), -1.0); }
            if j < n-1 { coo.insert(idx, i*n + (j+1), -1.0); }
        }
    }
    
    let matrix = coo.to_csr();
    
    // Create right-hand side (constant function)
    let b = vec![1.0; size];
    let mut x = vec![0.0; size];
    
    // Solve with different methods and compare
    let mut results = Vec::new();
    
    // CG solver
    let cg_solver = ConjugateGradient::new();
    let mut x_cg = x.clone();
    let info_cg = cg_solver.solve(&matrix, &b, &mut x_cg, 1e-8, 1000)?;
    results.push(("CG", info_cg, x_cg.clone()));
    
    // GMRES solver
    let gmres_solver = GMRES::new(20);
    let mut x_gmres = x.clone();
    let info_gmres = gmres_solver.solve(&matrix, &b, &mut x_gmres, 1e-8, 1000)?;
    results.push(("GMRES", info_gmres, x_gmres.clone()));
    
    // BiCGSTAB solver
    let bicgstab_solver = BiCGSTAB::new();
    let mut x_bicgstab = x.clone();
    let info_bicgstab = bicgstab_solver.solve(&matrix, &b, &mut x_bicgstab, 1e-8, 1000)?;
    results.push(("BiCGSTAB", info_bicgstab, x_bicgstab.clone()));
    
    // Verify all solvers converged and solutions are similar
    for (name, info, solution) in &results {
        assert!(info.converged, "{} solver did not converge", name);
        assert!(info.residual_norm < 1e-8, "{} residual too large: {}", name, info.residual_norm);
        
        // Verify solution by computing residual
        let ax = matrix.matvec(solution)?;
        let residual: f64 = ax.iter().zip(b.iter())
            .map(|(&ax_i, &b_i)| (ax_i - b_i).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(residual < 1e-6, "{} solution residual too large: {}", name, residual);
        
        println!("{}: {} iterations, residual: {:.2e}", 
                 name, info.iterations, info.residual_norm);
    }
    
    // Compare solutions (they should be very similar)
    let x_cg = &results[0].2;
    let x_gmres = &results[1].2;
    let x_bicgstab = &results[2].2;
    
    let diff_cg_gmres: f64 = x_cg.iter().zip(x_gmres.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    
    let diff_cg_bicgstab: f64 = x_cg.iter().zip(x_bicgstab.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    
    assert!(diff_cg_gmres < 1e-6, "CG and GMRES solutions differ too much: {}", diff_cg_gmres);
    assert!(diff_cg_bicgstab < 1e-6, "CG and BiCGSTAB solutions differ too much: {}", diff_cg_bicgstab);
    
    println!("All solvers produced consistent results!");
    
    Ok(())
}

/// Test GPU acceleration (if available)
#[tokio::test]
async fn test_gpu_acceleration() -> Result<()> {
    #[cfg(feature = "gpu-acceleration")]
    {
        use openalgebra::cuda::{CudaContext, CudaSolver};
        
        // Skip test if no GPU is available
        if std::env::var("CUDA_VISIBLE_DEVICES").is_err() {
            println!("Skipping GPU test - no CUDA device available");
            return Ok(());
        }
        
        let context_result = CudaContext::new(Some(0));
        if context_result.is_err() {
            println!("Skipping GPU test - CUDA initialization failed");
            return Ok(());
        }
        
        // Create a test matrix
        let mut coo = COOMatrix::<f64>::new(100, 100);
        for i in 0..100 {
            coo.insert(i, i, 2.0);
            if i > 0 { coo.insert(i, i-1, -1.0); }
            if i < 99 { coo.insert(i, i+1, -1.0); }
        }
        let matrix = coo.to_csr();
        
        // Test GPU solver
        let mut gpu_solver = CudaSolver::<f64>::new(Some(0))?;
        let b = vec![1.0; 100];
        let mut x = vec![0.0; 100];
        
        let info = gpu_solver.solve_cg(&matrix, &b, &mut x, 1e-8, 1000)?;
        
        assert!(info.converged, "GPU CG solver did not converge");
        assert!(info.residual_norm < 1e-8, "GPU CG residual too large: {}", info.residual_norm);
        
        println!("GPU solver completed successfully!");
    }
    
    #[cfg(not(feature = "gpu-acceleration"))]
    {
        println!("GPU acceleration not enabled - skipping GPU test");
    }
    
    Ok(())
}

/// Test distributed computing (if MPI is available)
#[tokio::test]
async fn test_distributed_computing() -> Result<()> {
    #[cfg(feature = "mpi")]
    {
        use openalgebra::distributed::{DistributedContext, DistributedCSRMatrix, DistributedVector, DistributedConjugateGradient, DistributedSolverConfig, LoadBalancingStrategy};
        
        let context = DistributedContext::new()?;
        
        // Create a test matrix
        let mut coo = COOMatrix::<f64>::new(20, 20);
        for i in 0..20 {
            coo.insert(i, i, 2.0);
            if i > 0 { coo.insert(i, i-1, -1.0); }
            if i < 19 { coo.insert(i, i+1, -1.0); }
        }
        let matrix = coo.to_csr();
        
        // Distribute the matrix
        let dist_matrix = DistributedCSRMatrix::from_csr_rowwise(&matrix, &context)?;
        
        // Create distributed vectors
        let mut dist_b = DistributedVector::zeros_uniform(20, &context);
        let mut dist_x = DistributedVector::zeros_uniform(20, &context);
        
        // Fill b with ones
        for i in 0..dist_b.local_data.len() {
            dist_b.local_data[i] = 1.0;
        }
        
        // Solve using distributed CG
        let config = DistributedSolverConfig {
            tolerance: 1e-8,
            max_iterations: 1000,
            restart: None,
            overlap_communication: false,
            load_balancing: LoadBalancingStrategy::RowWise,
        };
        
        let dist_solver = DistributedConjugateGradient::new(config);
        let info = dist_solver.solve(&dist_matrix, &dist_b, &mut dist_x, &context)?;
        
        assert!(info.converged, "Distributed CG solver did not converge");
        assert!(info.residual_norm < 1e-8, "Distributed CG residual too large: {}", info.residual_norm);
        
        println!("Distributed solver completed successfully on {} process(es)!", context.size());
    }
    
    #[cfg(not(feature = "mpi"))]
    {
        println!("MPI not enabled - skipping distributed test");
    }
    
    Ok(())
} 