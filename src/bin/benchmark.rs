/*!
# OpenAlgebra Benchmark Suite

Comprehensive benchmarking tool for OpenAlgebra library performance testing.
*/

use clap::{Arg, Command};
use openalgebra::{
    sparse::{COOMatrix, CSRMatrix, SparseMatrix},
    solvers::{ConjugateGradient, GMRES, BiCGSTAB, IterativeSolver},
    tensor::{SparseTensor, DenseTensor},
    preconditioners::{ILUPreconditioner, JacobiPreconditioner, Preconditioner},
    Config, init,
};
use std::time::Instant;
use rayon::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("OpenAlgebra Benchmark Suite")
        .version("1.0.0")
        .author("Nik Jois <nikjois@llamasearch.ai>")
        .about("Comprehensive performance benchmarks for OpenAlgebra")
        .arg(
            Arg::new("size")
                .short('s')
                .long("size")
                .value_name("SIZE")
                .help("Matrix size for benchmarks")
                .default_value("1000"),
        )
        .arg(
            Arg::new("iterations")
                .short('i')
                .long("iterations")
                .value_name("ITER")
                .help("Number of benchmark iterations")
                .default_value("10"),
        )
        .arg(
            Arg::new("threads")
                .short('t')
                .long("threads")
                .value_name("THREADS")
                .help("Number of threads to use")
                .default_value("0"),
        )
        .arg(
            Arg::new("benchmark")
                .short('b')
                .long("benchmark")
                .value_name("BENCH")
                .help("Specific benchmark to run")
                .value_parser(["all", "sparse", "solvers", "tensors", "preconditioners"])
                .default_value("all"),
        )
        .get_matches();

    // Initialize library
    init()?;

    let size: usize = matches.get_one::<String>("size").unwrap().parse()?;
    let iterations: usize = matches.get_one::<String>("iterations").unwrap().parse()?;
    let threads: usize = matches.get_one::<String>("threads").unwrap().parse()?;
    let benchmark_type = matches.get_one::<String>("benchmark").unwrap();

    // Configure threading
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()?;
    }

    println!("OpenAlgebra Benchmark Suite");
    println!("==========================");
    println!("Matrix size: {}", size);
    println!("Iterations: {}", iterations);
    println!("Threads: {}", rayon::current_num_threads());
    println!();

    match benchmark_type.as_str() {
        "all" => {
            run_sparse_benchmarks(size, iterations)?;
            run_solver_benchmarks(size, iterations)?;
            run_tensor_benchmarks(size, iterations)?;
            run_preconditioner_benchmarks(size, iterations)?;
        }
        "sparse" => run_sparse_benchmarks(size, iterations)?,
        "solvers" => run_solver_benchmarks(size, iterations)?,
        "tensors" => run_tensor_benchmarks(size, iterations)?,
        "preconditioners" => run_preconditioner_benchmarks(size, iterations)?,
        _ => unreachable!(),
    }

    Ok(())
}

fn run_sparse_benchmarks(size: usize, iterations: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("Sparse Matrix Benchmarks");
    println!("------------------------");

    // COO Matrix creation and insertion
    let start = Instant::now();
    for _ in 0..iterations {
        let mut coo = COOMatrix::<f64>::new(size, size);
        for i in 0..size {
            coo.insert(i, i, 2.0);
            if i > 0 {
                coo.insert(i, i - 1, -1.0);
            }
            if i < size - 1 {
                coo.insert(i, i + 1, -1.0);
            }
        }
    }
    let coo_time = start.elapsed();
    println!("COO Matrix Creation: {:.2?} per iteration", coo_time / iterations as u32);

    // COO to CSR conversion
    let mut coo = COOMatrix::<f64>::new(size, size);
    for i in 0..size {
        coo.insert(i, i, 2.0);
        if i > 0 {
            coo.insert(i, i - 1, -1.0);
        }
        if i < size - 1 {
            coo.insert(i, i + 1, -1.0);
        }
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _csr = coo.to_csr();
    }
    let conversion_time = start.elapsed();
    println!("COO to CSR Conversion: {:.2?} per iteration", conversion_time / iterations as u32);

    // CSR Matrix-Vector multiplication
    let csr = coo.to_csr();
    let x = vec![1.0; size];
    let start = Instant::now();
    for _ in 0..iterations {
        let _y = csr.matvec(&x);
    }
    let matvec_time = start.elapsed();
    println!("CSR Matrix-Vector Multiply: {:.2?} per iteration", matvec_time / iterations as u32);

    // Parallel Matrix-Vector multiplication
    let start = Instant::now();
    for _ in 0..iterations {
        let _y: Vec<f64> = (0..size)
            .into_par_iter()
            .map(|i| {
                let mut sum = 0.0;
                for j in csr.row_ptr[i]..csr.row_ptr[i + 1] {
                    sum += csr.values[j] * x[csr.col_indices[j]];
                }
                sum
            })
            .collect();
    }
    let parallel_matvec_time = start.elapsed();
    println!("Parallel CSR MatVec: {:.2?} per iteration", parallel_matvec_time / iterations as u32);

    println!();
    Ok(())
}

fn run_solver_benchmarks(size: usize, iterations: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("Iterative Solver Benchmarks");
    println!("---------------------------");

    // Create test matrix (tridiagonal)
    let mut coo = COOMatrix::<f64>::new(size, size);
    for i in 0..size {
        coo.insert(i, i, 2.0);
        if i > 0 {
            coo.insert(i, i - 1, -1.0);
        }
        if i < size - 1 {
            coo.insert(i, i + 1, -1.0);
        }
    }
    let matrix = coo.to_csr();
    let b = vec![1.0; size];

    // Conjugate Gradient
    let start = Instant::now();
    for _ in 0..iterations {
        let mut x = vec![0.0; size];
        let solver = ConjugateGradient::new();
        let _info = solver.solve(&matrix, &b, &mut x);
    }
    let cg_time = start.elapsed();
    println!("Conjugate Gradient: {:.2?} per iteration", cg_time / iterations as u32);

    // GMRES
    let start = Instant::now();
    for _ in 0..iterations {
        let mut x = vec![0.0; size];
        let solver = GMRES::new(30); // restart = 30
        let _info = solver.solve(&matrix, &b, &mut x);
    }
    let gmres_time = start.elapsed();
    println!("GMRES(30): {:.2?} per iteration", gmres_time / iterations as u32);

    // BiCGSTAB
    let start = Instant::now();
    for _ in 0..iterations {
        let mut x = vec![0.0; size];
        let solver = BiCGSTAB::new();
        let _info = solver.solve(&matrix, &b, &mut x);
    }
    let bicgstab_time = start.elapsed();
    println!("BiCGSTAB: {:.2?} per iteration", bicgstab_time / iterations as u32);

    println!();
    Ok(())
}

fn run_tensor_benchmarks(size: usize, iterations: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("Tensor Operation Benchmarks");
    println!("---------------------------");

    let shape = vec![size / 10, size / 10, size / 10];
    let total_elements = shape.iter().product::<usize>();

    // Sparse Tensor creation
    let start = Instant::now();
    for _ in 0..iterations {
        let mut tensor = SparseTensor::<f64>::new(shape.clone());
        for i in 0..std::cmp::min(1000, total_elements) {
            let indices = vec![i % shape[0], (i / shape[0]) % shape[1], i / (shape[0] * shape[1])];
            tensor.set(&indices, i as f64);
        }
    }
    let sparse_creation_time = start.elapsed();
    println!("Sparse Tensor Creation: {:.2?} per iteration", sparse_creation_time / iterations as u32);

    // Dense Tensor creation
    let start = Instant::now();
    for _ in 0..iterations {
        let _tensor = DenseTensor::<f64>::zeros(shape.clone());
    }
    let dense_creation_time = start.elapsed();
    println!("Dense Tensor Creation: {:.2?} per iteration", dense_creation_time / iterations as u32);

    // Tensor operations
    let mut sparse_tensor = SparseTensor::<f64>::new(shape.clone());
    for i in 0..std::cmp::min(1000, total_elements) {
        let indices = vec![i % shape[0], (i / shape[0]) % shape[1], i / (shape[0] * shape[1])];
        sparse_tensor.set(&indices, i as f64);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _dense = sparse_tensor.to_dense();
    }
    let conversion_time = start.elapsed();
    println!("Sparse to Dense Conversion: {:.2?} per iteration", conversion_time / iterations as u32);

    println!();
    Ok(())
}

fn run_preconditioner_benchmarks(size: usize, iterations: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("Preconditioner Benchmarks");
    println!("-------------------------");

    // Create test matrix
    let mut coo = COOMatrix::<f64>::new(size, size);
    for i in 0..size {
        coo.insert(i, i, 2.0);
        if i > 0 {
            coo.insert(i, i - 1, -1.0);
        }
        if i < size - 1 {
            coo.insert(i, i + 1, -1.0);
        }
    }
    let matrix = coo.to_csr();

    // Jacobi Preconditioner
    let start = Instant::now();
    for _ in 0..iterations {
        let _precond = JacobiPreconditioner::new(&matrix);
    }
    let jacobi_setup_time = start.elapsed();
    println!("Jacobi Setup: {:.2?} per iteration", jacobi_setup_time / iterations as u32);

    // ILU Preconditioner
    let start = Instant::now();
    for _ in 0..iterations {
        let _precond = ILUPreconditioner::new(&matrix, 0, 1e-6);
    }
    let ilu_setup_time = start.elapsed();
    println!("ILU(0) Setup: {:.2?} per iteration", ilu_setup_time / iterations as u32);

    // Preconditioner application
    let jacobi = JacobiPreconditioner::new(&matrix);
    let x = vec![1.0; size];
    let start = Instant::now();
    for _ in 0..iterations {
        let _y = jacobi.apply(&x);
    }
    let jacobi_apply_time = start.elapsed();
    println!("Jacobi Apply: {:.2?} per iteration", jacobi_apply_time / iterations as u32);

    println!();
    Ok(())
} 