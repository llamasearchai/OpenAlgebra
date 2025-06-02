/*!
# Sparse Operations Benchmarks

Performance benchmarks for OpenAlgebra sparse matrix operations.
*/

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use openalgebra::{
    sparse::{COOMatrix, CSRMatrix, SparseMatrix},
    solvers::{ConjugateGradient, GMRES, BiCGSTAB, IterativeSolver},
    preconditioners::{JacobiPreconditioner, ILUPreconditioner, Preconditioner},
};

fn create_tridiagonal_matrix(size: usize) -> COOMatrix<f64> {
    let mut matrix = COOMatrix::new(size, size);
    
    for i in 0..size {
        matrix.insert(i, i, 2.0);
        if i > 0 {
            matrix.insert(i, i - 1, -1.0);
        }
        if i < size - 1 {
            matrix.insert(i, i + 1, -1.0);
        }
    }
    
    matrix
}

fn create_laplacian_matrix(size: usize) -> COOMatrix<f64> {
    let mut matrix = COOMatrix::new(size, size);
    let n = (size as f64).sqrt() as usize;
    
    for i in 0..size {
        matrix.insert(i, i, 4.0);
        
        // Left neighbor
        if i % n != 0 {
            matrix.insert(i, i - 1, -1.0);
        }
        
        // Right neighbor
        if (i + 1) % n != 0 {
            matrix.insert(i, i + 1, -1.0);
        }
        
        // Top neighbor
        if i >= n {
            matrix.insert(i, i - n, -1.0);
        }
        
        // Bottom neighbor
        if i + n < size {
            matrix.insert(i, i + n, -1.0);
        }
    }
    
    matrix
}

fn bench_matrix_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_creation");
    
    for size in [100, 500, 1000, 2000].iter() {
        group.bench_with_input(
            BenchmarkId::new("coo_tridiagonal", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let matrix = create_tridiagonal_matrix(black_box(size));
                    black_box(matrix);
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("coo_laplacian", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let matrix = create_laplacian_matrix(black_box(size));
                    black_box(matrix);
                });
            },
        );
    }
    
    group.finish();
}

fn bench_matrix_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_conversion");
    
    for size in [100, 500, 1000, 2000].iter() {
        let coo_matrix = create_tridiagonal_matrix(*size);
        
        group.bench_with_input(
            BenchmarkId::new("coo_to_csr", size),
            &coo_matrix,
            |b, matrix| {
                b.iter(|| {
                    let csr = matrix.to_csr();
                    black_box(csr);
                });
            },
        );
        
        let csr_matrix = coo_matrix.to_csr();
        group.bench_with_input(
            BenchmarkId::new("csr_transpose", size),
            &csr_matrix,
            |b, matrix| {
                b.iter(|| {
                    let transposed = matrix.transpose();
                    black_box(transposed);
                });
            },
        );
    }
    
    group.finish();
}

fn bench_matrix_vector_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_vector_multiply");
    
    for size in [100, 500, 1000, 2000].iter() {
        let coo_matrix = create_tridiagonal_matrix(*size);
        let csr_matrix = coo_matrix.to_csr();
        let vector = vec![1.0; *size];
        
        group.bench_with_input(
            BenchmarkId::new("csr_matvec", size),
            &(&csr_matrix, &vector),
            |b, (matrix, vec)| {
                b.iter(|| {
                    let result = matrix.matvec(black_box(vec));
                    black_box(result);
                });
            },
        );
    }
    
    group.finish();
}

fn bench_iterative_solvers(c: &mut Criterion) {
    let mut group = c.benchmark_group("iterative_solvers");
    group.sample_size(10); // Reduce sample size for expensive operations
    
    for size in [100, 500, 1000].iter() {
        let coo_matrix = create_tridiagonal_matrix(*size);
        let csr_matrix = coo_matrix.to_csr();
        let b = vec![1.0; *size];
        
        group.bench_with_input(
            BenchmarkId::new("conjugate_gradient", size),
            &(&csr_matrix, &b),
            |bench, (matrix, rhs)| {
                bench.iter(|| {
                    let mut x = vec![0.0; matrix.rows()];
                    let solver = ConjugateGradient::new();
                    let info = solver.solve(black_box(matrix), black_box(rhs), black_box(&mut x));
                    black_box((x, info));
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("gmres", size),
            &(&csr_matrix, &b),
            |bench, (matrix, rhs)| {
                bench.iter(|| {
                    let mut x = vec![0.0; matrix.rows()];
                    let solver = GMRES::new(30);
                    let info = solver.solve(black_box(matrix), black_box(rhs), black_box(&mut x));
                    black_box((x, info));
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("bicgstab", size),
            &(&csr_matrix, &b),
            |bench, (matrix, rhs)| {
                bench.iter(|| {
                    let mut x = vec![0.0; matrix.rows()];
                    let solver = BiCGSTAB::new();
                    let info = solver.solve(black_box(matrix), black_box(rhs), black_box(&mut x));
                    black_box((x, info));
                });
            },
        );
    }
    
    group.finish();
}

fn bench_preconditioners(c: &mut Criterion) {
    let mut group = c.benchmark_group("preconditioners");
    
    for size in [100, 500, 1000].iter() {
        let coo_matrix = create_tridiagonal_matrix(*size);
        let csr_matrix = coo_matrix.to_csr();
        
        group.bench_with_input(
            BenchmarkId::new("jacobi_setup", size),
            &csr_matrix,
            |b, matrix| {
                b.iter(|| {
                    let precond = JacobiPreconditioner::new(black_box(matrix));
                    black_box(precond);
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("ilu_setup", size),
            &csr_matrix,
            |b, matrix| {
                b.iter(|| {
                    let precond = ILUPreconditioner::new(black_box(matrix), 0, 1e-6);
                    black_box(precond);
                });
            },
        );
        
        let jacobi_precond = JacobiPreconditioner::new(&csr_matrix);
        let vector = vec![1.0; *size];
        
        group.bench_with_input(
            BenchmarkId::new("jacobi_apply", size),
            &(&jacobi_precond, &vector),
            |b, (precond, vec)| {
                b.iter(|| {
                    let result = precond.apply(black_box(vec));
                    black_box(result);
                });
            },
        );
    }
    
    group.finish();
}

fn bench_preconditioned_solvers(c: &mut Criterion) {
    let mut group = c.benchmark_group("preconditioned_solvers");
    group.sample_size(10);
    
    for size in [100, 500, 1000].iter() {
        let coo_matrix = create_tridiagonal_matrix(*size);
        let csr_matrix = coo_matrix.to_csr();
        let b = vec![1.0; *size];
        
        group.bench_with_input(
            BenchmarkId::new("cg_jacobi", size),
            &(&csr_matrix, &b),
            |bench, (matrix, rhs)| {
                bench.iter(|| {
                    let mut x = vec![0.0; matrix.rows()];
                    let precond = JacobiPreconditioner::new(matrix);
                    let solver = ConjugateGradient::with_preconditioner(Box::new(precond));
                    let info = solver.solve(black_box(matrix), black_box(rhs), black_box(&mut x));
                    black_box((x, info));
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("cg_ilu", size),
            &(&csr_matrix, &b),
            |bench, (matrix, rhs)| {
                bench.iter(|| {
                    let mut x = vec![0.0; matrix.rows()];
                    let precond = ILUPreconditioner::new(matrix, 0, 1e-6);
                    let solver = ConjugateGradient::with_preconditioner(Box::new(precond));
                    let info = solver.solve(black_box(matrix), black_box(rhs), black_box(&mut x));
                    black_box((x, info));
                });
            },
        );
    }
    
    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    
    for size in [1000, 5000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("sparse_vs_dense", size),
            size,
            |b, &size| {
                b.iter(|| {
                    // Sparse matrix (tridiagonal)
                    let sparse = create_tridiagonal_matrix(black_box(size));
                    let sparse_csr = sparse.to_csr();
                    
                    // Dense matrix would be size * size * 8 bytes
                    let sparse_memory = sparse_csr.values.len() * 8 + 
                                       sparse_csr.col_indices.len() * 8 + 
                                       sparse_csr.row_ptr.len() * 8;
                    
                    black_box((sparse_csr, sparse_memory));
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_matrix_creation,
    bench_matrix_conversion,
    bench_matrix_vector_multiply,
    bench_iterative_solvers,
    bench_preconditioners,
    bench_preconditioned_solvers,
    bench_memory_usage
);

criterion_main!(benches); 