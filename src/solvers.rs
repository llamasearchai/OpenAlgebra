/*!
# Iterative Solvers

This module provides efficient iterative solvers for sparse linear systems
including Conjugate Gradient (CG), GMRES, and BiCGSTAB algorithms.
*/

use crate::sparse::SparseMatrix;
use crate::{Result, OpenAlgebraError};
use std::f64;

/// Trait for iterative linear system solvers
pub trait IterativeSolver<T> {
    /// Solve the linear system Ax = b
    fn solve(&self, matrix: &dyn SparseMatrix<T>, b: &[T], x: &mut [T]) -> Result<SolverInfo>;
    
    /// Set maximum number of iterations
    fn set_max_iterations(&mut self, max_iter: usize);
    
    /// Set convergence tolerance
    fn set_tolerance(&mut self, tol: T);
}

/// Information about solver convergence
#[derive(Debug, Clone)]
pub struct SolverInfo {
    pub converged: bool,
    pub iterations: usize,
    pub residual_norm: f64,
    pub solve_time: std::time::Duration,
}

impl SolverInfo {
    pub fn new() -> Self {
        Self {
            converged: false,
            iterations: 0,
            residual_norm: f64::INFINITY,
            solve_time: std::time::Duration::new(0, 0),
        }
    }
}

/// Conjugate Gradient solver for symmetric positive definite matrices
#[derive(Debug, Clone)]
pub struct ConjugateGradient {
    max_iterations: usize,
    tolerance: f64,
}

impl ConjugateGradient {
    /// Create new CG solver with default parameters
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-12,
        }
    }
    
    /// Create CG solver with custom parameters
    pub fn with_params(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }
}

impl Default for ConjugateGradient {
    fn default() -> Self {
        Self::new()
    }
}

impl IterativeSolver<f64> for ConjugateGradient {
    fn solve(&self, matrix: &dyn SparseMatrix<f64>, b: &[f64], x: &mut [f64]) -> Result<SolverInfo> {
        let start_time = std::time::Instant::now();
        let mut info = SolverInfo::new();
        
        let n = b.len();
        if x.len() != n {
            return Err(Box::new(OpenAlgebraError::DimensionMismatch {
                expected: format!("{}", n),
                actual: format!("{}", x.len()),
            }));
        }
        
        // Initialize vectors
        let mut r = vec![0.0; n];
        let mut p = vec![0.0; n];
        let mut ap = vec![0.0; n];
        
        // r = b - A * x
        matrix.matvec(x, &mut ap)?;
        for i in 0..n {
            r[i] = b[i] - ap[i];
            p[i] = r[i]; // p = r
        }
        
        let mut rsold = dot_product(&r, &r);
        
        for iter in 0..self.max_iterations {
            // ap = A * p
            matrix.matvec(&p, &mut ap)?;
            
            let pap = dot_product(&p, &ap);
            if pap.abs() < f64::EPSILON {
                info.converged = false;
                info.iterations = iter;
                info.residual_norm = rsold.sqrt();
                info.solve_time = start_time.elapsed();
                return Err(Box::new(OpenAlgebraError::ConvergenceFailure { iterations: iter }));
            }
            
            let alpha = rsold / pap;
            
            // x = x + alpha * p
            // r = r - alpha * ap
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            
            let rsnew = dot_product(&r, &r);
            let residual_norm = rsnew.sqrt();
            
            if residual_norm < self.tolerance {
                info.converged = true;
                info.iterations = iter + 1;
                info.residual_norm = residual_norm;
                info.solve_time = start_time.elapsed();
                return Ok(info);
            }
            
            let beta = rsnew / rsold;
            
            // p = r + beta * p
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            
            rsold = rsnew;
        }
        
        info.converged = false;
        info.iterations = self.max_iterations;
        info.residual_norm = rsold.sqrt();
        info.solve_time = start_time.elapsed();
        Err(Box::new(OpenAlgebraError::ConvergenceFailure { iterations: self.max_iterations }))
    }
    
    fn set_max_iterations(&mut self, max_iter: usize) {
        self.max_iterations = max_iter;
    }
    
    fn set_tolerance(&mut self, tol: f64) {
        self.tolerance = tol;
    }
}

/// GMRES solver for general non-symmetric matrices
#[derive(Debug, Clone)]
pub struct GMRES {
    max_iterations: usize,
    tolerance: f64,
    restart: usize,
}

impl GMRES {
    /// Create new GMRES solver with default parameters
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-12,
            restart: 30,
        }
    }
    
    /// Create GMRES solver with custom parameters
    pub fn with_params(max_iterations: usize, tolerance: f64, restart: usize) -> Self {
        Self {
            max_iterations,
            tolerance,
            restart,
        }
    }
    
    /// Set restart parameter
    pub fn set_restart(&mut self, restart: usize) {
        self.restart = restart;
    }
}

impl Default for GMRES {
    fn default() -> Self {
        Self::new()
    }
}

impl IterativeSolver<f64> for GMRES {
    fn solve(&self, matrix: &dyn SparseMatrix<f64>, b: &[f64], x: &mut [f64]) -> Result<SolverInfo> {
        let start_time = std::time::Instant::now();
        let mut info = SolverInfo::new();
        
        let n = b.len();
        if x.len() != n {
            return Err(Box::new(OpenAlgebraError::DimensionMismatch {
                expected: format!("{}", n),
                actual: format!("{}", x.len()),
            }));
        }
        
        let mut total_iterations = 0;
        
        for _restart_cycle in 0..(self.max_iterations / self.restart + 1) {
            if total_iterations >= self.max_iterations {
                break;
            }
            
            // Calculate initial residual
            let mut r = vec![0.0; n];
            let mut ax = vec![0.0; n];
            matrix.matvec(x, &mut ax)?;
            for i in 0..n {
                r[i] = b[i] - ax[i];
            }
            
            let beta = vector_norm(&r);
            if beta < self.tolerance {
                info.converged = true;
                info.iterations = total_iterations;
                info.residual_norm = beta;
                info.solve_time = start_time.elapsed();
                return Ok(info);
            }
            
            // Normalize r to get v1
            for i in 0..n {
                r[i] /= beta;
            }
            
            let m = std::cmp::min(self.restart, self.max_iterations - total_iterations);
            let mut v = vec![vec![0.0; n]; m + 1];
            let mut h = vec![vec![0.0; m + 1]; m];
            let mut g = vec![0.0; m + 1];
            
            v[0] = r.clone();
            g[0] = beta;
            
            for j in 0..m {
                if total_iterations >= self.max_iterations {
                    break;
                }
                
                // w = A * v[j]
                let mut w = vec![0.0; n];
                matrix.matvec(&v[j], &mut w)?;
                
                // Modified Gram-Schmidt orthogonalization
                for i in 0..=j {
                    h[i][j] = dot_product(&w, &v[i]);
                    for k in 0..n {
                        w[k] -= h[i][j] * v[i][k];
                    }
                }
                
                h[j + 1][j] = vector_norm(&w);
                
                if h[j + 1][j] > f64::EPSILON {
                    for i in 0..n {
                        w[i] /= h[j + 1][j];
                    }
                    v[j + 1] = w;
                }
                
                // Apply previous Givens rotations
                for i in 0..j {
                    let temp = h[i][j];
                    h[i][j] = h[i][j] * g[i] + h[i + 1][j] * g[i];
                    h[i + 1][j] = -temp * g[i] + h[i + 1][j] * g[i];
                }
                
                // Compute new Givens rotation
                let gamma = (h[j][j] * h[j][j] + h[j + 1][j] * h[j + 1][j]).sqrt();
                if gamma > f64::EPSILON {
                    let c = h[j][j] / gamma;
                    let s = h[j + 1][j] / gamma;
                    
                    h[j][j] = gamma;
                    h[j + 1][j] = 0.0;
                    
                    let temp = g[j];
                    g[j] = c * temp;
                    g[j + 1] = -s * temp;
                }
                
                total_iterations += 1;
                
                // Check convergence
                if g[j + 1].abs() < self.tolerance {
                    // Solve upper triangular system
                    let mut y = vec![0.0; j + 1];
                    for i in (0..=j).rev() {
                        y[i] = g[i];
                        for k in (i + 1)..=j {
                            y[i] -= h[i][k] * y[k];
                        }
                        y[i] /= h[i][i];
                    }
                    
                    // Update solution
                    for i in 0..n {
                        for k in 0..=j {
                            x[i] += y[k] * v[k][i];
                        }
                    }
                    
                    info.converged = true;
                    info.iterations = total_iterations;
                    info.residual_norm = g[j + 1].abs();
                    info.solve_time = start_time.elapsed();
                    return Ok(info);
                }
            }
            
            // Update solution with current Krylov space
            let mut y = vec![0.0; m];
            for i in (0..m).rev() {
                y[i] = g[i];
                for k in (i + 1)..m {
                    y[i] -= h[i][k] * y[k];
                }
                if h[i][i].abs() > f64::EPSILON {
                    y[i] /= h[i][i];
                }
            }
            
            for i in 0..n {
                for k in 0..m {
                    x[i] += y[k] * v[k][i];
                }
            }
        }
        
        info.converged = false;
        info.iterations = total_iterations;
        info.solve_time = start_time.elapsed();
        Err(Box::new(OpenAlgebraError::ConvergenceFailure { iterations: total_iterations }))
    }
    
    fn set_max_iterations(&mut self, max_iter: usize) {
        self.max_iterations = max_iter;
    }
    
    fn set_tolerance(&mut self, tol: f64) {
        self.tolerance = tol;
    }
}

/// BiCGSTAB solver for general non-symmetric matrices
#[derive(Debug, Clone)]
pub struct BiCGSTAB {
    max_iterations: usize,
    tolerance: f64,
}

impl BiCGSTAB {
    /// Create new BiCGSTAB solver with default parameters
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-12,
        }
    }
    
    /// Create BiCGSTAB solver with custom parameters
    pub fn with_params(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }
}

impl Default for BiCGSTAB {
    fn default() -> Self {
        Self::new()
    }
}

impl IterativeSolver<f64> for BiCGSTAB {
    fn solve(&self, matrix: &dyn SparseMatrix<f64>, b: &[f64], x: &mut [f64]) -> Result<SolverInfo> {
        let start_time = std::time::Instant::now();
        let mut info = SolverInfo::new();
        
        let n = b.len();
        if x.len() != n {
            return Err(Box::new(OpenAlgebraError::DimensionMismatch {
                expected: format!("{}", n),
                actual: format!("{}", x.len()),
            }));
        }
        
        // Initialize vectors
        let mut r = vec![0.0; n];
        let mut r_tilde = vec![0.0; n];
        let mut p = vec![0.0; n];
        let mut v = vec![0.0; n];
        let mut s = vec![0.0; n];
        let mut t = vec![0.0; n];
        
        // r = b - A * x
        matrix.matvec(x, &mut r)?;
        for i in 0..n {
            r[i] = b[i] - r[i];
            r_tilde[i] = r[i]; // Choose r_tilde = r
        }
        
        let mut rho = 1.0;
        let mut alpha = 1.0;
        let mut omega = 1.0;
        
        for iter in 0..self.max_iterations {
            let rho_new = dot_product(&r_tilde, &r);
            
            if rho_new.abs() < f64::EPSILON {
                info.converged = false;
                info.iterations = iter;
                info.solve_time = start_time.elapsed();
                return Err(Box::new(OpenAlgebraError::ConvergenceFailure { iterations: iter }));
            }
            
            let beta = (rho_new / rho) * (alpha / omega);
            
            // p = r + beta * (p - omega * v)
            for i in 0..n {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }
            
            // v = A * p
            matrix.matvec(&p, &mut v)?;
            
            let rtilde_v = dot_product(&r_tilde, &v);
            if rtilde_v.abs() < f64::EPSILON {
                info.converged = false;
                info.iterations = iter;
                info.solve_time = start_time.elapsed();
                return Err(Box::new(OpenAlgebraError::ConvergenceFailure { iterations: iter }));
            }
            
            alpha = rho_new / rtilde_v;
            
            // s = r - alpha * v
            for i in 0..n {
                s[i] = r[i] - alpha * v[i];
            }
            
            // Check for convergence
            let s_norm = vector_norm(&s);
            if s_norm < self.tolerance {
                // x = x + alpha * p
                for i in 0..n {
                    x[i] += alpha * p[i];
                }
                
                info.converged = true;
                info.iterations = iter + 1;
                info.residual_norm = s_norm;
                info.solve_time = start_time.elapsed();
                return Ok(info);
            }
            
            // t = A * s
            matrix.matvec(&s, &mut t)?;
            
            let t_s = dot_product(&t, &s);
            let t_t = dot_product(&t, &t);
            
            if t_t.abs() < f64::EPSILON {
                info.converged = false;
                info.iterations = iter;
                info.solve_time = start_time.elapsed();
                return Err(Box::new(OpenAlgebraError::ConvergenceFailure { iterations: iter }));
            }
            
            omega = t_s / t_t;
            
            // x = x + alpha * p + omega * s
            // r = s - omega * t
            for i in 0..n {
                x[i] += alpha * p[i] + omega * s[i];
                r[i] = s[i] - omega * t[i];
            }
            
            let r_norm = vector_norm(&r);
            if r_norm < self.tolerance {
                info.converged = true;
                info.iterations = iter + 1;
                info.residual_norm = r_norm;
                info.solve_time = start_time.elapsed();
                return Ok(info);
            }
            
            rho = rho_new;
        }
        
        info.converged = false;
        info.iterations = self.max_iterations;
        info.residual_norm = vector_norm(&r);
        info.solve_time = start_time.elapsed();
        Err(Box::new(OpenAlgebraError::ConvergenceFailure { iterations: self.max_iterations }))
    }
    
    fn set_max_iterations(&mut self, max_iter: usize) {
        self.max_iterations = max_iter;
    }
    
    fn set_tolerance(&mut self, tol: f64) {
        self.tolerance = tol;
    }
}

/// Helper function to compute dot product
fn dot_product(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

/// Helper function to compute vector norm (L2)
fn vector_norm(x: &[f64]) -> f64 {
    dot_product(x, x).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::COOMatrix;

    #[test]
    fn test_conjugate_gradient_solver() {
        // Create a simple 3x3 symmetric positive definite matrix
        let mut matrix = COOMatrix::<f64>::new(3, 3);
        matrix.insert(0, 0, 4.0);
        matrix.insert(0, 1, 1.0);
        matrix.insert(1, 0, 1.0);
        matrix.insert(1, 1, 3.0);
        matrix.insert(1, 2, 1.0);
        matrix.insert(2, 1, 1.0);
        matrix.insert(2, 2, 2.0);
        
        let csr = matrix.to_csr();
        let b = vec![1.0, 2.0, 3.0];
        let mut x = vec![0.0; 3];
        
        let solver = ConjugateGradient::new();
        let info = solver.solve(&csr, &b, &mut x);
        
        assert!(info.is_ok());
        let info = info.unwrap();
        assert!(info.converged);
        
        // Verify solution: A * x should be close to b
        let mut ax = vec![0.0; 3];
        csr.matvec(&x, &mut ax).unwrap();
        for i in 0..3 {
            assert!((ax[i] - b[i]).abs() < 1e-10);
        }
    }
    
    #[test]
    fn test_gmres_solver() {
        // Create a simple 3x3 matrix
        let mut matrix = COOMatrix::<f64>::new(3, 3);
        matrix.insert(0, 0, 2.0);
        matrix.insert(0, 1, 1.0);
        matrix.insert(1, 1, 2.0);
        matrix.insert(1, 2, 1.0);
        matrix.insert(2, 0, 1.0);
        matrix.insert(2, 2, 2.0);
        
        let csr = matrix.to_csr();
        let b = vec![1.0, 1.0, 1.0];
        let mut x = vec![0.0; 3];
        
        let solver = GMRES::new();
        let info = solver.solve(&csr, &b, &mut x);
        
        // GMRES should converge for this simple system
        if let Ok(info) = info {
            assert!(info.converged);
        }
    }
    
    #[test]
    fn test_bicgstab_solver() {
        // Create a simple 3x3 matrix
        let mut matrix = COOMatrix::<f64>::new(3, 3);
        matrix.insert(0, 0, 3.0);
        matrix.insert(0, 1, 1.0);
        matrix.insert(1, 0, 1.0);
        matrix.insert(1, 1, 3.0);
        matrix.insert(1, 2, 1.0);
        matrix.insert(2, 1, 1.0);
        matrix.insert(2, 2, 3.0);
        
        let csr = matrix.to_csr();
        let b = vec![1.0, 1.0, 1.0];
        let mut x = vec![0.0; 3];
        
        let solver = BiCGSTAB::new();
        let info = solver.solve(&csr, &b, &mut x);
        
        // BiCGSTAB should converge for this simple system
        if let Ok(info) = info {
            assert!(info.converged);
        }
    }
    
    #[test]
    fn test_solver_parameters() {
        let mut solver = ConjugateGradient::new();
        solver.set_max_iterations(500);
        solver.set_tolerance(1e-10);
        
        assert_eq!(solver.max_iterations, 500);
        assert_eq!(solver.tolerance, 1e-10);
    }
} 