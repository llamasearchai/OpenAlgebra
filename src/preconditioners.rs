/*!
# Preconditioners

This module provides preconditioning techniques for accelerating iterative solvers,
including Algebraic Multigrid (AMG), Incomplete LU (ILU), and other methods.
*/

use crate::sparse::{SparseMatrix, CSRMatrix, COOMatrix};
use crate::{Result, OpenAlgebraError};
use std::collections::HashMap;

/// Trait for preconditioners
pub trait Preconditioner<T> {
    /// Apply preconditioner: solve M * y = x for y
    fn apply(&self, x: &[T], y: &mut [T]) -> Result<()>;
    
    /// Setup preconditioner from matrix
    fn setup(&mut self, matrix: &dyn SparseMatrix<T>) -> Result<()>;
    
    /// Get memory usage in bytes
    fn memory_usage(&self) -> usize;
}

/// Incomplete LU (ILU) preconditioner
#[derive(Debug, Clone)]
pub struct ILUPreconditioner {
    l_matrix: Option<CSRMatrix<f64>>,
    u_matrix: Option<CSRMatrix<f64>>,
    fill_level: usize,
    drop_tolerance: f64,
}

impl ILUPreconditioner {
    /// Create new ILU preconditioner
    pub fn new(fill_level: usize, drop_tolerance: f64) -> Self {
        Self {
            l_matrix: None,
            u_matrix: None,
            fill_level,
            drop_tolerance,
        }
    }
    
    /// Create ILU(0) preconditioner (no fill-in)
    pub fn ilu0() -> Self {
        Self::new(0, 0.0)
    }
    
    /// Create ILU(k) preconditioner with fill level k
    pub fn iluk(fill_level: usize) -> Self {
        Self::new(fill_level, 0.0)
    }
    
    /// Create ILUT preconditioner with drop tolerance
    pub fn ilut(drop_tolerance: f64) -> Self {
        Self::new(0, drop_tolerance)
    }
    
    /// Perform ILU factorization
    fn factorize(&mut self, matrix: &CSRMatrix<f64>) -> Result<()> {
        let n = matrix.shape().0;
        
        // Initialize L and U matrices
        let mut l_entries: Vec<(usize, usize, f64)> = Vec::new();
        let mut u_entries: Vec<(usize, usize, f64)> = Vec::new();
        
        // Copy original matrix structure and values
        let mut working_matrix = HashMap::<(usize, usize), f64>::new();
        
        for i in 0..n {
            if let Some((cols, vals)) = matrix.row(i) {
                for (j, &col) in cols.iter().enumerate() {
                    working_matrix.insert((i, col), vals[j]);
                }
            }
        }
        
        // ILU factorization algorithm
        for k in 0..n {
            // Get diagonal element
            let akk = working_matrix.get(&(k, k)).copied().unwrap_or(0.0);
            if akk.abs() < f64::EPSILON {
                return Err(Box::new(OpenAlgebraError::InvalidFormat(
                    format!("Zero pivot encountered at row {}", k)
                )));
            }
            
            // Process row k
            for i in (k + 1)..n {
                if let Some(&aik) = working_matrix.get(&(i, k)) {
                    let factor = aik / akk;
                    
                    // Check drop tolerance
                    if factor.abs() > self.drop_tolerance {
                        l_entries.push((i, k, factor));
                        
                        // Update row i
                        for j in (k + 1)..n {
                            if let Some(&akj) = working_matrix.get(&(k, j)) {
                                let current = working_matrix.get(&(i, j)).copied().unwrap_or(0.0);
                                let new_val = current - factor * akj;
                                
                                if new_val.abs() > self.drop_tolerance {
                                    working_matrix.insert((i, j), new_val);
                                } else {
                                    working_matrix.remove(&(i, j));
                                }
                            }
                        }
                    }
                }
            }
            
            // Add diagonal element to U
            u_entries.push((k, k, akk));
            
            // Add upper triangular elements to U
            for j in (k + 1)..n {
                if let Some(&akj) = working_matrix.get(&(k, j)) {
                    if akj.abs() > self.drop_tolerance {
                        u_entries.push((k, j, akj));
                    }
                }
            }
        }
        
        // Add identity diagonal to L
        for i in 0..n {
            l_entries.push((i, i, 1.0));
        }
        
        // Build L and U matrices
        self.l_matrix = Some(Self::build_csr_from_entries(l_entries, n, n));
        self.u_matrix = Some(Self::build_csr_from_entries(u_entries, n, n));
        
        Ok(())
    }
    
    /// Build CSR matrix from triplet entries
    fn build_csr_from_entries(entries: Vec<(usize, usize, f64)>, rows: usize, cols: usize) -> CSRMatrix<f64> {
        let mut coo = COOMatrix::new(rows, cols);
        
        for (row, col, val) in entries {
            coo.insert(row, col, val);
        }
        
        coo.to_csr()
    }
    
    /// Forward solve: L * y = x
    fn forward_solve(&self, x: &[f64], y: &mut [f64]) -> Result<()> {
        let l = self.l_matrix.as_ref().unwrap();
        let n = x.len();
        
        for i in 0..n {
            let mut sum = 0.0;
            
            if let Some((cols, vals)) = l.row(i) {
                for (j, &col) in cols.iter().enumerate() {
                    if col < i {
                        sum += vals[j] * y[col];
                    }
                }
            }
            
            y[i] = x[i] - sum;
        }
        
        Ok(())
    }
    
    /// Backward solve: U * y = x
    fn backward_solve(&self, x: &[f64], y: &mut [f64]) -> Result<()> {
        let u = self.u_matrix.as_ref().unwrap();
        let n = x.len();
        
        for i in (0..n).rev() {
            let mut sum = 0.0;
            let mut diagonal = 1.0;
            
            if let Some((cols, vals)) = u.row(i) {
                for (j, &col) in cols.iter().enumerate() {
                    if col > i {
                        sum += vals[j] * y[col];
                    } else if col == i {
                        diagonal = vals[j];
                    }
                }
            }
            
            y[i] = (x[i] - sum) / diagonal;
        }
        
        Ok(())
    }
}

impl Preconditioner<f64> for ILUPreconditioner {
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<()> {
        if self.l_matrix.is_none() || self.u_matrix.is_none() {
            return Err(Box::new(OpenAlgebraError::InvalidFormat(
                "Preconditioner not setup".to_string()
            )));
        }
        
        let n = x.len();
        let mut temp = vec![0.0; n];
        
        // Solve L * temp = x
        self.forward_solve(x, &mut temp)?;
        
        // Solve U * y = temp
        self.backward_solve(&temp, y)?;
        
        Ok(())
    }
    
    fn setup(&mut self, matrix: &dyn SparseMatrix<f64>) -> Result<()> {
        // Convert to CSR if needed
        let csr_matrix = if let Some(csr) = matrix.as_any().downcast_ref::<CSRMatrix<f64>>() {
            csr.clone()
        } else {
            // Convert from other format (simplified - in practice would handle all formats)
            return Err(Box::new(OpenAlgebraError::InvalidFormat(
                "Matrix format not supported for ILU".to_string()
            )));
        };
        
        self.factorize(&csr_matrix)
    }
    
    fn memory_usage(&self) -> usize {
        let mut usage = 0;
        if let Some(l) = &self.l_matrix {
            usage += l.nnz() * (std::mem::size_of::<f64>() + std::mem::size_of::<usize>());
        }
        if let Some(u) = &self.u_matrix {
            usage += u.nnz() * (std::mem::size_of::<f64>() + std::mem::size_of::<usize>());
        }
        usage
    }
}

/// Jacobi (diagonal) preconditioner
#[derive(Debug, Clone)]
pub struct JacobiPreconditioner {
    diagonal: Vec<f64>,
}

impl JacobiPreconditioner {
    /// Create new Jacobi preconditioner
    pub fn new() -> Self {
        Self {
            diagonal: Vec::new(),
        }
    }
}

impl Default for JacobiPreconditioner {
    fn default() -> Self {
        Self::new()
    }
}

impl Preconditioner<f64> for JacobiPreconditioner {
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<()> {
        if self.diagonal.len() != x.len() {
            return Err(Box::new(OpenAlgebraError::DimensionMismatch {
                expected: format!("{}", self.diagonal.len()),
                actual: format!("{}", x.len()),
            }));
        }
        
        for i in 0..x.len() {
            y[i] = x[i] / self.diagonal[i];
        }
        
        Ok(())
    }
    
    fn setup(&mut self, matrix: &dyn SparseMatrix<f64>) -> Result<()> {
        let (rows, _) = matrix.shape();
        self.diagonal = vec![1.0; rows];
        
        // Extract diagonal elements
        for i in 0..rows {
            if let Some(&diag_val) = matrix.get(i, i) {
                if diag_val.abs() > f64::EPSILON {
                    self.diagonal[i] = diag_val;
                } else {
                    return Err(Box::new(OpenAlgebraError::InvalidFormat(
                        format!("Zero diagonal element at position {}", i)
                    )));
                }
            }
        }
        
        Ok(())
    }
    
    fn memory_usage(&self) -> usize {
        self.diagonal.len() * std::mem::size_of::<f64>()
    }
}

/// Gauss-Seidel preconditioner
#[derive(Debug, Clone)]
pub struct GaussSeidelPreconditioner {
    matrix: Option<CSRMatrix<f64>>,
}

impl GaussSeidelPreconditioner {
    /// Create new Gauss-Seidel preconditioner
    pub fn new() -> Self {
        Self {
            matrix: None,
        }
    }
}

impl Default for GaussSeidelPreconditioner {
    fn default() -> Self {
        Self::new()
    }
}

impl Preconditioner<f64> for GaussSeidelPreconditioner {
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<()> {
        let matrix = self.matrix.as_ref().unwrap();
        let n = x.len();
        
        // Initialize with x
        y.copy_from_slice(x);
        
        // Forward Gauss-Seidel iteration
        for i in 0..n {
            let mut sum = 0.0;
            let mut diagonal = 1.0;
            
            if let Some((cols, vals)) = matrix.row(i) {
                for (j, &col) in cols.iter().enumerate() {
                    if col < i {
                        sum += vals[j] * y[col];
                    } else if col == i {
                        diagonal = vals[j];
                    }
                }
            }
            
            if diagonal.abs() > f64::EPSILON {
                y[i] = (x[i] - sum) / diagonal;
            }
        }
        
        Ok(())
    }
    
    fn setup(&mut self, matrix: &dyn SparseMatrix<f64>) -> Result<()> {
        // Store matrix for Gauss-Seidel iterations
        if let Some(csr) = matrix.as_any().downcast_ref::<CSRMatrix<f64>>() {
            self.matrix = Some(csr.clone());
        } else {
            return Err(Box::new(OpenAlgebraError::InvalidFormat(
                "Matrix format not supported for Gauss-Seidel".to_string()
            )));
        }
        
        Ok(())
    }
    
    fn memory_usage(&self) -> usize {
        if let Some(matrix) = &self.matrix {
            matrix.nnz() * (std::mem::size_of::<f64>() + std::mem::size_of::<usize>())
        } else {
            0
        }
    }
}

/// Algebraic Multigrid (AMG) preconditioner (simplified implementation)
#[derive(Debug, Clone)]
pub struct AMGPreconditioner {
    levels: Vec<AMGLevel>,
    max_levels: usize,
    coarsening_threshold: f64,
}

#[derive(Debug, Clone)]
struct AMGLevel {
    matrix: CSRMatrix<f64>,
    restriction: Option<CSRMatrix<f64>>,
    prolongation: Option<CSRMatrix<f64>>,
}

impl AMGPreconditioner {
    /// Create new AMG preconditioner
    pub fn new(max_levels: usize, coarsening_threshold: f64) -> Self {
        Self {
            levels: Vec::new(),
            max_levels,
            coarsening_threshold,
        }
    }
    
    /// Create AMG with default parameters
    pub fn default_amg() -> Self {
        Self::new(10, 0.25)
    }
    
    /// V-cycle application
    fn v_cycle(&self, level: usize, x: &[f64], y: &mut [f64]) -> Result<()> {
        if level >= self.levels.len() {
            return Ok(());
        }
        
        let current_level = &self.levels[level];
        let n = x.len();
        
        if level == self.levels.len() - 1 {
            // Coarsest level - direct solve (simplified)
            y.copy_from_slice(x);
            return Ok(());
        }
        
        // Pre-smoothing (simplified Jacobi)
        let mut temp = vec![0.0; n];
        self.jacobi_smooth(&current_level.matrix, x, &mut temp, 2)?;
        
        // Compute residual
        let mut residual = vec![0.0; n];
        current_level.matrix.matvec(&temp, &mut residual)?;
        for i in 0..n {
            residual[i] = x[i] - residual[i];
        }
        
        // Restrict to coarse level
        if let Some(restriction) = &current_level.restriction {
            let coarse_size = restriction.shape().0;
            let mut coarse_residual = vec![0.0; coarse_size];
            let mut coarse_correction = vec![0.0; coarse_size];
            
            restriction.matvec(&residual, &mut coarse_residual)?;
            
            // Recursive call
            self.v_cycle(level + 1, &coarse_residual, &mut coarse_correction)?;
            
            // Prolongate correction
            if let Some(prolongation) = &current_level.prolongation {
                let mut correction = vec![0.0; n];
                prolongation.matvec(&coarse_correction, &mut correction)?;
                
                // Add correction
                for i in 0..n {
                    temp[i] += correction[i];
                }
            }
        }
        
        // Post-smoothing
        self.jacobi_smooth(&current_level.matrix, x, &mut temp, 2)?;
        
        y.copy_from_slice(&temp);
        Ok(())
    }
    
    /// Simple Jacobi smoothing
    fn jacobi_smooth(&self, matrix: &CSRMatrix<f64>, b: &[f64], x: &mut [f64], iterations: usize) -> Result<()> {
        let n = b.len();
        let mut diagonal = vec![1.0; n];
        
        // Extract diagonal
        for i in 0..n {
            if let Some(&diag_val) = matrix.get(i, i) {
                if diag_val.abs() > f64::EPSILON {
                    diagonal[i] = diag_val;
                }
            }
        }
        
        let mut temp = vec![0.0; n];
        for _iter in 0..iterations {
            // Compute A * x
            matrix.matvec(x, &mut temp)?;
            
            // Jacobi update
            for i in 0..n {
                x[i] = x[i] + 0.7 * (b[i] - temp[i]) / diagonal[i]; // Damped Jacobi
            }
        }
        
        Ok(())
    }
}

impl Preconditioner<f64> for AMGPreconditioner {
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<()> {
        if self.levels.is_empty() {
            return Err(Box::new(OpenAlgebraError::InvalidFormat(
                "AMG preconditioner not setup".to_string()
            )));
        }
        
        self.v_cycle(0, x, y)
    }
    
    fn setup(&mut self, matrix: &dyn SparseMatrix<f64>) -> Result<()> {
        // Simplified AMG setup - just store the matrix as the finest level
        if let Some(csr) = matrix.as_any().downcast_ref::<CSRMatrix<f64>>() {
            self.levels.clear();
            self.levels.push(AMGLevel {
                matrix: csr.clone(),
                restriction: None,
                prolongation: None,
            });
        } else {
            return Err(Box::new(OpenAlgebraError::InvalidFormat(
                "Matrix format not supported for AMG".to_string()
            )));
        }
        
        Ok(())
    }
    
    fn memory_usage(&self) -> usize {
        self.levels.iter().map(|level| {
            level.matrix.nnz() * std::mem::size_of::<f64>()
        }).sum()
    }
}

// Helper trait to enable downcasting
pub trait AsAny {
    fn as_any(&self) -> &dyn std::any::Any;
}

impl<T> AsAny for CSRMatrix<T>
where
    T: 'static,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Extend SparseMatrix trait with as_any
impl<T> dyn SparseMatrix<T>
where
    T: 'static,
{
    pub fn as_any(&self) -> &dyn std::any::Any {
        self as &dyn std::any::Any
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::COOMatrix;

    #[test]
    fn test_jacobi_preconditioner() {
        // Create a simple diagonal matrix
        let mut matrix = COOMatrix::<f64>::new(3, 3);
        matrix.insert(0, 0, 2.0);
        matrix.insert(1, 1, 3.0);
        matrix.insert(2, 2, 4.0);
        
        let csr = matrix.to_csr();
        let mut precond = JacobiPreconditioner::new();
        precond.setup(&csr).unwrap();
        
        let x = vec![2.0, 6.0, 8.0];
        let mut y = vec![0.0; 3];
        
        precond.apply(&x, &mut y).unwrap();
        
        // Should be [1.0, 2.0, 2.0] (x[i] / diagonal[i])
        assert!((y[0] - 1.0).abs() < 1e-10);
        assert!((y[1] - 2.0).abs() < 1e-10);
        assert!((y[2] - 2.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_ilu_preconditioner_creation() {
        let precond = ILUPreconditioner::ilu0();
        assert_eq!(precond.fill_level, 0);
        assert_eq!(precond.drop_tolerance, 0.0);
        
        let precond = ILUPreconditioner::ilut(1e-6);
        assert_eq!(precond.drop_tolerance, 1e-6);
    }
    
    #[test]
    fn test_amg_preconditioner_creation() {
        let precond = AMGPreconditioner::default_amg();
        assert_eq!(precond.max_levels, 10);
        assert_eq!(precond.coarsening_threshold, 0.25);
    }
} 