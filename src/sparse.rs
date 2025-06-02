/*!
# Sparse Matrix Operations

This module provides efficient sparse matrix representations and operations
including COO (Coordinate), CSR (Compressed Sparse Row), and CSC (Compressed Sparse Column) formats.
*/

use std::collections::HashMap;
use std::fmt::Debug;
use crate::{Result, OpenAlgebraError};

/// Trait for sparse matrix operations
pub trait SparseMatrix<T> {
    /// Get matrix dimensions (rows, cols)
    fn shape(&self) -> (usize, usize);
    
    /// Get number of non-zero elements
    fn nnz(&self) -> usize;
    
    /// Get element at position (row, col)
    fn get(&self, row: usize, col: usize) -> Option<&T>;
    
    /// Matrix-vector multiplication: y = A * x
    fn matvec(&self, x: &[T], y: &mut [T]) -> Result<()>;
    
    /// Transpose matrix
    fn transpose(&self) -> Result<Box<dyn SparseMatrix<T>>>;
}

/// COO (Coordinate) sparse matrix format
#[derive(Debug, Clone)]
pub struct COOMatrix<T> {
    rows: usize,
    cols: usize,
    row_indices: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<T>,
}

impl<T> COOMatrix<T>
where
    T: Clone + Default + PartialEq + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    /// Create new COO matrix with given dimensions
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
        }
    }
    
    /// Insert element at position (row, col)
    pub fn insert(&mut self, row: usize, col: usize, value: T) {
        if row >= self.rows || col >= self.cols {
            panic!("Index out of bounds");
        }
        
        self.row_indices.push(row);
        self.col_indices.push(col);
        self.values.push(value);
    }
    
    /// Convert to CSR format
    pub fn to_csr(self) -> CSRMatrix<T> {
        let mut csr = CSRMatrix::new(self.rows, self.cols);
        
        // Create a map to accumulate values for duplicate entries
        let mut entries: HashMap<(usize, usize), T> = HashMap::new();
        
        for ((row, col), value) in self.row_indices.iter()
            .zip(self.col_indices.iter())
            .zip(self.values.iter()) 
        {
            let key = (*row, *col);
            if let Some(existing) = entries.get_mut(&key) {
                *existing = existing.clone() + value.clone();
            } else {
                entries.insert(key, value.clone());
            }
        }
        
        // Sort entries by row, then by column
        let mut sorted_entries: Vec<_> = entries.into_iter().collect();
        sorted_entries.sort_by_key(|(k, _)| *k);
        
        // Build CSR structure
        csr.row_ptr.resize(self.rows + 1, 0);
        
        for ((row, col), value) in sorted_entries {
            csr.col_indices.push(col);
            csr.values.push(value);
            
            // Update row pointers
            for r in (row + 1)..=self.rows {
                csr.row_ptr[r] += 1;
            }
        }
        
        csr
    }
    
    /// Convert to CSC format
    pub fn to_csc(self) -> CSCMatrix<T> {
        let mut csc = CSCMatrix::new(self.rows, self.cols);
        
        // Create a map to accumulate values for duplicate entries
        let mut entries: HashMap<(usize, usize), T> = HashMap::new();
        
        for ((row, col), value) in self.row_indices.iter()
            .zip(self.col_indices.iter())
            .zip(self.values.iter()) 
        {
            let key = (*row, *col);
            if let Some(existing) = entries.get_mut(&key) {
                *existing = existing.clone() + value.clone();
            } else {
                entries.insert(key, value.clone());
            }
        }
        
        // Sort entries by column, then by row
        let mut sorted_entries: Vec<_> = entries.into_iter().collect();
        sorted_entries.sort_by_key(|((row, col), _)| (*col, *row));
        
        // Build CSC structure
        csc.col_ptr.resize(self.cols + 1, 0);
        
        for ((row, col), value) in sorted_entries {
            csc.row_indices.push(row);
            csc.values.push(value);
            
            // Update column pointers
            for c in (col + 1)..=self.cols {
                csc.col_ptr[c] += 1;
            }
        }
        
        csc
    }
}

impl<T> SparseMatrix<T> for COOMatrix<T>
where
    T: Clone + Default + PartialEq + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    
    fn nnz(&self) -> usize {
        self.values.len()
    }
    
    fn get(&self, row: usize, col: usize) -> Option<&T> {
        for i in 0..self.values.len() {
            if self.row_indices[i] == row && self.col_indices[i] == col {
                return Some(&self.values[i]);
            }
        }
        None
    }
    
    fn matvec(&self, x: &[T], y: &mut [T]) -> Result<()> {
        if x.len() != self.cols {
            return Err(Box::new(OpenAlgebraError::DimensionMismatch {
                expected: format!("{}", self.cols),
                actual: format!("{}", x.len()),
            }));
        }
        
        if y.len() != self.rows {
            return Err(Box::new(OpenAlgebraError::DimensionMismatch {
                expected: format!("{}", self.rows),
                actual: format!("{}", y.len()),
            }));
        }
        
        // Initialize y to zero
        for i in 0..y.len() {
            y[i] = T::default();
        }
        
        // Compute y = A * x
        for i in 0..self.values.len() {
            let row = self.row_indices[i];
            let col = self.col_indices[i];
            y[row] = y[row].clone() + (self.values[i].clone() * x[col].clone());
        }
        
        Ok(())
    }
    
    fn transpose(&self) -> Result<Box<dyn SparseMatrix<T>>> {
        let mut transposed = COOMatrix::new(self.cols, self.rows);
        
        for i in 0..self.values.len() {
            transposed.insert(
                self.col_indices[i],
                self.row_indices[i],
                self.values[i].clone(),
            );
        }
        
        Ok(Box::new(transposed))
    }
}

/// CSR (Compressed Sparse Row) matrix format
#[derive(Debug, Clone)]
pub struct CSRMatrix<T> {
    rows: usize,
    cols: usize,
    row_ptr: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<T>,
}

impl<T> CSRMatrix<T>
where
    T: Clone + Default + PartialEq + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    /// Create new CSR matrix with given dimensions
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            row_ptr: vec![0; rows + 1],
            col_indices: Vec::new(),
            values: Vec::new(),
        }
    }
    
    /// Get row slice for given row index
    pub fn row(&self, row: usize) -> Option<(&[usize], &[T])> {
        if row >= self.rows {
            return None;
        }
        
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        
        Some((
            &self.col_indices[start..end],
            &self.values[start..end],
        ))
    }
}

impl<T> SparseMatrix<T> for CSRMatrix<T>
where
    T: Clone + Default + PartialEq + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    
    fn nnz(&self) -> usize {
        self.values.len()
    }
    
    fn get(&self, row: usize, col: usize) -> Option<&T> {
        if let Some((cols, vals)) = self.row(row) {
            for (i, &c) in cols.iter().enumerate() {
                if c == col {
                    return Some(&vals[i]);
                }
            }
        }
        None
    }
    
    fn matvec(&self, x: &[T], y: &mut [T]) -> Result<()> {
        if x.len() != self.cols {
            return Err(Box::new(OpenAlgebraError::DimensionMismatch {
                expected: format!("{}", self.cols),
                actual: format!("{}", x.len()),
            }));
        }
        
        if y.len() != self.rows {
            return Err(Box::new(OpenAlgebraError::DimensionMismatch {
                expected: format!("{}", self.rows),
                actual: format!("{}", y.len()),
            }));
        }
        
        // Efficient CSR matrix-vector multiplication
        for row in 0..self.rows {
            let mut sum = T::default();
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            
            for i in start..end {
                let col = self.col_indices[i];
                sum = sum + (self.values[i].clone() * x[col].clone());
            }
            
            y[row] = sum;
        }
        
        Ok(())
    }
    
    fn transpose(&self) -> Result<Box<dyn SparseMatrix<T>>> {
        // Convert to COO, transpose, then back to CSR
        let mut coo = COOMatrix::new(self.cols, self.rows);
        
        for row in 0..self.rows {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            
            for i in start..end {
                let col = self.col_indices[i];
                coo.insert(col, row, self.values[i].clone());
            }
        }
        
        Ok(Box::new(coo.to_csr()))
    }
}

/// CSC (Compressed Sparse Column) matrix format
#[derive(Debug, Clone)]
pub struct CSCMatrix<T> {
    rows: usize,
    cols: usize,
    col_ptr: Vec<usize>,
    row_indices: Vec<usize>,
    values: Vec<T>,
}

impl<T> CSCMatrix<T>
where
    T: Clone + Default + PartialEq + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    /// Create new CSC matrix with given dimensions
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            col_ptr: vec![0; cols + 1],
            row_indices: Vec::new(),
            values: Vec::new(),
        }
    }
    
    /// Get column slice for given column index
    pub fn col(&self, col: usize) -> Option<(&[usize], &[T])> {
        if col >= self.cols {
            return None;
        }
        
        let start = self.col_ptr[col];
        let end = self.col_ptr[col + 1];
        
        Some((
            &self.row_indices[start..end],
            &self.values[start..end],
        ))
    }
}

impl<T> SparseMatrix<T> for CSCMatrix<T>
where
    T: Clone + Default + PartialEq + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    
    fn nnz(&self) -> usize {
        self.values.len()
    }
    
    fn get(&self, row: usize, col: usize) -> Option<&T> {
        if let Some((rows, vals)) = self.col(col) {
            for (i, &r) in rows.iter().enumerate() {
                if r == row {
                    return Some(&vals[i]);
                }
            }
        }
        None
    }
    
    fn matvec(&self, x: &[T], y: &mut [T]) -> Result<()> {
        if x.len() != self.cols {
            return Err(Box::new(OpenAlgebraError::DimensionMismatch {
                expected: format!("{}", self.cols),
                actual: format!("{}", x.len()),
            }));
        }
        
        if y.len() != self.rows {
            return Err(Box::new(OpenAlgebraError::DimensionMismatch {
                expected: format!("{}", self.rows),
                actual: format!("{}", y.len()),
            }));
        }
        
        // Initialize y to zero
        for i in 0..y.len() {
            y[i] = T::default();
        }
        
        // CSC matrix-vector multiplication
        for col in 0..self.cols {
            let x_val = x[col].clone();
            let start = self.col_ptr[col];
            let end = self.col_ptr[col + 1];
            
            for i in start..end {
                let row = self.row_indices[i];
                y[row] = y[row].clone() + (self.values[i].clone() * x_val.clone());
            }
        }
        
        Ok(())
    }
    
    fn transpose(&self) -> Result<Box<dyn SparseMatrix<T>>> {
        // Convert to COO, transpose, then to CSR
        let mut coo = COOMatrix::new(self.cols, self.rows);
        
        for col in 0..self.cols {
            let start = self.col_ptr[col];
            let end = self.col_ptr[col + 1];
            
            for i in start..end {
                let row = self.row_indices[i];
                coo.insert(col, row, self.values[i].clone());
            }
        }
        
        Ok(Box::new(coo.to_csr()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coo_matrix_creation() {
        let mut matrix = COOMatrix::<f64>::new(3, 3);
        matrix.insert(0, 0, 1.0);
        matrix.insert(1, 1, 2.0);
        matrix.insert(2, 2, 3.0);
        
        assert_eq!(matrix.shape(), (3, 3));
        assert_eq!(matrix.nnz(), 3);
        assert_eq!(matrix.get(0, 0), Some(&1.0));
        assert_eq!(matrix.get(1, 1), Some(&2.0));
        assert_eq!(matrix.get(2, 2), Some(&3.0));
    }
    
    #[test]
    fn test_coo_to_csr_conversion() {
        let mut coo = COOMatrix::<f64>::new(2, 2);
        coo.insert(0, 0, 1.0);
        coo.insert(0, 1, 2.0);
        coo.insert(1, 0, 3.0);
        coo.insert(1, 1, 4.0);
        
        let csr = coo.to_csr();
        assert_eq!(csr.shape(), (2, 2));
        assert_eq!(csr.nnz(), 4);
    }
    
    #[test]
    fn test_matrix_vector_multiplication() {
        let mut matrix = COOMatrix::<f64>::new(2, 2);
        matrix.insert(0, 0, 1.0);
        matrix.insert(0, 1, 2.0);
        matrix.insert(1, 0, 3.0);
        matrix.insert(1, 1, 4.0);
        
        let x = vec![1.0, 1.0];
        let mut y = vec![0.0; 2];
        
        matrix.matvec(&x, &mut y).unwrap();
        assert_eq!(y, vec![3.0, 7.0]);
    }
    
    #[test]
    fn test_csr_matrix_operations() {
        let mut coo = COOMatrix::<f64>::new(3, 3);
        for i in 0..3 {
            coo.insert(i, i, 2.0);
            if i > 0 {
                coo.insert(i, i-1, -1.0);
            }
            if i < 2 {
                coo.insert(i, i+1, -1.0);
            }
        }
        
        let csr = coo.to_csr();
        let x = vec![1.0, 1.0, 1.0];
        let mut y = vec![0.0; 3];
        
        csr.matvec(&x, &mut y).unwrap();
        assert_eq!(y, vec![1.0, 0.0, 1.0]); // Tridiagonal matrix result
    }
} 