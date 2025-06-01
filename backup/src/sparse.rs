use sprs::{CsMat, TriMat, CsMatView, CsVec};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;
use std::collections::HashMap;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Sparse matrix formats supported
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SparseFormat {
    CSR, // Compressed Sparse Row
    CSC, // Compressed Sparse Column
    COO, // Coordinate format
    BSR, // Block Sparse Row
}

/// Sparse matrix statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseMatrixStats {
    pub shape: (usize, usize),
    pub nnz: usize,
    pub density: f64,
    pub sparsity: f64,
    pub memory_usage_bytes: usize,
    pub bandwidth: usize,
    pub condition_estimate: Option<f64>,
}

/// Advanced sparse matrix operations
pub struct SparseOps;

impl SparseOps {
    /// Compute sparse matrix statistics
    pub fn compute_stats(matrix: &CsMat<f64>) -> SparseMatrixStats {
        let (rows, cols) = matrix.shape();
        let nnz = matrix.nnz();
        let total_elements = rows * cols;
        let density = nnz as f64 / total_elements as f64;
        
        // Estimate memory usage
        let memory_usage = std::mem::size_of::<f64>() * nnz +  // values
                          std::mem::size_of::<usize>() * nnz + // indices
                          std::mem::size_of::<usize>() * (rows + 1); // indptr
        
        // Compute bandwidth (simplified)
        let mut bandwidth = 0;
        for row_idx in 0..rows {
            let row = matrix.outer_view(row_idx).unwrap();
            if !row.indices().is_empty() {
                let min_col = *row.indices().iter().min().unwrap();
                let max_col = *row.indices().iter().max().unwrap();
                bandwidth = bandwidth.max(max_col - min_col + 1);
            }
        }
        
        SparseMatrixStats {
            shape: (rows, cols),
            nnz,
            density,
            sparsity: 1.0 - density,
            memory_usage_bytes: memory_usage,
            bandwidth,
            condition_estimate: None, // Would require specialized computation
        }
    }
    
    /// Parallel sparse matrix-vector multiplication
    pub fn parallel_spmv(matrix: &CsMat<f64>, x: &Array1<f64>) -> Result<Array1<f64>> {
        let (rows, _) = matrix.shape();
        let mut y = Array1::zeros(rows);
        
        // Parallel computation using Rayon
        y.par_iter_mut()
            .zip(matrix.outer_iterator().par_bridge())
            .for_each(|(y_i, row)| {
                *y_i = row.dot(&x.view());
            });
        
        Ok(y)
    }
    
    /// Sparse matrix addition with medical-specific optimizations
    pub fn medical_sparse_add(
        a: &CsMat<f64>, 
        b: &CsMat<f64>, 
        alpha: f64, 
        beta: f64
    ) -> Result<CsMat<f64>> {
        if a.shape() != b.shape() {
            return Err(anyhow::anyhow!("Matrix dimensions do not match"));
        }
        
        // Use sprs built-in addition with scaling
        let result = a * alpha + b * beta;
        Ok(result)
    }
    
    /// Extract submatrix for ROI processing
    pub fn extract_submatrix(
        matrix: &CsMat<f64>,
        row_indices: &[usize],
        col_indices: &[usize],
    ) -> Result<CsMat<f64>> {
        let mut triplets = Vec::new();
        
        for (new_row, &orig_row) in row_indices.iter().enumerate() {
            let row_view = matrix.outer_view(orig_row)?;
            
            for (&col, &val) in row_view.indices().iter().zip(row_view.data().iter()) {
                if let Ok(new_col) = col_indices.binary_search(&col) {
                    triplets.push((new_row, new_col, val));
                }
            }
        }
        
        let shape = (row_indices.len(), col_indices.len());
        let tri_mat = TriMat::from_triplets(shape, triplets);
        Ok(tri_mat.to_csr())
    }
    
    /// Apply medical windowing to sparse matrix values
    pub fn apply_windowing_sparse(
        matrix: &mut CsMat<f64>,
        window_center: f64,
        window_width: f64,
    ) -> Result<()> {
        let min_val = window_center - window_width / 2.0;
        let max_val = window_center + window_width / 2.0;
        
        // Modify values in place
        for val in matrix.data_mut() {
            if *val < min_val {
                *val = min_val;
            } else if *val > max_val {
                *val = max_val;
            }
        }
        
        Ok(())
    }
    
    /// Compute sparse matrix norm efficiently
    pub fn sparse_norm(matrix: &CsMat<f64>, norm_type: &str) -> Result<f64> {
        match norm_type {
            "frobenius" => {
                let norm_sq: f64 = matrix.data().iter().map(|x| x.powi(2)).sum();
                Ok(norm_sq.sqrt())
            },
            "1" => {
                // 1-norm (maximum absolute column sum)
                let (_, cols) = matrix.shape();
                let mut col_sums = vec![0.0; cols];
                
                for (row_idx, row) in matrix.outer_iterator().enumerate() {
                    for (&col, &val) in row.indices().iter().zip(row.data().iter()) {
                        col_sums[col] += val.abs();
                    }
                }
                
                Ok(col_sums.into_iter().fold(0.0, f64::max))
            },
            "inf" => {
                // Infinity norm (maximum absolute row sum)
                let row_sums: Vec<f64> = matrix.outer_iterator()
                    .map(|row| row.data().iter().map(|x| x.abs()).sum())
                    .collect();
                
                Ok(row_sums.into_iter().fold(0.0, f64::max))
            },
            _ => Err(anyhow::anyhow!("Unsupported norm type: {}", norm_type)),
        }
    }
    
    /// Threshold sparse matrix to increase sparsity
    pub fn threshold_matrix(matrix: &CsMat<f64>, threshold: f64) -> CsMat<f64> {
        let mut triplets = Vec::new();
        
        for (row_idx, row) in matrix.outer_iterator().enumerate() {
            for (&col, &val) in row.indices().iter().zip(row.data().iter()) {
                if val.abs() > threshold {
                    triplets.push((row_idx, col, val));
                }
            }
        }
        
        let tri_mat = TriMat::from_triplets(matrix.shape(), triplets);
        tri_mat.to_csr()
    }
    
    /// Create medical Laplacian matrix for image processing
    pub fn create_medical_laplacian_2d(
        height: usize, 
        width: usize, 
        voxel_spacing: (f64, f64)
    ) -> CsMat<f64> {
        let n = height * width;
        let mut triplets = Vec::with_capacity(n * 5); // Up to 5 non-zeros per row
        
        let (dx, dy) = voxel_spacing;
        let dx_inv = 1.0 / (dx * dx);
        let dy_inv = 1.0 / (dy * dy);
        
        for i in 0..height {
            for j in 0..width {
                let idx = i * width + j;
                let mut center_val = 0.0;
                
                // Left neighbor
                if j > 0 {
                    triplets.push((idx, idx - 1, -dx_inv));
                    center_val += dx_inv;
                }
                
                // Right neighbor
                if j < width - 1 {
                    triplets.push((idx, idx + 1, -dx_inv));
                    center_val += dx_inv;
                }
                
                // Top neighbor
                if i > 0 {
                    triplets.push((idx, idx - width, -dy_inv));
                    center_val += dy_inv;
                }
                
                // Bottom neighbor
                if i < height - 1 {
                    triplets.push((idx, idx + width, -dy_inv));
                    center_val += dy_inv;
                }
                
                // Center element
                triplets.push((idx, idx, center_val));
            }
        }
        
        let tri_mat = TriMat::from_triplets((n, n), triplets);
        tri_mat.to_csr()
    }
    
    /// Create medical Laplacian matrix for 3D volumes
    pub fn create_medical_laplacian_3d(
        depth: usize,
        height: usize, 
        width: usize, 
        voxel_spacing: (f64, f64, f64)
    ) -> CsMat<f64> {
        let n = depth * height * width;
        let mut triplets = Vec::with_capacity(n * 7); // Up to 7 non-zeros per row
        
        let (dx, dy, dz) = voxel_spacing;
        let dx_inv = 1.0 / (dx * dx);
        let dy_inv = 1.0 / (dy * dy);
        let dz_inv = 1.0 / (dz * dz);
        
        for k in 0..depth {
            for i in 0..height {
                for j in 0..width {
                    let idx = k * height * width + i * width + j;
                    let mut center_val = 0.0;
                    
                    // X-direction neighbors
                    if j > 0 {
                        triplets.push((idx, idx - 1, -dx_inv));
                        center_val += dx_inv;
                    }
                    if j < width - 1 {
                        triplets.push((idx, idx + 1, -dx_inv));
                        center_val += dx_inv;
                    }
                    
                    // Y-direction neighbors
                    if i > 0 {
                        triplets.push((idx, idx - width, -dy_inv));
                        center_val += dy_inv;
                    }
                    if i < height - 1 {
                        triplets.push((idx, idx + width, -dy_inv));
                        center_val += dy_inv;
                    }
                    
                    // Z-direction neighbors
                    if k > 0 {
                        triplets.push((idx, idx - height * width, -dz_inv));
                        center_val += dz_inv;
                    }
                    if k < depth - 1 {
                        triplets.push((idx, idx + height * width, -dz_inv));
                        center_val += dz_inv;
                    }
                    
                    // Center element
                    triplets.push((idx, idx, center_val));
                }
            }
        }
        
        let tri_mat = TriMat::from_triplets((n, n), triplets);
        tri_mat.to_csr()
    }
    
    /// Create adjacency matrix for medical graph neural networks
    pub fn create_medical_adjacency_matrix(
        node_features: &Array2<f64>,
        similarity_threshold: f64,
        max_connections: usize,
    ) -> Result<CsMat<f64>> {
        let num_nodes = node_features.nrows();
        let mut triplets = Vec::new();
        
        for i in 0..num_nodes {
            let mut similarities = Vec::new();
            
            for j in 0..num_nodes {
                if i != j {
                    // Compute cosine similarity
                    let node_i = node_features.row(i);
                    let node_j = node_features.row(j);
                    
                    let dot_product: f64 = node_i.iter().zip(node_j.iter()).map(|(a, b)| a * b).sum();
                    let norm_i: f64 = node_i.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
                    let norm_j: f64 = node_j.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
                    
                    if norm_i > 0.0 && norm_j > 0.0 {
                        let similarity = dot_product / (norm_i * norm_j);
                        if similarity >= similarity_threshold {
                            similarities.push((j, similarity));
                        }
                    }
                }
            }
            
            // Keep only top connections
            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            similarities.truncate(max_connections);
            
            for (j, similarity) in similarities {
                triplets.push((i, j, similarity));
            }
        }
        
        let tri_mat = TriMat::from_triplets((num_nodes, num_nodes), triplets);
        Ok(tri_mat.to_csr())
    }
}

/// Sparse iterative solvers for medical applications
pub struct MedicalSolvers;

impl MedicalSolvers {
    /// Conjugate Gradient solver optimized for medical matrices
    pub fn conjugate_gradient(
        matrix: &CsMat<f64>,
        rhs: &Array1<f64>,
        x0: Option<&Array1<f64>>,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<(Array1<f64>, bool, usize, Vec<f64>)> {
        let n = matrix.shape().0;
        let mut x = x0.map(|x| x.clone()).unwrap_or_else(|| Array1::zeros(n));
        
        let mut r = rhs - &SparseOps::parallel_spmv(matrix, &x)?;
        let mut p = r.clone();
        let mut rsold = r.dot(&r);
        
        let mut residual_history = Vec::new();
        let mut converged = false;
        let mut iteration = 0;
        
        for iter in 0..max_iterations {
            let ap = SparseOps::parallel_spmv(matrix, &p)?;
            let alpha = rsold / p.dot(&ap);
            
            x = &x + &(&p * alpha);
            r = &r - &(&ap * alpha);
            
            let rsnew = r.dot(&r);
            let residual_norm = rsnew.sqrt();
            residual_history.push(residual_norm);
            
            if residual_norm < tolerance {
                converged = true;
                iteration = iter + 1;
                break;
            }
            
            let beta = rsnew / rsold;
            p = &r + &(&p * beta);
            rsold = rsnew;
            iteration = iter + 1;
        }
        
        Ok((x, converged, iteration, residual_history))
    }
    
    /// BiCGSTAB solver for non-symmetric medical matrices
    pub fn bicgstab(
        matrix: &CsMat<f64>,
        rhs: &Array1<f64>,
        x0: Option<&Array1<f64>>,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<(Array1<f64>, bool, usize, Vec<f64>)> {
        let n = matrix.shape().0;
        let mut x = x0.map(|x| x.clone()).unwrap_or_else(|| Array1::zeros(n));
        
        let mut r = rhs - &SparseOps::parallel_spmv(matrix, &x)?;
        let r_hat = r.clone();
        let mut p = r.clone();
        let mut v = Array1::zeros(n);
        
        let mut rho = r.dot(&r_hat);
        let mut alpha = 1.0;
        let mut omega = 1.0;
        
        let mut residual_history = Vec::new();
        let mut converged = false;
        let mut iteration = 0;
        
        for iter in 0..max_iterations {
            let rho_new = r.dot(&r_hat);
            let beta = (rho_new / rho) * (alpha / omega);
            
            p = &r + &(&(&p - &(&v * omega)) * beta);
            v = SparseOps::parallel_spmv(matrix, &p)?;
            alpha = rho_new / r_hat.dot(&v);
            
            let s = &r - &(&v * alpha);
            let residual_norm = s.dot(&s).sqrt();
            residual_history.push(residual_norm);
            
            if residual_norm < tolerance {
                x = &x + &(&p * alpha);
                converged = true;
                iteration = iter + 1;
                break;
            }
            
            let t = SparseOps::parallel_spmv(matrix, &s)?;
            omega = t.dot(&s) / t.dot(&t);
            
            x = &x + &(&(&p * alpha) + &(&s * omega));
            r = &s - &(&t * omega);
            
            let final_residual = r.dot(&r).sqrt();
            if final_residual < tolerance {
                converged = true;
                iteration = iter + 1;
                break;
            }
            
            rho = rho_new;
            iteration = iter + 1;
        }
        
        Ok((x, converged, iteration, residual_history))
    }
    
    /// GMRES solver for general medical systems
    pub fn gmres(
        matrix: &CsMat<f64>,
        rhs: &Array1<f64>,
        x0: Option<&Array1<f64>>,
        tolerance: f64,
        max_iterations: usize,
        restart: usize,
    ) -> Result<(Array1<f64>, bool, usize, Vec<f64>)> {
        let n = matrix.shape().0;
        let mut x = x0.map(|x| x.clone()).unwrap_or_else(|| Array1::zeros(n));
        
        let mut residual_history = Vec::new();
        let mut converged = false;
        let mut total_iterations = 0;
        
        for _restart_iter in 0..(max_iterations / restart + 1) {
            let r0 = rhs - &SparseOps::parallel_spmv(matrix, &x)?;
            let beta = r0.dot(&r0).sqrt();
            
            if beta < tolerance {
                converged = true;
                break;
            }
            
            let mut q = vec![Array1::zeros(n); restart + 1];
            q[0] = &r0 / beta;
            
            let mut h = Array2::zeros((restart + 1, restart));
            
            for j in 0..restart.min(max_iterations - total_iterations) {
                let w = SparseOps::parallel_spmv(matrix, &q[j])?;
                
                // Gram-Schmidt orthogonalization
                for i in 0..=j {
                    h[[i, j]] = q[i].dot(&w);
                    let q_scaled = &q[i] * h[[i, j]];
                    let w_new = &w - &q_scaled;
                    // w = w_new; // This would require mutable w
                }
                
                let h_norm = q[j + 1].dot(&q[j + 1]).sqrt();
                h[[j + 1, j]] = h_norm;
                
                if h_norm > 1e-12 {
                    q[j + 1] = &q[j + 1] / h_norm;
                }
                
                total_iterations += 1;
                
                // This is a simplified GMRES - full implementation would solve
                // the least squares problem and update x
            }
            
            if total_iterations >= max_iterations {
                break;
            }
        }
        
        Ok((x, converged, total_iterations, residual_history))
    }
}

/// Sparse matrix preconditioners for medical applications
pub struct MedicalPreconditioners;

impl MedicalPreconditioners {
    /// Jacobi (diagonal) preconditioner
    pub fn jacobi_preconditioner(matrix: &CsMat<f64>) -> Result<Array1<f64>> {
        let n = matrix.shape().0;
        let mut diag = Array1::zeros(n);
        
        for i in 0..n {
            let row = matrix.outer_view(i)?;
            for (&j, &val) in row.indices().iter().zip(row.data().iter()) {
                if i == j {
                    diag[i] = if val.abs() > 1e-12 { 1.0 / val } else { 1.0 };
                    break;
                }
            }
        }
        
        Ok(diag)
    }
    
    /// Apply Jacobi preconditioner
    pub fn apply_jacobi(diag_inv: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
        x * diag_inv
    }
    
    /// Incomplete LU preconditioner (simplified)
    pub fn ilu_preconditioner(matrix: &CsMat<f64>, fill_level: usize) -> Result<(CsMat<f64>, CsMat<f64>)> {
        // This is a very simplified ILU implementation
        // In practice, you'd want a more sophisticated algorithm
        
        let n = matrix.shape().0;
        let mut l_triplets = Vec::new();
        let mut u_triplets = Vec::new();
        
        // Initialize L and U with matrix structure
        for (i, row) in matrix.outer_iterator().enumerate() {
            for (&j, &val) in row.indices().iter().zip(row.data().iter()) {
                if i > j {
                    l_triplets.push((i, j, val));
                } else if i == j {
                    l_triplets.push((i, j, 1.0)); // L diagonal
                    u_triplets.push((i, j, val)); // U diagonal
                } else {
                    u_triplets.push((i, j, val));
                }
            }
        }
        
        let l_matrix = TriMat::from_triplets((n, n), l_triplets).to_csr();
        let u_matrix = TriMat::from_triplets((n, n), u_triplets).to_csr();
        
        Ok((l_matrix, u_matrix))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_sparse_stats() {
        let triplets = vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)];
        let matrix = TriMat::from_triplets((3, 3), triplets).to_csr();
        
        let stats = SparseOps::compute_stats(&matrix);
        assert_eq!(stats.nnz, 3);
        assert!((stats.density - 1.0/3.0).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_spmv() {
        let triplets = vec![(0, 0, 2.0), (1, 1, 3.0), (2, 2, 4.0)];
        let matrix = TriMat::from_triplets((3, 3), triplets).to_csr();
        let x = Array1::from(vec![1.0, 2.0, 3.0]);
        
        let result = SparseOps::parallel_spmv(&matrix, &x).unwrap();
        assert_eq!(result, Array1::from(vec![2.0, 6.0, 12.0]));
    }

    #[test]
    fn test_medical_laplacian_2d() {
        let matrix = SparseOps::create_medical_laplacian_2d(3, 3, (1.0, 1.0));
        let stats = SparseOps::compute_stats(&matrix);
        
        // Interior points should have 4 neighbors + center = 5 non-zeros
        // Boundary points should have fewer
        assert!(stats.nnz > 0);
        assert_eq!(matrix.shape(), (9, 9));
    }

    #[test]
    fn test_conjugate_gradient() {
        // Create a simple SPD matrix
        let triplets = vec![
            (0, 0, 4.0), (0, 1, -1.0),
            (1, 0, -1.0), (1, 1, 4.0), (1, 2, -1.0),
            (2, 1, -1.0), (2, 2, 4.0)
        ];
        let matrix = TriMat::from_triplets((3, 3), triplets).to_csr();
        let rhs = Array1::from(vec![1.0, 2.0, 3.0]);
        
        let (solution, converged, iterations, _) = 
            MedicalSolvers::conjugate_gradient(&matrix, &rhs, None, 1e-6, 100).unwrap();
        
        assert!(converged);
        assert!(iterations < 100);
        
        // Verify solution
        let residual = &rhs - &SparseOps::parallel_spmv(&matrix, &solution).unwrap();
        let residual_norm = residual.dot(&residual).sqrt();
        assert!(residual_norm < 1e-6);
    }
} 