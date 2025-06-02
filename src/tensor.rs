/*!
# Sparse Tensor Operations

This module provides efficient sparse tensor representations and operations
for high-dimensional sparse data structures.
*/

use crate::{Result, OpenAlgebraError};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Trait for tensor operations
pub trait Tensor<T> {
    /// Get tensor shape
    fn shape(&self) -> &[usize];
    
    /// Get number of non-zero elements
    fn nnz(&self) -> usize;
    
    /// Get element at given indices
    fn get(&self, indices: &[usize]) -> Option<&T>;
    
    /// Set element at given indices
    fn set(&mut self, indices: &[usize], value: T) -> Result<()>;
    
    /// Contract tensor along specified axes
    fn contract(&self, axes: &[(usize, usize)]) -> Result<Box<dyn Tensor<T>>>;
}

/// Sparse tensor in COO (Coordinate) format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseTensor<T> {
    shape: Vec<usize>,
    indices: Vec<Vec<usize>>,
    values: Vec<T>,
}

impl<T> SparseTensor<T>
where
    T: Clone + Default + PartialEq + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    /// Create new sparse tensor with given shape
    pub fn new(shape: Vec<usize>) -> Self {
        Self {
            shape,
            indices: Vec::new(),
            values: Vec::new(),
        }
    }
    
    /// Create sparse tensor from dense data
    pub fn from_dense(data: &[T], shape: Vec<usize>) -> Result<Self> {
        let total_elements: usize = shape.iter().product();
        if data.len() != total_elements {
            return Err(Box::new(OpenAlgebraError::DimensionMismatch {
                expected: format!("{}", total_elements),
                actual: format!("{}", data.len()),
            }));
        }
        
        let mut tensor = Self::new(shape.clone());
        let mut flat_index = 0;
        
        // Convert flat index to multi-dimensional indices
        for (i, &value) in data.iter().enumerate() {
            if value != T::default() {
                let indices = flat_to_multi_index(i, &shape);
                tensor.indices.push(indices);
                tensor.values.push(value);
            }
            flat_index += 1;
        }
        
        Ok(tensor)
    }
    
    /// Insert element at given indices
    pub fn insert(&mut self, indices: Vec<usize>, value: T) -> Result<()> {
        if indices.len() != self.shape.len() {
            return Err(Box::new(OpenAlgebraError::DimensionMismatch {
                expected: format!("{}", self.shape.len()),
                actual: format!("{}", indices.len()),
            }));
        }
        
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return Err(Box::new(OpenAlgebraError::InvalidFormat(
                    format!("Index {} out of bounds for dimension {} of size {}", 
                            idx, i, self.shape[i])
                )));
            }
        }
        
        self.indices.push(indices);
        self.values.push(value);
        Ok(())
    }
    
    /// Convert to dense representation
    pub fn to_dense(&self) -> Vec<T> {
        let total_size: usize = self.shape.iter().product();
        let mut dense = vec![T::default(); total_size];
        
        for (indices, value) in self.indices.iter().zip(self.values.iter()) {
            let flat_index = multi_to_flat_index(indices, &self.shape);
            dense[flat_index] = value.clone();
        }
        
        dense
    }
    
    /// Element-wise addition
    pub fn add(&self, other: &SparseTensor<T>) -> Result<SparseTensor<T>> {
        if self.shape != other.shape {
            return Err(Box::new(OpenAlgebraError::DimensionMismatch {
                expected: format!("{:?}", self.shape),
                actual: format!("{:?}", other.shape),
            }));
        }
        
        let mut result = SparseTensor::new(self.shape.clone());
        let mut index_map: HashMap<Vec<usize>, T> = HashMap::new();
        
        // Add elements from self
        for (indices, value) in self.indices.iter().zip(self.values.iter()) {
            index_map.insert(indices.clone(), value.clone());
        }
        
        // Add elements from other
        for (indices, value) in other.indices.iter().zip(other.values.iter()) {
            if let Some(existing) = index_map.get_mut(indices) {
                *existing = existing.clone() + value.clone();
            } else {
                index_map.insert(indices.clone(), value.clone());
            }
        }
        
        // Build result tensor
        for (indices, value) in index_map {
            if value != T::default() {
                result.insert(indices, value)?;
            }
        }
        
        Ok(result)
    }
    
    /// Tensor product with another tensor
    pub fn tensor_product(&self, other: &SparseTensor<T>) -> Result<SparseTensor<T>> {
        let mut result_shape = self.shape.clone();
        result_shape.extend_from_slice(&other.shape);
        
        let mut result = SparseTensor::new(result_shape);
        
        for (self_indices, self_value) in self.indices.iter().zip(self.values.iter()) {
            for (other_indices, other_value) in other.indices.iter().zip(other.values.iter()) {
                let mut combined_indices = self_indices.clone();
                combined_indices.extend_from_slice(other_indices);
                
                let product_value = self_value.clone() * other_value.clone();
                result.insert(combined_indices, product_value)?;
            }
        }
        
        Ok(result)
    }
    
    /// Reshape tensor (preserving total number of elements)
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<SparseTensor<T>> {
        let old_size: usize = self.shape.iter().product();
        let new_size: usize = new_shape.iter().product();
        
        if old_size != new_size {
            return Err(Box::new(OpenAlgebraError::DimensionMismatch {
                expected: format!("{}", old_size),
                actual: format!("{}", new_size),
            }));
        }
        
        let mut result = SparseTensor::new(new_shape.clone());
        
        for (indices, value) in self.indices.iter().zip(self.values.iter()) {
            let flat_index = multi_to_flat_index(indices, &self.shape);
            let new_indices = flat_to_multi_index(flat_index, &new_shape);
            result.insert(new_indices, value.clone())?;
        }
        
        Ok(result)
    }
    
    /// Sum along specified axis
    pub fn sum_axis(&self, axis: usize) -> Result<SparseTensor<T>> {
        if axis >= self.shape.len() {
            return Err(Box::new(OpenAlgebraError::InvalidFormat(
                format!("Axis {} out of bounds for tensor with {} dimensions", 
                        axis, self.shape.len())
            )));
        }
        
        let mut result_shape = self.shape.clone();
        result_shape.remove(axis);
        
        if result_shape.is_empty() {
            result_shape.push(1); // Scalar result
        }
        
        let mut result = SparseTensor::new(result_shape.clone());
        let mut sum_map: HashMap<Vec<usize>, T> = HashMap::new();
        
        for (indices, value) in self.indices.iter().zip(self.values.iter()) {
            let mut result_indices = indices.clone();
            result_indices.remove(axis);
            
            if result_indices.is_empty() {
                result_indices.push(0); // Scalar case
            }
            
            if let Some(existing) = sum_map.get_mut(&result_indices) {
                *existing = existing.clone() + value.clone();
            } else {
                sum_map.insert(result_indices, value.clone());
            }
        }
        
        for (indices, value) in sum_map {
            if value != T::default() {
                result.insert(indices, value)?;
            }
        }
        
        Ok(result)
    }
}

impl<T> Tensor<T> for SparseTensor<T>
where
    T: Clone + Default + PartialEq + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    fn nnz(&self) -> usize {
        self.values.len()
    }
    
    fn get(&self, indices: &[usize]) -> Option<&T> {
        for (i, stored_indices) in self.indices.iter().enumerate() {
            if stored_indices == indices {
                return Some(&self.values[i]);
            }
        }
        None
    }
    
    fn set(&mut self, indices: &[usize], value: T) -> Result<()> {
        // Check if element already exists
        for (i, stored_indices) in self.indices.iter().enumerate() {
            if stored_indices == indices {
                self.values[i] = value;
                return Ok(());
            }
        }
        
        // Insert new element
        self.insert(indices.to_vec(), value)
    }
    
    fn contract(&self, axes: &[(usize, usize)]) -> Result<Box<dyn Tensor<T>>> {
        // Simple implementation for tensor contraction
        // This is a simplified version - a full implementation would be more complex
        let mut result_shape = self.shape.clone();
        
        // Remove contracted axes (in reverse order to maintain indices)
        let mut axes_to_remove: Vec<_> = axes.iter().flat_map(|(a, b)| vec![*a, *b]).collect();
        axes_to_remove.sort_unstable();
        axes_to_remove.reverse();
        
        for &axis in &axes_to_remove {
            if axis < result_shape.len() {
                result_shape.remove(axis);
            }
        }
        
        if result_shape.is_empty() {
            result_shape.push(1); // Scalar result
        }
        
        let result = SparseTensor::new(result_shape);
        Ok(Box::new(result))
    }
}

/// Dense tensor for comparison and conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseTensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
}

impl<T> DenseTensor<T>
where
    T: Clone + Default,
{
    /// Create new dense tensor with given shape
    pub fn new(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![T::default(); size],
            shape,
        }
    }
    
    /// Create dense tensor from data
    pub fn from_data(data: Vec<T>, shape: Vec<usize>) -> Result<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(Box::new(OpenAlgebraError::DimensionMismatch {
                expected: format!("{}", expected_size),
                actual: format!("{}", data.len()),
            }));
        }
        
        Ok(Self { data, shape })
    }
    
    /// Convert to sparse tensor
    pub fn to_sparse(&self) -> SparseTensor<T>
    where
        T: PartialEq,
    {
        let mut sparse = SparseTensor::new(self.shape.clone());
        
        for (i, value) in self.data.iter().enumerate() {
            if *value != T::default() {
                let indices = flat_to_multi_index(i, &self.shape);
                sparse.insert(indices, value.clone()).unwrap();
            }
        }
        
        sparse
    }
}

impl<T> Tensor<T> for DenseTensor<T>
where
    T: Clone + Default,
{
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    fn nnz(&self) -> usize {
        self.data.iter().filter(|&x| *x != T::default()).count()
    }
    
    fn get(&self, indices: &[usize]) -> Option<&T> {
        let flat_index = multi_to_flat_index(indices, &self.shape);
        self.data.get(flat_index)
    }
    
    fn set(&mut self, indices: &[usize], value: T) -> Result<()> {
        let flat_index = multi_to_flat_index(indices, &self.shape);
        if flat_index < self.data.len() {
            self.data[flat_index] = value;
            Ok(())
        } else {
            Err(Box::new(OpenAlgebraError::InvalidFormat(
                "Index out of bounds".to_string()
            )))
        }
    }
    
    fn contract(&self, _axes: &[(usize, usize)]) -> Result<Box<dyn Tensor<T>>> {
        // Simplified implementation
        let result = DenseTensor::new(vec![1]);
        Ok(Box::new(result))
    }
}

/// Convert flat index to multi-dimensional indices
fn flat_to_multi_index(flat_index: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = Vec::with_capacity(shape.len());
    let mut remaining = flat_index;
    
    for &dim_size in shape.iter().rev() {
        indices.push(remaining % dim_size);
        remaining /= dim_size;
    }
    
    indices.reverse();
    indices
}

/// Convert multi-dimensional indices to flat index
fn multi_to_flat_index(indices: &[usize], shape: &[usize]) -> usize {
    let mut flat_index = 0;
    let mut stride = 1;
    
    for (i, &idx) in indices.iter().enumerate().rev() {
        flat_index += idx * stride;
        stride *= shape[i];
    }
    
    flat_index
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_tensor_creation() {
        let tensor = SparseTensor::<f64>::new(vec![3, 3, 3]);
        assert_eq!(tensor.shape(), &[3, 3, 3]);
        assert_eq!(tensor.nnz(), 0);
    }
    
    #[test]
    fn test_sparse_tensor_insertion() {
        let mut tensor = SparseTensor::<f64>::new(vec![2, 2]);
        tensor.insert(vec![0, 0], 1.0).unwrap();
        tensor.insert(vec![1, 1], 2.0).unwrap();
        
        assert_eq!(tensor.nnz(), 2);
        assert_eq!(tensor.get(&[0, 0]), Some(&1.0));
        assert_eq!(tensor.get(&[1, 1]), Some(&2.0));
        assert_eq!(tensor.get(&[0, 1]), None);
    }
    
    #[test]
    fn test_sparse_tensor_from_dense() {
        let data = vec![1.0, 0.0, 2.0, 0.0];
        let shape = vec![2, 2];
        let tensor = SparseTensor::from_dense(&data, shape).unwrap();
        
        assert_eq!(tensor.nnz(), 2);
        assert_eq!(tensor.get(&[0, 0]), Some(&1.0));
        assert_eq!(tensor.get(&[1, 0]), Some(&2.0));
    }
    
    #[test]
    fn test_tensor_addition() {
        let mut tensor1 = SparseTensor::<f64>::new(vec![2, 2]);
        tensor1.insert(vec![0, 0], 1.0).unwrap();
        tensor1.insert(vec![1, 1], 2.0).unwrap();
        
        let mut tensor2 = SparseTensor::<f64>::new(vec![2, 2]);
        tensor2.insert(vec![0, 0], 3.0).unwrap();
        tensor2.insert(vec![0, 1], 4.0).unwrap();
        
        let result = tensor1.add(&tensor2).unwrap();
        assert_eq!(result.get(&[0, 0]), Some(&4.0)); // 1.0 + 3.0
        assert_eq!(result.get(&[0, 1]), Some(&4.0)); // 0.0 + 4.0
        assert_eq!(result.get(&[1, 1]), Some(&2.0)); // 2.0 + 0.0
    }
    
    #[test]
    fn test_tensor_reshape() {
        let mut tensor = SparseTensor::<f64>::new(vec![2, 3]);
        tensor.insert(vec![0, 0], 1.0).unwrap();
        tensor.insert(vec![1, 2], 2.0).unwrap();
        
        let reshaped = tensor.reshape(vec![3, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.nnz(), 2);
    }
    
    #[test]
    fn test_dense_tensor() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = DenseTensor::from_data(data, shape).unwrap();
        
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.get(&[0, 0]), Some(&1.0));
        assert_eq!(tensor.get(&[1, 1]), Some(&4.0));
    }
    
    #[test]
    fn test_dense_to_sparse_conversion() {
        let data = vec![1.0, 0.0, 0.0, 2.0];
        let shape = vec![2, 2];
        let dense = DenseTensor::from_data(data, shape).unwrap();
        let sparse = dense.to_sparse();
        
        assert_eq!(sparse.nnz(), 2);
        assert_eq!(sparse.get(&[0, 0]), Some(&1.0));
        assert_eq!(sparse.get(&[1, 1]), Some(&2.0));
    }
} 