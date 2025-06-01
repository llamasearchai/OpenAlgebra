//! Medical tensor module for sparse tensor operations

use crate::Result;

/// Medical tensor for sparse 3D/4D imaging data
pub struct MedicalTensor {
    shape: Vec<usize>,
    data: Vec<f64>,
    indices: Vec<Vec<usize>>,
    metadata: Option<MedicalMetadata>,
}

/// Medical metadata for tensors
#[derive(Clone, Debug)]
pub struct MedicalMetadata {
    pub patient_id: String,
    pub modality: String,
    pub voxel_spacing: Vec<f64>,
    pub anatomical_region: String,
}

impl MedicalTensor {
    /// Create a new medical tensor
    pub fn new(shape: Vec<usize>) -> Self {
        let shape_len = shape.len();
        Self {
            shape,
            data: Vec::new(),
            indices: vec![Vec::new(); shape_len],
            metadata: None,
        }
    }

    /// Create sparse tensor from dense data
    pub fn from_dense(dense_data: &[f64], shape: Vec<usize>, threshold: f64) -> Result<Self> {
        let mut tensor = Self::new(shape);
        
        // Convert dense to sparse based on threshold
        for (idx, &value) in dense_data.iter().enumerate() {
            if value.abs() > threshold {
                // Convert linear index to multi-dimensional coordinates
                let coords = tensor.linear_to_coords(idx)?;
                tensor.add_entry(coords, value)?;
            }
        }
        
        Ok(tensor)
    }

    /// Add entry to sparse tensor
    pub fn add_entry(&mut self, coords: Vec<usize>, value: f64) -> Result<()> {
        if coords.len() != self.shape.len() {
            return Err("Coordinate dimensions don't match tensor dimensions".into());
        }

        for (i, &coord) in coords.iter().enumerate() {
            if coord >= self.shape[i] {
                return Err("Index out of bounds".into());
            }
            self.indices[i].push(coord);
        }
        self.data.push(value);

        Ok(())
    }

    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Get density
    pub fn density(&self) -> f64 {
        let total_size: usize = self.shape.iter().product();
        self.nnz() as f64 / total_size as f64
    }

    /// Set medical metadata
    pub fn set_metadata(&mut self, metadata: MedicalMetadata) {
        self.metadata = Some(metadata);
    }

    /// Get medical metadata
    pub fn metadata(&self) -> Option<&MedicalMetadata> {
        self.metadata.as_ref()
    }

    /// Convert linear index to coordinates
    fn linear_to_coords(&self, linear_idx: usize) -> Result<Vec<usize>> {
        let mut coords = Vec::with_capacity(self.shape.len());
        let mut remaining = linear_idx;
        
        for &dim_size in self.shape.iter().rev() {
            coords.push(remaining % dim_size);
            remaining /= dim_size;
        }
        
        coords.reverse();
        Ok(coords)
    }
} 